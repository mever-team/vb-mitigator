"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import Parameter
import math



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = torch.device("cuda") if features.is_cuda else torch.device("cpu")
        device = features.device
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class EMAGPU:
    def __init__(self, label, device, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.device = device
        self.parameter = torch.zeros(label.size(0), device=device)
        self.updated = torch.zeros(label.size(0), device=device)
        self.num_class = label.max().item() + 1
        self.max_param_per_class = torch.zeros(self.num_class, device=device)

    def update(self, data, index):
        self.parameter[index] = (
            self.alpha * self.parameter[index]
            + (1 - self.alpha * self.updated[index]) * data
        )
        self.updated[index] = 1

        # update max_param_per_class
        batch_size = len(index)
        buffer = torch.zeros(batch_size, self.num_class, device=self.device)
        buffer[range(batch_size), self.label[index]] = self.parameter[index]
        cur_max = buffer.max(dim=0).values
        global_max = torch.maximum(cur_max, self.max_param_per_class)
        label_set_indices = self.label[index].unique()
        self.max_param_per_class[label_set_indices] = global_max[label_set_indices]

    def max_loss(self, label):
        return self.max_param_per_class[label]


class GeneralizedCECriterion(nn.Module):
    def __init__(self, q=0.7, reduction="none"):
        super(GeneralizedCECriterion, self).__init__()
        self.q = q
        self.is_mean_loss = reduction == "mean"

    def forward(self, logits, targets):
        p = torch.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        loss = F.cross_entropy(logits, targets, reduction="none") * loss_weight

        if self.is_mean_loss:
            loss = torch.mean(loss)

        return loss


class BBLoss(nn.Module):
    def __init__(self, confusion_matrix):
        super().__init__()
        self.confusion_matrix = confusion_matrix.cuda()
        self.min_prob = 1e-9

    def forward(self, logits, labels, biases):
        prior = self.confusion_matrix[biases]
        logits += torch.log(prior + self.min_prob)
        label_loss = F.cross_entropy(logits, labels)

        return label_loss

class WBLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.min_prob = 1e-9

    def forward(self, logits, labels, prior):
        # print(logits[0])
        # print(prior[0], torch.log(prior[0] + self.min_prob))
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        prior[prior>0.2] = 1
        logits[batch_indices, labels] += torch.log(prior + self.min_prob)
        # print(logits[0])
        label_loss = F.cross_entropy(logits, labels)

        return label_loss


class pattern_norm(torch.nn.Module):
    def __init__(self, scale=1.0):
        super(pattern_norm, self).__init__()
        self.scale = scale

    def forward(self, input):
        sizes = input.size()
        if len(sizes) > 2:
            input = input.view(-1, np.prod(sizes[1:]))
            input = torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12)
            input = input.view(sizes)
        return input


class Hook:
    def __init__(self, module, backward=False):
        self.module = module
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


# For each discriminatory class, orthogonalize samples
def abs_orthogonal_blind(output, gram, target_labels, bias_labels):
    bias_classes = torch.unique(bias_labels)
    orthogonal_loss = torch.tensor(0.0).to(output.device)
    M_tot = 0.0

    for bias_class in bias_classes:
        bias_mask = (bias_labels == bias_class).type(torch.float).unsqueeze(dim=1)
        bias_mask = torch.tril(
            torch.mm(bias_mask, torch.transpose(bias_mask, 0, 1)), diagonal=-1
        )
        M = bias_mask.sum()
        M_tot += M

        if M > 0:
            orthogonal_loss += torch.abs(torch.sum(gram * bias_mask))

    if M_tot > 0:
        orthogonal_loss /= M_tot
    return orthogonal_loss


# For each target class, parallelize samples belonging to
# different discriminatory classes
def abs_parallel(gram, target_labels, bias_labels):
    target_classes = torch.unique(target_labels)
    bias_classes = torch.unique(bias_labels)
    parallel_loss = torch.tensor(0.0).to(gram.device)
    M_tot = 0.0

    for target_class in target_classes:
        class_mask = (target_labels == target_class).type(torch.float).unsqueeze(dim=1)

        for idx, bias_class in enumerate(bias_classes):
            bias_mask = (bias_labels == bias_class).type(torch.float).unsqueeze(dim=1)

            for other_bias_class in bias_classes[idx:]:
                if other_bias_class == bias_class:
                    continue

                other_bias_mask = (
                    (bias_labels == other_bias_class).type(torch.float).unsqueeze(dim=1)
                )
                mask = torch.tril(
                    torch.mm(
                        class_mask * bias_mask,
                        torch.transpose(class_mask * other_bias_mask, 0, 1),
                    ),
                    diagonal=-1,
                )
                M = mask.sum()
                M_tot += M

                if M > 0:
                    parallel_loss -= torch.sum((1.0 + gram) * mask * 0.5)

    if M_tot > 0:
        parallel_loss = 1.0 + (parallel_loss / M_tot)

    return parallel_loss


def abs_regu(hook, target_labels, bias_labels, alpha=1.0, beta=1.0, sum=True):
    D = hook.output
    if len(D.size()) > 2:
        D = D.view(-1, np.prod((D.size()[1:])))

    gram_matrix = torch.tril(torch.mm(D, torch.transpose(D, 0, 1)), diagonal=-1)
    # not really needed, just for safety for approximate repr
    gram_matrix = torch.clamp(gram_matrix, -1, 1.0)

    zero = torch.tensor(0.0).to(target_labels.device)
    R_ortho = (
        abs_orthogonal_blind(D, gram_matrix, target_labels, bias_labels)
        if alpha != 0
        else zero
    )
    R_parallel = (
        abs_parallel(gram_matrix, target_labels, bias_labels) if beta != 0 else zero
    )

    if sum:
        return alpha * R_ortho + beta * R_parallel
    return alpha * R_ortho, beta * R_parallel


class EnDLoss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.min_prob = 1e-9

        print(f"EnDLoss - alpha: {alpha} beta: {beta}")

    def forward(self, logits, labels, biases, feats):
        label_loss = F.cross_entropy(logits, labels)

        gram_matrix = torch.tril(
            torch.mm(feats, torch.transpose(feats, 0, 1)), diagonal=-1
        )
        # not really needed, just for safety for approximate repr
        gram_matrix = torch.clamp(gram_matrix, -1, 1.0)

        R_ortho = abs_orthogonal_blind(feats, gram_matrix, labels, biases)
        R_parallel = abs_parallel(gram_matrix, labels, biases)

        bias_loss = self.alpha * R_ortho + self.beta * R_parallel

        return label_loss, bias_loss





def kd_loss(logits_student, logits_teacher, temperature=2):
    batch_size = logits_student.shape[0]
    
    # Convert targets to one-hot encoding
    targets_one_hot = F.one_hot(logits_teacher, num_classes=2).float()
    
    # Apply soft labels (0.6 for correct class, 0.4 for incorrect class)
    soft_targets = 0.3 + (0.4 * targets_one_hot)

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(soft_targets / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class SoftLabelLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature


    def forward(self, logits, targets, **kwargs):
        loss_kd =  kd_loss(
            logits, targets, self.temperature
        )

        return loss_kd
    

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T=2):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class SubcenterArcMarginProduct(nn.Module):
    r"""Modified implementation from https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py#L10
        """
    def __init__(self, in_features, out_features, K=3, s=30.0, m=0.50, easy_margin=False):
        super(SubcenterArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.K = K
        self.weight = Parameter(torch.FloatTensor(out_features*self.K, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        if self.K > 1:
            cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
            subcenter = cosine.argmax(dim=2)
            cosine, _ = torch.max(cosine, axis=2)
        
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        #cos(phi+m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
    
    def predict(self, input, label):
        m = 0.0
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        th = math.cos(math.pi - m)
        mm = math.sin(math.pi - m) * m
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        if self.K > 1:
            cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
            subcenter = cosine.argmax(dim=2)
            cosine, _ = torch.max(cosine, axis=2)
        
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        #cos(phi+m)
        phi = cosine * cos_m - sine * sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > th, phi, cosine - mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
    
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=-0.2, m_b=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        # m_b = 0.5
        self.cos_m_b = math.cos(m_b)
        self.sin_m_b = math.sin(m_b)
        self.th_b = math.cos(math.pi - m_b)
        self.mm_b = math.sin(math.pi - m_b) * m_b

    def forward_def(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    # def forward_def(self, input, targets):
    #     """
    #     input: shape [B, D] — feature vectors
    #     weight: shape [C, D] — class weights (should be L2-normalized)

    #     Returns:
    #         Loss that encourages any class prediction to exceed the margin threshold
    #     """
    #     # Normalize inputs and weights
    #     input_norm = F.normalize(input, dim=1)
    #     weight_norm = F.normalize(self.weight, dim=1)

    #     # Cosine similarity between input and all class weights
    #     cosine = F.linear(input_norm, weight_norm)  # shape: [B, C]

    #     # Max confidence per sample
    #     max_cosine, _ = torch.max(cosine, dim=1)  # shape: [B]
    #     # print(max_cosine)
    #     # Penalize samples with max cosine < margin
    #     loss = F.relu(self.cos_m - max_cosine)  # encourages high max cosine


    #     probs = F.softmax(cosine.detach(), dim=1)  # [B, C]
    #     mean_probs = probs.mean(dim=0)  # [C]
    #     entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))  # scalar
    #     max_entropy = torch.log(torch.tensor(probs.shape[1], device=probs.device))  # log(C)
    #     diversity_loss = (max_entropy - entropy)  # encourage entropy to be high


    #     return loss, diversity_loss


    def forward(self, input, label, margins_per_sample):
        # Normalize input and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # [batch_size, num_classes]
        
        # Create one-hot encoding of labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # Gather cosine values at the label positions (cos(θ_yi))
        cosine_theta_yi = (cosine * one_hot).sum(dim=1)  # [batch_size]

        # Compute per-sample sine values
        sine_theta_yi = torch.sqrt((1.0 - cosine_theta_yi**2).clamp(0, 1))  # [batch_size]

        # Compute cos(m) and sin(m) for each sample
        cos_m = torch.cos(margins_per_sample)  # [batch_size]
        sin_m = torch.sin(margins_per_sample)  # [batch_size]

        # Compute phi = cos(θ + m) = cosθ * cosm - sinθ * sinm
        phi_theta_yi = cosine_theta_yi * cos_m - sine_theta_yi * sin_m  # [batch_size]

        # Optionally apply easy margin logic
        if self.easy_margin:
            phi_theta_yi = torch.where(cosine_theta_yi > 0, phi_theta_yi, cosine_theta_yi)
        else:
            th = torch.cos(math.pi - margins_per_sample)
            mm = torch.sin(math.pi - margins_per_sample) * margins_per_sample
            phi_theta_yi = torch.where(cosine_theta_yi > th, phi_theta_yi, cosine_theta_yi - mm)

        # Replace the logits at label positions with phi
        output = cosine.clone()
        output.scatter_(1, label.view(-1, 1), phi_theta_yi.view(-1, 1))

        # Scale output
        output *= self.s

        return output

    def predict(self, input, label):
        m=0
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        th = math.cos(math.pi - m)
        mm = math.sin(math.pi - m) * m

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * cos_m - sine * sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > th, phi, cosine - mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    from https://github.com/CoinCheung/pytorch-loss/blob/master/label_smooth.py
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss
    


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


eps = 1e-7

class SCELoss(nn.Module):
    def __init__(self, num_classes=10, a=1, b=1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce.mean()
        return loss


class RCELoss(nn.Module):
    def __init__(self, num_classes=10, scale=1.0):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        loss = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * loss.mean()

class NRCELoss(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NRCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        norm = 1 / 4 * (self.num_classes - 1)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * norm * rce.mean()


class NCELoss(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * loss.mean()

class MAELoss(nn.Module):
    def __init__(self, num_classes=10, scale=2.0):
        super(MAELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * loss.mean()

class NMAE(nn.Module):
    def __init__(self, num_classes=10, scale=1.0):
        super(NMAE, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        norm = 1 / (self.num_classes - 1)
        loss = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * norm * loss.mean()

class GCELoss(nn.Module):
    def __init__(self, num_classes=10, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()

class NGCELoss(nn.Module):
    def __init__(self, num_classes=10, q=0.7, scale=1.0):
        super(NGCELoss, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        loss = numerators / denominators
        return self.scale * loss.mean()

class AGCELoss(nn.Module):
    def __init__(self, num_classes=10, a=1, q=2, eps=eps, scale=1.):
        super(AGCELoss, self).__init__()
        self.a = a
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = ((self.a+1)**self.q - torch.pow(self.a + torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean() * self.scale

class AUELoss(nn.Module):
    def __init__(self, num_classes=10, a=1.5, q=0.9, eps=eps, scale=1.0):
        super(AUELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.q = q
        self.eps = eps
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (torch.pow(self.a - torch.sum(label_one_hot * pred, dim=1), self.q) - (self.a-1)**self.q)/ self.q
        return loss.mean() * self.scale

class ANormLoss(nn.Module):
    def __init__(self, num_classes=10, a=1.5, p=0.9, scale=1.0):
        super(ANormLoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.p = p
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-5, max=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = torch.sum(torch.pow(torch.abs(self.a * label_one_hot-pred), self.p), dim=1) - (self.a-1)**self.p
        return loss.mean() * self.scale / self.p


class AExpLoss(torch.nn.Module):
    def __init__(self, num_classes=10, a=3, scale=1.0):
        super(AExpLoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = torch.exp(-torch.sum(label_one_hot * pred, dim=1) / self.a)
        return loss.mean() * self.scale

class NCEandRCE(nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.rce = RCELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)

class NCEandMAE(nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10):
        super(NCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.mae = MAELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.mae(pred, labels)

class NLNL(torch.nn.Module):
    def __init__(self, train_loader, num_classes=10, ln_neg=1):
        super(NLNL, self).__init__()
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.
        if not hasattr(train_loader.dataset, 'targets'):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (torch.from_numpy(np.array(train_loader.dataset.targets)) == i).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight.cuda()
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        labels_neg = (labels.unsqueeze(-1).repeat(1, self.ln_neg)
                      + torch.LongTensor(len(labels), self.ln_neg).cuda().random_(1, self.num_classes)) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes-1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(labels)*self.ln_neg

        s_neg = torch.log(torch.clamp(1. - F.softmax(pred, 1), min=1e-5, max=1.))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size()).cuda()
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
        loss = ((loss+loss_neg) / (float((labels >= 0).sum())+float((labels_neg[:, 0] >= 0).sum())))
        return loss

class FocalLoss(torch.nn.Module):
    '''
        https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    '''

    def __init__(self, gamma=0.5, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, gamma=0.5, num_classes=10, alpha=None, size_average=True, scale=1.0):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class NFLandRCE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10, gamma=0.5):
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(gamma=gamma, num_classes=num_classes, scale=alpha)
        self.rce = RCELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.rce(pred, labels)


class NFLandMAE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10, gamma=0.5):
        super(NFLandMAE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(gamma=gamma, num_classes=num_classes, scale=alpha)
        self.mae = MAELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.mae(pred, labels)


class NCEandAGCE(torch.nn.Module):
    def __init__(self, alpha=1., beta = 1., num_classes=10, a=3, q=1.5):
        super(NCEandAGCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.agce = AGCELoss(num_classes=num_classes, a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.agce(pred, labels)


class NCEandAUE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10, a=6, q=1.5):
        super(NCEandAUE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.aue = AUELoss(num_classes=num_classes, a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.aue(pred, labels)

class NCEandAEL(torch.nn.Module):
    def __init__(self, alpha=1., beta=4., num_classes=10, a=2.5):
        super(NCEandAEL, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.aue = AExpLoss(num_classes=num_classes, a=a, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.aue(pred, labels)


class NFLandAGCE(torch.nn.Module):
    def __init__(self, alpha=1., beta = 1., num_classes=10, a=3, q=2):
        super(NFLandAGCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedFocalLoss(num_classes=num_classes, scale=alpha)
        self.agce = AGCELoss(num_classes=num_classes, a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.agce(pred, labels)


class NFLandAUE(torch.nn.Module):
    def __init__(self, alpha=1., beta = 1., num_classes=10, a=1.5, q=0.9):
        super(NFLandAUE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedFocalLoss(num_classes=num_classes, scale=alpha)
        self.aue = AUELoss(num_classes=num_classes, a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.aue(pred, labels)


class NFLandAEL(torch.nn.Module):
    def __init__(self, alpha=1., beta = 1., num_classes=10, a=3):
        super(NFLandAEL, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedFocalLoss(num_classes=num_classes, scale=alpha)
        self.ael = AExpLoss(num_classes=num_classes, a=a, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.ael(pred, labels)

class ANLandRCE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10, a=2, p=0.7):
        super(ANLandRCE, self).__init__()
        self.num_classes = num_classes
        self.anl = ANormLoss(num_classes=num_classes, a=a, p=p, scale=alpha)
        self.rce = RCELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.anl(pred, labels) + self.rce(pred, labels)

class NCEandANL(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10, a=2, p=0.7):
        super(NCEandANL, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.anl = ANormLoss(num_classes=num_classes, a=a, p=p, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.anl(pred, labels)
    


class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
    


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, issame):
        output1 = F.normalize(output1, dim=1)
        output2 = F.normalize(output2, dim=1)
        euclidean_distance = F.pairwise_distance(output1, output2)
        # pos = issame * torch.pow(euclidean_distance, 2)
        # neg = (1-issame) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        # loss_contrastive = torch.mean( pos + neg )
        # return loss_contrastive
        # Compute pairwise distances
        distances = F.pairwise_distance(output1, output2, p=2)

        # Identify positive (same target) and negative (different target) pairs
        positive_mask = issame.float()
        negative_mask = 1.0 - positive_mask

        # Loss components
        positive_loss = (distances ** 2) * positive_mask
        negative_loss = (F.relu(self.margin - distances) ** 2) * negative_mask

        # Combine and average
        loss = (positive_loss + negative_loss).mean()
        return loss