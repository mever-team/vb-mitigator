"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
