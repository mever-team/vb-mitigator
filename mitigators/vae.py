import torch
import numpy as np
from .base_trainer import BaseTrainer
from models.builder import get_model, get_bcc
import torch.nn.functional as F
from models.beta_vae import BetaVAE
from models.classification_head import Classifier
# Use logistic regression to test informativeness of each latent dim
from sklearn.linear_model import LogisticRegression
import numpy as np
import torchvision.utils as vutils
import os

class VAETrainer(BaseTrainer):

    def _setup_models(self):
        super()._setup_models()
        self.vae = BetaVAE(beta=40)
        self.vae.to(self.device)
        return

    def _setup_optimizer(self):
        super()._setup_optimizer()
        parameters = [p for p in self.vae.parameters() if p.requires_grad]
        self.optimizer_vae = torch.optim.Adam(
            parameters,
            lr=1e-4,
            betas=(0.9, 0.999)
        )
    
    def _method_specific_setups(self):
        self.train_vae(self.dataloaders["train"])

        # Probe digit-relevant latents
        self.useful_dims = self.probe_latents(self.dataloaders["train"])

        # Train classifier
        self.model.set_input_dim(input_dim=len(self.useful_dims)) 
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self._setup_scheduler()

    

    def save_reconstructions(self, dataloader, epoch, output_dir="reconstructions", num_images=8):
        self.vae.eval()
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for batch in dataloader:
                x = batch["inputs"].to(self.device)
                x = x.to(self.device)
                x_recon, _, _ = self.vae(x)
                break  # just take the first batch

        # Denormalize if necessary (e.g., if you used Normalize in transform)
        x = x[:num_images].detach().cpu()
        x_recon = x_recon[:num_images].detach().cpu()

        # Save original images
        vutils.save_image(x, os.path.join(output_dir, f"epoch_{epoch}_original.png"), nrow=num_images, normalize=True)
        # Save reconstructions
        vutils.save_image(x_recon, os.path.join(output_dir, f"epoch_{epoch}_recon.png"), nrow=num_images, normalize=True)

        print(f"[✓] Saved reconstructions to {output_dir}/epoch_{epoch}_*.png")
        
    def train_vae(self, dataloader, epochs=100):
        self.vae.train()
        final_beta = 1.0

        for epoch in range(epochs):
            current_beta = 10 # min(final_beta, final_beta * (epoch+1) / epochs)
            self.vae.beta = current_beta
            total_loss = 0
            recon_loss = 0
            kl_loss = 0
            for batch in dataloader:
                x = batch["inputs"].to(self.device)
                x_recon, mu, logvar = self.vae(x)
                loss, recon, kl = self.vae.loss_function(x, x_recon, mu, logvar)

                self.optimizer_vae.zero_grad()
                loss.backward()
                self.optimizer_vae.step()
                total_loss += loss.item()
                recon_loss += recon.item()
                kl_loss += kl.item()
            print(f"[Epoch {epoch+1}] VAE Loss: {total_loss / len(dataloader):.4f}, Recon Loss: {recon_loss / len(dataloader):.4f}, KL Loss: {kl_loss / len(dataloader):.4f}")
            self.save_reconstructions(dataloader=self.dataloaders["train"], epoch=epoch)
            self.vae.train()
        self.vae.eval()

    def probe_latents(self, dataloader):
        self.vae.eval()
        zs, labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                x = batch["inputs"].to(self.device)
                y = batch["targets"].to(self.device)
                mu = self.vae.encode_mu(x)
                zs.append(mu.cpu())
                labels.append(y)

        zs = torch.cat(zs, dim=0)
        labels = torch.cat(labels, dim=0)



        accs = []
        for i in range(zs.shape[1]):
            clf = LogisticRegression(max_iter=1000)
            acc = clf.fit(zs[:, i:i+1].detach().cpu().numpy(), labels.detach().cpu().numpy()).score(zs[:, i:i+1].detach().cpu().numpy(), labels.detach().cpu().numpy())
            accs.append(acc)
        print(accs)
        useful_dims = [i for i, acc in enumerate(accs) if acc < 0.3]  # threshold
        print(f"Digit-relevant dimensions: {useful_dims}")
        return useful_dims

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        with torch.no_grad():
            mu = self.vae.encode_mu(inputs)
            z = mu[:, self.useful_dims]

        outputs = self.model(z)

        loss = self.criterion(
            outputs, targets
        )
        self.optimizer.zero_grad()
        self._loss_backward(loss)
        self._optimizer_step()
        return {"train_cls_loss": loss}

    def _val_iter(self, batch):
        batch_dict = {}
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        with torch.no_grad():
            mu = self.vae.encode_mu(inputs)
            z = mu[:, self.useful_dims]

        outputs = self.model(z)

        loss = self.criterion(outputs, targets)
        batch_dict["predictions"] = torch.argmax(outputs, dim=1)
        batch_dict["targets"] = batch["targets"]
        for b in self.biases:
            batch_dict[b] = batch[b]
        return batch_dict, loss