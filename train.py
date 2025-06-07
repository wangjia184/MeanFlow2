from __future__ import annotations
from datetime import datetime
import math
import os

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms as T, datasets
from torchvision.utils import make_grid, save_image
from ema_pytorch import EMA
from accelerate import Accelerator
from tqdm import tqdm

from network import MFDiT
from meanflow import MeanFlow


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


class MNISTTrainer(nn.Module):
    def __init__(
        self,
        input_size: int = 32,
        batch_size: int = 48,
        lr: float = 1e-4,
        n_steps: int = 12000,
        sample_interval: int = 500,
        checkpoint_every: int = 2000,
        ema_decay: float = 0.999,
        cfg_scale: float = 2.0,
        output_dir: str = '.',
        mixed_precision: str = 'fp16',
        load_checkpoint: str | None = None,
    ):
        super().__init__()
        # accelerator for mixed precision
        self.accel = Accelerator(mixed_precision=mixed_precision)
        self.device = self.accel.device

        # directories
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

        # dataset and dataloader
        transform = T.Compose([T.Resize((input_size, input_size)), T.ToTensor()])
        dataset = datasets.MNIST(root='mnist', train=True, download=True, transform=transform)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
        self.icdl = cycle(dl)

        # model and optimizer
        self.model = MFDiT(
            input_size=input_size,
            patch_size=2,
            in_channels=1,
            dim=384,
            depth=12,
            num_heads=6,
            num_classes=10,
        )
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        # loss & sampling helper
        self.meanflow = MeanFlow(
            channels=1,
            image_size=input_size,
            num_classes=10,
            flow_ratio=0.50,
            time_dist=['lognorm', -0.4, 1.0],
            cfg_ratio=0.10,
            cfg_scale=cfg_scale,
            cfg_uncond='u',
        )

        # prepare model, optimizer, dataloader
        self.model, self.optimizer, dl = self.accel.prepare(self.model, self.optimizer, dl)

        # EMA (wraps model)
        self.ema = EMA(self.model, beta=ema_decay)
        self.ema.ema_model.to(self.device)

        # training settings
        self.n_steps = n_steps
        self.sample_interval = sample_interval
        self.checkpoint_every = checkpoint_every
        self.global_step = 0

        # tensorboard writer
        run_name = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, '../.runs', run_name))

        # optionally load
        if load_checkpoint is not None and self.is_main:
            self.load(load_checkpoint)

    @property
    def is_main(self) -> bool:
        return self.accel.is_main_process

    def save(self, name: str):
        """
        Save model, EMA, optimizer, and global_step
        """
        if not self.is_main:
            return
        package = {
            'model': self.accel.get_state_dict(self.model),
            'ema_model': self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }
        path = os.path.join(self.output_dir, 'checkpoints', name)
        torch.save(package, path)

    def load(self, name: str):
        """
        Load model, EMA, optimizer, and resume global_step
        """
        path = os.path.join(self.output_dir, 'checkpoints', name)
        package = torch.load(path)
        self.model.load_state_dict(package['model'])
        self.optimizer.load_state_dict(package['optimizer'])
        self.ema.load_state_dict(package['ema_model'])
        self.global_step = package.get('global_step', 0)
        print(f"Loaded checkpoint {name}, starting from step {self.global_step}")

    def train(self):
        self.model.train()
        pbar = tqdm(total=self.n_steps, initial=self.global_step, dynamic_ncols=True)
        while self.global_step < self.n_steps:
            x, c = next(self.icdl)
            x = x.to(self.device)
            c = c.to(self.device)

            # forward + loss
            loss, mse_val = self.meanflow.loss(self.model, x, c)

            # backward + step
            self.accel.backward(loss)
            self.optimizer.step()

            # compute grad norm before zeroing
            grad_norm = 0.0
            if self.is_main:
                grads = [p.grad for g in self.optimizer.param_groups for p in g['params'] if p.grad is not None]
                grad_norm = math.sqrt(sum(p.norm(2).item()**2 for p in grads)) if grads else 0.0

            self.optimizer.zero_grad()

            # EMA update
            if self.is_main:
                self.ema.update()

            # step count and logging
            self.global_step += 1
            pbar.update(1)
            if self.is_main:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/MSE', mse_val.item(), self.global_step)
                self.writer.add_scalar('Train/GradNorm', grad_norm, self.global_step)
                pbar.set_postfix(step=self.global_step, loss=f"{loss.item():.4f}", mse=f"{mse_val.item():.4f}", grad_norm=f"{grad_norm:.4f}")

            # sampling
            if self.is_main and self.global_step % self.sample_interval == 0:
                with torch.no_grad():
                    samples = self.meanflow.sample_each_class(self.ema.ema_model, 1)
                grid = make_grid(samples, nrow=10)
                img_path = os.path.join(self.output_dir, 'images', f'step_{self.global_step}.png')
                save_image(grid, img_path)
                self.writer.add_image('Samples', grid, self.global_step)

            # checkpoint
            if self.is_main and self.global_step % self.checkpoint_every == 0:
                self.save(f'checkpoint_{self.global_step}.pt')

        # final save
        if self.is_main:
            self.save(f'final_{self.global_step}.pt')
            self.writer.close()


def main():
    trainer = MNISTTrainer(load_checkpoint='checkpoint_6000.pt')
    trainer.train()


if __name__ == '__main__':
    main()
