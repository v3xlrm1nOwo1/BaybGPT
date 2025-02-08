# Libraries
import os
import json
import math
import wandb
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

# Local files
"""
 - None
"""


logger = logging.getLogger(__name__)


class TrainerConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# learning rate decay scheduler (cosine with warmup)
class CosineWithWarmupLR(_LRScheduler):
    def __init__(self, optimizer, total_steps=19073, warmup_steps=715, min_lr=0, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.cosine_decay_steps = total_steps - warmup_steps
        super(CosineWithWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # 1) linear warmup for warmup_iters steps
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        
        # 2) if last_epoch > lr_decay_iters, return min learning rate
        elif self.last_epoch > self.total_steps:
            # After the total steps, return the minimum learning rate
            return [self.min_lr for _ in self.base_lrs]
        
        # 3) in between, use cosine decay down to min learning rate
        else:
            # Cosine decay phase
            elapsed_steps = self.last_epoch - self.warmup_steps
            decay_ratio = elapsed_steps / self.cosine_decay_steps
            return [
                self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * decay_ratio))
                for base_lr in self.base_lrs
            ]


class Trainer:
    def __init__(self, model, trainset, evalset, train_config):
        self.model = model
        self.trainset = trainset
        self.evalset = evalset
        self.config = train_config
        
        wandb.init(project="BaybGPT-runs", name=f"BaybGPT_model_\
                   {train_config.max_epoch}_epoch_{train_config.train_batch_size}_batch_{train_config.learning_rate:.0e}_LR\
                    _{train_config.seed}_seed")

        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        self.device = "cuda" if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

    def save_checkpoints(self, ckpt_id):
        model = self.model
        ckpt_folder = self.config.ckpt_path
        torch.save(model.state_dict(), f"{ckpt_folder}/{ckpt_id}.pth")    

    def generate_text(self, model, num_tokens, device):
        idx = torch.zeros((1,1), dtype=torch.long).to(self.device)
        token_ids = model.generate(idx=idx, max_new_tokens=num_tokens, device=device)
        text = self.config.tokenizer.decode(token_ids.squeeze())
        return text

    def train(self):
        config = self.config
        model = self.model
        optimizer = model.configure_optimizers(device=self.device, learning_rate=self.config.learning_rate)

        lr_steps = int(len(self.trainset) / config.train_batch_size * config.max_epoch)
        scheduler = CosineWithWarmupLR(
            optimizer=optimizer, 
            total_steps=lr_steps, 
            warmup_steps=int(0.1*lr_steps), 
            min_lr=config.learning_rate * 0.1,
            last_epoch=-1,
        )

        def train_loop(train_dataloader, epoch_idx=1):
            model.train()
            losses = []
            for itr, (input, targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="TrainBaybGPT"):
                input = input.to(device=self.device)
                targets = targets.to(device=self.device)

                optimizer.zero_grad()

                _, loss = model(idx=input, targets=targets)
                
                losses.append(loss.item())

                loss.backward()

                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                train_metrics = {
                    "train/train_itr": itr, 
                    "train/train_loss": loss.item(), 
                    "norm": norm, 
                    "train/train_lr": scheduler.get_last_lr()[0]
                }
                wandb.log(train_metrics)

                if itr%1000 == 0:
                    generated_text = self.generate_text(model, num_tokens=config.num_generated_tokens, device=self.device)
                    if config.show_generated_text:
                        print(f"\n{generated_text}\n")
                    state_generated_text = {
                            "epoch": epoch_idx,
                            "model": "BaybGPT",
                            "generated_text": generated_text,
                            "train_itr": itr
                        }
                    try:
                        if os.path.exists(config.generatation_save_path):
                            with open(config.generatation_save_path, 'r') as file:
                                data = json.load(file)
                                data.append(state_generated_text)
                        else:
                            data = [state_generated_text]

                        with open(file=config.generatation_save_path, mode="w", encoding="utf-8") as file:
                            json.dump(data, file, indent=4) 

                    except IOError as e:
                        print(f"Error writing to JSON file: {e}")

            return float(np.mean(losses))        

        def eval_loop(eval_dataloader):
            model.eval()
            losses = []
            for itr, (input, targets) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="EvalBaybGPT"):

                input = input.to(device=self.device)
                targets = targets.to(device=self.device)
                with torch.no_grad():
                    _, loss = model(idx=input, targets=targets)

                losses.append(loss.item())
                val_metrics = {
                    "val/val_itr": itr, 
                    "val/val_loss": loss
                }

                wandb.log(val_metrics)

            return float(np.mean(losses))

        train_dataloader = DataLoader(
            self.trainset,
            batch_size=config.train_batch_size,
            num_workers=config.num_workers,
            drop_last=True,
            shuffle=True,
        )

        eval_dataloader = DataLoader(
            self.evalset,
            batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            drop_last=True,
            shuffle=False,
        )

        best_loss = float("inf")
        for epoch in range(config.max_epoch):
            epoch_idx = (epoch + 1)
            logger.info(f"===============Epoch: [{epoch_idx}/{config.max_epoch}]===============")
            train_loss = train_loop(train_dataloader, epoch_idx=epoch_idx)
            eval_loss = eval_loop(eval_dataloader)

            print(f"Train Loos: [{train_loss}] - Eval Loss: [{eval_loss}]")

            goodModel = eval_loss < best_loss
            if config.ckpt_path is not None and goodModel:
                best_loss = eval_loss
                self.save_checkpoints(f"{config.max_epoch}epoch_best_model_{config.train_batch_size}batch_{config.learning_rate:.0e}LR_{config.seed}Seed_{train_loss:.4f}train_loss_{eval_loss:.4f}eval_loss")
        
        self.save_checkpoints(f"{config.max_epoch}epoch_last_model_{config.train_batch_size}batch_{config.learning_rate:.0e}LR_{config.seed}Seed_{train_loss:.4f}train_loss_{eval_loss:.4f}eval_loss")
        wandb.finish()
