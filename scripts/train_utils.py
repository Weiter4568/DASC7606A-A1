import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import amp
from tqdm import tqdm
import time
import math
import torch.nn.functional as F
import numpy as np

class EMA:
    """
    Exponential Moving Average for both trainable parameters and floating buffers (e.g., BN running stats).
    - 兼容 DataParallel/DDP（自动走 model.module）
    - update() 放在 optimizer.step 之后
    - apply_to()/restore() 同时交换 params + buffers
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow_params = {}
        self.shadow_buffers = {}
        self.backup_params = {}
        self.backup_buffers = {}
        self.num_updates = 0

        tracked = model.module if hasattr(model, "module") else model

        # 跟踪需梯度的参数
        for n, p in tracked.named_parameters():
            if p.requires_grad:
                self.shadow_params[n] = p.detach().clone()

        # 跟踪浮点 buffers（BN running_mean/var 等）
        for n, b in tracked.named_buffers():
            if torch.is_floating_point(b):
                self.shadow_buffers[n] = b.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        tracked = model.module if hasattr(model, "module") else model
        d = self.decay
        self.num_updates += 1

        for n, p in tracked.named_parameters():
            if p.requires_grad and n in self.shadow_params:
                self.shadow_params[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

        for n, b in tracked.named_buffers():
            if torch.is_floating_point(b) and n in self.shadow_buffers:
                self.shadow_buffers[n].mul_(d).add_(b.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        tracked = model.module if hasattr(model, "module") else model
        self.backup_params = {}
        self.backup_buffers = {}

        # swap params
        for n, p in tracked.named_parameters():
            if p.requires_grad and n in self.shadow_params:
                self.backup_params[n] = p.detach().clone()
                p.data.copy_(self.shadow_params[n])

        # swap buffers
        for n, b in tracked.named_buffers():
            if torch.is_floating_point(b) and n in self.shadow_buffers:
                self.backup_buffers[n] = b.detach().clone()
                b.data.copy_(self.shadow_buffers[n])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        tracked = model.module if hasattr(model, "module") else model

        for n, p in tracked.named_parameters():
            if p.requires_grad and n in self.backup_params:
                p.data.copy_(self.backup_params[n])

        for n, b in tracked.named_buffers():
            if torch.is_floating_point(b) and n in self.backup_buffers:
                b.data.copy_(self.backup_buffers[n])

        self.backup_params.clear()
        self.backup_buffers.clear()

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": {k: v.cpu() for k, v in self.shadow_params.items()},
            "shadow_buffers": {k: v.cpu() for k, v in self.shadow_buffers.items()},
        }

    def load_state_dict(self, state, device=None):
        self.decay = state.get("decay", self.decay)
        self.num_updates = state.get("num_updates", 0)
        dev = device or "cpu"
        self.shadow_params  = {k: v.to(dev) for k, v in state["shadow_params"].items()}
        self.shadow_buffers = {k: v.to(dev) for k, v in state["shadow_buffers"].items()}

# ---- Replace your SAM class with this wrapper (no inheritance) ----
class SAM:
    """
    Sharpness-Aware Minimization wrapper (composition).
    - Holds a base optimizer (e.g., SGD).
    - Provides first_step/second_step for two-step update.
    - Proxies state_dict()/load_state_dict() to base optimizer so main can save/load.
    """
    def __init__(self, params, base_optimizer, rho=0.05, **defaults):
        self.rho = rho
        # instantiate the real optimizer
        self.base_optimizer = base_optimizer(params, **defaults)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state  # for compatibility

    @torch.no_grad()
    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm(p=2).to(device))
        return torch.norm(torch.stack(norms), p=2) if norms else torch.tensor(0., device=device)

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        scale = self._grad_norm()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * (self.rho / (scale + 1e-12))
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.base_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.base_optimizer.zero_grad(set_to_none=True)

    # ---- passthrough helpers so main/scheduler work as usual ----
    def zero_grad(self):
        self.base_optimizer.zero_grad(set_to_none=True)

    # DO NOT expose step(); we use first_step/second_step
    def step(self, *args, **kwargs):
        raise NotImplementedError("Use first_step/second_step with SAM.")

    # crucial: allow checkpointing
    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


def mixup_cutmix_collate(batch, alpha=1.0, mixup_prob=0.5, cutmix_prob=0.5, num_classes=100):
    """Batch-level collate_fn implementing Mixup and CutMix."""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)

    onehot = torch.zeros(len(labels), num_classes, dtype=torch.float)
    onehot.scatter_(1, labels.view(-1, 1), 1.0)

    # 随机决定本 batch 是否进行混合
    r = np.random.rand()
    if r < mixup_prob:
        # ---- Mixup ----
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(len(imgs))
        mixed_imgs = lam * imgs + (1 - lam) * imgs[index]
        mixed_labels = lam * onehot + (1 - lam) * onehot[index]
    elif r < mixup_prob + cutmix_prob:
        # ---- CutMix ----
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(len(imgs))
        bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
        mixed_imgs = imgs.clone()
        mixed_imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2)))
        mixed_labels = lam * onehot + (1 - lam) * onehot[index]
    else:
        # ---- No mix ----
        mixed_imgs, mixed_labels = imgs, onehot

    return mixed_imgs, mixed_labels

def rand_bbox(size, lam):
    """Generate random rectangle bbox for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def load_transforms(train: bool = False, 
                    use_randaugment: bool = False,
                    image_size: int = 32):
    """
    Load data transformations for CIFAR-100.
    Args:
        train (bool): True for training transform (random augmentations), 
                      False for validation/test transform.
        use_randaugment (bool): Whether to use RandAugment for stronger augmentation.
        image_size (int): target image size (default 32 for CIFAR)
    Returns:
        torchvision.transforms.Compose
    """
    if train:
        transform_list = [
            transforms.RandomCrop(image_size, padding=4),     # 随机裁剪 + 填充
            transforms.RandomHorizontalFlip(),                # 随机水平翻转
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),       # 轻微颜色扰动
        ]
        # 可选的更强随机增强（优先使用 torchvision.transforms.RandAugment；退而用 AutoAugment）
        if use_randaugment:
            try:
                from torchvision.transforms import RandAugment
                transform_list.append(RandAugment(num_ops=3, magnitude=10))
            except Exception:
                try:
                    from torchvision.transforms import AutoAugment, AutoAugmentPolicy
                    transform_list.append(AutoAugment(AutoAugmentPolicy.CIFAR10))
                except Exception:
                    pass

        # 基本归一化
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ]
        # RandomErasing 需在张量域执行，放在 Normalize 之后
        if use_randaugment:
            transform_list.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'))
        return transforms.Compose(transform_list)
    else:
        # 验证/测试集使用确定性变换
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])

def load_data(data_dir, batch_size, num_workers: int = 8, mixup_alpha: float = 0.8, mixup_prob: float = 1.0, cutmix_prob: float = 0.0):
    """
    Load the data from the data directory and split it into training and validation sets
    This function is similar to the cell 2. Data Preparation in 04_model_training.ipynb

    Args:
        data_dir: The directory to load the data from
        batch_size: The batch size to use for the data loaders
    Returns:
        train_loader: The training data loader
        val_loader: The validation data loader
    """

    # Load the train dataset from the augmented data directory
    train_dataset = datasets.ImageFolder(root=data_dir + "/train", transform=load_transforms(train=True, use_randaugment=True))

    # Load the validation dataset from the raw data directory
    val_dataset = datasets.ImageFolder(root=data_dir + "/val", transform=load_transforms(train=False))

    # Create data loaders for training and validation
    from scripts.train_utils import mixup_cutmix_collate  # 导入上面的函数

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda b: mixup_cutmix_collate(
            b,
            alpha=mixup_alpha,
            mixup_prob=mixup_prob,
            cutmix_prob=cutmix_prob,
            num_classes=len(train_dataset.classes)
        )
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(256, batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Print dataset summary
    print(f"Dataset loaded from: {data_dir}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class names: {train_dataset.classes}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    return train_loader, val_loader


def define_loss_and_optimizer(model: nn.Module,
                              lr: float,
                              weight_decay: float,
                              optimizer_type: str = "sgd",
                              label_smoothing: float = 0.1,
                              epochs: int = 300,
                              warmup: int = 5):
    # Loss：配合 Mixup/CutMix 建议开 LS
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # param groups：BN/bias 不做 weight decay（稳增益）
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.dim() > 1 and "bn" not in n.lower() and not n.endswith("bias"):
            decay.append(p)
        else:
            no_decay.append(p)

    # 基础优化器（被 SAM 包裹）
    def _sgd(params, **kw):
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, **kw)

    optimizer = SAM(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        base_optimizer=_sgd,
        rho=0.05   # 关键超参：推荐 0.05（可在 0.03~0.1 微调）
    )

    # Warmup + Cosine（绑定 base_optimizer）
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        t = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer.base_optimizer, lr_lambda=lr_lambda)

    return criterion, optimizer, scheduler


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(-soft_targets * F.log_softmax(logits, dim=1), dim=1))

def train_epoch(model,
                dataloader,
                criterion,
                optimizer,
                device,
                use_amp: bool = True,
                scaler=None,           # 从 main 传进来的 GradScaler
                log_interval: int = 0):
    """
    Train one epoch. 支持：
      - Mixup/CutMix 软标签
      - AMP (torch.amp.autocast) + 外部传入的 scaler
      - SAM 两步更新
      - EMA（每步更新）
    返回: (epoch_loss, epoch_acc)
    """
    if use_amp and scaler is None:
        raise ValueError("use_amp=True 但未传入 GradScaler，请从 main 传入 scaler 或将 use_amp=False。")

    model.train()
    running_loss, correct, total = 0.0, 0, 0
    start = time.time()

    for step, (inputs, labels) in enumerate(dataloader, start=1):
        inputs = inputs.to(device, non_blocking=True)

        # 判定是否为软标签（来自 mixup/cutmix 的 one-hot/soft）
        use_soft = isinstance(labels, torch.Tensor) and labels.ndim > 1
        if use_soft:
            labels_soft = labels.to(device, non_blocking=True).float()          # for loss
            labels_hard = labels.argmax(dim=1).to(device, non_blocking=True)    # for acc
        else:
            labels_hard = labels.to(device, non_blocking=True)                  # for acc & loss

        optimizer.zero_grad()

        # ===== SAM 分支 =====
        if hasattr(optimizer, "first_step") and hasattr(optimizer, "second_step"):
            if use_amp:
                # 第一次前向/反向（不 unscale_）
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out1 = model(inputs)
                    loss1 = soft_cross_entropy(out1, labels_soft) if use_soft else criterion(out1, labels_hard)
                scaler.scale(loss1).backward()
                optimizer.first_step(zero_grad=True)

                # 第二次前向/反向（在 second_step 前 unscale_ 一次）
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out2 = model(inputs)
                    loss2 = soft_cross_entropy(out2, labels_soft) if use_soft else criterion(out2, labels_hard)
                scaler.scale(loss2).backward()
                scaler.unscale_(optimizer.base_optimizer)
                # (如需梯度裁剪，请在此处执行 clip)
                optimizer.second_step(zero_grad=True)
                scaler.update()

                outputs_for_metrics = out2
                loss_for_metrics = loss2
            else:
                out1 = model(inputs)
                loss1 = soft_cross_entropy(out1, labels_soft) if use_soft else criterion(out1, labels_hard)
                loss1.backward()
                optimizer.first_step(zero_grad=True)

                out2 = model(inputs)
                loss2 = soft_cross_entropy(out2, labels_soft) if use_soft else criterion(out2, labels_hard)
                loss2.backward()
                optimizer.second_step(zero_grad=True)

                outputs_for_metrics = out2
                loss_for_metrics = loss2

        # ===== 普通优化器分支 =====
        else:
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = soft_cross_entropy(outputs, labels_soft) if use_soft else criterion(outputs, labels_hard)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                outputs_for_metrics = outputs
                loss_for_metrics = loss
            else:
                outputs = model(inputs)
                loss = soft_cross_entropy(outputs, labels_soft) if use_soft else criterion(outputs, labels_hard)
                loss.backward()
                optimizer.step()

                outputs_for_metrics = outputs
                loss_for_metrics = loss



        # 统计
        bs = labels_hard.size(0)
        running_loss += float(loss_for_metrics.item()) * bs
        preds = outputs_for_metrics.argmax(dim=1)
        total += bs
        correct += (preds == labels_hard).sum().item()

        if log_interval and (step % log_interval == 0):
            cur_acc = 100.0 * correct / max(1, total)
            print(f"[Step {step:5d}] loss={loss_for_metrics.item():.4f} acc={cur_acc:.2f}%")

    epoch_seconds = time.time() - start
    epoch_loss = running_loss / max(1, total)
    epoch_acc = 100.0 * correct / max(1, total)
    print(f"Epoch done | time: {epoch_seconds:.2f}s | loss: {epoch_loss:.4f} | acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc



def validate_epoch(model, dataloader, criterion, device):
    """
    Validate the model (no tqdm, with timing)
    Args:
        model: model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: torch.device
    Returns:
        (epoch_loss, epoch_acc)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    start = time.time()

    for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if labels.ndim > 1:
                labels = labels.argmax(dim=1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    epoch_seconds = time.time() - start
    epoch_loss = running_loss / max(1, total)
    epoch_acc = 100.0 * correct / max(1, total)

    print(f"Validation done | time: {epoch_seconds:.2f}s | loss: {epoch_loss:.4f} | acc: {epoch_acc:.2f}%")

    return epoch_loss, epoch_acc

def save_checkpoint(state, filename):
    """
    Save model checkpoint
    Args:
        state: Checkpoint state
        filename: Path to save checkpoint
    """
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    Args:
        filename: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
    Returns:
        Checkpoint state
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint, model

def save_metrics(metrics: str, filename: str = "training_metrics.txt"):
    """
    Save training metrics to a file
    Args:
        metrics: Metrics string to save
        filename: Path to save metrics
    """
    with open(filename, 'w') as f:
        f.write(metrics)
