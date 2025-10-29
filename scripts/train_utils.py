import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
        # 可选的更强随机增强
        if use_randaugment:
            try:
                from transforms import RandAugment
                transform_list.append(RandAugment(num_ops=3, magnitude=10))
                transform_list.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'))
            except Exception:
                pass  # torchvision 旧版本可忽略

        # 基本归一化
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ]
        return transforms.Compose(transform_list)
    else:
        # 验证/测试集使用确定性变换
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])

def load_data(data_dir, batch_size):
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
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda b: mixup_cutmix_collate(
            b,
            alpha=1.0,
            mixup_prob=0.5,
            cutmix_prob=0.5,
            num_classes=len(train_dataset.classes)
        )
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(256, batch_size),
        shuffle=False,
        num_workers=2,
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
                              warmup: int = 5,
                              ema_decay: float = 0.999):
    """
    返回：criterion, optimizer, scheduler, ema
    - SGD + Nesterov（推荐 WRN）或 AdamW/Adam
    - Warmup + Cosine LR
    - CrossEntropy + Label Smoothing
    - EMA 初始化（默认 0.999）
    """
    # 1) Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 2) Optimizer
    opt_type = optimizer_type.lower()
    if opt_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              nesterov=True, weight_decay=weight_decay)
    elif opt_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay, betas=(0.9, 0.999))
    elif opt_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # 3) Scheduler: warmup + cosine
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        t = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 4) EMA
    ema = EMA(model, decay=ema_decay)

    return criterion, optimizer, scheduler, ema


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """CE for soft/one-hot labels."""
    return torch.mean(torch.sum(-soft_targets * F.log_softmax(logits, dim=1), dim=1))

def train_epoch(model,
                dataloader,
                criterion,             # 仍可传 CrossEntropyLoss；软标签时会自动改用 SCE
                optimizer,
                device,
                ema=None,              # EMA 对象或 None
                use_amp: bool = True,
                scaler=None,           # torch.amp.GradScaler；use_amp=True 时建议传入
                log_interval: int = 0,  # >0 时按步数间隔打印批次日志
               ):
    """
    Train the model for one epoch (handles mixup/cutmix soft labels).
    Returns:
        epoch_loss (float), epoch_acc (float)
    """
    if use_amp and scaler is None:
        raise ValueError("use_amp=True 但未传入 GradScaler，请传入 scaler 或将 use_amp 设为 False。")

    model.train()
    running_loss, correct, total = 0.0, 0, 0
    start = time.time()

    for step, (inputs, labels) in enumerate(dataloader, start=1):
        inputs = inputs.to(device, non_blocking=True)

        # --- 判定是否为软标签（来自 mixup/cutmix 的 one-hot/soft） ---
        use_soft = isinstance(labels, torch.Tensor) and labels.ndim > 1
        if use_soft:
            labels_soft = labels.to(device, non_blocking=True).float()     # for loss
            labels_hard = labels.argmax(dim=1).to(device, non_blocking=True)  # for acc
        else:
            labels_hard = labels.to(device, non_blocking=True)             # for acc & loss

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = soft_cross_entropy(outputs, labels_soft) if use_soft else criterion(outputs, labels_hard)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = soft_cross_entropy(outputs, labels_soft) if use_soft else criterion(outputs, labels_hard)
            loss.backward()
            optimizer.step()

        # ---- EMA: 前10步后更新 ----
        if ema is not None:
            ema.update(model)

        # ---- 统计 ----
        bs = labels_hard.size(0)
        running_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        total += bs
        correct += (preds == labels_hard).sum().item()

        if log_interval and (step % log_interval == 0):
            cur_acc = 100.0 * correct / max(1, total)
            print(f"[Step {step:5d}] loss={loss.item():.4f} acc={cur_acc:.2f}%")

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
