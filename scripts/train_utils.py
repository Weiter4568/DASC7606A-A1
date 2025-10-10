import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def load_transforms(train=True):
    mean, std = (0.5071,0.4865,0.4409), (0.2673,0.2564,0.2762)  # CIFAR-100; CIFAR-10 也可用 (0.4914,0.4822,0.4465)/(0.2023,0.1994,0.2010)
    if train:
        tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return tf

def load_data(train_dir, batch_size, val_ratio=0.1, num_workers=4):
    full = datasets.ImageFolder(root=train_dir, transform=load_transforms(train=True))
    n_total = len(full); n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    train_set, val_set = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(123))
    # 验证集不做随机增广
    val_set.dataset.transform = load_transforms(train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ---- 2) 优化器 + 调度器 ----
class _CompatScheduler:
    """Wrap schedulers that don't take a metric so they accept step(val_loss)."""
    def __init__(self, sched): self.sched = sched
    def step(self, *_args, **_kwargs): self.sched.step()
    def state_dict(self): return self.sched.state_dict()
    def load_state_dict(self, s): return self.sched.load_state_dict(s)

def define_loss_and_optimizer(model, lr, weight_decay):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 小幅 smoothing 提升泛化
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    # 余弦退火到 0（T_max 在 main.py 里看不到 epoch，这里约个上限；实际每轮 step 一次）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = _CompatScheduler(scheduler)  # 兼容 main.py 的 scheduler.step(val_loss)
    return criterion, optimizer, scheduler

# ---- 3) 训练/验证（含 AMP + 梯度裁剪）----
def train_epoch(model, loader, criterion, optimizer, device, max_norm=1.0):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda' and torch.cuda.is_available()))
    total_loss, correct, n = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        if max_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer); scaler.update()

        total_loss += loss.item() * inputs.size(0)
        pred = outputs.argmax(1)
        correct += (pred == targets).sum().item()
        n += inputs.size(0)
    return total_loss / n, 100.0 * correct / n


def validate_epoch(model, dataloader, criterion, device):
    """
    Validate the model
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
    Returns:
        Average loss and accuracy for the validation set
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

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

    return checkpoint

def save_metrics(metrics: str, filename: str = "training_metrics.txt"):
    """
    Save training metrics to a file
    Args:
        metrics: Metrics string to save
        filename: Path to save metrics
    """
    with open(filename, 'w') as f:
        f.write(metrics)
