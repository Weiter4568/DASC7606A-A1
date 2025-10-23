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

def load_data(train_dir, batch_size, num_workers=8):
    """
    Load training and validation data separately.
    train_dir: directory for training data (e.g., data_c100/augmented/train)
    val_dir: directory for validation data (e.g., data_c100/raw/val, inferred automatically)
    """
    # Assume validation directory is "../raw/val" parallel to train_dir = ".../augmented/train"
    # E.g., train_dir="data_c100/augmented/train" -> val_dir="data_c100/raw/val"
    base_data_dir = os.path.dirname(os.path.dirname(train_dir))  # up to data_c100
    val_dir = os.path.join(base_data_dir, "raw", "val")

    train_set = datasets.ImageFolder(root=train_dir, transform=load_transforms(train=True))
    val_set   = datasets.ImageFolder(root=val_dir, transform=load_transforms(train=False))

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
    # 使用更强的标签平滑和焦点损失
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)  # 增加标签平滑
    
    # 使用SGD优化器，通常在大数据集上表现更好
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, 
                         momentum=0.9, nesterov=True)
    
    # 使用更复杂的学习率调度策略
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr*5, epochs=200, 
        steps_per_epoch=1, pct_start=0.3,
        anneal_strategy='cos'
    )
    scheduler = _CompatScheduler(scheduler)
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
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

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
