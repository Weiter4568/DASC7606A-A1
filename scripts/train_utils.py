import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import math

class EMA:
    """
    Exponential Moving Average of model parameters AND buffers (e.g., BN running stats).
    - 跟踪: 仅 requires_grad 的 parameters + 所有浮点 buffers
    - 支持: DDP(DataParallel) 的 model.module
    - 提供: warm start (靠外部控制何时 apply_to)
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow_params = {}
        self.shadow_buffers = {}
        self.backup_params = {}
        self.backup_buffers = {}

        tracked = model.module if hasattr(model, "module") else model

        # 1) track trainable parameters
        for n, p in tracked.named_parameters():
            if p.requires_grad:
                self.shadow_params[n] = p.detach().clone()

        # 2) track floating buffers (BN running stats, etc.)
        for n, b in tracked.named_buffers():
            if torch.is_floating_point(b):
                self.shadow_buffers[n] = b.detach().clone()

        self.num_updates = 0  # 记录更新次数（可用于 bias-correction 等）

    @torch.no_grad()
    def update(self, model: nn.Module):
        tracked = model.module if hasattr(model, "module") else model
        d = self.decay
        self.num_updates += 1

        # 可选的 bias-corrected 动态衰减（更快贴近当前权重）
        # d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

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
        # swap buffers (e.g., BN running stats)
        for n, b in tracked.named_buffers():
            if torch.is_floating_point(b) and n in self.shadow_buffers:
                self.backup_buffers[n] = b.detach().clone()
                b.data.copy_(self.shadow_buffers[n])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        tracked = model.module if hasattr(model, "module") else model
        # restore params
        for n, p in tracked.named_parameters():
            if p.requires_grad and n in self.backup_params:
                p.data.copy_(self.backup_params[n])
        # restore buffers
        for n, b in tracked.named_buffers():
            if torch.is_floating_point(b) and n in self.backup_buffers:
                b.data.copy_(self.backup_buffers[n])
        self.backup_params.clear()
        self.backup_buffers.clear()

    def state_dict(self):
        # 统一保存到 CPU，防止跨设备报错
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
        self.shadow_params = {k: v.to(dev) for k, v in state["shadow_params"].items()}
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

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



def train_epoch(model,
                dataloader,
                criterion,
                optimizer,
                device,
                ema=None,                 # 传入你的 EMA 对象（可为 None）
                use_amp: bool = True,    # 是否使用自动混合精度
                scaler=None,              # torch.cuda.amp.GradScaler；use_amp=True 时建议传入
                log_interval: int = 0):   # >0 时按步数间隔打印批次日志
    """
    Train the model for one epoch (no tqdm; prints epoch time).
    Returns:
        epoch_loss (float), epoch_acc (float), epoch_seconds (float)
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    start = time.time()

    for step, (inputs, labels) in enumerate(dataloader, start=1):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            if scaler is None:
                raise ValueError("use_amp=True 但未传入 GradScaler，请传入 scaler 或将 use_amp 设为 False。")
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # ---- EMA: 每个优化步后更新 ----
        if ema is not None:
            ema.update(model)

        # ---- 统计 ----
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total += batch_size
        correct += predicted.eq(labels).sum().item()

        # 可选：按间隔打印批次日志
        if log_interval and (step % log_interval == 0):
            cur_acc = 100.0 * correct / max(1, total)
            print(f"[Step {step:5d}] loss={loss.item():.4f} acc={cur_acc:.2f}%")

    epoch_seconds = time.time() - start
    epoch_loss = running_loss / max(1, total)
    epoch_acc = 100.0 * correct / max(1, total)

    # 这里直接打印每个 epoch 的用时（也可以在外层根据返回值自行打印）
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
        (epoch_loss, epoch_acc, epoch_seconds)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    start = time.time()

    for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

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
