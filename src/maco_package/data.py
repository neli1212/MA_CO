# =====================================================================
# Imports
# =====================================================================
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset,
    ConcatDataset,
    DataLoader,
)
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms

# =====================================================================
# Dataset & Class Mapping 
# =====================================================================

def collect_classes(train_dirs):
    """
    Collects all unique class names from training shards.

    Args:
        train_dirs (list[Path]): Training shard directories.

    Returns:
        list[str]: Sorted class names.
    """
    classes = set()
    for d in train_dirs:
        for cls in d.iterdir():
            if cls.is_dir():
                classes.add(cls.name)
    return sorted(classes)


def build_class_mapping(classes):
    """
    Builds a class-to-index mapping.

    Args:
        classes (list[str]): Class names.

    Returns:
        dict: {class_name: index}
    """
    return {cls: i for i, cls in enumerate(classes)}


def build_transform():
    """
    Returns standard ImageNet preprocessing transform.

    Returns:
        transforms.Compose: Transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])


class RemappedDataset(Dataset):
    """
    Remaps ImageFolder labels to a global index.
    """
    def __init__(self, imgfolder, class_to_idx):
        self.ds = imgfolder
        self.class_to_idx = class_to_idx
        self.old_classes = imgfolder.classes

    def __getitem__(self, idx):
        img, old_label = self.ds[idx]
        cls_name = self.old_classes[old_label]
        new_label = self.class_to_idx[cls_name]
        return img, new_label

    def __len__(self):
        return len(self.ds)


def build_datasets(root):
    """
    Builds training and validation datasets.

    Args:
        root (str | Path): Dataset root directory.

    Returns:
        tuple: (train_dataset, val_dataset, class_to_idx)
    """
    root = Path(root)

    train_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("train.")
    )
    val_dir = root / "val.X"

    classes = collect_classes(train_dirs)
    class_to_idx = build_class_mapping(classes)
    tf = build_transform()

    train_parts = []
    for d in train_dirs:
        raw = ImageFolder(d, transform=tf)
        train_parts.append(RemappedDataset(raw, class_to_idx))

    train_dataset = ConcatDataset(train_parts)
    val_dataset = RemappedDataset(
        ImageFolder(val_dir, transform=tf),
        class_to_idx
    )

    return train_dataset, val_dataset, class_to_idx


def build_loaders(root, batch_size=64):
    """
    Builds training and validation dataloaders.

    Args:
        root (str): Dataset root.
        batch_size (int): Batch size.

    Returns:
        tuple: (train_loader, val_loader, class_to_idx)
    """
    train_ds, val_ds, class_to_idx = build_datasets(root)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, class_to_idx


def get_calibration_batch(loader, num_batches=1):
    """
    Collects calibration samples for PTQ.

    Args:
        loader: DataLoader source.
        num_batches (int): Number of batches.

    Returns:
        torch.Tensor: Calibration images.
    """
    imgs = []
    for i, (batch_imgs, _) in enumerate(loader):
        imgs.append(batch_imgs)
        if i + 1 == num_batches:
            break
    return torch.cat(imgs, dim=0)


# =====================================================================
# Fine-tuning 
# =====================================================================

def finetuneModel(
    model,
    train_dataset,
    val_dataset,
    num_classes,
    batch_size=64,
    epochs=5,
    lr=1e-3,
    full_finetune=False
):
    """
    Fine-tunes a model on a new classification task.

    Args:
        model: Pretrained model.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        num_classes (int): Output classes.
        batch_size (int): Batch size.
        epochs (int): Training epochs.
        lr (float): Learning rate.
        full_finetune (bool): Train full model if True.

    Returns:
        nn.Module: Fine-tuned model (CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            last = model.classifier[-1]
            if isinstance(last, nn.Linear):
                in_features = last.in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)

    elif hasattr(model, "model") and hasattr(model.model, "classifier"):
        if isinstance(model.model.classifier, nn.Sequential):
            last = model.model.classifier[-1]
            if isinstance(last, nn.Linear):
                in_features = last.in_features
                model.model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.model.classifier.in_features
            model.model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Model architecture head not recognized.")

    if full_finetune:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False

        if hasattr(model, "fc"):
            head = model.fc
        elif hasattr(model, "classifier"):
            head = model.classifier
        elif hasattr(model, "model") and hasattr(model.model, "classifier"):
            head = model.model.classifier

        for p in head.parameters():
            p.requires_grad = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    def train_one_epoch(model, loader):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc="Finetuning", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}")
        return total_loss / len(loader)

    def evaluate(model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

    model = model.to(device)
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader)
        acc = evaluate(model, val_loader)
        print(f"[Epoch {epoch+1}] loss={loss:.4f} acc={acc*100:.2f}%")

    return model.cpu()
