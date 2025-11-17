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

def collect_classes(train_dirs):
    """
    Scan all training shard directories and collect all class names.

    Args:
        train_dirs (list[Path]): List of training shard directories.

    Returns:
        list[str]: Sorted list of class names (e.g. ['n01440764', ...]).
    """
    classes = set()
    for d in train_dirs:
        for cls in d.iterdir():
            if cls.is_dir():
                classes.add(cls.name)
    return sorted(classes)


def build_class_mapping(classes):
    """
    Create a mapping from class name → integer index.

    Args:
        classes (list[str]): List of class folder names.

    Returns:
        dict: {class_name: index}
    """
    return {cls: i for i, cls in enumerate(classes)}


def build_transform():
    """
    Standard ImageNet preprocessing transform.

    Returns:
        torchvision.transforms.Compose
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
    Wraps an ImageFolder dataset and remaps labels to a custom class_to_idx.

    This is needed because each shard (train.X1/train.X2/...) has its own
    ImageFolder-generated label mapping, but we want a unified mapping.
    """
    def __init__(self, imgfolder, class_to_idx):
        self.ds = imgfolder
        self.class_to_idx = class_to_idx
        self.old_classes = imgfolder.classes  # the original class names

    def __getitem__(self, idx):
        img, old_label = self.ds[idx]
        cls_name = self.old_classes[old_label]
        new_label = self.class_to_idx[cls_name]
        return img, new_label

    def __len__(self):
        return len(self.ds)


def build_datasets(root):
    """
    Build unified train + validation datasets from ImageNet-100 folder layout:

        root/
           train.X1/
           train.X2/
           train.X3/
           train.X4/
           val.X/

    Args:
        root (str or Path): Dataset root folder.

    Returns:
        train_dataset: Concat of all training shards.
        val_dataset: Validation set.
        class_to_idx: Unified mapping for all classes.
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
    val_dataset   = RemappedDataset(ImageFolder(val_dir, transform=tf),
                                    class_to_idx)

    return train_dataset, val_dataset, class_to_idx


def build_loaders(root, batch_size=64):
    """
    Build train + val dataloaders.

    Args:
        root (str): Dataset root.
        batch_size (int): Loader batch size.

    Returns:
        train_loader, val_loader, class_to_idx
    """
    train_ds, val_ds, class_to_idx = build_datasets(root)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, class_to_idx


def get_calibration_batch(loader, num_batches=1):
    """
    Collect a calibration batch (for PTQ).

    Args:
        loader: dataloader
        num_batches: number of batches to extract

    Returns:
        torch.Tensor: Concatenated tensor of images.
    """
    imgs = []
    for i, (batch_imgs, _) in enumerate(loader):
        imgs.append(batch_imgs)
        if i + 1 == num_batches:
            break
    return torch.cat(imgs, dim=0)


# =====================================================================
# Fine-tuning Utility
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
    Fine-tune a pretrained model on a new dataset.

    Automatically replaces classifier head depending on architecture:
        - model.fc for ResNet
        - model.classifier[-1] for MobileNet/EfficientNet

    Args:
        model: pretrained model
        train_dataset: training dataset
        val_dataset: validation dataset
        num_classes: output classes
        batch_size: dataloader batch size
        epochs: number of fine-tuning epochs
        lr: learning rate
        full_finetune: if True, unfreeze all layers

    Returns:
        torch.nn.Module (model on CPU)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------
    # Replace final head
    # -------------------------------------------------------
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        # ResNet
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        # MobileNet / EfficientNet
        last = model.classifier[-1]
        if isinstance(last, nn.Linear):
            in_features = last.in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("Unsupported classifier layout.")

    elif hasattr(model, "model") and hasattr(model.model, "classifier"):
        # VGG16 inside QuantizableVGG16 wrapper
        last = model.model.classifier[-1]
        if isinstance(last, nn.Linear):
            in_features = last.in_features
            model.model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("Unsupported VGG classifier layout.")

    else:
        raise ValueError("Model has no recognized classifier head.")

    # -------------------------------------------------------
    # Freeze backbone / unfreeze everything
    # -------------------------------------------------------
    if full_finetune:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze only the classifier head
        if hasattr(model, "fc"):
            head = model.fc                           # ResNet
        elif hasattr(model, "classifier"):
            head = model.classifier                   # MobileNet/EfficientNet
        elif hasattr(model, "model") and hasattr(model.model, "classifier"):
            head = model.model.classifier             # QuantizableVGG16 wrapper
        else:
            raise ValueError("Model has no accessible classifier head.")

        for p in head.parameters():
            p.requires_grad = True

    # -------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # -------------------------------------------------------
    # Loss + Optimizer
    # -------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # -------------------------------------------------------
    # Training / Eval loops
    # -------------------------------------------------------
    def train_one_epoch(model, loader):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc="Finetuning", leave=False)

        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
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
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

    # -------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------
    model = model.to(device)

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader)
        acc = evaluate(model, val_loader)
        print(f"[Epoch {epoch+1}] loss={loss:.4f} acc={acc*100:.2f}%")

    # Move back to CPU for saving or quantization
    model.cpu()
    return model
