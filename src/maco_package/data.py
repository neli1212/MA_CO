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
import copy
import gc
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
    """
    train_ds, val_ds, class_to_idx = build_datasets(root)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, class_to_idx


def get_calibration_batch(loader, num_batches=1):
    """
    Collects calibration samples for PTQ.
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
    batch_size=128,  
    epochs=10,       
    lr=0.001,        
    full_finetune=False,
    patience=3
):
    """
    Fine-tunes a pretrained model on a new classification task.

    Args:
        model (nn.Module): Pretrained model to be fine-tuned.
        train_dataset (Dataset): Dataset used for training.
        val_dataset (Dataset): Dataset used for validation.
        num_classes (int): Number of output classes for the new task.
        batch_size (int): Number of samples per gradient update.
        epochs (int): Maximum number of training passes over the dataset.
        lr (float): Learning rate for the SGD optimizer.
        full_finetune (bool): If True, trains all parameters. If False, only the head.
        patience (int, optional): Epochs to wait for improvement before early stopping.

    Returns:
        tuple: (finetuned_model, best_epoch)
    """
    # 1. Setup & Cleanup
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Fine-tuning on {device} ---")


    
    head_module = None
    layer_name = ""

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        head_module = model.fc
        layer_name = "fc"
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            head_module = model.classifier[-1]
            layer_name = "classifier[-1]"
        elif isinstance(model.classifier, nn.Linear):
            head_module = model.classifier
            layer_name = "classifier"

    if head_module is not None and isinstance(head_module, nn.Linear):
        if head_module.out_features != num_classes:
            print(f"🔄 Replacing head '{layer_name}' ({head_module.out_features} -> {num_classes})")
            new_head = nn.Linear(head_module.in_features, num_classes)
            
            if layer_name == "fc":
                model.fc = new_head
            elif layer_name == "classifier":
                model.classifier = new_head
            elif layer_name == "classifier[-1]":
                model.classifier[-1] = new_head
        else:
            print(f"✅ Head '{layer_name}' matches num_classes ({num_classes})")
    else:
        pass

    model = model.to(device)

    # 3. Freezing Logic
    if full_finetune:
        print("🔓 Full Fine-tune: All layers unfrozen.")
        for p in model.parameters():
            p.requires_grad = True
    else:
        print("🔒 Linear Probe: Backprop only on head.")
        for p in model.parameters():
            p.requires_grad = False
        
        # Unfreeze only the head
        if hasattr(model, "fc"):
            for p in model.fc.parameters(): p.requires_grad = True
        elif hasattr(model, "classifier"):
            for p in model.classifier.parameters(): p.requires_grad = True

    # 4. DataLoaders 
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=6, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=6, pin_memory=True
    )

    # 5. Optimizer 
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = torch.optim.SGD(
        trainable_params, 
        lr=lr, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # 6. Training Loop
    best_acc = 0.0
    stagnant = 0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}")
            
        # Validation
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        print(f"[Epoch {epoch+1}] loss={avg_loss:.4f} acc={acc*100:.2f}%")
        
        # Save Best
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            stagnant = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            if patience is not None:
                stagnant += 1
                if stagnant >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    # 7. Always return the BEST model, not the LAST model
    print(f"🏆 Loading Best Model from Epoch {best_epoch} (Acc: {best_acc*100:.2f}%)")
    model.load_state_dict(best_state)
    
    return model.cpu(), best_epoch