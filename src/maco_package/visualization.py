# =====================================================================
# Imports
# =====================================================================
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# =====================================================================
# Constants
# =====================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# =====================================================================
# Image Display Utility
# =====================================================================
def show_image(img, denorm=False):
    """
    Display an image (PIL image, file path, or tensor).

    Args:
        img: PIL.Image, file path string, or torch.Tensor.
        denorm (bool): If True, undo ImageNet normalization for tensors.

    Returns:
        None
    """

    # ------------------------------------------------------
    # Load if file path
    # ------------------------------------------------------
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")

    # ------------------------------------------------------
    # PIL image → numpy array
    # ------------------------------------------------------
    if isinstance(img, Image.Image):
        arr = np.array(img)

    elif isinstance(img, torch.Tensor):
        t = img.detach().cpu()

        # Remove batch dimension
        if t.ndim == 4 and t.shape[0] == 1:
            t = t.squeeze(0)

        # CHW → HWC
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)

        arr = t.numpy()

        # Optional ImageNet denormalization
        if denorm:
            arr = arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)

        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)

    else:
        raise TypeError("img must be a PIL.Image, file path, or torch.Tensor.")

    plt.imshow(arr)
    plt.axis("off")
    plt.show()


# =====================================================================
# Top-5 Prediction Visualization
# =====================================================================
def ShowTop5(model, img, class_names):
    """
    Compute and display the top-5 predictions for an image.

    Args:
        model (torch.nn.Module): The model (can be FP32 or INT8).
        img: PIL.Image, file path string, or torch.Tensor.
        class_names (list[str]): List of class names matching model outputs.

    Behavior:
        - Image is shown using `show_image`
        - Top-5 labels + probabilities are printed
        - Always runs inference on CPU
    """

    device = torch.device("cpu")
    net = model.to(device).eval()

    if isinstance(img, torch.Tensor):
        if img.ndim == 3:
            x = img.unsqueeze(0).to(device)  # CHW
        elif img.ndim == 4:
            x = img.to(device)              # BCHW
        else:
            raise ValueError("Tensor must be CHW or BCHW.")

    else:
        # PIL or path → preprocess normally
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")

        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = net(x).softmax(1).cpu().numpy()[0]


    top5 = np.argsort(probs)[-5:][::-1]
    show_image(img, denorm=False)

    print("\nTop-5 predictions:")
    for i in top5:
        print(f"{class_names[i]}: {probs[i] * 100:.2f}%")


# =====================================================================
# Random Dataset Sample Viewer
# =====================================================================
def show_random_sample(dataset, class_to_idx, synset_to_name, return_sample=False):
    """
    Show a random sample from an ImageFolder dataset.

    Args:
        dataset: torchvision.datasets.ImageFolder or compatible dataset.
        class_to_idx (dict): e.g. {"n01440764": 0, ...}
        synset_to_name (dict): e.g. {"n01440764": "tench", ...}
        return_sample (bool): If True, return the PIL image.

    Prints:
        The human-readable label for the sample.
    """

    idx_to_synset = {v: k for k, v in class_to_idx.items()}
    sample_index = random.randint(0, len(dataset) - 1)

    img_tensor, label_index = dataset[sample_index]

    synset = idx_to_synset[label_index]
    name = synset_to_name.get(synset, synset)

    print("Label:", name)
    unnorm = (
        img_tensor.clone()
        * torch.tensor(IMAGENET_STD)[:, None, None]
        + torch.tensor(IMAGENET_MEAN)[:, None, None]
    ).clamp(0, 1)

    pil_img = to_pil_image(unnorm)

    show_image(unnorm, denorm=False)

    if return_sample:
        return pil_img
