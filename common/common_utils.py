# import gdown
# import zipfile
# import os


# def download_file_from_google_drive(
#     drive_url, output_path, extract=False, extract_dir=None
# ):
#     print("Downloading file...")
#     gdown.download(drive_url, output_path, quiet=False, fuzzy=True)
#     print(f"Downloaded '{output_path}' successfully.")

#     if extract and extract_dir:
#         os.makedirs(extract_dir, exist_ok=True)

#         print(f"Extracting '{output_path}' to '{extract_dir}'...")
#         with zipfile.ZipFile(output_path, "r") as zip_ref:
#             zip_ref.extractall(extract_dir)

#         print("Extraction complete!")


# def zip_folder(folder_path, output_path):
#     with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
#         for root, _, files in os.walk(folder_path):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 arcname = os.path.relpath(file_path, folder_path)
#                 zipf.write(file_path, arcname)
#     print(f"Folder '{folder_path}' has been zipped into '{output_path}'.")


# if __name__ == "__main__":
#     download_file_from_google_drive("https://drive.google.com/file/d/1RLcLZbdq7qhqDgl4elywKSZX3Sq7Nwd3/view",
#                                     "/root/quoc-huy/CMOS/Cityscapes-BSR.zip",
#                                     extract=True,
#                                     extract_dir="/root/media/quoc-huy/Cityscapes-BSR")

import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def crop_and_show_images(
    image: Image.Image,
    bounding_boxes: list,
    title: str = "",
    colors: list = ["red"],
    cols=1,
):
    cropped = []
    colors = itertools.cycle(colors)
    draw = ImageDraw.Draw(image)

    for box, color in zip(bounding_boxes, colors):
        draw.rectangle(box, outline=color, width=2)
        cropped.append(image.crop(box))

    rows = math.ceil(len(cropped) / cols)
    fig = plt.figure(figsize=(4 + cols * 3, rows * 3))
    ax_main = plt.subplot2grid((rows, cols + 2), (0, 0), rowspan=rows, colspan=2)
    ax_main.imshow(image)
    ax_main.axis("off")

    for i, crop in enumerate(cropped):
        r, c = divmod(i, cols)
        ax = plt.subplot2grid((rows, cols + 2), (r, 2 + c))
        ax.imshow(crop)
        ax.axis("off")

    plt.tight_layout()
    plt.title(title)
    plt.show()


def combine_images_horizontal(img1, img2, img3, spacing=0):
    """
    Combine 3 images into 1 row with 3 columns.

    Args:
        img1, img2, img3: PIL Image objects or file paths
        spacing: space between images in pixels (default: 0)

    Returns:
        PIL Image object with combined images
    """
    if isinstance(img1, str):
        img1 = Image.open(img1)
    if isinstance(img2, str):
        img2 = Image.open(img2)
    if isinstance(img3, str):
        img3 = Image.open(img3)

    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    img3 = img3.convert("RGB")

    w1, h1 = img1.size
    w2, h2 = img2.size
    w3, h3 = img3.size

    img1 = img1.crop((160, 40, w1 - 160, h1 - 40))
    img2 = img2.crop((160, 40, w2 - 160, h2 - 40))
    img3 = img3.crop((160, 40, w3 - 160, h3 - 40))

    w1, h1 = img1.size
    w2, h2 = img2.size
    w3, h3 = img3.size

    max_height = max(h1, h2, h3)

    if h1 != max_height:
        new_w1 = int(w1 * max_height / h1)
        img1 = img1.resize((new_w1, max_height), Image.LANCZOS)
        w1 = new_w1

    if h2 != max_height:
        new_w2 = int(w2 * max_height / h2)
        img2 = img2.resize((new_w2, max_height), Image.LANCZOS)
        w2 = new_w2

    if h3 != max_height:
        new_w3 = int(w3 * max_height / h3)
        img3 = img3.resize((new_w3, max_height), Image.LANCZOS)
        w3 = new_w3

    total_width = w1 + w2 + w3 + spacing * 2
    combined = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

    combined.paste(img1, (0, 0))
    combined.paste(img2, (w1 + spacing, 0))
    combined.paste(img3, (w1 + w2 + spacing * 2, 0))

    return combined


def visualize_sr_comparison(
    lr_img_path, sr_images_dict, bbox, scale_factor=4, color="red", figsize=(15, 8)
):
    """
    Visualize super-resolution comparison with LR image and cropped patches from SR images.

    Parameters:
    -----------
    lr_img_path : str
        Path to the Low Resolution (original/input) image
    sr_images_dict : dict
        Dictionary with method names as keys and FULL SR image paths/arrays as values
        Example: {'Bicubic': 'path/to/bicubic_sr.png', 'ESRGAN': sr_array, ...}
    bbox : tuple
        Bounding box coordinates (x, y, width, height) defined on the LR image
    scale_factor : int
        Super-resolution scale factor (2x, 4x, 8x, etc.). Default is 4.
        The bbox coordinates will be scaled accordingly for SR images
    figsize : tuple
        Figure size (width, height)
    """
    lr_img = Image.open(lr_img_path)
    x, y, w, h = bbox

    sr_x, sr_y = x * scale_factor, y * scale_factor
    sr_w, sr_h = w * scale_factor, h * scale_factor

    n_methods = len(sr_images_dict)
    n_cols = min(4, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        n_rows, n_cols + 1, width_ratios=[2] + [1] * n_cols, hspace=0.3, wspace=0.3
    )

    ax_lr = fig.add_subplot(gs[:, 0])
    ax_lr.imshow(lr_img)
    ax_lr.set_title(f"LR Image", fontsize=12, fontweight="bold")
    ax_lr.axis("off")

    rect = patches.Rectangle(
        (x, y), w, h, linewidth=2, edgecolor=color, facecolor="none"
    )
    ax_lr.add_patch(rect)

    for idx, (method_name, sr_img_path) in enumerate(sr_images_dict.items()):
        row = idx // n_cols
        col = idx % n_cols + 1

        ax = fig.add_subplot(gs[row, col])

        if isinstance(sr_img_path, str):
            sr_img = Image.open(sr_img_path)
        elif isinstance(sr_img_path, np.ndarray):
            sr_img = Image.fromarray(sr_img_path)
        else:
            sr_img = sr_img_path

        sr_patch = sr_img.crop((sr_x, sr_y, sr_x + sr_w, sr_y + sr_h))

        ax.imshow(sr_patch)
        ax.set_title(
            f"{method_name}",
            fontsize=10,
            fontweight="bold",
        )
        ax.axis("off")
        
    plt.tight_layout()
    return fig

def visualize_images(lr_path: str, methods: dict, bbox: tuple, scale=4, figsize=(20, 12)):
    """
    methods: dict like {
        'Method1': [{'path': 'sr1.png', 'type': 'image'}, {'path': 'kernel1.png', 'type': 'kernel'}],
        'Method2': [...],
        ...
    }
    """
    x, y, w, h = bbox
    sr_x, sr_y, sr_w, sr_h = x * scale, y * scale, w * scale, h * scale
    
    n_methods = len(methods)
    n_rows = 2  # SR row, Pred Kernel row
    n_cols = n_methods + 1
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, n_cols, 
                          width_ratios=[3.0] + [1] * n_methods, height_ratios=[1, 1],
                          hspace=-0.5, wspace=0.1)
    
    ax_lr = fig.add_subplot(gs[:, 0])
    lr_img = Image.open(lr_path)
    ax_lr.imshow(lr_img)
    ax_lr.set_ylabel("LR Input", fontsize=10, fontweight="bold", rotation=0, labelpad=40, va="center")
    ax_lr.axis("off")
    
    rect = patches.Rectangle(
        (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
    )
    ax_lr.add_patch(rect)
    
    lr_patch = lr_img.crop((x, y, x + w, y + h))
    
    ax_ins = inset_axes(ax_lr, width="40%", height="40%", loc='lower left', borderpad=0)
    ax_ins.imshow(lr_patch)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    for spine in ax_ins.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)
    
    # Row 1-2: SR patches and Predicted Kernels for each method
    for idx, (method_name, img_list) in enumerate(methods.items()):        
        # SR patch
        col_idx = idx + 1
        ax_sr = fig.add_subplot(gs[0, col_idx])
        sr_img = Image.open(img_list[0]["path"])
        if img_list[0]["type"] == "image":
            sr_patch = sr_img.crop((sr_x, sr_y, sr_x + sr_w, sr_y + sr_h))
        else:
            sr_patch = sr_img
        ax_sr.imshow(sr_patch)
        ax_sr.set_title(method_name, fontsize=10, fontweight="bold")
        if col_idx == 1:
            ax_sr.set_ylabel("SR", fontsize=10, fontweight="bold", rotation=0, labelpad=40, va="center")
        ax_sr.axis("off")
        
        # Predicted Kernel
        ax_kernel = fig.add_subplot(gs[1, col_idx])
        kernel_img = Image.open(img_list[1]["path"])
        ax_kernel.imshow(kernel_img)
        if col_idx == 1:
            ax_kernel.set_ylabel("Pred Kernel", fontsize=10, fontweight="bold", rotation=0, labelpad=40, va="center")
        ax_kernel.axis("off")
    
    plt.show()

def show_list_sr_hr(sr_path: list[str], hr_path: list[str] | None, figsize=(12, 4)):
    row, col = 2, len(sr_path) if hr_path else 1
    fig, axs = plt.subplots(nrows=row, ncols=col, figsize=figsize)
    if hr_path:
        for r in range(row):
            for c in range(col):
                img = Image.open(hr_path[c]) if r == 0 else Image.open(sr_path[c])
                axs[r, c].imshow(img, aspect="auto")
                axs[r, c].axis("off")
                if c == 0:
                    title = "HR Image" if r == 0 else "Output"
                    axs[r, c].text(-0.3, 0.5, title, 
                                transform=axs[r, c].transAxes,
                                fontsize=12, 
                                va='center', 
                                ha='center',
                                rotation=0)
    else:
        sr_images = [Image.open(path) for path in sr_path]
        
        max_width = max(img.size[0] for img in sr_images)
        max_height = max(img.size[1] for img in sr_images)
        sr_images = [img.resize((max_width, max_height), Image.LANCZOS) for img in sr_images]
        for r in range(row):
            img = sr_images[r]
            axs[r].imshow(img)
            axs[r].axis("off")
            axs[r].set_aspect('equal')
    
    plt.subplots_adjust(hspace=0.05)            
    plt.tight_layout()
    plt.show()
    return fig

if __name__ == "__main__":
    # sr_path = [
    #     # "/root/quoc-huy/all-tested-results/combined/img_013.png",
    #     "/root/quoc-huy/all-tested-results/combined/img_027.png",
    #     "/root/quoc-huy/all-tested-results/combined/img_090.png",
    # ]
    # hr_path = [
    #     "/root/media/quoc-huy/eval-wir/srbenchmarks/Urban100/HR/img_001.png",
    #     "/root/media/quoc-huy/eval-wir/srbenchmarks/Urban100/HR/img_004.png",
    #     "/root/media/quoc-huy/eval-wir/srbenchmarks/Urban100/HR/img_018.png",
    #     "/root/media/quoc-huy/eval-wir/srbenchmarks/Urban100/HR/img_019.png",
    #     "/root/media/quoc-huy/eval-wir/srbenchmarks/Urban100/HR/img_067.png"
    # ]
    # hr_path = None
    # img = show_list_sr_hr(sr_path, hr_path)
    # plt.savefig('/root/quoc-huy/all-tested-results/combined/manet_results.png', dpi=300, bbox_inches='tight')
    bounding_box = (140, 30, 25, 25)

    lr_path = "/root/quoc-huy/materials/Urban100/imgs/img_090.png"
    img_dicts = {
        "SR": [
            {"path": "/root/quoc-huy/all-tested-results/DAN/Urban100/imgs/img_090.png", "type": "image"}, 
            {"path": "/root/quoc-huy/all-tested-results/DAN/Urban100/kernels/img_090.png", "type": "kernel"}
        ],
        "Ground truth": [
            {"path": "/root/media/quoc-huy/eval-wir/srbenchmarks/Urban100/HR/img_090.png", "type": "image"}, 
            {"path": "/root/quoc-huy/materials/Urban100/kernels/img_090.png", "type": "kernel"}
        ],
    }
    fig = visualize_images(lr_path, img_dicts, bbox=bounding_box, figsize=(20, 12))

    plt.savefig('/root/quoc-huy/all-tested-results/combined/img_090.png', dpi=300, bbox_inches='tight')
