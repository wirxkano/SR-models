import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
                          hspace=-0.77, wspace=0.1)
    
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

if __name__ == "__main__":
    bounding_box = (200, 60, 20, 20)

    lr_path = "/root/quoc-huy/materials/Urban100/imgs/img_052.png"
    img_dicts = {
        "DAN": [
            {"path": "/root/quoc-huy/all-tested-results/DAN/Urban100/imgs/img_052.png", "type": "image"}, 
            {"path": "/root/quoc-huy/all-tested-results/DAN/Urban100/kernels/img_052.png", "type": "kernel"}
        ],
        "DCLS": [
            {"path": "/root/quoc-huy/all-tested-results/DCLS/Urban100/imgs/img_052.png", "type": "image"}, 
            {"path": "/root/quoc-huy/all-tested-results/DCLS/Urban100/kernels/img_052.png", "type": "kernel"}
        ],
        "ASBSR": [
            {"path": "/root/quoc-huy/all-tested-results/ASBSR/Urban100/imgs/img_052.png", "type": "image"}, 
            {"path": "/root/quoc-huy/all-tested-results/ASBSR/Urban100/kernels/img_052.png", "type": "kernel"}
        ],
        "MANet": [
            {"path": "/root/quoc-huy/all-tested-results/MANet/Urban100/imgs/img_052.png", "type": "image"}, 
            {"path": "/root/quoc-huy/all-tested-results/MANet/Urban100/kernels/img_052.png", "type": "kernel"}
        ],
        "KOALAnet": [
            {"path": "/root/quoc-huy/all-tested-results/KOALAnet/Urban100/imgs/img_052.png", "type": "image"}, 
            {"path": "/root/quoc-huy/all-tested-results/KOALAnet/Urban100/kernels/img_052.png", "type": "kernel"}
        ],
        "Ground truth": [
            {"path": "/root/media/quoc-huy/eval-wir/srbenchmarks/Urban100/HR/img_052.png", "type": "image"}, 
            {"path": "/root/quoc-huy/materials/Urban100/kernels/img_052.png", "type": "kernel"}
        ],
    }
    fig = visualize_images(lr_path, img_dicts, bbox=bounding_box)

    plt.savefig('/root/quoc-huy/all-tested-results/combined/img_052.png', dpi=300, bbox_inches='tight')
    # plt.show()