import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def visualize_images(lr_paths: list[str], methods: dict, bboxes: list[tuple], scale=4, figsize=(20, 12)):
    """
    lr_paths: list of LR image paths
    methods: dict like {
        'Method1': [
            [{'path': 'sr1_img1.png', 'type': 'image'}, {'path': 'kernel1_img1.png', 'type': 'kernel'}],
            [{'path': 'sr1_img2.png', 'type': 'image'}, {'path': 'kernel1_img2.png', 'type': 'kernel'}],
            ...
        ],
        'Method2': [...],
        ...
    }
    bboxes: list of tuples, one for each LR image
    """
    n_lr_images = len(lr_paths)
    n_methods = len(methods)
    n_rows = n_lr_images * 2  # SR row + Pred Kernel row for each LR image
    n_cols = n_methods + 1
    
    fig = plt.figure(figsize=figsize)
    
    # Create height ratios for each LR image section
    height_ratios = [1, 1] * n_lr_images
    
    gs = fig.add_gridspec(n_rows, n_cols, 
                          width_ratios=[2.0] + [1] * n_methods, 
                          height_ratios=height_ratios,
                          hspace=0.2, wspace=-0.4)
    
    # Process each LR image
    lr_imgs = [Image.open(path) for path in lr_paths]
        
    min_width = min(img.size[0] for img in lr_imgs)
    min_height = min(img.size[1] for img in lr_imgs)
    lr_imgs = [img.crop((0, 0, min_width, min_height)) for img in lr_imgs]
    for lr_idx, (lr_img, bbox) in enumerate(zip(lr_imgs, bboxes)):
        x, y, w, h = bbox
        sr_x, sr_y, sr_w, sr_h = x * scale, y * scale, w * scale, h * scale
        
        base_row = lr_idx * 2  # Starting row for this LR image
        
        # LR image with bounding box and inset
        ax_lr = fig.add_subplot(gs[base_row:base_row+2, 0])
        ax_lr.imshow(lr_img)
        
        if lr_idx == 0:
            ax_lr.set_ylabel("LR Input", fontsize=10, fontweight="bold", 
                           rotation=0, labelpad=40, va="center")
        
        ax_lr.axis("off")
        
        # Add bounding box
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax_lr.add_patch(rect)
        
        # Add inset
        lr_patch = lr_img.crop((x, y, x + w, y + h))
        ax_ins = inset_axes(ax_lr, width="40%", height="40%", 
                           loc='lower left', borderpad=0)
        ax_ins.imshow(lr_patch)
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        for spine in ax_ins.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)
        
        # Process each method for this LR image
        for method_idx, (method_name, img_lists) in enumerate(methods.items()):
            col_idx = method_idx + 1
            img_list = img_lists[lr_idx]  # Get images for current LR
            
            # SR patch
            ax_sr = fig.add_subplot(gs[base_row, col_idx])
            sr_img = Image.open(img_list[0]["path"])
            if img_list[0]["type"] == "image":
                sr_patch = sr_img.crop((sr_x, sr_y, sr_x + sr_w, sr_y + sr_h))
            else:
                sr_patch = sr_img
            ax_sr.imshow(sr_patch)
            
            # Add title only for first LR image
            if lr_idx == 0:
                ax_sr.set_title(method_name, fontsize=10, fontweight="bold")
            
            # Add row label only for first method and first LR image
            if col_idx == 1 and lr_idx == 0:
                ax_sr.set_ylabel("SR", fontsize=10, fontweight="bold", 
                               rotation=0, labelpad=40, va="center")
            ax_sr.axis("off")
            
            # Predicted Kernel
            ax_kernel = fig.add_subplot(gs[base_row + 1, col_idx])
            kernel_img = Image.open(img_list[1]["path"])
            ax_kernel.imshow(kernel_img)
            
            # Add row label only for first method and first LR image
            if col_idx == 1 and lr_idx == 0:
                ax_kernel.set_ylabel("Pred Kernel", fontsize=10, fontweight="bold", 
                                   rotation=0, labelpad=40, va="center")
            ax_kernel.axis("off")
    
    plt.tight_layout()
    # plt.show()
    return fig

lr_paths = [
    "/root/quoc-huy/materials/Urban100/imgs/img_027.png",
    "/root/quoc-huy/materials/Urban100/imgs/img_090.png"
    ]
bboxes = [(140, 30, 25, 25), (130, 95, 25, 25)]

methods = {
    'SR': [
        [
            {"path": "/root/quoc-huy/all-tested-results/DAN/Urban100/imgs/img_027.png", "type": "image"}, 
            {"path": "/root/quoc-huy/all-tested-results/DAN/Urban100/kernels/img_027.png", "type": "kernel"}
        ],
        [
            {"path": "/root/quoc-huy/all-tested-results/DAN/Urban100/imgs/img_090.png", "type": "image"}, 
            {"path": "/root/quoc-huy/all-tested-results/DAN/Urban100/kernels/img_090.png", "type": "kernel"}
        ]
    ],
    'Ground truth': [
        [
            {"path": "/root/media/quoc-huy/eval-wir/srbenchmarks/Urban100/HR/img_027.png", "type": "image"}, 
            {"path": "/root/quoc-huy/materials/Urban100/kernels/img_027.png", "type": "kernel"}
        ],
        [
            {"path": "/root/media/quoc-huy/eval-wir/srbenchmarks/Urban100/HR/img_090.png", "type": "image"}, 
            {"path": "/root/quoc-huy/materials/Urban100/kernels/img_090.png", "type": "kernel"}
        ]
    ]
}

visualize_images(lr_paths, methods, bboxes)
plt.savefig('/root/quoc-huy/all-tested-results/combined/manet_results.png', dpi=300, bbox_inches='tight')