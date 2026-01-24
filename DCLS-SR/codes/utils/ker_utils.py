import torch

pca_matrix = torch.load(
    "/root/quoc-huy/DCLS-SR/pca_matrix/DCLS/pca_matrix.pth", map_location=lambda storage, loc: storage
)

def reconstruct_from_pca(alpha, k=21):
    """
    alpha:      (B, d) or (d,)
    pca_matrix: (k*k, d)
    """  
    if alpha.dim() == 1:
        # single kernel
        kernel = torch.matmul(pca_matrix, alpha)   # (k*k,)
        kernel = kernel.view(k, k)
        kernel = torch.relu(kernel)
        kernel = kernel / kernel.sum()
        # kernel = torch.matmul(pca_matrix, alpha)   # (k*k,)
        # kernel = torch.softmax(kernel, dim=0)
        # kernel = kernel.view(k, k)
    else:
        # batch of kernels
        B = alpha.size(0)
        kernel = torch.matmul(alpha, pca_matrix.t())  # (B, k*k)
        kernel = kernel.view(B, k, k)
        kernel = torch.relu(kernel)
        kernel = kernel / kernel.sum(dim=(-2, -1), keepdim=True)

    return kernel
