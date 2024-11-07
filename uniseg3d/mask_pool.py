import torch
import torch.nn.functional as F


def mask_pool(x, mask):
    """
    Args:
        x: [D, M]
        mask: [N, M]
    """
    with torch.no_grad():
        mask = mask.detach()
        mask = (mask > 0).to(mask.dtype)
        denorm = mask.sum(dim=(-1), keepdim=True) + 1e-8

    mask_pooled_x = torch.einsum(
        "dm,nm->nd",
        x,
        mask / denorm,
    )
    return mask_pooled_x
