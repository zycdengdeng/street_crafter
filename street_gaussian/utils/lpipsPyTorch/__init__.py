import torch

from .modules.lpips import LPIPS


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    if not hasattr(lpips, 'criterion'):
        lpips.criterion = {}

    key = f"{net_type}_{version}"
    if key not in lpips.criterion:
        lpips.criterion[key] = LPIPS(net_type, version).to(device)

    criterion = lpips.criterion[key]
    return criterion(x, y)
