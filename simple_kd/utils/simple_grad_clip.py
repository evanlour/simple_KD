import torch


def _unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)


def _adaptive_clip_grad(
    parameters,
    clip_factor: float = 0.01,
    eps: float = 1e-3,
    norm_type: float = 2.0,
    ignore_1D: bool = False,
):
    """Implementation of AGC gradient clipping.
    This code is directly adapted from the timm repository

    https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/agc.py by Ross Wightman

    Reference to the original paper:

    @article{brock2021high,
        author={Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan},
        title={High-Performance Large-Scale Image Recognition Without Normalization},
        journal={arXiv preprint arXiv:},
        url={https://arxiv.org/abs/2102.06171}
        year={2021}
    }

    The original JAX implementation(paper authors) can be found at https://github.com/deepmind/deepmind-research/tree/master/nfnets.
    Phil Wang's PyTorch gist: https://gist.github.com/lucidrains/0d6560077edac419ab5d3aa29e674d5c
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is None:
            continue
        if ignore_1D and p.ndim <= 1:
            continue

        p_data = p.detach()
        g_data = p.grad.detach()
        max_norm = (
            _unitwise_norm(p_data, norm_type=norm_type)
            .clamp_(min=eps)
            .mul_(clip_factor)
        )
        grad_norm = _unitwise_norm(g_data, norm_type=norm_type)
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grads)


class SimpleGradClip:
    """Wrapper class providing gradient clipping strategies (value, norm, and AGC).

    Supports:
    - "value": Elementwise gradient value clipping via torch.nn.utils.clip_grad_value_.
    - "norm": Total gradient norm clipping via torch.nn.utils.clip_grad_norm_.
    - "agc": Adaptive Gradient Clipping (Brock et al., 2021) adapted from timm.

    Code implementation for the Adaptive Gradient Clipping is from https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/agc.py.

    Args:
        clip_type (str, optional): Clipping strategy. One of "value", "norm", or "agc". Defaults to "value".
        clip_value (float, optional): Maximum clip value (used as max_value for value clipping and max_norm for norm clipping). Defaults to 2.0.
        norm_type (float, optional): Norm type used for norm clipping and AGC. Defaults to 2.0.
        clip_factor (float, optional): Clipping threshold factor for AGC (ratio of grad norm to param norm). Defaults to 0.02.
        eps (float, optional): Epsilon floor to prevent zero-norm division in AGC. Defaults to 1e-3.
        error_if_nonfinite (bool, optional): If True, raises error if total norm is NaN/inf. Defaults to False.
        foreach (bool, optional): Use faster foreach implementation in PyTorch if available. Defaults to False.
        ignore_1D (bool, optional): If True, skips 1D parameters (biases, batchnorm weights) in AGC. Defaults to False.

    Raises:
        ValueError: If an unsupported clip_type is specified.
    """

    def __init__(
        self,
        clip_type: str = "value",
        clip_value: float = 2.0,
        norm_type: float = 2.0,
        clip_factor: float = 0.02,
        eps: float = 1e-3,
        error_if_nonfinite: bool = False,
        foreach: bool = False,
        ignore_1D: bool = False,
    ):
        super().__init__()
        if clip_type == "value":
            self.inner_fn = torch.nn.utils.clip_grad_value_
            self.gradient_params = {"clip_value": clip_value, "foreach": foreach}
        elif clip_type == "norm":
            self.inner_fn = torch.nn.utils.clip_grad_norm_
            self.gradient_params = {
                "max_norm": clip_value,
                "norm_type": norm_type,
                "foreach": foreach,
                "error_if_nonfinite": error_if_nonfinite,
            }
        elif clip_type == "agc":
            self.inner_fn = _adaptive_clip_grad
            self.gradient_params = {
                "clip_factor": clip_factor,
                "norm_type": norm_type,
                "eps": eps,
                "ignore_1D": ignore_1D,
            }
        else:
            raise ValueError(
                f"Invalid clip_type: '{clip_type}'. Supported: 'value', 'norm', 'agc'."
            )

    def __call__(self, parameters) -> torch.Tensor | None:
        """Applies gradient clipping in-place to the given parameters.

        Args:
            parameters (Iterable[Tensor] or Tensor): Model parameters whose .grad attributes will be clipped.

        Returns:
            torch.Tensor or None: Total norm of parameters if norm clipping is used, else None.
        """
        return self.inner_fn(parameters, **self.gradient_params)
