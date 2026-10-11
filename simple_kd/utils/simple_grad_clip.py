import torch


def _unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)


def _adaptive_clip_grad(parameters, clip_factor: float = 0.01, eps: float = 1e-3, norm_type: float = 2.0, ignore_1D: bool = False):
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
        max_norm = _unitwise_norm(p_data, norm_type=norm_type).clamp_(min=eps).mul_(clip_factor)
        grad_norm = _unitwise_norm(g_data, norm_type=norm_type)
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grads)

class SimpleGradClip:
    def __init__(self,
                clip_type: str = "value",
                clip_value: float = 2.0,
                norm_type: float = 2.0,
                clip_factor: float = 0.02,
                eps: float = 1e-3,
                error_if_nonfinite: bool = False,
                foreach: bool = False,
                ignore_1D: bool = False):
        """This is a simple wrapper class for gradient clipping.

        Args:
            clip_type (str, optional): Accepts "value" and "norm" from pytorch, along with "agc" supplied from the timm repository. Defaults to "value".
            clip_value (float, optional): The clip value, used as is for value clipping and as max_norm for norm clipping. Defaults to 2.0.
            norm_type (float, optional): The norm type for the gradient norm and agc methods. Defaults to 2.0.
            clip_factor (float, optional): The clip factor used by the agc method, ignored otherwise. Defaults to 0.02.
            eps (float, optional): The eps valued used by the agc method, ignored otherwise. Defaults to 1e-3.
            error_if_nonfinite (bool, optional): Toggle error for the gradient norm method. Defaults to False.
            foreach (bool, optional): Toggle for method use for the gradient norm and value methods. Defaults to None.

        Raises:
            Exception: _description_
        """
        super().__init__()
        if clip_type == "value":
            self.inner_fn = torch.nn.utils.clip_grad_value_
            self.gradient_params = {
                "clip_value": clip_value,
                "foreach": foreach
            }
        elif clip_type == "norm":
            self.inner_fn = torch.nn.utils.clip_grad_norm_
            self.gradient_params = {
                "max_norm": clip_value,
                "norm_type": norm_type,
                "foreach": foreach,
                "error_if_nonfinite": error_if_nonfinite
            }
        elif clip_type == "agc":
            self.inner_fn = _adaptive_clip_grad
            self.gradient_params = {
                "clip_factor": clip_factor,
                "norm_type": norm_type,
                "eps": eps,
                "ignore_1D": ignore_1D
            }
        else:
            raise Exception("Non Valid method selected.")

    def __call__(self, x):
        return self.inner_fn(x, **self.gradient_params)
