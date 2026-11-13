import math

import torch
import torch.nn.functional as F

from .functional import _contrast_loss


class Embed(torch.nn.Module):
    """Linear projection into the shared contrastive space, followed by L2 normalization."""

    def __init__(self, in_dim: int, out_dim: int = 128) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return F.normalize(x, p=2, dim=1)


class ContrastMemory(torch.nn.Module):
    """Memory buffer supplying a large number of negatives.

    Two buffers are kept: one stores the student projection of past samples, the other the teacher projection. Each forward pass uses the opposite buffer as negatives, then updates the slot of every sample with a momentum average.

    Args:
        feat_dim (int): Dimension of the shared projection space.
        n_data (int): Number of samples in the training set.
        nce_n (int, optional): Number of negatives paired with each positive. Defaults to 16384.
        nce_t (float, optional): Temperature of the contrastive distribution. Defaults to 0.07.
        nce_mom (float, optional): Momentum used to update the memory buffers. Defaults to 0.5.
    """

    def __init__(
        self,
        feat_dim: int,
        n_data: int,
        nce_n: int = 16384,
        nce_t: float = 0.07,
        nce_mom: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_data = n_data
        self.nce_n = nce_n
        self.nce_t = nce_t
        self.nce_mom = nce_mom
        self.register_buffer("Z_t", torch.tensor(-1.0))
        self.register_buffer("Z_s", torch.tensor(-1.0))
        stdv = 1.0 / math.sqrt(feat_dim / 3.0)
        self.register_buffer(
            "memory_s", torch.rand(n_data, feat_dim).mul_(2 * stdv).add_(-stdv)
        )
        self.register_buffer(
            "memory_t", torch.rand(n_data, feat_dim).mul_(2 * stdv).add_(-stdv)
        )

        # write position used when no dataset indices are given, not saved in checkpoints
        self.register_buffer(
            "_cursor", torch.zeros((), dtype=torch.long), persistent=False
        )

    def _next_index(self, batch_size: int) -> torch.Tensor:
        """Assign consecutive memory slots to a batch and advance the cursor."""
        offset = torch.arange(batch_size, device=self.memory_s.device)  # type: ignore
        index = (self._cursor + offset) % self.n_data
        self._cursor.fill_((self._cursor + batch_size) % self.n_data)  # type: ignore
        return index

    def _sample_negatives(self, index: torch.Tensor) -> torch.Tensor:
        """Sample K negatives per anchor and put the positive slot first."""
        negatives = torch.randint(
            0,
            self.n_data,
            (index.numel(), self.nce_n),
            device=self.memory_s.device,  # type: ignore
        )  # type: ignore
        return torch.cat([index.view(-1, 1), negatives], dim=1)

    def forward(
        self,
        student_stages: torch.Tensor,
        teacher_stages: torch.Tensor,
        index: torch.Tensor | None = None,
        contrast_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match student and teacher features against the memory buffers.

        Args:
            student_stages (torch.Tensor): L2-normalized student features of shape (batch, feat_dim).
            teacher_stages (torch.Tensor): L2-normalized teacher features of shape (batch, feat_dim).
            index (torch.Tensor | None, optional): Dataset index of each sample, used for the memory update. If None, consecutive slots are assigned with an internal cursor, or taken from the first column of ``contrast_idx`` when that is provided. Defaults to None.
            contrast_idx (torch.Tensor | None, optional): Precomputed memory indices of shape (batch, K + 1), positive slot first. If None, K negatives are sampled uniformly. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Scores for the student-anchored and teacher-anchored losses, each of shape (batch, K + 1).
        """
        batch_size = student_stages.size(0)
        feat_dim = self.memory_s.size(1)  # type: ignore
        device = self.memory_s.device

        if index is None:
            if contrast_idx is not None:
                index = contrast_idx[:, 0]
            else:
                index = self._next_index(batch_size)
        index = index.reshape(-1).to(device=device, dtype=torch.long)  # type: ignore
        if index.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} indices, got {index.numel()}.")

        if contrast_idx is None:
            contrast_idx = self._sample_negatives(index)
        else:
            contrast_idx = contrast_idx.to(device=device, dtype=torch.long)  # type: ignore
            if contrast_idx.shape != (batch_size, self.nce_n + 1):
                raise ValueError(
                    f"contrast_idx must have shape {(batch_size, self.nce_n + 1)}, "
                    f"got {tuple(contrast_idx.shape)}."
                )

        # using teacher as anchor, choose positive and negatives over the student side
        weight_s = torch.index_select(
            self.memory_s,  #  type: ignore
            0,
            contrast_idx.reshape(-1),
        ).detach()
        weight_s = weight_s.view(batch_size, self.nce_n + 1, feat_dim)
        out_t = torch.bmm(weight_s, teacher_stages.reshape(batch_size, feat_dim, 1))
        out_t = torch.exp(out_t / self.nce_t).squeeze(-1)

        # using student as anchor, choose positive and negatives over the teacher side
        weight_t = torch.index_select(
            self.memory_t,  #  type: ignore
            0,
            contrast_idx.reshape(-1),
        ).detach()
        weight_t = weight_t.view(batch_size, self.nce_n + 1, feat_dim)
        out_s = torch.bmm(weight_t, student_stages.reshape(batch_size, feat_dim, 1))
        out_s = torch.exp(out_s / self.nce_t).squeeze(-1)

        # set Z once, on the first forward pass
        if self.Z_t.item() < 0:  # type: ignore
            with torch.no_grad():
                self.Z_t.copy_((out_t.mean() * self.n_data).detach())  # type: ignore
        if self.Z_s.item() < 0:  # type: ignore
            with torch.no_grad():
                self.Z_s.copy_((out_s.mean() * self.n_data).detach())  # type: ignore

        out_t = out_t / self.Z_t.clamp(1e-8)  # type: ignore
        out_s = out_s / self.Z_s.clamp(1e-8)  # type: ignore

        # momentum update of the memory slots of the current samples
        with torch.no_grad():
            pos_mem_t = torch.index_select(self.memory_t, 0, index)  # type: ignore
            pos_mem_t.mul_(self.nce_mom)
            pos_mem_t.add_(teacher_stages.detach(), alpha=1.0 - self.nce_mom)
            pos_mem_t = F.normalize(pos_mem_t, p=2, dim=1)
            self.memory_t.index_copy_(0, index, pos_mem_t)  # type: ignore

            pos_mem_s = torch.index_select(self.memory_s, 0, index)  # type: ignore
            pos_mem_s.mul_(self.nce_mom)
            pos_mem_s.add_(student_stages.detach(), alpha=1.0 - self.nce_mom)
            pos_mem_s = F.normalize(pos_mem_s, p=2, dim=1)
            self.memory_s.index_copy_(0, index, pos_mem_s)  # type: ignore

        return out_s, out_t


class CRDLoss(torch.nn.Module):
    """Contrastive Representation Distillation loss with its own projection layers and memory.

    The original paper (CONTRASTIVE REPRESENTATION DISTILLATION) can be found at https://arxiv.org/pdf/1910.10699.

    The original code implementation can be found at https://github.com/HobbitLong/RepDistiller.

    Args:
        n_data (int): Number of samples in the training set, i.e. M in Eq. (19). Should match the dataset size.
        student_dim (int): Dimension of the flattened student feature.
        teacher_dim (int): Dimension of the flattened teacher feature.
        feat_dim (int, optional): Dimension of the shared projection space. Defaults to 128.
        nce_n (int, optional): Number of negatives paired with each positive. Defaults to 16384.
        nce_t (float, optional): Temperature of the contrastive distribution. Defaults to 0.07.
        nce_mom (float, optional): Momentum used to update the memory buffers. Defaults to 0.5.
    """

    def __init__(
        self,
        n_data: int,
        student_dim: int,
        teacher_dim: int,
        feat_dim: int = 128,
        nce_n: int = 16384,
        nce_t: float = 0.07,
        nce_mom: float = 0.5,
        student_gamma: float = 1.0,
        teacher_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_s = Embed(student_dim, feat_dim)
        self.embed_t = Embed(teacher_dim, feat_dim)
        self.memory = ContrastMemory(feat_dim, n_data, nce_n, nce_t, nce_mom)
        self.student_gamma = student_gamma
        self.teacher_gamma = teacher_gamma

    def forward(
        self,
        student_stages: torch.Tensor,
        teacher_stages: torch.Tensor,
        index: torch.Tensor | None = None,
        contrast_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the CRD loss.

        Args:
            student_stages (torch.Tensor): Raw student features, flattened per sample before projection.
            teacher_stages (torch.Tensor): Raw teacher features, flattened per sample before projection.
            index (torch.Tensor | None, optional): Dataset index of each sample, used for the memory update. If None, the internal cursor is used. Defaults to None.
            contrast_idx (torch.Tensor | None, optional): Precomputed memory indices of shape (batch, K + 1), positive slot first. If None, they are sampled uniformly. Defaults to None.

        Returns:
            torch.Tensor: The distillation loss.
        """
        if isinstance(student_stages, list):
            student_stages = student_stages[0]
        if isinstance(teacher_stages, list):
            teacher_stages = teacher_stages[0]
        embedded_student = self.embed_s(student_stages)
        embedded_teacher = self.embed_t(teacher_stages)
        student_out, teacher_out = self.memory(
            embedded_student, embedded_teacher, index=index, contrast_idx=contrast_idx
        )
        student_loss = (
            _contrast_loss(student_out, self.memory.n_data) * self.student_gamma
        )  # type: ignore
        teacher_loss = (
            _contrast_loss(teacher_out, self.memory.n_data) * self.teacher_gamma
        )  # type: ignore
        return student_loss + teacher_loss
