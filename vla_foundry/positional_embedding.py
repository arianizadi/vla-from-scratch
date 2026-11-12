import torch

# Disable torch.compile/dynamo for cache updates to avoid in-graph buffer overwrites
from torch._dynamo import disable as _dynamo_disable  # type: ignore

from vla_foundry.model_utils import Float32Module


def identity_with_cast(q, k, v, offset: int = 0):
    return q.to(v.dtype), k.to(v.dtype), v


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, offset: int = 0):
    # NOTE: This could probably be moved to Triton
    assert cos.shape[1] >= offset + x.shape[1], (
        f"Offset and/or input sequence is too large,\
        \n offset: {offset}, seq_len: {x.shape[1]}, max: {cos.shape[1]}"
    )

    # Handle a possible sequence length mismatch in between q and k
    cos_out = cos[:, offset : offset + x.shape[1], :, :]
    sin_out = sin[:, offset : offset + x.shape[1], :, :]

    return (x * cos_out) + (rotate_half(x) * sin_out)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox


    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim_model: int, seq_len: int, *_, legacy_autocast_table: bool = False, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        self.dim_model = dim_model
        # True reproduces the table of checkpoints trained before the table was computed in float32
        # (see _update_cos_sin_tables). Never use it for a new run.
        self.legacy_autocast_table = legacy_autocast_table
        self.register_buffer("inv_freq", torch.zeros(self.dim_model // 2))

        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
        self.seq_len = seq_len
        self.reset_parameters()

    def reset_parameters(self):
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim_model, 2).float() / self.dim_model))
        self._update_cos_sin_tables(self.seq_len)

    @_dynamo_disable
    def _update_cos_sin_tables(self, seq_len: int = None, device: torch.device = None, dtype: torch.dtype = None):
        # If no seq_len is provided, use the cached one
        # If the seq_len is smaller than the cached one it is included in the cached one so no need to update
        if seq_len is None or seq_len < self._seq_len_cached:
            seq_len = self._seq_len_cached

        # Reset the tables if the sequence length has increased,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len > self._seq_len_cached or self._cos_cached.device != device or self._cos_cached.dtype != dtype:
            self._seq_len_cached = seq_len
            if self.legacy_autocast_table:
                with torch.no_grad():
                    t = torch.arange(seq_len, device=device, dtype=torch.float32)
                    freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(dtype))
                    emb = torch.cat((freqs, freqs), dim=-1).to(device)
                    self._cos_cached = emb.cos().to(dtype)[None, :, None, :]
                    self._sin_cached = emb.sin().to(dtype)[None, :, None, :]
                return
            # The angles are computed in float32 with autocast switched off, and only the finished
            # cos / sin values are cast to the working dtype. Under bfloat16 autocast the product
            # position x frequency would otherwise be formed in bfloat16, which keeps 8 significant bits:
            # every position above 256 collapses onto an even one and above 512 onto a multiple of 4, so
            # neighbouring tokens get the same rotation and the transformer cannot tell them apart
            # (641 distinct positions out of 2048).
            device_type = device.type if isinstance(device, torch.device) else "cpu"
            with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
                t = torch.arange(seq_len, device=device, dtype=torch.float32)
                freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
                emb = torch.cat((freqs, freqs), dim=-1)

                # Create fresh tensors to avoid aliasing and ensure assignment happens outside compiled graphs
                self._cos_cached = emb.cos().to(dtype)[None, :, None, :]
                self._sin_cached = emb.sin().to(dtype)[None, :, None, :]

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_cos_sin_tables(k.shape[1] + offset, device=k.device, dtype=k.dtype)
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, offset),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, offset),
        )


class RotaryWithCast(RotaryEmbedding):
    def forward(self, q, k, v, offset: int = 0):
        q, k = super().forward(q, k, offset)
        return q.to(v.dtype), k.to(v.dtype), v


def get_pos_embed(model_params):
    head_dim = model_params.hidden_dim // model_params.n_heads
    if model_params.positional_embedding_type == "rotary":
        return RotaryWithCast(head_dim, model_params.max_seq_len)
    elif model_params.positional_embedding_type == "rotary_legacy_autocast":
        # Only for evaluating checkpoints trained before the rotary table was computed in float32.
        return RotaryWithCast(head_dim, model_params.max_seq_len, legacy_autocast_table=True)
    elif model_params.positional_embedding_type == "rotary_float32":
        # TODO: Currently results in some slowdowns. Need to investigate.
        return Float32Module(RotaryWithCast(head_dim, model_params.max_seq_len), cast_outputs_back=True)
    elif model_params.positional_embedding_type == "none":
        return identity_with_cast
    else:
        raise RuntimeError(f"Unknown positional embedding type {model_params.positional_embedding_type}")
