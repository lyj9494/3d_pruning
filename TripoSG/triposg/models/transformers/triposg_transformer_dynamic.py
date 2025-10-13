# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AdaLayerNorm, AdaLayerNormZero
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


@dataclass
class Transformer1DModelOutput_(BaseOutput):
    """
    The output of [`Transformer3DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on the `encoder_hidden_states` input. If
            `return_dict` is `False`, this is the only output.
    """

    sample: torch.FloatTensor = None


def get_activation(act_fn: str):
    if act_fn == "gelu":
        return nn.GELU()
    elif act_fn == "gelu-approximate":
        return nn.GELU(approximate="tanh")
    elif act_fn == "geglu":
        return nn.GEGLU()
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


class TripoSGFeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        act_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        act_fn: str = "geglu",
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.Sequential(
            get_activation(act_fn),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )
        self.proj_in = nn.Linear(dim, inner_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.net(hidden_states)
        return hidden_states


# MODIFIED: Added DynamicMlp
class DynamicMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        if mask is not None:
            x = x * mask.unsqueeze(1)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# MODIFIED: Added DynamicAttention
class DynamicAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, mask=None, **cross_attention_kwargs):
        # The `Attention` class can call different attention processors / implementations
        # here, so we simply pass along the `mask` argument to the processor.
        return super().forward(hidden_states, encoder_hidden_states, attention_mask, mask=mask, **cross_attention_kwargs)


@maybe_allow_in_graph
class TripoSGTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the cross-attention latent dimensions to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (`int`, *optional*):
            The number of diffusion steps used during training. See `Transformer2DModel` for more details.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if you want to use {norm_type}."
            )

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        
        # MODIFIED: Use DynamicAttention
        self.attn1 = DynamicAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None:
            # We currently only use AdaLayerNormZero for self-attention blocks.
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is a cross-attention
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        
        # MODIFIED: Use DynamicMlp
        self.ff = DynamicMlp(dim, ff_inner_dim, act_layer=lambda: get_activation(activation_fn), drop=dropout)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        mask: Optional[torch.FloatTensor] = None, # MODIFIED: Added mask
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # MODIFIED: Pass mask to attention
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            mask=mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states, mask=mask)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class TripoSGTransformerModel(ModelMixin, ConfigMixin):
    """
    A 3D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
        sample_size (`int`, *optional*): The width of the latent images of the unet.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"`, or `"ada_norm_zero"`.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether or not to use elementwise affine parameters for the normalization layer.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = 4,
        out_channels: Optional[int] = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        only_cross_attention: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_layers * attention_head_dim

        # 1. Input
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 2. Timestep embeddings
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)

        # 3. Positional embeddings
        self.pos_embed = nn.Embedding(num_vector_embeds, inner_dim)

        # 4. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TripoSGTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    ff_inner_dim=ff_inner_dim,
                    ff_mult=ff_mult,
                    activation_fn=activation_fn,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_type=norm_type,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                )
                for d in range(num_layers)
            ]
        )

        # 5. Output
        self.out_channels = in_channels if out_channels is None else out_channels
        self.proj_out = nn.Linear(inner_dim, self.out_channels)

        self.gradient_checkpointing = False

    # MODIFIED: Added add_mask method
    def add_mask(self, num_layers):
        self.masks = torch.nn.Parameter(torch.ones(num_layers, 2), requires_grad=True)
        self.num_layers = num_layers

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (TripoSGTransformerBlock, Attention)):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        mask: Optional[torch.FloatTensor] = None, # MODIFIED: Added mask
    ):
        """
        The [`TripoSGTransformerModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            timestep (`torch.LongTensor`, *optional*):
                Timesteps to be used for conditioning.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        # 1. Input
        batch, _, seq_len = hidden_states.shape
        residual = hidden_states
        hidden_states = self.proj_in(hidden_states)

        # 2. Timestep embedding
        if timestep is not None:
            t_emb = self.time_proj(timestep)
            t_emb = t_emb.to(dtype=self.dtype)
            emb = self.timestep_embedder(t_emb)
        else:
            emb = None

        # 3. Positional embedding
        pos_embed = self.pos_embed.weight.unsqueeze(0).repeat(batch, 1, 1)
        hidden_states = hidden_states + pos_embed

        # 4. Blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training:
                # MODIFIED: Pass mask to block
                current_mask = STE.apply(self.masks[i])
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=emb,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    mask=current_mask,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=emb,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    mask=mask,
                )

        # 5. Output
        hidden_states = self.proj_out(hidden_states)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput_(sample=hidden_states)


# MODIFIED: Added STE
class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        return (i > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
