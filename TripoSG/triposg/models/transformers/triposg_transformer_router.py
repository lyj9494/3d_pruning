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
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        inner_dim=None,  # MODIFIED: Added inner_dim
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.proj_in = nn.Linear(dim, inner_dim)
        self.net = nn.Sequential(
            get_activation(activation_fn),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.net(hidden_states)
        return hidden_states


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
        self.attn1 = Attention(
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
        self.ff = TripoSGFeedForward(dim, dropout=dropout, activation_fn=activation_fn, inner_dim=ff_inner_dim, mult=ff_mult)

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
        # MODIFIED: Added reuse parameters
        reuse_att=None, 
        reuse_mlp=None,
        reuse_att_weight=0, 
        reuse_mlp_weight=0,
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

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        
        # MODIFIED: Added attention reuse logic
        if reuse_att is not None:
            attn_output = reuse_att * reuse_att_weight + attn_output * (1 - reuse_att_weight)

        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)

            attn_output_cross = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output_cross + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        # MODIFIED: Added mlp reuse logic
        if reuse_mlp is not None:
            ff_output = reuse_mlp * reuse_mlp_weight + ff_output * (1 - reuse_mlp_weight)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        # MODIFIED: Return attention and mlp output
        return hidden_states, (attn_output, ff_output)


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
        
        # MODIFIED: Added reset method
        self.reset()

    # MODIFIED: Added reset method
    def reset(self):
        self.reuse_feature = [None] * len(self.transformer_blocks)

    # MODIFIED: Added add_router method
    def add_router(self, num_steps, timestep_map):
        self.routers = torch.nn.ModuleList([
            Router(2 * len(self.transformer_blocks)) for _ in range(num_steps)
        ])
        self.timestep_map = {timestep: i for i, timestep in enumerate(timestep_map)}

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
        # MODIFIED: Added router-related parameters
        thres=None, 
        activate_router=False, 
        fix_reuse_feature=False, 
        ori=False,
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

        # MODIFIED: Added router logic
        if activate_router and not ori:
            router_idx = self.timestep_map[timestep[0].item()]
            scores = self.routers[router_idx]()
            if thres is None:
                weights = scores
            else:
                weights = STE.apply(scores, thres)
            router_l1_loss = scores.sum()

        # 4. Blocks
        for i, block in enumerate(self.transformer_blocks):
            att, mlp = None, None
            reuse_att_weight, reuse_mlp_weight = 0, 0
            
            if not ori:
                if self.reuse_feature[i] is not None:
                    att, mlp = self.reuse_feature[i]
                if activate_router:
                    reuse_att_weight, reuse_mlp_weight = weights[2*i], weights[2*i+1]

            hidden_states, reuse_feature = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=emb,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                reuse_att=att, 
                reuse_mlp=mlp,
                reuse_att_weight=reuse_att_weight,
                reuse_mlp_weight=reuse_mlp_weight,
            )
            
            if not fix_reuse_feature and not ori:
                self.reuse_feature[i] = reuse_feature

        # 5. Output
        hidden_states = self.proj_out(hidden_states)

        if not return_dict:
            if activate_router and not ori:
                return (hidden_states, router_l1_loss)
            else:
                return (hidden_states,)

        if activate_router and not ori:
            return Transformer1DModelOutput_(sample=hidden_states), router_l1_loss
        else:
            return Transformer1DModelOutput_(sample=hidden_states)


# MODIFIED: Added STE and Router
class STE(torch.autograd.Function):
     @staticmethod
     def forward(ctx, i, thres=0.5):
         return (i>thres).float()

     @staticmethod
     def backward(ctx, grad_output):
         return grad_output, None

class Router(nn.Module):
    def __init__(self, num_choises):
        super().__init__()
        self.num_choises = num_choises
        self.prob = torch.nn.Parameter(torch.randn(num_choises), requires_grad=True)
        self.activation = torch.nn.Sigmoid()
        
    def forward(self, x=None):
        return self.activation(self.prob)
