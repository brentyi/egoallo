from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from typing import Literal, assert_never

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float
from loguru import logger
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor, nn

from .fncsmpl import SmplhModel, SmplhShapedAndPosed
from .tensor_dataclass import TensorDataclass
from .transforms import SE3, SO3


def project_rotmats_via_svd(
    rotmats: Float[Tensor, "*batch 3 3"],
) -> Float[Tensor, "*batch 3 3"]:
    u, s, vh = torch.linalg.svd(rotmats)
    del s
    return torch.einsum("...ij,...jk->...ik", u, vh)


class EgoDenoiseTraj(TensorDataclass):
    """Data structure for denoising. Contains tensors that we are denoising, as
    well as utilities for packing + unpacking them."""

    betas: Float[Tensor, "*#batch timesteps 16"]
    """Body shape parameters. We don't really need the timesteps axis here,
    it's just for convenience."""

    body_rotmats: Float[Tensor, "*#batch timesteps 21 3 3"]
    """Local orientations for each body joint."""

    contacts: Float[Tensor, "*#batch timesteps 21"]
    """Contact boolean for each joint."""

    hand_rotmats: Float[Tensor, "*#batch timesteps 30 3 3"] | None
    """Local orientations for each body joint."""

    @staticmethod
    def get_packed_dim(include_hands: bool) -> int:
        packed_dim = 16 + 21 * 9 + 21
        if include_hands:
            packed_dim += 30 * 9
        return packed_dim

    def apply_to_body(self, body_model: SmplhModel) -> SmplhShapedAndPosed:
        device = self.betas.device
        dtype = self.betas.dtype
        assert self.hand_rotmats is not None
        shaped = body_model.with_shape(self.betas)
        posed = shaped.with_pose(
            T_world_root=SE3.identity(device=device, dtype=dtype).parameters(),
            local_quats=SO3.from_matrix(
                torch.cat([self.body_rotmats, self.hand_rotmats], dim=-3)
            ).wxyz,
        )
        return posed

    def pack(self) -> Float[Tensor, "*#batch timesteps d_state"]:
        """Pack trajectory into a single flattened vector."""
        (*batch, time, num_joints, _, _) = self.body_rotmats.shape
        assert num_joints == 21
        return torch.cat(
            [
                x.reshape((*batch, time, -1))
                for x in vars(self).values()
                if x is not None
            ],
            dim=-1,
        )

    @classmethod
    def unpack(
        cls,
        x: Float[Tensor, "*#batch timesteps d_state"],
        include_hands: bool,
        project_rotmats: bool = False,
    ) -> EgoDenoiseTraj:
        """Unpack trajectory from a single flattened vector.

        Args:
            x: Packed trajectory.
            project_rotmats: If True, project the rotation matrices to SO(3) via SVD.
        """
        (*batch, time, d_state) = x.shape
        assert d_state == cls.get_packed_dim(include_hands)

        if include_hands:
            betas, body_rotmats_flat, contacts, hand_rotmats_flat = torch.split(
                x, [16, 21 * 9, 21, 30 * 9], dim=-1
            )
            body_rotmats = body_rotmats_flat.reshape((*batch, time, 21, 3, 3))
            hand_rotmats = hand_rotmats_flat.reshape((*batch, time, 30, 3, 3))
            assert betas.shape == (*batch, time, 16)
        else:
            betas, body_rotmats_flat, contacts = torch.split(
                x, [16, 21 * 9, 21], dim=-1
            )
            body_rotmats = body_rotmats_flat.reshape((*batch, time, 21, 3, 3))
            hand_rotmats = None
            assert betas.shape == (*batch, time, 16)

        if project_rotmats:
            # We might want to handle the -1 determinant case as well.
            body_rotmats = project_rotmats_via_svd(body_rotmats)

        return EgoDenoiseTraj(
            betas=betas,
            body_rotmats=body_rotmats,
            contacts=contacts,
            hand_rotmats=hand_rotmats,
        )


@dataclass(frozen=True)
class EgoDenoiserConfig:
    max_t: int = 1000
    fourier_enc_freqs: int = 3
    d_latent: int = 512
    d_feedforward: int = 2048
    d_noise_emb: int = 1024
    num_heads: int = 4
    encoder_layers: int = 6
    decoder_layers: int = 6
    dropout_p: float = 0.0
    activation: Literal["gelu", "relu"] = "gelu"

    positional_encoding: Literal["transformer", "rope"] = "rope"
    noise_conditioning: Literal["token", "film"] = "token"

    xattn_mode: Literal["kv_from_cond_q_from_x", "kv_from_x_q_from_cond"] = (
        "kv_from_cond_q_from_x"
    )

    include_canonicalized_cpf_rotation_in_cond: bool = True
    include_hands: bool = True
    """Whether to include hand joints (+15 per hand) in the denoised state."""

    cond_param: Literal[
        "ours", "canonicalized", "absolute", "absrel", "absrel_global_deltas"
    ] = "ours"
    """Which conditioning parameterization to use.

    "ours" is the default, we try to be clever and design something with nice
        equivariance properties.
    "canonicalized" contains a transformation that's canonicalized to aligned
        to the first frame.
    "absolute" is the naive case, where we just pass in transformations
        directly.
    """

    include_hand_positions_cond: bool = False
    """Whether to include hand positions in the conditioning information."""

    @cached_property
    def d_cond(self) -> int:
        """Dimensionality of conditioning vector."""

        if self.cond_param == "ours":
            d_cond = 0
            d_cond += 12  # Relative CPF pose, flattened 3x4 matrix.
            d_cond += 1  # Floor height.
            if self.include_canonicalized_cpf_rotation_in_cond:
                d_cond += 9  # Canonicalized CPF rotation, flattened 3x3 matrix.
        elif self.cond_param == "canonicalized":
            d_cond = 12
        elif self.cond_param == "absolute":
            d_cond = 12
        elif self.cond_param == "absrel":
            # Both absolute and relative!
            d_cond = 24
        elif self.cond_param == "absrel_global_deltas":
            # Both absolute and relative!
            d_cond = 24
        else:
            assert_never(self.cond_param)

        # Add two 3D positions to the conditioning dimension if we're including
        # hand conditioning.
        if self.include_hand_positions_cond:
            d_cond = d_cond + 6

        d_cond = d_cond + d_cond * self.fourier_enc_freqs * 2  # Fourier encoding.
        return d_cond

    def make_cond(
        self,
        T_cpf_tm1_cpf_t: Float[Tensor, "batch time 7"],
        T_world_cpf: Float[Tensor, "batch time 7"],
        hand_positions_wrt_cpf: Float[Tensor, "batch time 6"] | None,
    ) -> Float[Tensor, "batch time d_cond"]:
        """Construct conditioning information from CPF pose."""

        (batch, time, _) = T_cpf_tm1_cpf_t.shape

        # Construct device pose conditioning.
        if self.cond_param == "ours":
            # Compute conditioning terms. +Z is up in the world frame. We want
            # the translation to be invariant to translations in the world X/Y
            # directions.
            height_from_floor = T_world_cpf[..., 6:7]

            cond_parts = [
                SE3(T_cpf_tm1_cpf_t).as_matrix()[..., :3, :].reshape((batch, time, 12)),
                height_from_floor,
            ]
            if self.include_canonicalized_cpf_rotation_in_cond:
                # We want the rotation to be invariant to rotations around the
                # world Z axis. Visualization of what's happening here:
                #
                # https://gist.github.com/brentyi/9226d082d2707132af39dea92b8609f6
                #
                # (The coordinate frame may differ by some axis-swapping
                # compared to the exact equations in the paper. But to the
                # network these will all look the same.)
                R_world_cpf = SE3(T_world_cpf).rotation().wxyz
                forward_cpf = R_world_cpf.new_tensor([0.0, 0.0, 1.0])
                forward_world = SO3(R_world_cpf) @ forward_cpf
                assert forward_world.shape == (batch, time, 3)
                R_canonical_world = SO3.from_z_radians(
                    -torch.arctan2(forward_world[..., 1], forward_world[..., 0])
                ).wxyz
                assert R_canonical_world.shape == (batch, time, 4)
                cond_parts.append(
                    (SO3(R_canonical_world) @ SO3(R_world_cpf))
                    .as_matrix()
                    .reshape((batch, time, 9)),
                )
            cond = torch.cat(cond_parts, dim=-1)
        elif self.cond_param == "canonicalized":
            # Align the first timestep.
            # Put poses so start is at origin, facing forward.
            R_world_cpf = SE3(T_world_cpf[:, 0:1, :]).rotation().wxyz
            forward_cpf = R_world_cpf.new_tensor([0.0, 0.0, 1.0])
            forward_world = SO3(R_world_cpf) @ forward_cpf
            assert forward_world.shape == (batch, 1, 3)
            R_canonical_world = SO3.from_z_radians(
                -torch.arctan2(forward_world[..., 1], forward_world[..., 0])
            ).wxyz
            assert R_canonical_world.shape == (batch, 1, 4)

            R_canonical_cpf = SO3(R_canonical_world) @ SE3(T_world_cpf).rotation()
            t_canonical_cpf = SO3(R_canonical_world) @ SE3(T_world_cpf).translation()
            t_canonical_cpf = t_canonical_cpf - t_canonical_cpf[:, 0:1, :]

            cond = (
                SE3.from_rotation_and_translation(R_canonical_cpf, t_canonical_cpf)
                .as_matrix()[..., :3, :4]
                .reshape((batch, time, 12))
            )
        elif self.cond_param == "absolute":
            cond = SE3(T_world_cpf).as_matrix()[..., :3, :4].reshape((batch, time, 12))
        elif self.cond_param == "absrel":
            cond = torch.concatenate(
                [
                    SE3(T_world_cpf)
                    .as_matrix()[..., :3, :4]
                    .reshape((batch, time, 12)),
                    SE3(T_cpf_tm1_cpf_t)
                    .as_matrix()[..., :3, :4]
                    .reshape((batch, time, 12)),
                ],
                dim=-1,
            )
        elif self.cond_param == "absrel_global_deltas":
            cond = torch.concatenate(
                [
                    SE3(T_world_cpf)
                    .as_matrix()[..., :3, :4]
                    .reshape((batch, time, 12)),
                    SE3(T_cpf_tm1_cpf_t)
                    .rotation()
                    .as_matrix()
                    .reshape((batch, time, 9)),
                    (
                        SE3(T_world_cpf).rotation()
                        @ SE3(T_cpf_tm1_cpf_t).inverse().translation()
                    ).reshape((batch, time, 3)),
                ],
                dim=-1,
            )
        else:
            assert_never(self.cond_param)

        # Condition on hand poses as well.
        # We didn't use this for the paper.
        if self.include_hand_positions_cond:
            if hand_positions_wrt_cpf is None:
                logger.warning(
                    "Model is looking for hand conditioning but none was provided. Passing in zeros."
                )
                hand_positions_wrt_cpf = torch.zeros(
                    (batch, time, 6), device=T_world_cpf.device
                )
            assert hand_positions_wrt_cpf.shape == (batch, time, 6)
            cond = torch.cat([cond, hand_positions_wrt_cpf], dim=-1)

        cond = fourier_encode(cond, freqs=self.fourier_enc_freqs)
        assert cond.shape == (batch, time, self.d_cond)
        return cond


class EgoDenoiser(nn.Module):
    """Denoising network for human motion.

    Inputs are noisy trajectory, conditioning information, and timestep.
    Output is denoised trajectory.
    """

    def __init__(self, config: EgoDenoiserConfig):
        super().__init__()

        self.config = config
        Activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]

        # MLP encoders and decoders for each modality we want to denoise.
        modality_dims: dict[str, int] = {
            "betas": 16,
            "body_rotmats": 21 * 9,
            "contacts": 21,
        }
        if config.include_hands:
            modality_dims["hand_rotmats"] = 30 * 9

        assert sum(modality_dims.values()) == self.get_d_state()
        self.encoders = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(modality_dim, config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, config.d_latent),
                )
                for k, modality_dim in modality_dims.items()
            }
        )
        self.decoders = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(config.d_latent, config.d_latent),
                    nn.LayerNorm(normalized_shape=config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, modality_dim),
                )
                for k, modality_dim in modality_dims.items()
            }
        )

        # Helpers for converting between input dimensionality and latent dimensionality.
        self.latent_from_cond = nn.Linear(config.d_cond, config.d_latent)

        # Noise embedder.
        self.noise_emb = nn.Embedding(
            # index 0 will be t=1
            # index 999 will be t=1000
            num_embeddings=config.max_t,
            embedding_dim=config.d_noise_emb,
        )
        self.noise_emb_token_proj = (
            nn.Linear(config.d_noise_emb, config.d_latent, bias=False)
            if config.noise_conditioning == "token"
            else None
        )

        # Encoder / decoder layers.
        # Inputs are conditioning (current noise level, observations); output
        # is encoded conditioning information.
        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        d_latent=config.d_latent,
                        d_noise_emb=config.d_noise_emb,
                        d_feedforward=config.d_feedforward,
                        n_heads=config.num_heads,
                        dropout_p=config.dropout_p,
                        activation=config.activation,
                        include_xattn=False,  # No conditioning for encoder.
                        use_rope_embedding=config.positional_encoding == "rope",
                        use_film_noise_conditioning=config.noise_conditioning == "film",
                        xattn_mode=config.xattn_mode,
                    )
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        d_latent=config.d_latent,
                        d_noise_emb=config.d_noise_emb,
                        d_feedforward=config.d_feedforward,
                        n_heads=config.num_heads,
                        dropout_p=config.dropout_p,
                        activation=config.activation,
                        include_xattn=True,  # Include conditioning for the decoder.
                        use_rope_embedding=config.positional_encoding == "rope",
                        use_film_noise_conditioning=config.noise_conditioning == "film",
                        xattn_mode=config.xattn_mode,
                    )
                )
                for _ in range(config.decoder_layers)
            ]
        )

    def get_d_state(self) -> int:
        return EgoDenoiseTraj.get_packed_dim(self.config.include_hands)

    def forward(
        self,
        x_t_packed: Float[Tensor, "batch time state_dim"],
        t: Float[Tensor, "batch"],
        *,
        T_world_cpf: Float[Tensor, "batch time 7"],
        T_cpf_tm1_cpf_t: Float[Tensor, "batch time 7"],
        project_output_rotmats: bool,
        # Observed hand positions, relative to the CPF.
        hand_positions_wrt_cpf: Float[Tensor, "batch time 6"] | None,
        # Attention mask for using shorter sequences.
        mask: Bool[Tensor, "batch time"] | None,
        # Mask for when to drop out / keep conditioning information.
        cond_dropout_keep_mask: Bool[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch time state_dim"]:
        """Predict a denoised trajectory. Note that `t` refers to a noise
        level, not a timestep."""
        config = self.config

        x_t = EgoDenoiseTraj.unpack(x_t_packed, include_hands=self.config.include_hands)
        (batch, time, num_body_joints, _, _) = x_t.body_rotmats.shape
        assert num_body_joints == 21

        # Encode the trajectory into a single vector per timestep.
        x_t_encoded = (
            self.encoders["betas"](x_t.betas.reshape((batch, time, -1)))
            + self.encoders["body_rotmats"](x_t.body_rotmats.reshape((batch, time, -1)))
            + self.encoders["contacts"](x_t.contacts)
        )
        if self.config.include_hands:
            assert x_t.hand_rotmats is not None
            x_t_encoded = x_t_encoded + self.encoders["hand_rotmats"](
                x_t.hand_rotmats.reshape((batch, time, -1))
            )
        assert x_t_encoded.shape == (batch, time, config.d_latent)

        # Embed the diffusion noise level.
        assert t.shape == (batch,)
        noise_emb = self.noise_emb(t - 1)
        assert noise_emb.shape == (batch, config.d_noise_emb)

        # Prepare conditioning information.
        cond = config.make_cond(
            T_cpf_tm1_cpf_t,
            T_world_cpf=T_world_cpf,
            hand_positions_wrt_cpf=hand_positions_wrt_cpf,
        )

        # Randomly drop out conditioning information; this serves as a
        # regularizer that aims to improve sample diversity.
        if cond_dropout_keep_mask is not None:
            assert cond_dropout_keep_mask.shape == (batch,)
            cond = cond * cond_dropout_keep_mask[:, None, None]

        # Prepare encoder and decoder inputs.
        if config.positional_encoding == "rope":
            pos_enc = 0
        elif config.positional_encoding == "transformer":
            pos_enc = make_positional_encoding(
                d_latent=config.d_latent,
                length=time,
                dtype=cond.dtype,
            )[None, ...].to(x_t_encoded.device)
            assert pos_enc.shape == (1, time, config.d_latent)
        else:
            assert_never(config.positional_encoding)

        encoder_out = self.latent_from_cond(cond) + pos_enc
        decoder_out = x_t_encoded + pos_enc

        # Append the noise embedding to the encoder and decoder inputs.
        # This is weird if we're using rotary embeddings!
        if self.noise_emb_token_proj is not None:
            noise_emb_token = self.noise_emb_token_proj(noise_emb)
            assert noise_emb_token.shape == (batch, config.d_latent)
            encoder_out = torch.cat([noise_emb_token[:, None, :], encoder_out], dim=1)
            decoder_out = torch.cat([noise_emb_token[:, None, :], decoder_out], dim=1)
            assert (
                encoder_out.shape
                == decoder_out.shape
                == (batch, time + 1, config.d_latent)
            )
            num_tokens = time + 1
        else:
            num_tokens = time

        # Compute attention mask. This needs to be a fl
        if mask is None:
            attn_mask = None
        else:
            assert mask.shape == (batch, time)
            assert mask.dtype == torch.bool
            if self.noise_emb_token_proj is not None:  # Account for noise token.
                mask = torch.cat([mask.new_ones((batch, 1)), mask], dim=1)
            # Last two dimensions of mask are (query, key). We're masking out only keys;
            # it's annoying for the softmax to mask out entire rows without getting NaNs.
            attn_mask = mask[:, None, None, :].repeat(1, 1, num_tokens, 1)
            assert attn_mask.shape == (batch, 1, num_tokens, num_tokens)
            assert attn_mask.dtype == torch.bool

        # Forward pass through transformer.
        for layer in self.encoder_layers:
            encoder_out = layer(encoder_out, attn_mask, noise_emb=noise_emb)
        for layer in self.decoder_layers:
            decoder_out = layer(
                decoder_out, attn_mask, noise_emb=noise_emb, cond=encoder_out
            )

        # Remove the extra token corresponding to the noise embedding.
        if self.noise_emb_token_proj is not None:
            decoder_out = decoder_out[:, 1:, :]
        assert isinstance(decoder_out, Tensor)
        assert decoder_out.shape == (batch, time, config.d_latent)

        packed_output = torch.cat(
            [
                # Project rotation matrices for body_rotmats via SVD,
                (
                    project_rotmats_via_svd(
                        modality_decoder(decoder_out).reshape((-1, 3, 3))
                    ).reshape(
                        (batch, time, {"body_rotmats": 21, "hand_rotmats": 30}[key] * 9)
                    )
                    # if enabled,
                    if project_output_rotmats
                    and key in ("body_rotmats", "hand_rotmats")
                    # otherwise, just decode normally.
                    else modality_decoder(decoder_out)
                )
                for key, modality_decoder in self.decoders.items()
            ],
            dim=-1,
        )
        assert packed_output.shape == (batch, time, self.get_d_state())

        # Return packed output.
        return packed_output


@cache
def make_positional_encoding(
    d_latent: int, length: int, dtype: torch.dtype
) -> Float[Tensor, "length d_latent"]:
    """Computes standard Transformer positional encoding."""
    pe = torch.zeros(length, d_latent, dtype=dtype)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_latent, 2).float() * (-np.log(10000.0) / d_latent)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    assert pe.shape == (length, d_latent)
    return pe


def fourier_encode(
    x: Float[Tensor, "*#batch channels"], freqs: int
) -> Float[Tensor, "*#batch channels+2*freqs*channels"]:
    """Apply Fourier encoding to a tensor."""
    *batch_axes, x_dim = x.shape
    coeffs = 2.0 ** torch.arange(freqs, device=x.device)
    scaled = (x[..., None] * coeffs).reshape((*batch_axes, x_dim * freqs))
    return torch.cat(
        [
            x,
            torch.sin(torch.cat([scaled, scaled + torch.pi / 2.0], dim=-1)),
        ],
        dim=-1,
    )


@dataclass(frozen=True)
class TransformerBlockConfig:
    d_latent: int
    d_noise_emb: int
    d_feedforward: int
    n_heads: int
    dropout_p: float
    activation: Literal["gelu", "relu"]
    include_xattn: bool
    use_rope_embedding: bool
    use_film_noise_conditioning: bool
    xattn_mode: Literal["kv_from_cond_q_from_x", "kv_from_x_q_from_cond"]


class TransformerBlock(nn.Module):
    """An even-tempered Transformer block."""

    def __init__(self, config: TransformerBlockConfig) -> None:
        super().__init__()
        self.sattn_qkv_proj = nn.Linear(
            config.d_latent, config.d_latent * 3, bias=False
        )
        self.sattn_out_proj = nn.Linear(config.d_latent, config.d_latent, bias=False)

        self.layernorm1 = nn.LayerNorm(config.d_latent)
        self.layernorm2 = nn.LayerNorm(config.d_latent)

        assert config.d_latent % config.n_heads == 0
        self.rotary_emb = (
            RotaryEmbedding(config.d_latent // config.n_heads)
            if config.use_rope_embedding
            else None
        )

        if config.include_xattn:
            self.xattn_kv_proj = nn.Linear(
                config.d_latent, config.d_latent * 2, bias=False
            )
            self.xattn_q_proj = nn.Linear(config.d_latent, config.d_latent, bias=False)
            self.xattn_layernorm = nn.LayerNorm(config.d_latent)
            self.xattn_out_proj = nn.Linear(
                config.d_latent, config.d_latent, bias=False
            )

        self.norm_no_learnable = nn.LayerNorm(
            config.d_feedforward, elementwise_affine=False, bias=False
        )
        self.activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]()
        self.dropout = nn.Dropout(config.dropout_p)

        self.mlp0 = nn.Linear(config.d_latent, config.d_feedforward)
        self.mlp_film_cond_proj = (
            zero_module(
                nn.Linear(config.d_noise_emb, config.d_feedforward * 2, bias=False)
            )
            if config.use_film_noise_conditioning
            else None
        )
        self.mlp1 = nn.Linear(config.d_feedforward, config.d_latent)
        self.config = config

    def forward(
        self,
        x: Float[Tensor, "batch tokens d_latent"],
        attn_mask: Bool[Tensor, "batch 1 tokens tokens"] | None,
        noise_emb: Float[Tensor, "batch d_noise_emb"],
        cond: Float[Tensor, "batch tokens d_latent"] | None = None,
    ) -> Float[Tensor, "batch tokens d_latent"]:
        config = self.config
        (batch, time, d_latent) = x.shape

        # Self-attention.
        # We put layer normalization after the residual connection.
        x = self.layernorm1(x + self._sattn(x, attn_mask))

        # Include conditioning.
        if config.include_xattn:
            assert cond is not None
            x = self.xattn_layernorm(x + self._xattn(x, attn_mask, cond=cond))

        mlp_out = x
        mlp_out = self.mlp0(mlp_out)
        mlp_out = self.activation(mlp_out)

        # FiLM-style conditioning.
        if self.mlp_film_cond_proj is not None:
            scale, shift = torch.chunk(
                self.mlp_film_cond_proj(noise_emb), chunks=2, dim=-1
            )
            assert scale.shape == shift.shape == (batch, config.d_feedforward)
            mlp_out = (
                self.norm_no_learnable(mlp_out) * (1.0 + scale[:, None, :])
                + shift[:, None, :]
            )

        mlp_out = self.dropout(mlp_out)
        mlp_out = self.mlp1(mlp_out)

        x = self.layernorm2(x + mlp_out)
        assert x.shape == (batch, time, d_latent)
        return x

    def _sattn(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        """Multi-head self-attention."""
        config = self.config
        q, k, v = rearrange(
            self.sattn_qkv_proj(x),
            "b t (qkv nh dh) -> qkv b nh t dh",
            qkv=3,
            nh=config.n_heads,
        )
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=config.dropout_p, attn_mask=attn_mask
        )
        x = self.dropout(x)
        x = rearrange(x, "b nh t dh -> b t (nh dh)", nh=config.n_heads)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=config.dropout_p
        )
        x = self.dropout(x)
        x = rearrange(x, "b nh t dh -> b t (nh dh)", nh=config.n_heads)
        x = self.sattn_out_proj(x)
        return x

    def _xattn(self, x: Tensor, attn_mask: Tensor | None, cond: Tensor) -> Tensor:
        """Multi-head cross-attention."""
        config = self.config
        k, v = rearrange(
            self.xattn_kv_proj(
                {
                    "kv_from_cond_q_from_x": cond,
                    "kv_from_x_q_from_cond": x,
                }[self.config.xattn_mode]
            ),
            "b t (qk nh dh) -> qk b nh t dh",
            qk=2,
            nh=config.n_heads,
        )
        q = rearrange(
            self.xattn_q_proj(
                {
                    "kv_from_cond_q_from_x": x,
                    "kv_from_x_q_from_cond": cond,
                }[self.config.xattn_mode]
            ),
            "b t (nh dh) -> b nh t dh",
            nh=config.n_heads,
        )
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=config.dropout_p, attn_mask=attn_mask
        )
        x = rearrange(x, "b nh t dh -> b t (nh dh)")
        x = self.xattn_out_proj(x)

        return x


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module
