"""Training loss configuration."""

import dataclasses
from typing import Literal

import torch.utils.data
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel

from . import network
from .data.amass import EgoTrainingData
from .sampling import CosineNoiseScheduleConstants
from .transforms import SO3


@dataclasses.dataclass(frozen=True)
class TrainingLossConfig:
    cond_dropout_prob: float = 0.0
    beta_coeff_weights: tuple[float, ...] = tuple(1 / (i + 1) for i in range(16))
    loss_weights: dict[str, float] = dataclasses.field(
        default_factory={
            "betas": 0.1,
            "body_rotmats": 1.0,
            "contacts": 0.1,
            # We don't have many hands in the AMASS dataset...
            "hand_rotmats": 0.01,
        }.copy
    )
    weight_loss_by_t: Literal["emulate_eps_pred"] = "emulate_eps_pred"
    """Weights to apply to the loss at each noise level."""


class TrainingLossComputer:
    """Helper class for computing the training loss. Contains a single method
    for computing a training loss."""

    def __init__(self, config: TrainingLossConfig, device: torch.device) -> None:
        self.config = config
        self.noise_constants = (
            CosineNoiseScheduleConstants.compute(timesteps=1000)
            .to(device)
            .map(lambda tensor: tensor.to(torch.float32))
        )

        # Emulate loss weight that would be ~equivalent to epsilon prediction.
        #
        # This will penalize later errors (close to the end of sampling) much
        # more than earlier ones (at the start of sampling).
        assert self.config.weight_loss_by_t == "emulate_eps_pred"
        weight_t = self.noise_constants.alpha_bar_t / (
            1 - self.noise_constants.alpha_bar_t
        )
        # Pad for numerical stability, and scale between [padding, 1.0].
        padding = 0.01
        self.weight_t = weight_t / weight_t[1] * (1.0 - padding) + padding

    def compute_denoising_loss(
        self,
        model: network.EgoDenoiser | DistributedDataParallel | OptimizedModule,
        unwrapped_model: network.EgoDenoiser,
        train_batch: EgoTrainingData,
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute a training loss for the EgoDenoiser model.

        Returns:
            A tuple (loss tensor, dictionary of things to log).
        """
        log_outputs: dict[str, Tensor | float] = {}

        batch, time, num_joints, _ = train_batch.body_quats.shape
        assert num_joints == 21
        if unwrapped_model.config.include_hands:
            assert train_batch.hand_quats is not None
            x_0 = network.EgoDenoiseTraj(
                betas=train_batch.betas.expand((batch, time, 16)),
                body_rotmats=SO3(train_batch.body_quats).as_matrix(),
                contacts=train_batch.contacts,
                hand_rotmats=SO3(train_batch.hand_quats).as_matrix(),
            )
        else:
            x_0 = network.EgoDenoiseTraj(
                betas=train_batch.betas.expand((batch, time, 16)),
                body_rotmats=SO3(train_batch.body_quats).as_matrix(),
                contacts=train_batch.contacts,
                hand_rotmats=None,
            )
        x_0_packed = x_0.pack()
        device = x_0_packed.device
        assert x_0_packed.shape == (batch, time, unwrapped_model.get_d_state())

        # Diffuse.
        t = torch.randint(
            low=1,
            high=unwrapped_model.config.max_t + 1,
            size=(batch,),
            device=device,
        )
        eps = torch.randn(x_0_packed.shape, dtype=x_0_packed.dtype, device=device)
        assert self.noise_constants.alpha_bar_t.shape == (
            unwrapped_model.config.max_t + 1,
        )
        alpha_bar_t = self.noise_constants.alpha_bar_t[t, None, None]
        assert alpha_bar_t.shape == (batch, 1, 1)
        x_t_packed = (
            torch.sqrt(alpha_bar_t) * x_0_packed + torch.sqrt(1.0 - alpha_bar_t) * eps
        )

        hand_positions_wrt_cpf: Tensor | None = None
        if unwrapped_model.config.include_hand_positions_cond:
            # Joints 19 and 20 are the hand positions.
            hand_positions_wrt_cpf = train_batch.joints_wrt_cpf[:, :, 19:21, :].reshape(
                (batch, time, 6)
            )

            # Exclude hand positions for some items in the batch. We'll just do
            # this by passing in zeros.
            hand_positions_wrt_cpf = torch.where(
                # Uniformly drop out with some uniformly sampled probability.
                # :)
                (
                    torch.rand((batch, time, 1), device=device)
                    < torch.rand((batch, 1, 1), device=device)
                ),
                hand_positions_wrt_cpf,
                0.0,
            )

        # Denoise.
        x_0_packed_pred = model.forward(
            x_t_packed=x_t_packed,
            t=t,
            T_world_cpf=train_batch.T_world_cpf,
            T_cpf_tm1_cpf_t=train_batch.T_cpf_tm1_cpf_t,
            hand_positions_wrt_cpf=hand_positions_wrt_cpf,
            project_output_rotmats=False,
            mask=train_batch.mask,
            cond_dropout_keep_mask=torch.rand((batch,), device=device)
            > self.config.cond_dropout_prob
            if self.config.cond_dropout_prob > 0.0
            else None,
        )
        assert isinstance(x_0_packed_pred, torch.Tensor)
        x_0_pred = network.EgoDenoiseTraj.unpack(
            x_0_packed_pred, include_hands=unwrapped_model.config.include_hands
        )

        weight_t = self.weight_t[t].to(device)
        assert weight_t.shape == (batch,)

        def weight_and_mask_loss(
            loss_per_step: Float[Tensor, "b t d"],
            # bt stands for "batch time"
            bt_mask: Bool[Tensor, "b t"] = train_batch.mask,
            bt_mask_sum: Int[Tensor, ""] = torch.sum(train_batch.mask),
        ) -> Float[Tensor, ""]:
            """Weight and mask per-timestep losses (squared errors)."""
            _, _, d = loss_per_step.shape
            assert loss_per_step.shape == (batch, time, d)
            assert bt_mask.shape == (batch, time)
            assert weight_t.shape == (batch,)
            return (
                # Sum across b axis.
                torch.sum(
                    # Sum across t axis.
                    torch.sum(
                        # Mean across d axis.
                        torch.mean(loss_per_step, dim=-1) * bt_mask,
                        dim=-1,
                    )
                    * weight_t
                )
                / bt_mask_sum
            )

        loss_terms: dict[str, Tensor | float] = {
            "betas": weight_and_mask_loss(
                # (b, t, 16)
                (x_0_pred.betas - x_0.betas) ** 2
                # (16,)
                * x_0.betas.new_tensor(self.config.beta_coeff_weights),
            ),
            "body_rotmats": weight_and_mask_loss(
                # (b, t, 21 * 3 * 3)
                (x_0_pred.body_rotmats - x_0.body_rotmats).reshape(
                    (batch, time, 21 * 3 * 3)
                )
                ** 2,
            ),
            "contacts": weight_and_mask_loss((x_0_pred.contacts - x_0.contacts) ** 2),
        }

        # Include hand objective.
        # We didn't use this in the paper.
        if unwrapped_model.config.include_hands:
            assert x_0_pred.hand_rotmats is not None
            assert x_0.hand_rotmats is not None
            assert x_0.hand_rotmats.shape == (batch, time, 30, 3, 3)

            # Detect whether or not hands move in a sequence.
            # We should only supervise sequences where the hands are actully tracked / move;
            # we mask out hands in AMASS sequences where they are not tracked.
            gt_hand_flatmat = x_0.hand_rotmats.reshape((batch, time, -1))
            hand_motion = (
                torch.sum(  # (b,) from (b, t)
                    torch.sum(  # (b, t) from (b, t, d)
                        torch.abs(gt_hand_flatmat - gt_hand_flatmat[:, 0:1, :]), dim=-1
                    )
                    # Zero out changes in masked frames.
                    * train_batch.mask,
                    dim=-1,
                )
                > 1e-5
            )
            assert hand_motion.shape == (batch,)

            hand_bt_mask = torch.logical_and(hand_motion[:, None], train_batch.mask)
            loss_terms["hand_rotmats"] = torch.sum(
                weight_and_mask_loss(
                    (x_0_pred.hand_rotmats - x_0.hand_rotmats).reshape(
                        batch, time, 30 * 3 * 3
                    )
                    ** 2,
                    bt_mask=hand_bt_mask,
                    # We want to weight the loss by the number of frames where
                    # the hands actually move, but gradients here can be too
                    # noisy and put NaNs into mixed-precision training when we
                    # inevitably sample too few frames. So we clip the
                    # denominator.
                    bt_mask_sum=torch.maximum(
                        torch.sum(hand_bt_mask), torch.tensor(256, device=device)
                    ),
                )
            )
            # self.log(
            #     "train/hand_motion_proportion",
            #     torch.sum(hand_motion) / batch,
            # )
        else:
            loss_terms["hand_rotmats"] = 0.0

        assert loss_terms.keys() == self.config.loss_weights.keys()

        # Log loss terms.
        for name, term in loss_terms.items():
            log_outputs[f"loss_term/{name}"] = term

        # Return loss.
        loss = sum([loss_terms[k] * self.config.loss_weights[k] for k in loss_terms])
        assert isinstance(loss, Tensor)
        assert loss.shape == ()
        log_outputs["train_loss"] = loss

        return loss, log_outputs
