"""Example script for computing body metrics on the test split of the AMASS dataset.

This is not the exact script we used for the paper metrics, but should have the
details that matter matched. Below are some metrics from this script when our
released checkpoint is passed in.

For --subseq-len 128:

     mpjpe 118.340 +/- 1.350             (in paper: 119.7 +/- 1.3)
     pampjpe 100.026 +/- 1.349           (in paper: 101.1 +/- 1.3)
     T_head 0.006 +/- 0.000              (in paper: 0.0062 +/- 0.0001)
     foot_contact (GND) 1.000 +/- 0.000  (in paper: 1.0 +/- 0.0)
     foot_skate 0.417 +/- 0.017          (not reported in paper)


For --subseq-len 32:

     mpjpe 129.193 +/- 1.108             (in paper: 129.8 +/- 1.1)
     pampjpe 109.489 +/- 1.147           (in paper: 109.8 +/- 1.1)
     T_head 0.006 +/- 0.000              (in paper: 0.0064 +/- 0.0001)
     foot_contact (GND) 0.985 +/- 0.003  (in paper: 0.98 +/- 0.00)
     foot_skate 0.185 +/- 0.005          (not reported in paper)
"""

from pathlib import Path

import jax.tree
import numpy as np
import torch.optim.lr_scheduler
import torch.utils.data
import tyro

from egoallo import fncsmpl
from egoallo.data.amass import EgoAmassHdf5Dataset
from egoallo.fncsmpl_extensions import get_T_world_root_from_cpf_pose
from egoallo.inference_utils import load_denoiser
from egoallo.metrics_helpers import (
    compute_foot_contact,
    compute_foot_skate,
    compute_head_trans,
    compute_mpjpe,
)
from egoallo.sampling import run_sampling_with_stitching
from egoallo.transforms import SE3, SO3


def main(
    dataset_hdf5_path: Path,
    dataset_files_path: Path,
    subseq_len: int = 128,
    guidance_inner: bool = False,
    checkpoint_dir: Path = Path("./egoallo_checkpoint_april13/checkpoints_3000000/"),
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
    num_samples: int = 1,
) -> None:
    """Compute body metrics on the test split of the AMASS dataset."""
    device = torch.device("cuda")

    # Setup.
    denoiser_network = load_denoiser(checkpoint_dir).to(device)
    dataset = EgoAmassHdf5Dataset(
        dataset_hdf5_path,
        dataset_files_path,
        splits=("test",),
        # We need an extra timestep in order to compute the relative CPF pose. (T_cpf_tm1_cpf_t)
        subseq_len=subseq_len + 1,
        cache_files=True,
        slice_strategy="deterministic",
        random_variable_len_proportion=0.0,
    )
    body_model = fncsmpl.SmplhModel.load(smplh_npz_path).to(device)

    metrics = list[dict[str, np.ndarray]]()

    for i in range(len(dataset)):
        sequence = dataset[i].to(device)

        samples = run_sampling_with_stitching(
            denoiser_network,
            body_model=body_model,
            guidance_mode="no_hands",
            guidance_inner=guidance_inner,
            guidance_post=True,
            Ts_world_cpf=sequence.T_world_cpf,
            hamer_detections=None,
            aria_detections=None,
            num_samples=num_samples,
            floor_z=0.0,
            device=device,
            guidance_verbose=False,
        )

        assert samples.hand_rotmats is not None
        assert samples.betas.shape == (num_samples, subseq_len, 16)
        assert samples.body_rotmats.shape == (num_samples, subseq_len, 21, 3, 3)
        assert samples.hand_rotmats.shape == (num_samples, subseq_len, 30, 3, 3)
        assert sequence.hand_quats is not None

        # We'll only use the body joint rotations.
        pred_posed = body_model.with_shape(samples.betas).with_pose(
            T_world_root=SE3.identity(device, torch.float32).wxyz_xyz,
            local_quats=SO3.from_matrix(
                torch.cat([samples.body_rotmats, samples.hand_rotmats], dim=2)
            ).wxyz,
        )
        pred_posed = pred_posed.with_new_T_world_root(
            get_T_world_root_from_cpf_pose(pred_posed, sequence.T_world_cpf[1:, ...])
        )

        label_posed = body_model.with_shape(sequence.betas[1:, ...]).with_pose(
            sequence.T_world_root[1:, ...],
            torch.cat(
                [
                    sequence.body_quats[1:, ...],
                    sequence.hand_quats[1:, ...],
                ],
                dim=1,
            ),
        )

        metrics.append(
            {
                "mpjpe": compute_mpjpe(
                    label_T_world_root=label_posed.T_world_root,
                    label_Ts_world_joint=label_posed.Ts_world_joint[:, :21, :],
                    pred_T_world_root=pred_posed.T_world_root,
                    pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                    per_frame_procrustes_align=False,
                ),
                "pampjpe": compute_mpjpe(
                    label_T_world_root=label_posed.T_world_root,
                    label_Ts_world_joint=label_posed.Ts_world_joint[:, :21, :],
                    pred_T_world_root=pred_posed.T_world_root,
                    pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                    per_frame_procrustes_align=True,
                ),
                # We didn't report foot skating metrics in the paper. It's not
                # really meaningful: since we optimize foot skating in the
                # guidance optimizer, it's easy to "cheat" this metric.
                "foot_skate": compute_foot_skate(
                    pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                ),
                "foot_contact (GND)": compute_foot_contact(
                    pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                ),
                "T_head": compute_head_trans(
                    label_Ts_world_joint=label_posed.Ts_world_joint[:, :21, :],
                    pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                ),
            }
        )

        print("=" * 80)
        print("=" * 80)
        print("=" * 80)
        print(f"Metrics ({i}/{len(dataset)} processed)")
        for k, v in jax.tree.map(
            lambda *x: f"{np.mean(x):.3f} +/- {np.std(x) / np.sqrt(len(metrics) * num_samples):.3f}",
            *metrics,
        ).items():
            print("\t", k, v)
        print("=" * 80)
        print("=" * 80)
        print("=" * 80)


if __name__ == "__main__":
    tyro.cli(main)
