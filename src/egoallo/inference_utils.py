"""Functions that are useful for inference scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from jaxtyping import Float
from projectaria_tools.core import mps  # type: ignore
from projectaria_tools.core.data_provider import create_vrs_data_provider
from safetensors import safe_open
from torch import Tensor

from .network import EgoDenoiser, EgoDenoiserConfig
from .tensor_dataclass import TensorDataclass
from .transforms import SE3


def load_denoiser(checkpoint_dir: Path) -> EgoDenoiser:
    """Load a denoiser model."""
    checkpoint_dir = checkpoint_dir.absolute()
    experiment_dir = checkpoint_dir.parent

    config = yaml.load(
        (experiment_dir / "model_config.yaml").read_text(), Loader=yaml.Loader
    )
    assert isinstance(config, EgoDenoiserConfig)

    model = EgoDenoiser(config)
    with safe_open(checkpoint_dir / "model.safetensors", framework="pt") as f:  # type: ignore
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    model.load_state_dict(state_dict)

    return model


@dataclass(frozen=True)
class InferenceTrajectoryPaths:
    """Paths for running EgoAllo on a single sequence from Project Aria.

    Our basic assumptions here are:
    1. VRS file for images: there is exactly one VRS file in the trajectory root directory.
    2. Aria MPS point cloud: there is either one semidense_points.csv.gz file or one global_points.csv.gz file.
        - Its parent directory should contain other Aria MPS artifacts. (like poses)
        - This is optionally used for guidance.
    3. HaMeR outputs: The hamer_outputs.pkl file may or may not exist in the trajectory root directory.
        - This is optionally used for guidance.
    4. Aria MPS wrist/palm poses: There may be zero or one wrist_and_palm_poses.csv file.
        - This is optionally used for guidance.
    5. Scene splat/ply file: There may be a splat.ply or scene.splat file.
        - This is only used for visualization.
    """

    vrs_file: Path
    slam_root_dir: Path
    points_path: Path
    hamer_outputs: Path | None
    wrist_and_palm_poses_csv: Path | None
    splat_path: Path | None

    @staticmethod
    def find(traj_root: Path) -> InferenceTrajectoryPaths:
        vrs_files = tuple(traj_root.glob("**/*.vrs"))
        assert len(vrs_files) == 1, f"Found {len(vrs_files)} VRS files!"

        points_paths = tuple(traj_root.glob("**/semidense_points.csv.gz"))
        assert len(points_paths) <= 1, f"Found multiple points files! {points_paths}"
        if len(points_paths) == 0:
            points_paths = tuple(traj_root.glob("**/global_points.csv.gz"))
        assert len(points_paths) == 1, f"Found {len(points_paths)} files!"

        hamer_outputs = traj_root / "hamer_outputs.pkl"
        if not hamer_outputs.exists():
            hamer_outputs = None

        wrist_and_palm_poses_csv = tuple(traj_root.glob("**/wrist_and_palm_poses.csv"))
        if len(wrist_and_palm_poses_csv) == 0:
            wrist_and_palm_poses_csv = None
        else:
            assert (
                len(wrist_and_palm_poses_csv) == 1
            ), "Found multiple wrist and palm poses files!"

        splat_path = traj_root / "splat.ply"
        if not splat_path.exists():
            splat_path = traj_root / "scene.splat"
        if not splat_path.exists():
            print("No scene splat found.")
            splat_path = None
        else:
            print("Found splat at", splat_path)

        return InferenceTrajectoryPaths(
            vrs_file=vrs_files[0],
            slam_root_dir=points_paths[0].parent,
            points_path=points_paths[0],
            hamer_outputs=hamer_outputs,
            wrist_and_palm_poses_csv=wrist_and_palm_poses_csv[0]
            if wrist_and_palm_poses_csv
            else None,
            splat_path=splat_path,
        )


class InferenceInputTransforms(TensorDataclass):
    """Some relevant transforms for inference."""

    Ts_world_cpf: Float[Tensor, "timesteps 7"]
    Ts_world_device: Float[Tensor, "timesteps 7"]
    pose_timesteps: tuple[float, ...]

    @staticmethod
    def load(
        vrs_path: Path,
        slam_root_dir: Path,
        fps: int = 30,
    ) -> InferenceInputTransforms:
        """Read some useful transforms via MPS + the VRS calibration."""
        # Read device poses.
        closed_loop_path = slam_root_dir / "closed_loop_trajectory.csv"
        if not closed_loop_path.exists():
            # Aria digital twins.
            closed_loop_path = slam_root_dir / "aria_trajectory.csv"
        closed_loop_traj = mps.read_closed_loop_trajectory(str(closed_loop_path))  # type: ignore

        provider = create_vrs_data_provider(str(vrs_path))
        device_calib = provider.get_device_calibration()
        T_device_cpf = device_calib.get_transform_device_cpf().to_matrix()

        # Get downsampled CPF frames.
        aria_fps = len(closed_loop_traj) / (
            closed_loop_traj[-1].tracking_timestamp.total_seconds()
            - closed_loop_traj[0].tracking_timestamp.total_seconds()
        )
        num_poses = len(closed_loop_traj)
        print(f"Loaded {num_poses=} with {aria_fps=}, visualizing at {fps=}")
        Ts_world_device = []
        Ts_world_cpf = []
        out_timestamps_secs = []
        for i in range(0, num_poses, int(aria_fps // fps)):
            T_world_device = closed_loop_traj[i].transform_world_device.to_matrix()
            assert T_world_device.shape == (4, 4)
            Ts_world_device.append(T_world_device)
            Ts_world_cpf.append(T_world_device @ T_device_cpf)
            out_timestamps_secs.append(
                closed_loop_traj[i].tracking_timestamp.total_seconds()
            )

        return InferenceInputTransforms(
            Ts_world_device=SE3.from_matrix(torch.from_numpy(np.array(Ts_world_device)))
            .parameters()
            .to(torch.float32),
            Ts_world_cpf=SE3.from_matrix(torch.from_numpy(np.array(Ts_world_cpf)))
            .parameters()
            .to(torch.float32),
            pose_timesteps=tuple(out_timestamps_secs),
        )
