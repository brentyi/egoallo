from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import tyro
import viser
from projectaria_tools.core.data_provider import (
    VrsDataProvider,
    create_vrs_data_provider,
)
from projectaria_tools.core.sensor_data import TimeDomain
from tqdm import tqdm

from egoallo import fncsmpl
from egoallo.data.aria_mps import load_point_cloud_and_find_ground
from egoallo.hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from egoallo.inference_utils import InferenceTrajectoryPaths
from egoallo.network import EgoDenoiseTraj
from egoallo.transforms import SE3, SO3
from egoallo.vis_helpers import visualize_traj_and_hand_detections


def main(
    search_root_dir: Path,
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
) -> None:
    """Visualization script for outputs from EgoAllo.

    Arguments:
        search_root_dir: Root directory where inputs/outputs are stored. All
            NPZ files in this directory will be assumed to be outputs from EgoAllo.
        smplh_npz_path: Path to the SMPLH model NPZ file.
    """
    device = torch.device("cuda")

    body_model = fncsmpl.SmplhModel.load(smplh_npz_path).to(device)

    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    def get_file_list():
        return ["None"] + sorted(
            str(p.relative_to(search_root_dir))
            for p in search_root_dir.glob("**/egoallo_outputs/*.npz")
        )

    options = get_file_list()
    file_dropdown = server.gui.add_dropdown("File", options=options)

    refresh_file_list = server.gui.add_button("Refresh File List")

    @refresh_file_list.on_click
    def _(_) -> None:
        file_dropdown.options = get_file_list()

    trajectory_folder = server.gui.add_folder("Trajectory")

    current_file = "None"
    loop_cb = lambda: None

    while True:
        loop_cb()
        if current_file != file_dropdown.value:
            current_file = file_dropdown.value

            # Clear the scene.
            server.scene.reset()

            if current_file != "None":
                # Clear the folder by removing then re-adding it.
                # Perhaps we should expose some API for looping through children?
                trajectory_folder.remove()
                trajectory_folder = server.gui.add_folder("Trajectory")

                with trajectory_folder:
                    npz_path = Path(search_root_dir / current_file).resolve()
                    loop_cb = load_and_visualize(
                        server,
                        npz_path,
                        body_model,
                        device=device,
                    )
                    args = npz_path.parent / (npz_path.stem + "_args.yaml")
                    if args.exists():
                        with server.gui.add_folder("Args"):
                            server.gui.add_markdown(
                                "```\n" + args.read_text() + "\n```"
                            )


def load_and_visualize(
    server: viser.ViserServer,
    npz_path: Path,
    body_model: fncsmpl.SmplhModel,
    device: torch.device,
) -> Callable[[], int]:
    # Here's how we saved:
    #
    # np.savez(
    #     out_path,
    #     Ts_world_cpf=Ts_world_cpf[1:, :].numpy(force=True),
    #     Ts_world_root=Ts_world_root.numpy(force=True),
    #     body_quats=posed.local_quats[..., :21, :].numpy(force=True),
    #     left_hand_quats=posed.local_quats[..., 21:36, :].numpy(force=True),
    #     right_hand_quats=posed.local_quats[..., 36:51, :].numpy(force=True),
    #     betas=traj.betas.numpy(force=True),
    #     frame_nums=np.arange(args.start_index, args.start_index + args.traj_length),
    #     timestamps_ns=(np.array(pose_timestamps_sec) * 1e9).astype(np.int64),
    # )
    outputs = np.load(npz_path)
    expected_keys = [
        "Ts_world_cpf",
        "Ts_world_root",
        "body_quats",
        "left_hand_quats",
        "right_hand_quats",
        "betas",
        "frame_nums",
        "timestamps_ns",
    ]
    assert all(
        key in outputs for key in expected_keys
    ), f"Missing keys in NPZ file. Expected: {expected_keys}, Found: {list(outputs.keys())}"
    (num_samples, timesteps, _, _) = outputs["body_quats"].shape

    # We assume the directory structure is:
    # - some trajectory root
    #     - outputs
    #         -  the npz file
    traj_dir = npz_path.resolve().parent.parent
    paths = InferenceTrajectoryPaths.find(traj_dir)

    provider = create_vrs_data_provider(str(paths.vrs_file))
    device_calib = provider.get_device_calibration()
    T_device_cpf = SE3(
        torch.from_numpy(
            device_calib.get_transform_device_cpf().to_quat_and_translation()
        )
    )
    assert T_device_cpf.wxyz_xyz.shape == (1, 7)
    pose_timestamps_sec = outputs["timestamps_ns"] / 1e9

    Ts_world_device = (
        SE3(torch.from_numpy(outputs["Ts_world_cpf"])) @ T_device_cpf.inverse()
    ).wxyz_xyz

    # Get temporally corresponded HaMeR detections.
    if paths.hamer_outputs is not None:
        hamer_detections = CorrespondedHamerDetections.load(
            paths.hamer_outputs,
            pose_timestamps_sec,
        )
    else:
        print("No hand detections found.")
        hamer_detections = None

    # Get temporally corresponded Aria wrist and palm estimates.
    if paths.wrist_and_palm_poses_csv is not None:
        aria_detections = CorrespondedAriaHandWristPoseDetections.load(
            paths.wrist_and_palm_poses_csv,
            pose_timestamps_sec,
            Ts_world_device=Ts_world_device.numpy(force=True),
        )
    else:
        aria_detections = None

    if paths.splat_path is not None:
        print("Found splat at", paths.splat_path)
    else:
        print("No scene splat found.")

    # Get point cloud + floor.
    points_data, floor_z = load_point_cloud_and_find_ground(
        paths.points_path, "filtered"
    )

    traj = EgoDenoiseTraj(
        betas=torch.from_numpy(outputs["betas"]).to(device),
        body_rotmats=SO3(
            torch.from_numpy(outputs["body_quats"]),
        )
        .as_matrix()
        .to(device),
        # We weren't saving contacts originally. We added it September 28th.
        contacts=torch.zeros((num_samples, timesteps, 21), device=device)
        if "contacts" not in outputs
        else torch.from_numpy(outputs["contacts"]).to(device),
        hand_rotmats=SO3(
            torch.from_numpy(
                np.concatenate(
                    [
                        outputs["left_hand_quats"],
                        outputs["right_hand_quats"],
                    ],
                    axis=-2,
                )
            ).to(device)
        ).as_matrix(),
    )
    Ts_world_cpf = torch.from_numpy(outputs["Ts_world_cpf"]).to(device)

    def get_ego_video(
        start_index: int,
        end_index: int,
        total_duration: float,
    ) -> bytes:
        """Helper function that returns the egocentric video corresponding to
        some start/end pose index."""
        assert isinstance(provider, VrsDataProvider)
        rgb_stream_id = provider.get_stream_id_from_label("camera-rgb")
        assert rgb_stream_id is not None
        camera_fps = provider.get_configuration(rgb_stream_id).get_nominal_rate_hz()
        print(f"{camera_fps=}")

        start_ns = int(outputs["timestamps_ns"][start_index])
        first_ns = provider.get_first_time_ns(rgb_stream_id, TimeDomain.RECORD_TIME)

        image_start_index = int((start_ns - first_ns) / 1e9 * camera_fps)
        image_end_index = min(
            int(image_start_index + (end_index - start_index) / 30.0 * camera_fps) + 5,
            provider.get_num_data(rgb_stream_id),
        )

        frames = []
        for i in tqdm(range(image_start_index, image_end_index)):
            image_data = provider.get_image_data_by_index(rgb_stream_id, i)[0]
            image_array = image_data.to_numpy_array().copy()
            image_array = cv2.resize(
                image_array, (800, 800), interpolation=cv2.INTER_AREA
            )
            image_array = cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
            frames.append(image_array)

        fps = len(frames) / total_duration
        output = io.BytesIO()
        iio.imwrite(
            output,
            frames,
            fps=fps,
            extension=".mp4",
            codec="libx264",
            pixelformat="yuv420p",
            quality=None,
            ffmpeg_params=["-crf", "23"],
        )
        return output.getvalue()

    return visualize_traj_and_hand_detections(
        server,
        Ts_world_cpf,
        traj,
        body_model,
        hamer_detections,
        aria_detections,
        points_data,
        paths.splat_path,
        floor_z=floor_z,
        get_ego_video=get_ego_video,
    )


if __name__ == "__main__":
    tyro.cli(main)
