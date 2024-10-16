"""Script to run HaMeR on VRS data and save outputs to a pickle file."""

import pickle
import shutil
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import tyro
from egoallo.hand_detection_structs import (
    SavedHamerOutputs,
    SingleHandHamerOutputWrtCamera,
)
from hamer_helper import HamerHelper
from projectaria_tools.core import calibration
from projectaria_tools.core.data_provider import (
    VrsDataProvider,
    create_vrs_data_provider,
)
from tqdm.auto import tqdm

from egoallo.inference_utils import InferenceTrajectoryPaths


def main(traj_root: Path, overwrite: bool = False) -> None:
    """Run HaMeR for on trajectory. We'll save outputs to
    `traj_root/hamer_outputs.pkl` and `traj_root/hamer_outputs_render".

    Arguments:
        traj_root: The root directory of the trajectory. We assume that there's
            a VRS file in this directory.
        overwrite: If True, overwrite any existing HaMeR outputs.
    """

    paths = InferenceTrajectoryPaths.find(traj_root)

    vrs_path = paths.vrs_file
    assert vrs_path.exists()
    pickle_out = traj_root / "hamer_outputs.pkl"
    hamer_render_out = traj_root / "hamer_outputs_render"  # This is just for debugging.
    run_hamer_and_save(vrs_path, pickle_out, hamer_render_out, overwrite)


def run_hamer_and_save(
    vrs_path: Path, pickle_out: Path, hamer_render_out: Path, overwrite: bool
) -> None:
    if not overwrite:
        assert not pickle_out.exists()
        assert not hamer_render_out.exists()
    else:
        pickle_out.unlink(missing_ok=True)
        shutil.rmtree(hamer_render_out, ignore_errors=True)

    hamer_render_out.mkdir(exist_ok=True)
    hamer_helper = HamerHelper()

    # VRS data provider setup.
    provider = create_vrs_data_provider(str(vrs_path.absolute()))
    assert isinstance(provider, VrsDataProvider)
    rgb_stream_id = provider.get_stream_id_from_label("camera-rgb")
    assert rgb_stream_id is not None

    num_images = provider.get_num_data(rgb_stream_id)
    print(f"Found {num_images=}")

    # Get calibrations.
    device_calib = provider.get_device_calibration()
    assert device_calib is not None
    camera_calib = device_calib.get_camera_calib("camera-rgb")
    assert camera_calib is not None
    pinhole = calibration.get_linear_camera_calibration(1408, 1408, 450)

    # Compute camera extrinsics!
    sophus_T_device_camera = device_calib.get_transform_device_sensor("camera-rgb")
    sophus_T_cpf_camera = device_calib.get_transform_cpf_sensor("camera-rgb")
    assert sophus_T_device_camera is not None
    assert sophus_T_cpf_camera is not None
    T_device_cam = np.concatenate(
        [
            sophus_T_device_camera.rotation().to_quat().squeeze(axis=0),
            sophus_T_device_camera.translation().squeeze(axis=0),
        ]
    )
    T_cpf_cam = np.concatenate(
        [
            sophus_T_cpf_camera.rotation().to_quat().squeeze(axis=0),
            sophus_T_cpf_camera.translation().squeeze(axis=0),
        ]
    )
    assert T_device_cam.shape == T_cpf_cam.shape == (7,)

    # Dict from capture timestamp in nanoseconds to fields we care about.
    detections_left_wrt_cam: dict[int, SingleHandHamerOutputWrtCamera | None] = {}
    detections_right_wrt_cam: dict[int, SingleHandHamerOutputWrtCamera | None] = {}

    pbar = tqdm(range(num_images))
    for i in pbar:
        image_data, image_data_record = provider.get_image_data_by_index(
            rgb_stream_id, i
        )
        undistorted_image = calibration.distort_by_calibration(
            image_data.to_numpy_array(), pinhole, camera_calib
        )

        hamer_out_left, hamer_out_right = hamer_helper.look_for_hands(
            undistorted_image,
            focal_length=450,
        )
        timestamp_ns = image_data_record.capture_timestamp_ns

        if hamer_out_left is None:
            detections_left_wrt_cam[timestamp_ns] = None
        else:
            detections_left_wrt_cam[timestamp_ns] = {
                "verts": hamer_out_left["verts"],
                "keypoints_3d": hamer_out_left["keypoints_3d"],
                "mano_hand_pose": hamer_out_left["mano_hand_pose"],
                "mano_hand_betas": hamer_out_left["mano_hand_betas"],
                "mano_hand_global_orient": hamer_out_left["mano_hand_global_orient"],
            }

        if hamer_out_right is None:
            detections_right_wrt_cam[timestamp_ns] = None
        else:
            detections_right_wrt_cam[timestamp_ns] = {
                "verts": hamer_out_right["verts"],
                "keypoints_3d": hamer_out_right["keypoints_3d"],
                "mano_hand_pose": hamer_out_right["mano_hand_pose"],
                "mano_hand_betas": hamer_out_right["mano_hand_betas"],
                "mano_hand_global_orient": hamer_out_right["mano_hand_global_orient"],
            }

        composited = undistorted_image
        composited = hamer_helper.composite_detections(
            composited,
            hamer_out_left,
            border_color=(255, 100, 100),
            focal_length=450,
        )
        composited = hamer_helper.composite_detections(
            composited,
            hamer_out_right,
            border_color=(100, 100, 255),
            focal_length=450,
        )
        composited = put_text(
            composited,
            "L detections: "
            + (
                "0" if hamer_out_left is None else str(hamer_out_left["verts"].shape[0])
            ),
            0,
            color=(255, 100, 100),
            font_scale=10.0 / 2880.0 * undistorted_image.shape[0],
        )
        composited = put_text(
            composited,
            "R detections: "
            + (
                "0"
                if hamer_out_right is None
                else str(hamer_out_right["verts"].shape[0])
            ),
            1,
            color=(100, 100, 255),
            font_scale=10.0 / 2880.0 * undistorted_image.shape[0],
        )
        composited = put_text(
            composited,
            f"ns={timestamp_ns}",
            2,
            color=(255, 255, 255),
            font_scale=10.0 / 2880.0 * undistorted_image.shape[0],
        )

        print(f"Saving image {i:06d} to {hamer_render_out / f'{i:06d}.jpeg'}")
        iio.imwrite(
            str(hamer_render_out / f"{i:06d}.jpeg"),
            np.concatenate(
                [
                    # Darken input image, just for contrast...
                    (undistorted_image * 0.6).astype(np.uint8),
                    composited,
                ],
                axis=1,
            ),
            quality=90,
        )

    outputs = SavedHamerOutputs(
        mano_faces_right=hamer_helper.get_mano_faces("right"),
        mano_faces_left=hamer_helper.get_mano_faces("left"),
        detections_right_wrt_cam=detections_right_wrt_cam,
        detections_left_wrt_cam=detections_left_wrt_cam,
        T_device_cam=T_device_cam,
        T_cpf_cam=T_cpf_cam,
    )
    with open(pickle_out, "wb") as f:
        pickle.dump(outputs, f)


def put_text(
    image: np.ndarray,
    text: str,
    line_number: int,
    color: tuple[int, int, int],
    font_scale: float,
) -> np.ndarray:
    """Put some text on the top-left corner of an image."""
    image = image.copy()
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(
        image,
        text=text,
        org=(2, 1 + int(15 * font_scale * (line_number + 1))),
        fontFace=font,
        fontScale=font_scale,
        color=(0, 0, 0),
        thickness=max(int(font_scale), 1),
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text=text,
        org=(2, 1 + int(15 * font_scale * (line_number + 1))),
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=max(int(font_scale), 1),
        lineType=cv2.LINE_AA,
    )
    return image


if __name__ == "__main__":
    tyro.cli(main)
