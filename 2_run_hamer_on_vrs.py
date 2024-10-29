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

from projectaria_tools.core import calibration
from projectaria_tools.core.data_provider import (
    VrsDataProvider,
    create_vrs_data_provider,
)
from _hamer_helper import HamerHelper
import projectaria_tools.core.mps as mps
from projectaria_tools.core.sensor_data import DEVICE_TIME
from tqdm.auto import tqdm

from egoallo.inference_utils import InferenceTrajectoryPaths


def main(traj_root: Path, detector: str = "hamer",overwrite: bool = False, wilor_home: str = "/secondary/home/annie/code/WiLoR") -> None:
    """Run HaMeR for on trajectory. We'll save outputs to
    `traj_root/hamer_outputs.pkl` and `traj_root/hamer_outputs_render".

    Arguments:
        traj_root: The root directory of the trajectory. We assume that there's
            a VRS file in this directory.
        detector: The detector to use. Can be "wilor", "aria", or "hamer".
        overwrite: If True, overwrite any existing HaMeR outputs.
        wilor_home: The path to the WiLoR home directory. Only used if `detector`
            is "wilor".
    """

    paths = InferenceTrajectoryPaths.find(traj_root)
    vrs_path = paths.vrs_file
    assert vrs_path.exists()
    
    if detector == "wilor":
        render_out_path = traj_root / "wilor_outputs_unrot_render"  # This is just for debugging.
        pickle_out = traj_root / "wilor_outputs.pkl"
        run_wilor_and_save(vrs_path, pickle_out, render_out_path, overwrite, wilor_home)
    elif detector == "aria":
        render_out_path = traj_root / "hamer_aria_outputs_render"  # This is just for debugging.
        wrist_and_palm_poses_path = traj_root / "hand_tracking/wrist_and_palm_poses.csv"
        online_path = traj_root / "slam/online_calibration.jsonl"
        pickle_out = traj_root / "aria_outputs.pkl"
        run_aria_hamer_and_save(vrs_path, pickle_out, render_out_path, wrist_and_palm_poses_path, online_path, overwrite)
    elif detector == "hamer":
        render_out_path = traj_root / "hamer_outputs_render"  # This is just for debugging.
        pickle_out = traj_root / "hamer_outputs.pkl"
        run_hamer_and_save(vrs_path, pickle_out, render_out_path, overwrite)
    else:
        raise ValueError(f"Unknown detector: {detector}")


def run_hamer_and_save(
    vrs_path: Path, pickle_out: Path, render_out_path: Path, overwrite: bool
) -> None:
    if not overwrite:
        assert not pickle_out.exists()
        assert not render_out_path.exists()
    else:
        pickle_out.unlink(missing_ok=True)
        shutil.rmtree(render_out_path, ignore_errors=True)

    render_out_path.mkdir(exist_ok=True)
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

        print(f"Saving image {i:06d} to {render_out_path / f'{i:06d}.jpeg'}")
        iio.imwrite(
            str(render_out_path / f"{i:06d}.jpeg"),
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

def run_wilor_and_save(
    vrs_path: Path, pickle_out: Path, render_out_path: Path, overwrite: bool, wilor_home: str
) -> None:
    from _wilor_helper import WiLoRHelper
    if not overwrite:
        assert not pickle_out.exists()
        assert not render_out_path.exists()
    else:
        pickle_out.unlink(missing_ok=True)
        shutil.rmtree(render_out_path, ignore_errors=True)
    render_out_path.mkdir(exist_ok=True)
    wilor_helper = WiLoRHelper(wilor_home)
    hamer_helper = HamerHelper(other_detector=True)

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
        
        hamer_out_left, hamer_out_right = wilor_helper.look_for_hands(
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

        print(f"Saving image {i:06d} to {render_out_path / f'{i:06d}.jpeg'}")
        iio.imwrite(
            str(render_out_path / f"{i:06d}.jpeg"),
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
        mano_faces_right=wilor_helper.get_mano_faces("right"),
        mano_faces_left=wilor_helper.get_mano_faces("left"),
        detections_right_wrt_cam=detections_right_wrt_cam,
        detections_left_wrt_cam=detections_left_wrt_cam,
        T_device_cam=T_device_cam,
        T_cpf_cam=T_cpf_cam,
    )
    with open(pickle_out, "wb") as f:
        pickle.dump(outputs, f)


def run_aria_hamer_and_save(
    vrs_path: Path, pickle_out: Path, render_out_path: Path, wrist_and_palm_poses_path: Path, online_calib_path: Path, overwrite: bool
) -> None:
    from aria_utils import per_image_hand_tracking,get_online_calib,x_y_around
    if not overwrite:
        assert not pickle_out.exists()
        assert not render_out_path.exists()
    else:
        pickle_out.unlink(missing_ok=True)
        shutil.rmtree(render_out_path, ignore_errors=True)

    render_out_path.mkdir(exist_ok=True)
    hamer_helper = HamerHelper(other_detector=True)

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

    wrist_and_palm_poses_path = str(wrist_and_palm_poses_path)
    online_calib_path = str(online_calib_path)

    rgb_calib = get_online_calib(online_calib_path, "camera-rgb")
    wrist_and_palm_poses = mps.hand_tracking.read_wrist_and_palm_poses(wrist_and_palm_poses_path)

    pbar = tqdm(range(num_images))

    for i in pbar:
        image_data, image_data_record = provider.get_image_data_by_index(
            rgb_stream_id, i
        )
        undistorted_image = calibration.distort_by_calibration(
            image_data.to_numpy_array(), pinhole, camera_calib
        )

        timestamp_ns = image_data_record.capture_timestamp_ns
        l_existed, r_existed, l_point, r_point = per_image_hand_tracking(timestamp_ns, wrist_and_palm_poses, pinhole, camera_calib, rgb_calib)
        if l_existed:
            l_box = x_y_around(l_point[0], l_point[1],pinhole,offset=80)
        else:
            l_box = None

        if r_existed:
            r_box = x_y_around(r_point[0], r_point[1],pinhole,offset=80)
        else:
            r_box = None


        hamer_out_left, hamer_out_right = hamer_helper.get_det_from_boxes(
            undistorted_image,
            l_existed,
            r_existed,
            l_box,
            r_box,
            focal_length=450,
        )

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

        print(f"Saving image {i:06d} to {render_out_path / f'{i:06d}.jpeg'}")
        # bbox on undistorted image
        if l_existed:
            min_l_p_x, min_l_p_y, max_l_p_x, max_l_p_y = l_box
            max_l_p_x, min_l_p_x, max_l_p_y, min_l_p_y = int(max_l_p_x), int(min_l_p_x), int(max_l_p_y), int(min_l_p_y)

            cv2.rectangle(composited, (max_l_p_x, max_l_p_y), (min_l_p_x, min_l_p_y), (255, 100, 100),2)
        if r_existed:
            min_r_p_x, min_r_p_y, max_r_p_x, max_r_p_y = r_box
            max_r_p_x, min_r_p_x, max_r_p_y, min_r_p_y = int(max_r_p_x), int(min_r_p_x), int(max_r_p_y), int(min_r_p_y)
            
            cv2.rectangle(composited, (max_r_p_x, max_r_p_y), (min_r_p_x, min_r_p_y), (100, 100, 255),2)
            
        
        iio.imwrite(
            str(render_out_path / f"{i:06d}.jpeg"),
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
