"""Data structure definition that we use for hand detections.

We'll run HaMeR, produce the dictionary defined by `SavedHamerOutputs`, then
pickle this dictionary.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol, TypedDict, cast

import numpy as np
import torch
from jaxtyping import Float, Int
from projectaria_tools.core import mps
from projectaria_tools.core.mps.utils import get_nearest_wrist_and_palm_pose
from torch import Tensor

from .tensor_dataclass import TensorDataclass
from .transforms import SE3, SO3


class SingleHandHamerOutputWrtCamera(TypedDict):
    """Hand outputs with respect to the camera frame. For use in pickle files."""

    verts: np.ndarray
    keypoints_3d: np.ndarray
    mano_hand_pose: np.ndarray
    mano_hand_betas: np.ndarray
    mano_hand_global_orient: np.ndarray


class SavedHamerOutputs(TypedDict):
    """Outputs from the HAMeR hand detection algorithm. This is the structure
    to pickle.

    `detections_left_wrt_cam` and `detections_right_wrt_cam` use nanosecond
    timestamps as keys.
    """

    mano_faces_right: np.ndarray
    mano_faces_left: np.ndarray

    detections_left_wrt_cam: dict[int, SingleHandHamerOutputWrtCamera | None]
    detections_right_wrt_cam: dict[int, SingleHandHamerOutputWrtCamera | None]

    T_device_cam: np.ndarray  # wxyz_xyz
    T_cpf_cam: np.ndarray  # wxyz_xyz


class AriaHandWristPoseWrtWorld(TensorDataclass):
    confidence: Float[Tensor, "n_detections"]
    wrist_position: Float[Tensor, "n_detections 3"]
    wrist_normal: Float[Tensor, "n_detections 3"]

    palm_position: Float[Tensor, "n_detections 3"]
    palm_normal: Float[Tensor, "n_detections 3"]

    indices: Int[Tensor, "n_detections"]


class CorrespondedAriaHandWristPoseDetections(TensorDataclass):
    detections_left_concat: AriaHandWristPoseWrtWorld | None
    detections_right_concat: AriaHandWristPoseWrtWorld | None

    @staticmethod
    def load(
        wrist_and_palm_poses_csv_path: Path,
        target_timestamps_sec: tuple[float, ...],
        Ts_world_device: Float[np.ndarray, "timesteps 7"],
    ) -> CorrespondedAriaHandWristPoseDetections:
        # API from runtime inspection of `projectaria_tools` outputs.
        class WristAndPalmNormals(Protocol):
            wrist_normal_device: np.ndarray
            palm_normal_device: np.ndarray

        class OneSide(Protocol):
            confidence: float
            wrist_position_device: np.ndarray
            palm_position_device: np.ndarray
            wrist_and_palm_normal_device: WristAndPalmNormals

        wp_poses = mps.hand_tracking.read_wrist_and_palm_poses(
            str(wrist_and_palm_poses_csv_path)
        )
        detections_left = list[OneSide]()
        detections_right = list[OneSide]()
        indices_left = list[int]()
        indices_right = list[int]()
        for i, time_sec in enumerate(target_timestamps_sec):
            wp_pose = get_nearest_wrist_and_palm_pose(wp_poses, int(time_sec * 1e9))
            if (
                wp_pose is None
                or abs(wp_pose.tracking_timestamp.total_seconds() - time_sec)
                >= 1.0 / 30.0
            ):
                continue

            if wp_pose.left_hand.confidence > 0.7:
                indices_left.append(i)
                detections_left.append(wp_pose.left_hand)

            if wp_pose.right_hand.confidence > 0.7:
                indices_right.append(i)
                detections_right.append(wp_pose.right_hand)

        def form_detections_concat(
            detections: list[OneSide], indices: list[int]
        ) -> AriaHandWristPoseWrtWorld | None:
            assert len(detections) == len(indices)
            if len(indices) == 0:
                return None

            Tslice_world_device = SE3(
                torch.from_numpy(Ts_world_device[np.array(indices), :]).to(
                    dtype=torch.float32
                )
            )
            Rslice_world_device = SO3(
                torch.from_numpy(Ts_world_device[np.array(indices), :4]).to(
                    dtype=torch.float32
                )
            )

            return AriaHandWristPoseWrtWorld(
                confidence=torch.from_numpy(
                    np.array([d.confidence for d in detections])
                ),
                wrist_position=Tslice_world_device
                @ torch.from_numpy(
                    np.array(
                        [d.wrist_position_device for d in detections], dtype=np.float32
                    )
                ),
                wrist_normal=Rslice_world_device
                @ torch.from_numpy(
                    np.array(
                        [
                            d.wrist_and_palm_normal_device.wrist_normal_device
                            for d in detections
                        ],
                        dtype=np.float32,
                    )
                ),
                palm_position=Tslice_world_device
                @ torch.from_numpy(
                    np.array(
                        [d.palm_position_device for d in detections], dtype=np.float32
                    )
                ),
                palm_normal=Rslice_world_device
                @ torch.from_numpy(
                    np.array(
                        [
                            d.wrist_and_palm_normal_device.palm_normal_device
                            for d in detections
                        ],
                        dtype=np.float32,
                    )
                ),
                indices=torch.from_numpy(np.array(indices, dtype=np.int64)),
            )

        return CorrespondedAriaHandWristPoseDetections(
            detections_left_concat=form_detections_concat(
                detections_left, indices_left
            ),
            detections_right_concat=form_detections_concat(
                detections_right, indices_right
            ),
        )


class SingleHandHamerOutputWrtCameraConcatenated(TensorDataclass):
    verts: Float[Tensor, "n_detections n_verts 3"]
    keypoints_3d: Float[Tensor, "n_detections n_keypoints 3"]
    mano_hand_global_orient: Float[Tensor, "n_detections 3 3"]
    single_hand_quats: Float[Tensor, "n_detections 15 3"]
    indices: Int[Tensor, "n_detections"]


class CorrespondedHamerDetections(TensorDataclass):
    mano_faces_right: Tensor
    mano_faces_left: Tensor
    detections_left_tuple: tuple[None | SingleHandHamerOutputWrtCamera, ...]
    detections_right_tuple: tuple[None | SingleHandHamerOutputWrtCamera, ...]
    T_cpf_cam: Tensor
    focal_length: float

    # Concatenated detections will be None if there are no detections at all.
    detections_left_concat: None | SingleHandHamerOutputWrtCameraConcatenated
    detections_right_concat: None | SingleHandHamerOutputWrtCameraConcatenated

    def get_length(self) -> int:
        assert len(self.detections_left_tuple) == len(self.detections_right_tuple)
        return len(self.detections_left_tuple)

    def slice(self, start_index: int, end_index: int) -> CorrespondedHamerDetections:
        """Slice the hand detections. Removes unused hand detections, and
        shifts indices as necessary."""

        assert start_index < end_index

        def _get_detections_in_window(
            detections_side_concat: None | SingleHandHamerOutputWrtCameraConcatenated,
        ) -> None | SingleHandHamerOutputWrtCameraConcatenated:
            if detections_side_concat is None:
                return None
            else:
                indices = detections_side_concat.indices
                indices_mask = (indices >= start_index) & (indices < end_index)
                out = detections_side_concat.map(lambda x: x[indices_mask].clone())
                out.indices -= start_index
                return out

        return CorrespondedHamerDetections(
            self.mano_faces_right,
            self.mano_faces_left,
            self.detections_left_tuple[start_index:end_index],
            self.detections_right_tuple[start_index:end_index],
            T_cpf_cam=self.T_cpf_cam,
            focal_length=self.focal_length,
            detections_left_concat=_get_detections_in_window(
                self.detections_left_concat
            ),
            detections_right_concat=_get_detections_in_window(
                self.detections_right_concat
            ),
        )

    @staticmethod
    def load(
        hand_pkl_path: Path,
        target_timestamps_sec: tuple[float, ...],
    ) -> CorrespondedHamerDetections:
        """Helper which takes as input:

        (1) A path to a pickle file containing hand detections through time.

            See feb25_hamer_outputs_from_vrs.py for how this is generated.

        (2) A set of target timestamps, sorted, in seconds.

        We then output a data structure that has hand detections (or `None`) for each target timestamp.
        """

        with open(hand_pkl_path, "rb") as f:
            hamer_out = cast(SavedHamerOutputs, pickle.load(f))

        def match_detections_to_targets(
            detections_wrt_cam: dict[int, None | SingleHandHamerOutputWrtCamera],
        ) -> list[None | SingleHandHamerOutputWrtCamera]:
            # Approximate the frame rate of the detections.
            est_fps = len(detections_wrt_cam) / (
                (max(detections_wrt_cam.keys()) - min(detections_wrt_cam.keys())) / 1e9
            )
            # Usually framerate is either 10 FPS or 30 FPS. We might want to
            # run on 1 FPS video in the future, we can tweak this assert if we
            # run into that...
            assert 5 < est_fps < 40

            # Get nanosecond timestamps within our target timestamp window.
            # Note that input dictionary keys are nanosecond timestamps!
            detect_ns = sorted(
                [
                    time_ns
                    for time_ns in detections_wrt_cam.keys()
                    if time_ns / 1e9 >= target_timestamps_sec[0] - 1 / est_fps
                    and time_ns / 1e9 <= target_timestamps_sec[-1] + 1 / est_fps
                ]
            )
            delta_matrix = np.abs(
                np.array(target_timestamps_sec)[:, None]
                - np.array(detect_ns)[None, :] / 1e9
            )

            # For each target, which is the closest detection?
            best_det_from_target = np.argmin(delta_matrix, axis=-1)

            # For each detection, which is the closest target?
            best_target_from_det = np.argmin(delta_matrix, axis=0)

            # Get detection list; we do a cycle-consistency check to make sure
            # we get a 1-to-1 mapping.
            out: list[None | SingleHandHamerOutputWrtCamera] = []
            for i in range(len(target_timestamps_sec)):
                if best_target_from_det[best_det_from_target[i]] == i:
                    out.append(detections_wrt_cam[detect_ns[best_det_from_target[i]]])
                else:
                    out.append(None)
            return out

        detections_left = match_detections_to_targets(
            hamer_out["detections_left_wrt_cam"]
        )
        detections_right = match_detections_to_targets(
            hamer_out["detections_right_wrt_cam"]
        )
        assert (
            len(detections_left) == len(detections_right) == len(target_timestamps_sec)
        )

        def make_concat_detections(
            detections_side: list[None | SingleHandHamerOutputWrtCamera],
            detections_other_side: list[None | SingleHandHamerOutputWrtCamera],
        ) -> None | SingleHandHamerOutputWrtCameraConcatenated:
            detections_side_concat = None

            # Filter out HaMeR detections that are in the same location as each
            # other.
            # Sometimes we have a left detections and a right detection both in
            # the same location. This filters both out.
            detections_side_filtered: list[None | SingleHandHamerOutputWrtCamera] = []
            for i, d in enumerate(detections_side):
                if d is None:
                    detections_side_filtered.append(None)
                    continue
                num_d = d["verts"].shape[0]

                keep_mask = np.ones(d["keypoints_3d"].shape[0], dtype=bool)
                for offset in range(-15, 15):
                    i_offset = i + offset
                    if i_offset < 0 or i_offset >= len(detections_other_side):
                        continue

                    d_other = detections_other_side[i_offset]
                    if d_other is None:
                        # detections_side_filtered.append(d)
                        continue

                    num_d_other = d_other["verts"].shape[0]

                    dist_matrix = np.linalg.norm(
                        d["keypoints_3d"][:, None, 0, :]
                        - d_other["keypoints_3d"][None, :, 0, :],
                        axis=-1,
                    )
                    assert dist_matrix.shape == (num_d, num_d_other)
                    keep_mask = np.logical_and(
                        keep_mask, np.all(dist_matrix > 0.1, axis=-1)
                    )

                if keep_mask.sum() == 0:
                    detections_side_filtered.append(None)
                else:
                    detections_side_filtered.append(
                        cast(
                            SingleHandHamerOutputWrtCamera,
                            {k: cast(np.ndarray, v)[keep_mask] for k, v in d.items()},
                        )
                    )
            del detections_side

            detections_side_not_none = [d is not None for d in detections_side_filtered]
            if not any(detections_side_not_none):
                return None
            (valid_detection_indices,) = np.where(detections_side_not_none)

            # We should be done with these.
            del detections_side_not_none
            del detections_other_side

            detections_side_concat = SingleHandHamerOutputWrtCameraConcatenated(
                verts=torch.from_numpy(
                    np.stack(
                        # Currently: we always just take the first hand detection.
                        [
                            d["verts"][0]
                            for d in detections_side_filtered
                            if d is not None
                        ]
                    )
                ).to(torch.float32),
                keypoints_3d=torch.from_numpy(
                    np.stack(
                        [
                            # Currently: we always just take the first hand detection.
                            d["keypoints_3d"][0]
                            for d in detections_side_filtered
                            if d is not None
                        ]
                    )
                ).to(torch.float32),
                mano_hand_global_orient=torch.from_numpy(
                    np.stack(
                        [
                            # Currently: we always just take the first hand detection.
                            d["mano_hand_global_orient"][0]
                            for d in detections_side_filtered
                            if d is not None
                        ]
                    )
                ).to(torch.float32),
                single_hand_quats=SO3.from_matrix(
                    torch.from_numpy(
                        np.stack(
                            [
                                # Currently: we always just take the first hand detection.
                                d["mano_hand_pose"][0]
                                for d in detections_side_filtered
                                if d is not None
                            ]
                        )
                    ).to(torch.float32)
                ).wxyz,
                indices=torch.from_numpy(valid_detection_indices),
            )
            return detections_side_concat

        return CorrespondedHamerDetections(
            mano_faces_right=torch.from_numpy(
                hamer_out["mano_faces_right"].astype(np.int64)
            ),
            mano_faces_left=torch.from_numpy(
                hamer_out["mano_faces_left"].astype(np.int64)
            ),
            detections_left_tuple=tuple(detections_left),
            detections_right_tuple=tuple(detections_right),
            T_cpf_cam=torch.from_numpy(hamer_out["T_cpf_cam"]).to(torch.float32),
            focal_length=450,
            detections_left_concat=make_concat_detections(
                detections_left, detections_right
            ),
            detections_right_concat=make_concat_detections(
                detections_right, detections_left
            ),
        )
