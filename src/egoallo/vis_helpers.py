import time
from pathlib import Path
from typing import Callable, TypedDict

import numpy as np
import numpy.typing as npt
import torch
import trimesh
import viser
import viser.transforms as vtf
from jaxtyping import Float
from plyfile import PlyData
from torch import Tensor

from . import fncsmpl, fncsmpl_extensions, network
from .hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from .transforms import SE3, SO3


class SplatArgs(TypedDict):
    centers: npt.NDArray[np.floating]
    """(N, 3)."""
    rgbs: npt.NDArray[np.floating]
    """(N, 3). Range [0, 1]."""
    opacities: npt.NDArray[np.floating]
    """(N, 1). Range [0, 1]."""
    covariances: npt.NDArray[np.floating]
    """(N, 3, 3)."""


def load_splat_file(splat_path: Path, center: bool = False) -> SplatArgs:
    """Load an antimatter15-style splat file."""
    start_time = time.time()
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = (
        # Each Gaussian is serialized as:
        # - position (vec3, float32)
        3 * 4
        # - xyz (vec3, float32)
        + 3 * 4
        # - rgba (vec4, uint8)
        + 4
        # - ijkl (vec4, uint8), where 0 => -1, 255 => 1.
        + 4
    )
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian

    # Reinterpret cast to dtypes that we want to extract.
    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = vtf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    centers = splat_uint8[:, 0:12].copy().view(np.float32)
    if center:
        centers -= np.mean(centers, axis=0, keepdims=True)
    print(
        f"Splat file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": centers,
        # Colors should have shape (N, 3).
        "rgbs": splat_uint8[:, 24:27] / 255.0,
        "opacities": splat_uint8[:, 27:28] / 255.0,
        # Covariances should have shape (N, 3, 3).
        "covariances": covariances,
    }


def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatArgs:
    """Load Gaussians stored in a PLY file."""
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

    Rs = vtf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
    }


def add_splat_to_viser(
    splat_or_ply_path: Path, server: viser.ViserServer, z_offset: float = 0.0
) -> None:
    """Add some Gaussian splats to the Viser server."""
    if splat_or_ply_path.suffix.lower() == ".ply":
        splat_args = load_ply_file(splat_or_ply_path)
    elif splat_or_ply_path.suffix.lower() == ".splat":
        splat_args = load_splat_file(splat_or_ply_path)
    else:
        assert False
    server.scene.add_gaussian_splats(
        "/gaussian_splats",
        centers=splat_args["centers"],
        rgbs=splat_args["rgbs"],
        opacities=splat_args["opacities"],
        covariances=splat_args["covariances"],
        position=(0.0, 0.0, z_offset),
    )


def visualize_traj_and_hand_detections(
    server: viser.ViserServer,
    Ts_world_cpf: Float[Tensor, "timesteps 7"],
    traj: network.EgoDenoiseTraj | None,
    body_model: fncsmpl.SmplhModel,
    hamer_detections: CorrespondedHamerDetections | None = None,
    aria_detections: CorrespondedAriaHandWristPoseDetections | None = None,
    points_data: np.ndarray | None = None,
    splat_path: Path | None = None,
    floor_z: float = 0.0,
    show_joints: bool = False,
    get_ego_video: Callable[[int, int, float], bytes] | None = None,
) -> Callable[[], int]:
    """Chaotic mega-function for visualization. Returns a callback that should
    be called repeatedly in a loop."""

    timesteps = Ts_world_cpf.shape[0]

    server.scene.add_grid(
        "/ground",
        plane="xy",
        cell_color=(80, 80, 80),
        section_color=(50, 50, 50),
        position=(0.0, 0.0, floor_z),
    )

    if points_data is not None:
        point_cloud = server.scene.add_point_cloud(
            "/aria_points",
            points=points_data,
            colors=np.cos(points_data + np.arange(3)) / 3.0
            + 0.7,  # Make points colorful :)
            point_size=0.01,
            # point_size=0.1,
            point_shape="sparkle",
        )
        size_slider = server.gui.add_slider(
            "Point cloud size", min=0.001, max=0.05, step=0.001, initial_value=0.005
        )

        @size_slider.on_update
        def _(_) -> None:
            if point_cloud is not None:
                point_cloud.point_size = size_slider.value

    if splat_path is not None:
        add_splat_to_viser(splat_path, server)  # , z_offset=-floor_z)

    if traj is not None:
        betas = traj.betas
        timesteps = betas.shape[1]
        sample_count = betas.shape[0]
        assert betas.shape == (sample_count, timesteps, 16)
        body_quats = SO3.from_matrix(traj.body_rotmats).wxyz
        assert body_quats.shape == (sample_count, timesteps, 21, 4)
        device = body_quats.device

        if traj.hand_rotmats is not None:
            hand_quats = SO3.from_matrix(traj.hand_rotmats).wxyz
            left_hand_quats = hand_quats[..., :15, :]
            right_hand_quats = hand_quats[..., 15:30, :]
        else:
            left_hand_quats = None
            right_hand_quats = None

        shaped = body_model.with_shape(torch.mean(betas, dim=1, keepdim=True))
        fk_outputs = shaped.with_pose_decomposed(
            T_world_root=SE3.identity(
                device=device, dtype=body_quats.dtype
            ).parameters(),
            body_quats=body_quats,
            left_hand_quats=left_hand_quats,
            right_hand_quats=right_hand_quats,
        )

        assert Ts_world_cpf.shape == (timesteps, 7)
        T_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
            # Batch axes of fk_outputs are (num_samples, time).
            # Batch axes of Ts_world_cpf are (time,).
            fk_outputs,
            Ts_world_cpf[None, ...],
        )
        fk_outputs = fk_outputs.with_new_T_world_root(T_world_root)
    else:
        shaped = None
        fk_outputs = None
        sample_count = 0

    glasses_mesh = trimesh.load("./data/glasses.stl")
    assert isinstance(glasses_mesh, trimesh.Trimesh)
    glasses_mesh.visual.face_colors = [10, 20, 20, 255]  # type: ignore

    cpf_handle = server.scene.add_frame(
        "/cpf",
        show_axes=True,
        axes_length=0.05,
        axes_radius=0.004,
    )
    server.scene.add_mesh_trimesh("/cpf/glasses", glasses_mesh, scale=0.001 * 1.05)

    # TODO: remove
    # hamer_detections = None
    # aria_detections = None

    joint_position_handles: list[viser.SceneNodeHandle] = []
    timestep_handles: list[viser.FrameHandle] = []
    hamer_handles: list[viser.MeshHandle | viser.PointCloudHandle] = []
    aria_handles: list[viser.SceneNodeHandle] = []
    for t in range(Ts_world_cpf.shape[0]):
        timestep_handles.append(
            server.scene.add_frame(f"/timesteps/{t}", show_axes=False)
        )

        # Joints.
        if show_joints and fk_outputs is not None:
            assert traj is not None
            for j in range(sample_count):
                joints_colors = np.zeros((21, 3))
                joints_colors[:, 0] = traj.contacts[j, t, :].numpy(force=True)
                joints_colors[:, 2] = 1.0 - traj.contacts[j, t, :].numpy(force=True)
                joint_position_handles.append(
                    server.scene.add_point_cloud(
                        f"/timesteps/{t}/joints",
                        points=fk_outputs.Ts_world_joint[j, t, :21, 4:7].numpy(
                            force=True
                        ),
                        colors=joints_colors,
                        point_shape="circle",
                        point_size=0.02,
                    )
                )

        # Visualize HaMeR outputs.
        if hamer_detections is not None:
            T_world_cam = SE3(Ts_world_cpf[t]) @ SE3(hamer_detections.T_cpf_cam)
            server.scene.add_frame(
                f"/timesteps/{t}/cpf/cam",
                show_axes=True,
                axes_length=0.025,
                axes_radius=0.003,
                wxyz=T_world_cam.wxyz_xyz[..., :4].numpy(force=True),
                position=T_world_cam.wxyz_xyz[..., 4:7].numpy(force=True),
            )
            hands_l = hamer_detections.detections_left_tuple[t]
            hands_r = hamer_detections.detections_right_tuple[t]
            if hands_l is not None:
                for j in range(hands_l["verts"].shape[0]):
                    hamer_handles.append(
                        server.scene.add_mesh_simple(
                            f"/timesteps/{t}/cpf/cam/left_hand{j}",
                            vertices=hands_l["verts"][j],
                            faces=hamer_detections.mano_faces_left.numpy(force=True),
                            visible=False,
                        )
                    )
                    hamer_handles.append(
                        server.scene.add_point_cloud(
                            f"/timesteps/{t}/cpf/cam/lefft_keypoints3d",
                            points=hands_l["keypoints_3d"][j],
                            colors=(255, 127, 0),
                            point_size=0.008,
                            point_shape="square",
                            visible=False,
                        )
                    )
            if hands_r is not None:
                for j in range(hands_r["verts"].shape[0]):
                    hamer_handles.append(
                        server.scene.add_mesh_simple(
                            f"/timesteps/{t}/cpf/cam/right_hand{j}",
                            vertices=hands_r["verts"][j],
                            faces=hamer_detections.mano_faces_right.numpy(force=True),
                            visible=False,
                        )
                    )
                    hamer_handles.append(
                        server.scene.add_point_cloud(
                            f"/timesteps/{t}/cpf/cam/right_keypoints3d",
                            points=hands_r["keypoints_3d"][j],
                            colors=(0, 127, 255),
                            point_size=0.008,
                            point_shape="square",
                            visible=False,
                        )
                    )

        # Visualize Aria detections.
        if aria_detections is not None:
            for side in ("left", "right"):
                detections = {
                    "left": aria_detections.detections_left_concat,
                    "right": aria_detections.detections_right_concat,
                }[side]
                if detections is None:
                    continue
                indices = detections.indices
                index = torch.searchsorted(indices, t)
                if index < len(indices) and indices[index] == t:  # found?
                    aria_handles.append(
                        server.scene.add_spline_catmull_rom(
                            f"/timesteps/{t}/aria_detections/{side}",
                            np.array(
                                [
                                    detections.wrist_position[index].numpy(force=True),
                                    detections.palm_position[index].numpy(force=True),
                                ]
                            ),
                            line_width=3.0,
                            color=(255, 0, 0) if side == "left" else (0, 255, 0),
                            visible=False,
                        )
                    )

    body_handles = (
        [
            server.scene.add_mesh_skinned(
                f"/persons/{i}",
                vertices=shaped.verts_zero[i, 0, :, :].numpy(force=True),
                faces=body_model.faces.numpy(force=True),
                bone_wxyzs=vtf.SO3.identity(
                    batch_axes=(body_model.get_num_joints() + 1,)
                ).wxyz,
                bone_positions=np.concatenate(
                    [
                        np.zeros((1, 3)),
                        # Indices are (batch, time, joint, positions).
                        shaped.joints_zero[i, :, :, :]
                        .numpy(force=True)
                        .squeeze(axis=0),
                    ],
                    axis=0,
                ),
                color=(152, 93, 229),
                skin_weights=body_model.weights.numpy(force=True),
            )
            for i in range(sample_count)
        ]
        if shaped is not None
        else []
    )

    gui_attach = server.gui.add_checkbox("Attach camera to CPF", initial_value=False)
    gui_attach_dist = server.gui.add_number("Attach distance", initial_value=0.3)
    gui_show_body = server.gui.add_checkbox("Show body", initial_value=True)
    gui_show_glasses = server.gui.add_checkbox("Show glasses", initial_value=True)
    gui_show_cpf_axes = server.gui.add_checkbox("Show CPF axes", initial_value=False)
    gui_wireframe = server.gui.add_checkbox("Wireframe", initial_value=False)
    gui_smpl_opacity = server.gui.add_slider(
        "SMPL Opacity", initial_value=1.0, min=0.0, max=1.0, step=0.01
    )
    gui_hamer_opacity = server.gui.add_slider(
        "HaMeR Opacity", initial_value=1.0, min=0.0, max=1.0, step=0.01
    )

    @gui_smpl_opacity.on_update
    def _(_) -> None:
        for handle in body_handles:
            handle.opacity = gui_smpl_opacity.value

    @gui_hamer_opacity.on_update
    def _(_) -> None:
        for handle in hamer_handles:
            if isinstance(handle, viser.MeshHandle):
                handle.opacity = gui_hamer_opacity.value

    gui_show_hamer_hands = server.gui.add_checkbox(
        "Show HaMeR hands", initial_value=False
    )
    gui_show_aria_hands = server.gui.add_checkbox(
        "Show wrist detections", initial_value=False
    )
    gui_body_color = server.gui.add_rgb("Body color", initial_value=(152, 93, 229))

    if show_joints:
        gui_show_joints = server.gui.add_checkbox("Show joints", initial_value=True)

        @gui_show_joints.on_update
        def _(_) -> None:
            for handle in joint_position_handles:
                handle.visible = gui_show_joints.value

    @gui_show_body.on_update
    def _(_) -> None:
        for handle in body_handles:
            handle.visible = gui_show_body.value

    @gui_show_glasses.on_update
    def _(_) -> None:
        # The glasses are a child of the CPF frame.
        cpf_handle.visible = gui_show_glasses.value

    @gui_show_cpf_axes.on_update
    def _(_) -> None:
        cpf_handle.show_axes = gui_show_cpf_axes.value

    @gui_wireframe.on_update
    def _(_) -> None:
        for handle in body_handles:
            handle.wireframe = gui_wireframe.value

    @gui_show_hamer_hands.on_update
    def _(_) -> None:
        for handle in hamer_handles:
            handle.visible = gui_show_hamer_hands.value

    @gui_show_aria_hands.on_update
    def _(_) -> None:
        for handle in aria_handles:
            handle.visible = gui_show_aria_hands.value

    @gui_body_color.on_update
    def _(_) -> None:
        for handle in body_handles:
            handle.color = gui_body_color.value

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=timesteps - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_start_end = server.gui.add_multi_slider(
            "Start/end",
            min=0,
            max=timesteps - 1,
            initial_value=(0, timesteps - 1),
            step=1,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=15
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % timesteps

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % timesteps

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    Ts_world_cpf_numpy = Ts_world_cpf.numpy(force=True)

    def do_update() -> None:
        t = gui_timestep.value
        cpf_handle.wxyz = Ts_world_cpf_numpy[t, :4]
        cpf_handle.position = Ts_world_cpf_numpy[t, 4:7]

        if gui_attach.value:
            for client in server.get_clients().values():
                client.camera.wxyz = (
                    vtf.SO3(cpf_handle.wxyz) @ vtf.SO3.from_z_radians(np.pi)
                ).wxyz
                client.camera.position = cpf_handle.position - vtf.SO3(
                    cpf_handle.wxyz
                ) @ np.array([0.0, 0.0, gui_attach_dist.value])

        if fk_outputs is not None:
            for i in range(sample_count):
                for b, bone_handle in enumerate(body_handles[i].bones):
                    if b == 0:
                        bone_transform = fk_outputs.T_world_root[i, t].numpy(force=True)
                    else:
                        bone_transform = fk_outputs.Ts_world_joint[i, t, b - 1].numpy(
                            force=True
                        )
                    bone_handle.wxyz = bone_transform[:4]
                    bone_handle.position = bone_transform[4:7]

        for ii, timestep_frame in enumerate(timestep_handles):
            timestep_frame.visible = t == ii

    get_viser_file = server.gui.add_button("Get .viser file")

    if get_ego_video is not None:
        ego_video = server.gui.add_button("Get Ego Video")

        @ego_video.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            notif = event.client.add_notification(
                "Getting video...", body="", loading=True, with_close_button=False
            )
            ego_video_bytes = get_ego_video(
                gui_start_end.value[0],
                gui_start_end.value[1],
                (gui_start_end.value[1] - gui_start_end.value[0]) / gui_framerate.value,
            )
            notif.remove()
            event.client.send_file_download("ego_video.mp4", ego_video_bytes)

    prev_time = time.time()
    handle = None

    def loop_cb() -> int:
        start, end = gui_start_end.value
        duration = end - start

        if get_viser_file.value is False:
            nonlocal prev_time
            now = time.time()
            sleepdur = 1.0 / gui_framerate.value - (now - prev_time)
            if sleepdur > 0.0:
                time.sleep(sleepdur)
            prev_time = now
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1 - start) % duration + start
            do_update()
            return gui_timestep.value
        else:
            # Save trajectory.
            nonlocal handle
            if handle is None:
                handle = server._start_scene_recording()
                handle.set_loop_start()
                gui_timestep.value = start

            assert handle is not None
            handle.insert_sleep(1.0 / gui_framerate.value)
            gui_timestep.value = (gui_timestep.value + 1 - start) % duration + start

            if gui_timestep.value == start:
                get_viser_file.value = False
                server.send_file_download(
                    "recording.viser", content=handle.end_and_serialize()
                )
                handle = None

            do_update()

            return gui_timestep.value

    return loop_cb
