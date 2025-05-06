from loguru import logger as guru
import os
from einops import rearrange
import torch
import torch.nn as nn
from typing import Tuple, Dict

from smplx import SMPLLayer, SMPLHLayer
from smplx.vertex_ids import vertex_ids
from smplx.utils import Struct
from smplx.lbs import lbs as smpl_lbs

from ..geometry import convert_rotation
from ..util.tensor import pad_dim

from .specs import SMPL_JOINTS
from .utils import (
    forward_kinematics,
    inverse_kinematics,
    select_vert_params,
    get_verts_with_transforms,
)


class BodyModel(nn.Module):
    """
    Wrapper around SMPLX body model class.
    """

    def __init__(
        self,
        bm_path,
        model_type: str = "smplh",
        use_pca: bool = True,
        num_pca_comps: int = 6,
        batch_size: int = 1,
        use_vtx_selector: bool = True,
        **kwargs,
    ):
        """
        Creates the body model object at the given path.
        :param bm_path: path to the body model file
        :param model_type: one of [smpl, smplh]
        :param use_vtx_selector:
            if true, returns additional vertices as joints that correspond to OpenPose joints
        """
        super().__init__()
        assert model_type in ["smpl", "smplh"]
        self.model_type = model_type

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        self.use_vtx_selector = use_vtx_selector
        cur_vertex_ids = None
        if self.use_vtx_selector:
            cur_vertex_ids = vertex_ids[model_type]
        kwargs["vertex_ids"] = cur_vertex_ids

        ext = os.path.splitext(bm_path)[-1][1:]
        if model_type == "smpl":
            cls = SMPLLayer
            self.hand_dim = 0
            self.num_betas = 10
        else:
            cls = SMPLHLayer
            self.hand_dim = cls.NUM_HAND_JOINTS * 3 if not use_pca else num_pca_comps
            self.num_betas = 16

        self.batch_size = batch_size
        self.num_joints = cls.NUM_JOINTS + 1  # include root
        self.num_body_joints = cls.NUM_BODY_JOINTS
        # create body model without default parameters
        self.bm = cls(
            bm_path,
            ext=ext,
            num_betas=self.num_betas,
            use_pca=use_pca,
            num_pca_comps=num_pca_comps,
            batch_size=batch_size,
            **kwargs,
        )
        guru.info(f"loading body model from {bm_path}, batch size {batch_size}")

        # make our own default buffers
        self.var_dims = {
            "root_orient": 3,
            "pose_body": cls.NUM_BODY_JOINTS * 3,
            "pose_hand": self.hand_dim * 2,
            "betas": self.num_betas,
            "trans": 3,
        }
        guru.info(f"variable dims {self.var_dims}")
        for name, sh in self.var_dims.items():
            self.register_buffer(name, torch.zeros(batch_size, sh))

        # save the template joints
        v_template = self.bm.v_template  # (V, 3)
        J_regressor = self.bm.J_regressor  # (J, V)
        joint_template = torch.matmul(J_regressor, v_template)[None]  # type: ignore

        # save the extra joints from the vertex template

        # (1, J, 3)
        self.register_buffer("joint_template", joint_template)
        self.parents = self.bm.parents

        shapedirs = self.bm.shapedirs  # (V, 3, B)
        j_shapedirs = rearrange(
            torch.einsum("jv,v...->j...", J_regressor, shapedirs), "a b c -> (a b) c"
        )
        # (J * 3, B)
        self.register_buffer("joint_shapedirs", j_shapedirs)
        # (B, J * 3)
        # because overparameterized, use the fewest smpl joints
        J = len(SMPL_JOINTS)
        self.register_buffer(
            "joint_shapedirs_pinv", torch.linalg.pinv(j_shapedirs[: J * 3])
        )
        self._recompute_inverse_beta_mat = False
        # self.register_buffer("joint_shapedirs_pinv", torch.linalg.pinv(j_shapedirs))
        # self._recompute_inverse_beta_mat = True

        for p in self.parameters():
            p.requires_grad_(False)

    def _fill_default_vars(self, model_args) -> Tuple[Dict, Dict]:
        """
        fill in the missing variables with defaults padded to correct batch size
        """
        B = self._get_batch_size(**model_args)

        model_vars = {}
        for name in self.var_dims:
            var = model_args.pop(name, None)
            if var is None:
                var = self._get_default_model_var(name, B)
            model_vars[name] = var
        return model_vars, model_args

    def _get_batch_size(self, **model_args) -> int:
        """
        get the batch size of the input args
        """
        B = self.batch_size
        for name in self.var_dims:
            if name in model_args and type(var := model_args[name]) == torch.Tensor:
                B = var.shape[0]
                break
        return B

    def _get_default_model_var(self, name: str, batch_size: int):
        """
        if we have the desired variable, return it, otherwise return the default value
        get model var with desired batch size
        """
        return pad_dim(getattr(self, name), batch_size)

    def get_full_pose_mats(self, model_args, add_mean: bool = True):
        """
        get the full pose from provided model args
        """
        B = self._get_batch_size(**model_args)
        names = ["root_orient", "pose_body", "pose_hand"]
        model_vars = {
            k: model_args.get(k, self._get_default_model_var(k, B)) for k in names
        }
        root_mat = model_vars["root_orient"]
        if root_mat.ndim == 2:
            root_mat = convert_rotation(
                root_mat.unsqueeze(-2), "aa", "mat"
            )  # (B, 1, 3, 3)
        body_mat = model_vars["pose_body"]
        if body_mat.ndim == 2:
            body_mat = convert_rotation(
                body_mat.reshape(B, -1, 3), "aa", "mat"
            )  # (B, J, 3, 3)
        hand_mat = self.get_hand_pose_mat(
            model_vars["pose_hand"], add_mean=add_mean
        )  # (B, H, 3, 3)
        full_pose = torch.cat([root_mat, body_mat, hand_mat], dim=-3)
        return full_pose

    def get_hand_pose_mat(
        self, pose_hand: torch.Tensor, add_mean: bool = True
    ) -> torch.Tensor:
        """
        get the hand joint rotations if applicable
        :param pose_hand (*, D)
        """
        if self.hand_dim == 0:
            return pose_hand

        B = pose_hand.shape[0]
        if self.use_pca:
            left_hand_pose = torch.einsum(
                "...i,ij->...j",
                pose_hand[..., : self.hand_dim],
                self.bm.left_hand_components,
            )
            right_hand_pose = torch.einsum(
                "...i,ij->...j",
                pose_hand[..., self.hand_dim :],
                self.bm.right_hand_components,
            )
            pose_hand = torch.cat([left_hand_pose, right_hand_pose], dim=-1)
            if add_mean:
                J = self.num_body_joints + 1
                hand_mean = self.bm.pose_mean[..., 3 * J :]  # type: ignore
                pose_hand += hand_mean

        if pose_hand.ndim == 2:
            pose_hand = convert_rotation(pose_hand.reshape(B, -1, 3), "aa", "mat")

        return pose_hand

    def forward_joints(self, **kwargs):
        """
        forward on joints only
        returns (*, J, 3) joints
        """
        model_vars, _ = self._fill_default_vars(kwargs)

        rot_mats = self.get_full_pose_mats(model_vars)
        B = rot_mats.shape[0]
        shape_diffs = torch.einsum(
            "ij,nj->ni", self.joint_shapedirs, model_vars["betas"]
        )
        shape_diffs = shape_diffs.reshape(B, -1, 3)
        joints_shaped = self.joint_template + shape_diffs

        joints_local, rel_transforms = forward_kinematics(
            rot_mats, joints_shaped, self.parents  # type: ignore
        )
        if self.use_vtx_selector:
            extra_joints = self.get_extra_joints(
                model_vars["betas"], rot_mats, rel_transforms
            )
            joints_local = torch.cat([joints_local, extra_joints], dim=-2)

        return joints_local + model_vars["trans"].unsqueeze(-2)

    def get_extra_joints(self, betas, pose_mats, rel_transforms):
        vtx_idcs = self.bm.vertex_joint_selector.extra_joints_idxs
        v_template, shapedirs, posedirs, lbs_weights = select_vert_params(
            vtx_idcs,  # type: ignore
            self.bm.v_template,  # type: ignore
            self.bm.shapedirs,  # type: ignore
            self.bm.posedirs,  # type: ignore
            self.bm.lbs_weights,  # type: ignore
        )
        return get_verts_with_transforms(
            betas,
            pose_mats,
            rel_transforms,
            v_template,
            shapedirs,
            posedirs,
            lbs_weights,
        )

    def inverse_joints(self, joints: torch.Tensor, **kwargs):
        """
        get the unposed joints (template pose)
        """
        model_vars, _ = self._fill_default_vars(kwargs)
        rot_mats = self.get_full_pose_mats(model_vars)
        joints_local = joints - model_vars["trans"].unsqueeze(-2)
        return inverse_kinematics(rot_mats, joints_local, self.parents)  # type: ignore

    def joints_to_beta(self, joint_unposed: torch.Tensor) -> torch.Tensor:
        """
        get the nearest beta such that
        joint_unposed = joint_template + A @ beta
        :param (*, J, 3) joints
        """
        # get the residual with the template
        J = len(SMPL_JOINTS)
        if self._recompute_inverse_beta_mat:
            self.joint_shapedirs_pinv = torch.linalg.pinv(self.joint_shapedirs[: J * 3])  # type: ignore
            self._recompute_inverse_beta_mat = False
        dims = joint_unposed.shape[:-2]
        joint_unposed = joint_unposed[..., :J, :]
        joint_template = self.joint_template[..., :J, :]  # type: ignore
        joint_delta = (joint_unposed - joint_template).reshape(*dims, J * 3)
        betas = torch.einsum(
            "ij,...j->...i", self.joint_shapedirs_pinv, joint_delta
        )  # (*, B)
        return betas

    def forward(self, **kwargs):
        """
        forward pass of smpl model
        expects kwargs in [root_orient, pose_body, pose_hand, betas, trans]
        to have same leading dimension if included, otherwise will pad itself
        """
        model_vars, kwargs = self._fill_default_vars(kwargs)

        rot_mats = self.get_full_pose_mats(model_vars)
        verts, joints = smpl_lbs(
            model_vars["betas"],
            rot_mats,  # type: ignore
            self.bm.v_template,  # type: ignore
            self.bm.shapedirs,  # type: ignore
            self.bm.posedirs,  # type: ignore
            self.bm.J_regressor,  # type: ignore
            self.bm.parents,  # type: ignore
            self.bm.lbs_weights,  # type: ignore
            pose2rot=False,
        )
        joints = self.bm.vertex_joint_selector(verts, joints)
        trans = model_vars["trans"].unsqueeze(-2)
        joints += trans
        verts += trans

        out = {
            "v": verts,
            "f": self.bm.faces_tensor,
            "Jtr": joints,
            "full_pose": rot_mats,
        }

        if not self.use_vtx_selector:  # don't need extra joints
            out["Jtr"] = out["Jtr"][:, : self.num_joints]

        return Struct(**out)
