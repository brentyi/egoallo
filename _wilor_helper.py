from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from dataclasses import dataclass
from torch import Tensor
from typing import Literal, TypedDict
from ultralytics import YOLO 
from jaxtyping import Float, Int

import sys

LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)


# same in _hamer_helper.py
class HandOutputsWrtCamera(TypedDict):
    """Hand outputs with respect to the camera frame."""

    verts: Float[np.ndarray, "num_hands 778 3"]
    keypoints_3d: Float[np.ndarray, "num_hands 21 3"]
    mano_hand_pose: Float[np.ndarray, "num_hands 15 3 3"]
    mano_hand_betas: Float[np.ndarray, "num_hands 10"]
    mano_hand_global_orient: Float[np.ndarray, "num_hands 1 3 3"]
    faces: Int[np.ndarray, "mesh_faces 3"]

# same in _hamer_helper.py
@dataclass(frozen=True)
class _RawHamerOutputs:
    """A typed wrapper for outputs from HaMeR."""

    # Comments here are what I got when printing out the shapes of different
    # HaMeR outputs.

    # pred_cam torch.Size([1, 3])
    pred_cam: Float[Tensor, "num_hands 3"]
    # pred_mano_params global_orient torch.Size([1, 1, 3, 3])
    pred_mano_global_orient: Float[Tensor, "num_hands 1 3 3"]
    # pred_mano_params hand_pose torch.Size([1, 15, 3, 3])
    pred_mano_hand_pose: Float[Tensor, "num_hands 15 3 3"]
    # pred_mano_params betas torch.Size([1, 10])
    pred_mano_hand_betas: Float[Tensor, "num_hands 10"]
    # pred_cam_t torch.Size([1, 3])
    pred_cam_t: Float[Tensor, "num_hands 3"]

    # focal length from model is ignored
    # focal_length torch.Size([1, 2])
    # focal_length: Float[Tensor, "num_hands 2"]

    # pred_keypoints_3d torch.Size([1, 21, 3])
    pred_keypoints_3d: Float[Tensor, "num_hands 21 3"]
    # pred_vertices torch.Size([1, 778, 3])
    pred_vertices: Float[Tensor, "num_hands 778 3"]
    # pred_keypoints_2d torch.Size([1, 21, 2])
    pred_keypoints_2d: Float[Tensor, "num_hands 21 3"]

    pred_right: Float[Tensor, "num_hands"]
    """A given hand is a right hand if this value is >0.5."""

    # These aren't technically HaMeR outputs, but putting them here for convenience.
    mano_faces_right: Tensor
    mano_faces_left: Tensor

class WiLoRHelper:
    def __init__(self, wilor_home: str="./"):
        sys.path.append(wilor_home)
        global load_wilor
        global recursive_to
        global ViTDetDataset
        global Renderer
        global cam_crop_to_full
        
        from wilor.models import load_wilor
        from wilor.utils import recursive_to
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.utils.renderer import Renderer, cam_crop_to_full
        checkpoint_path = os.path.join(wilor_home, 'pretrained_models/wilor_final.ckpt')
        cfg_path = os.path.join(wilor_home, 'pretrained_models/model_config.yaml')
        original_dir = os.getcwd()
        os.chdir(wilor_home)
        model, model_cfg = load_wilor(checkpoint_path = checkpoint_path ,
            cfg_path= cfg_path)
        os.chdir(original_dir)
        detector = YOLO(os.path.join(wilor_home, 'pretrained_models/detector.pt'))
        # Setup the renderer
        self.renderer = Renderer(model_cfg, faces=model.mano.faces)
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._model = model.to(self.device)
        self._model_cfg = model_cfg
        self._detector = detector.to(self.device)

    def look_for_hands(
        self,
        image: np.ndarray,
        focal_length: float | None = None,
        rescale_factor: float = 2.0,
    ) -> tuple[HandOutputsWrtCamera | None, HandOutputsWrtCamera | None]:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        detections = self._detector(image, conf = 0.3, verbose=False)[0]
        # breakpoint()
        bboxes    = []
        is_right  = []
        for det in detections: 
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())
        
        if len(bboxes) == 0:
            return None, None
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(self._model_cfg, image, boxes, right, rescale_factor=rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        outputs: list[_RawHamerOutputs] = []
        
        for batch in dataloader: 
            batch = recursive_to(batch, self.device)

            with torch.no_grad():
                out = self._model(batch) 
                
            multiplier    = (2*batch['right']-1)
            pred_cam      = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center    = batch["box_center"].float()
            box_size      = batch["box_size"].float()
            img_size      = batch["img_size"].float()
            if focal_length is None:
                # All of the img_size rows should be the same. I think.
                focal_length = self._model_cfg.EXTRA.FOCAL_LENGTH / self._model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            focal_length = float(focal_length)
            scaled_focal_length = focal_length
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            pred_cam_t_full = torch.tensor(pred_cam_t_full, device=self.device)
            hamer_out = _RawHamerOutputs(
                mano_faces_left=torch.from_numpy(
                    self._model.mano.faces[:, [0, 2, 1]].astype(np.int64)
                ).to(device=self.device),
                mano_faces_right=torch.from_numpy(
                    self._model.mano.faces.astype(np.int64)
                ).to(device=self.device),
                pred_cam=out["pred_cam"],
                pred_mano_global_orient=out["pred_mano_params"]["global_orient"],
                pred_mano_hand_pose=out["pred_mano_params"]["hand_pose"],
                pred_mano_hand_betas=out["pred_mano_params"]["betas"],
                pred_cam_t=pred_cam_t_full,
                pred_keypoints_3d=out["pred_keypoints_3d"],
                pred_vertices=out["pred_vertices"],
                pred_keypoints_2d=out["pred_keypoints_2d"],
                pred_right=batch["right"],
            )

            outputs.append(hamer_out)

        assert len(outputs) > 0

        stacked_outputs = _RawHamerOutputs(
            **{
                field_name: torch.cat([getattr(x, field_name) for x in outputs], dim=0)
                for field_name in vars(outputs[0]).keys()
            },
        )
        # begin new brent stuff
        verts = stacked_outputs.pred_vertices.numpy(force=True)
        keypoints_3d = stacked_outputs.pred_keypoints_3d.numpy(force=True)
        pred_cam_t = stacked_outputs.pred_cam_t.numpy(force=True)
        mano_hand_pose = stacked_outputs.pred_mano_hand_pose.numpy(force=True)
        mano_hand_betas = stacked_outputs.pred_mano_hand_betas.numpy(force=True)
        R_camera_hand = stacked_outputs.pred_mano_global_orient.squeeze(dim=1).numpy(
            force=True
        )

        is_right = (stacked_outputs.pred_right > 0.5).numpy(force=True)
        is_left = ~is_right

        detections_right_wrt_cam: HandOutputsWrtCamera | None
        if np.sum(is_right) == 0:
            detections_right_wrt_cam = None
        else:
            detections_right_wrt_cam = {
                "verts": verts[is_right] + pred_cam_t[is_right, None, :],
                "keypoints_3d": keypoints_3d[is_right] + pred_cam_t[is_right, None, :],
                "mano_hand_pose": mano_hand_pose[is_right],
                "mano_hand_betas": mano_hand_betas[is_right],
                "mano_hand_global_orient": R_camera_hand[is_right],
                "faces": self.get_mano_faces("right"),
            }

        detections_left_wrt_cam: HandOutputsWrtCamera | None
        if np.sum(is_left) == 0:
            detections_left_wrt_cam = None
        else:

            def flip_rotmats(rotmats: np.ndarray) -> np.ndarray:
                assert rotmats.shape[-2:] == (3, 3)
                from viser import transforms

                logspace = transforms.SO3.from_matrix(rotmats).log()
                logspace[..., 1] *= -1
                logspace[..., 2] *= -1
                return transforms.SO3.exp(logspace).as_matrix()

            detections_left_wrt_cam = {
                "verts": verts[is_left] * np.array([-1, 1, 1])
                + pred_cam_t[is_left, None, :],
                "keypoints_3d": keypoints_3d[is_left] * np.array([-1, 1, 1])
                + pred_cam_t[is_left, None, :],
                "mano_hand_pose": flip_rotmats(mano_hand_pose[is_left]),
                "mano_hand_betas": mano_hand_betas[is_left],
                "mano_hand_global_orient": flip_rotmats(R_camera_hand[is_left]),
                "faces": self.get_mano_faces("left"),
            }
        # end new brent stuff
        return detections_left_wrt_cam, detections_right_wrt_cam
    
    def get_mano_faces(self, side: Literal["left", "right"]) -> np.ndarray:
        if side == "left":
            return self._model.mano.faces[:, [0, 2, 1]].copy()
        else:
            return self._model.mano.faces.copy()

def example_demo():
    parser = argparse.ArgumentParser(description='WiLoR demo code')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'], help='List of file extensions to consider')

    args = parser.parse_args()

    # Download and load checkpoints
    model, model_cfg = load_wilor(checkpoint_path = './pretrained_models/wilor_final.ckpt' , cfg_path= './pretrained_models/model_config.yaml')
    detector = YOLO('./pretrained_models/detector.pt')
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    renderer_side = Renderer(model_cfg, faces=model.mano.faces)
    
    device   = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model    = model.to(device)
    detector = detector.to(device)
    model.eval()

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        detections = detector(img_cv2, conf = 0.3, verbose=False)[0]
        bboxes    = []
        is_right  = []
        for det in detections: 
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())
        
        if len(bboxes) == 0:
            # basename = os.path.basename(img_path).split('.')[0]
            # cv2.imwrite(os.path.join(args.out_folder, f'{basename}.jpg'), img_cv2)
            # print(os.path.join(args.out_folder, f'{basename}.jpg'))
            continue
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints= []
        all_kpts  = []

        for batch in dataloader: 
            batch = recursive_to(batch, device)
    
            with torch.no_grad():
                out = model(batch) 
                
            multiplier    = (2*batch['right']-1)
            pred_cam      = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center    = batch["box_center"].float()
            box_size      = batch["box_size"].float()
            img_size      = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            
            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                
                verts  = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                
                is_right    = batch['right'][n].cpu().numpy()
                verts[:,0]  = (2*is_right-1)*verts[:,0]
                joints[:,0] = (2*is_right-1)*joints[:,0]
                cam_t = pred_cam_t_full[n]
                kpts_2d = project_full_img(verts, cam_t, scaled_focal_length, img_size[n])
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_joints.append(joints)
                all_kpts.append(kpts_2d)
                
                
                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_PURPLE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{n}.obj'))

        # Render front view
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            final_img = 255*input_img_overlay[:, :, ::-1]
        else:
            final_img = img_cv2
        for i in range(len(bboxes)):
            if right[i] == 0:
                final_img = cv2.rectangle(final_img, (int(bboxes[i][0]), int(bboxes[i][1])), (int(bboxes[i][2]), int(bboxes[i][3])), (255, 100, 100), 2)
            else:
                final_img = cv2.rectangle(final_img, (int(bboxes[i][0]), int(bboxes[i][1])), (int(bboxes[i][2]), int(bboxes[i][3])), (100, 100, 255), 2)
        cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}.jpg'), final_img)

def project_full_img(points, cam_trans, focal_length, img_res): 
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3) 
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:] 
    
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]

