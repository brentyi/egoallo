import pickle
from pathlib import Path
from typing import Literal

import torch
import tyro
import viser

from egoallo import guidance_optimizer_jax, guidance_optimizer_torch
from egoallo.fncsmpl import SmplhModel
from egoallo.hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from egoallo.network import EgoDenoiseTraj
from egoallo.vis_helpers import visualize_traj_and_hand_detections


def main(
    optimizer: Literal["ours", "lbfgs", "adam"],
    inputs_path: Path = Path("./data/guidance_optimization_inputs.pkl"),
) -> None:
    with open(inputs_path, "rb") as f:
        inputs = pickle.load(f)

    Ts_world_cpf: torch.Tensor = inputs["Ts_world_cpf"]
    traj: EgoDenoiseTraj = inputs["traj"]
    body_model: SmplhModel = inputs["body_model"]
    guidance_mode: guidance_optimizer_jax.GuidanceMode = inputs["guidance_mode"]
    phase: Literal["inner", "post"] = inputs["phase"]
    hamer_detections: CorrespondedHamerDetections = inputs["hamer_detections"]
    aria_detections: CorrespondedAriaHandWristPoseDetections = inputs["aria_detections"]

    server = viser.ViserServer()

    orig_traj = traj
    if optimizer in ("lbfgs", "adam"):
        traj, meta = guidance_optimizer_torch.do_guidance_optimization_torch(
            Ts_world_cpf=Ts_world_cpf,
            traj=orig_traj,
            body_model=body_model,
            guidance_mode=guidance_mode,
            phase=phase,
            optimizer=optimizer,
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
        )

    elif optimizer == "ours":
        traj, meta = guidance_optimizer_jax.do_guidance_optimization(
            Ts_world_cpf=Ts_world_cpf,
            traj=orig_traj,
            body_model=body_model,
            guidance_mode=guidance_mode,
            phase=phase,
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
        )
        traj, meta = guidance_optimizer_jax.do_guidance_optimization(
            Ts_world_cpf=Ts_world_cpf,
            traj=orig_traj,
            body_model=body_model,
            guidance_mode=guidance_mode,
            phase="post",
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
        )
        traj, meta = guidance_optimizer_jax.do_guidance_optimization(
            Ts_world_cpf=Ts_world_cpf,
            traj=orig_traj,
            body_model=body_model,
            guidance_mode=guidance_mode,
            phase=phase,
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
        )

    cb = visualize_traj_and_hand_detections(
        server,
        Ts_world_cpf,
        traj,
        body_model,
        hamer_detections,
        aria_detections,
    )

    while True:
        cb()


if __name__ == "__main__":
    tyro.cli(main)
