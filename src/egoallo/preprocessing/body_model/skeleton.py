import numpy as np
import torch

from .specs import SMPL_PARENTS, SMPL_JOINTS


__all__ = [
    "NUM_KINEMATIC_CHAINS",
    "smpl_kinematic_tree",
    "joint_angles_rel_to_glob",
    "joint_angles_glob_to_rel",
]


NUM_KINEMATIC_CHAINS = 5


def smpl_kinematic_tree():
    """
    get the SMPL kinematic tree as a list of chains of joint indices
    """
    joint_idcs = list(range(len(SMPL_JOINTS)))
    tree = []
    chains = {}  # key: last vertex so far, value: chain(s))))
    for joint in joint_idcs[::-1]:
        parent = SMPL_PARENTS[joint]
        if parent in chains or parent < 0:
            continue
        chains[parent] = [parent] + chains.pop(joint, [joint])
    tree = []
    for joint, chain in chains.items():
        parent = SMPL_PARENTS[joint]
        if parent >= 0:
            chain = [parent] + chain
        tree.insert(0, np.array(chain))
    return tree


def joint_angles_rel_to_glob(rel_mats):
    """
    convert joint angles
    from relative (wrt to previous branch on kinematic chain)
    to global (wrt to root of skeleton)
    :param rotation matrices (*, 21, 3, 3)
    return (*, 21, 3, 3)
    """
    assert rel_mats.shape[-3] == len(SMPL_JOINTS) - 1
    glob_mats = torch.zeros_like(rel_mats)

    # aggregate transforms from parent to children
    kin_tree = smpl_kinematic_tree()
    for chain in kin_tree:
        for pidx, cidx in zip(chain[:-1], chain[1:]):
            # R_c0 = R_cp * R_p0
            if pidx == 0:
                glob_mats[..., cidx - 1, :, :] = rel_mats[..., cidx - 1, :, :]
            else:
                glob_mats[..., cidx - 1, :, :] = torch.matmul(
                    rel_mats[..., cidx - 1, :, :], glob_mats[..., pidx - 1, :, :]
                )
    return glob_mats


def joint_angles_glob_to_rel(glob_mats):
    """
    convert joint angles
    from global (wrt to root of skeleton)
    to relative (wrt to previous branch on kinematic chain)
    :param rotation matrices (*, 21, 3, 3)
    return (*, 21, 3, 3)
    """
    rel_mats = torch.zeros_like(glob_mats)
    assert glob_mats.shape[-3] == len(SMPL_JOINTS) - 1

    # add the root matrix to global rotations
    dims = glob_mats.shape[:-3]
    I = (
        torch.eye(3, device=glob_mats.device)
        .reshape(*(1,) * len(dims), 1, 3, 3)
        .expand(*dims, 1, 3, 3)
    )
    glob_mats = torch.cat([I, glob_mats], dim=-3)

    # invert transforms from parent to children
    kin_tree = smpl_kinematic_tree()
    for chain in kin_tree:
        pidx, cidx = chain[:-1], chain[1:]
        # R_cp = R_c0 * R_0p
        rel_mats[..., cidx - 1, :, :] = torch.matmul(
            glob_mats[..., cidx, :, :], glob_mats[..., pidx, :, :].transpose(-1, -2)
        )
    return rel_mats
