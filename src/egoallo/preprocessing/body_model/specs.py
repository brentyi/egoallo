import numpy as np

SMPL_JOINTS = {
    "hips": 0,
    "leftUpLeg": 1,
    "rightUpLeg": 2,
    "spine": 3,
    "leftLeg": 4,
    "rightLeg": 5,
    "spine1": 6,
    "leftFoot": 7,
    "rightFoot": 8,
    "spine2": 9,
    "leftToeBase": 10,
    "rightToeBase": 11,
    "neck": 12,
    "leftShoulder": 13,
    "rightShoulder": 14,
    "head": 15,
    "leftArm": 16,
    "rightArm": 17,
    "leftForeArm": 18,
    "rightForeArm": 19,
    "leftHand": 20,
    "rightHand": 21,
}

SMPL_PARENTS = np.array(
    [
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        12,
        12,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
    ]
)

# reflect joints
RIGHT_CHAIN = np.array([2, 5, 8, 11, 14, 17, 19, 21])
LEFT_CHAIN = np.array([1, 4, 7, 10, 13, 16, 18, 20])
REFLECT_PERM = np.array(
    [
        0,
        2,
        1,
        3,
        5,
        4,
        6,
        8,
        7,
        9,
        11,
        10,
        12,
        14,
        13,
        15,
        17,
        16,
        19,
        18,
        21,
        20,
    ]
)
POSE_REFLECT_PERM = np.concatenate([3 * i + np.arange(3) for i in REFLECT_PERM], axis=0)

# root, left knee, right knee, left heel, right heel,
# left toe, right toe, left hand, right hand
CONTACT_JOINTS = [
    "hips",
    "leftLeg",
    "rightLeg",
    "leftFoot",
    "rightFoot",
    "leftToeBase",
    "rightToeBase",
    "leftHand",
    "rightHand",
]
CONTACT_INDS = [SMPL_JOINTS[joint] for joint in CONTACT_JOINTS]

FEET_JOINTS = [
    "leftToeBase",
    "rightToeBase",
]
FEET_INDS = [SMPL_JOINTS[joint] for joint in FEET_JOINTS]

# chosen virtual mocap markers that are "keypoints" to work with
KEYPT_VERTS = [
    4404,
    920,
    3076,
    3169,
    823,
    4310,
    1010,
    1085,
    4495,
    4569,
    6615,
    3217,
    3313,
    6713,
    6785,
    3383,
    6607,
    3207,
    1241,
    1508,
    4797,
    4122,
    1618,
    1569,
    5135,
    5040,
    5691,
    5636,
    5404,
    2230,
    2173,
    2108,
    134,
    3645,
    6543,
    3123,
    3024,
    4194,
    1306,
    182,
    3694,
    4294,
    744,
]


"""
Openpose
"""
OP_NUM_JOINTS = 25
# OP_IGNORE_JOINTS = [1, 9, 12]  # neck and left/right hip
OP_IGNORE_JOINTS = [1]  # neck
OP_EDGE_LIST = [
    [1, 8],
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [8, 9],
    [9, 10],
    [10, 11],
    [8, 12],
    [12, 13],
    [13, 14],
    [1, 0],
    [0, 15],
    [15, 17],
    [0, 16],
    [16, 18],
    [14, 19],
    [19, 20],
    [14, 21],
    [11, 22],
    [22, 23],
    [11, 24],
]
# indices to map an openpose detection to its flipped version
OP_FLIP_MAP = [
    0,
    1,
    5,
    6,
    7,
    2,
    3,
    4,
    8,
    12,
    13,
    14,
    9,
    10,
    11,
    16,
    15,
    18,
    17,
    22,
    23,
    24,
    19,
    20,
    21,
]


# From https://github.com/vchoutas/smplify-x/blob/master/smplifyx/utils.py
# Please see license for usage restrictions.
def smpl_to_openpose(
    model_type="smplh",
    use_hands=False,
    use_face=False,
    use_face_contour=False,
    openpose_format="coco25",
):
    """Returns the indices of the permutation that maps SMPL to OpenPose

    Parameters
    ----------
    model_type: str, optional
        The type of SMPL-like model that is used. The default mapping
        returned is for the SMPLX model
    use_hands: bool, optional
        Flag for adding to the returned permutation the mapping for the
        hand keypoints. Defaults to True
    use_face: bool, optional
        Flag for adding to the returned permutation the mapping for the
        face keypoints. Defaults to True
    use_face_contour: bool, optional
        Flag for appending the facial contour keypoints. Defaults to False
    openpose_format: bool, optional
        The output format of OpenPose. For now only COCO-25 and COCO-19 is
        supported. Defaults to 'coco25'

    """
    if openpose_format.lower() == "coco25":
        if model_type == "smpl":
            return np.array(
                [
                    24,
                    12,
                    17,
                    19,
                    21,
                    16,
                    18,
                    20,
                    0,
                    2,
                    5,
                    8,
                    1,
                    4,
                    7,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                ],
                dtype=np.int32,
            )
        elif model_type == "smplh":
            body_mapping = np.array(
                [
                    52,
                    12,
                    17,
                    19,
                    21,
                    16,
                    18,
                    20,
                    0,
                    2,
                    5,
                    8,
                    1,
                    4,
                    7,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                ],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        34,
                        35,
                        36,
                        63,
                        22,
                        23,
                        24,
                        64,
                        25,
                        26,
                        27,
                        65,
                        31,
                        32,
                        33,
                        66,
                        28,
                        29,
                        30,
                        67,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        49,
                        50,
                        51,
                        68,
                        37,
                        38,
                        39,
                        69,
                        40,
                        41,
                        42,
                        70,
                        46,
                        47,
                        48,
                        71,
                        43,
                        44,
                        45,
                        72,
                    ],
                    dtype=np.int32,
                )
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == "smplx":
            body_mapping = np.array(
                [
                    55,
                    12,
                    17,
                    19,
                    21,
                    16,
                    18,
                    20,
                    0,
                    2,
                    5,
                    8,
                    1,
                    4,
                    7,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                    63,
                    64,
                    65,
                ],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        37,
                        38,
                        39,
                        66,
                        25,
                        26,
                        27,
                        67,
                        28,
                        29,
                        30,
                        68,
                        34,
                        35,
                        36,
                        69,
                        31,
                        32,
                        33,
                        70,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        52,
                        53,
                        54,
                        71,
                        40,
                        41,
                        42,
                        72,
                        43,
                        44,
                        45,
                        73,
                        49,
                        50,
                        51,
                        74,
                        46,
                        47,
                        48,
                        75,
                    ],
                    dtype=np.int32,
                )

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(
                    76, 127 + 17 * use_face_contour, dtype=np.int32
                )
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError("Unknown model type: {}".format(model_type))
    elif openpose_format == "coco19":
        if model_type == "smpl":
            return np.array(
                [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28],
                dtype=np.int32,
            )
        elif model_type == "smplh":
            body_mapping = np.array(
                [52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 53, 54, 55, 56],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        34,
                        35,
                        36,
                        57,
                        22,
                        23,
                        24,
                        58,
                        25,
                        26,
                        27,
                        59,
                        31,
                        32,
                        33,
                        60,
                        28,
                        29,
                        30,
                        61,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        49,
                        50,
                        51,
                        62,
                        37,
                        38,
                        39,
                        63,
                        40,
                        41,
                        42,
                        64,
                        46,
                        47,
                        48,
                        65,
                        43,
                        44,
                        45,
                        66,
                    ],
                    dtype=np.int32,
                )
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == "smplx":
            body_mapping = np.array(
                [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        37,
                        38,
                        39,
                        60,
                        25,
                        26,
                        27,
                        61,
                        28,
                        29,
                        30,
                        62,
                        34,
                        35,
                        36,
                        63,
                        31,
                        32,
                        33,
                        64,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        52,
                        53,
                        54,
                        65,
                        40,
                        41,
                        42,
                        66,
                        43,
                        44,
                        45,
                        67,
                        49,
                        50,
                        51,
                        68,
                        46,
                        47,
                        48,
                        69,
                    ],
                    dtype=np.int32,
                )

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(
                    70, 70 + 51 + 17 * use_face_contour, dtype=np.int32
                )
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError("Unknown model type: {}".format(model_type))
    else:
        raise ValueError("Unknown joint format: {}".format(openpose_format))
