import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider,calibration
# from projectaria_tools.core.mps.utils import get_nearest_pose
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import DEVICE_TIME,CLOSEST
from projectaria_tools.core.mps.utils import get_nearest_wrist_and_palm_pose
import os

def x_y_rot90(x, y, w, h):
    return h-y, x

def x_y_undistort(x, y, w, h, pinhole, calib):
    x, y = int(x), int(y)
    # a zero numpy array of shape (w, h) with coordinate (x,y) having value 1
    point = np.zeros((w, h))
    offset = 2
    for i in range(x-offset, x+offset+1):
        for j in range(y-offset, y+offset+1):
            if i>=0 and i<w and j>=0 and j<h:
                point[i, j] = 1
    undistorted_point = calibration.distort_by_calibration(point, pinhole, calib)
    x, y = np.where(undistorted_point != 0)

    if len(x)==0:
        return -1, -1
    x = np.mean(x)
    y = np.mean(y)
    # x, y = x_y_rot90(x, y, w, h)
    return x, y

def x_y_around(x, y, pinhole,offset=70):
    w,h = pinhole.get_image_size()
    x_max = min(x+offset, w-1)
    x_min = max(x-offset, 0)
    y_max = min(y+offset, h-1)
    y_min = max(y-offset, 0)
    return np.array([x_min, y_min, x_max, y_max])

def init_plotting():
    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'xtick.labelbottom': False,
        'xtick.bottom': False,
        'ytick.labelleft': False,
        'ytick.left': False,
        'xtick.labeltop': False,
        'xtick.top': False,
        'ytick.labelright': False,
        'ytick.right': False
    })

def get_offset_abs_list(time_offset=15):
    offset_abs_list = 0
    for i in range(1, time_offset+1):
        offset_abs_list.append(i)
        offset_abs_list.append(-i)
    return offset_abs_list

def per_image_hand_tracking(query_timestamp_ns, wrist_and_palm_poses, pinhole, camera_calibration, rgb_calib):
    """
    Returns:
    rgb_image: np.ndarray,
    l_existed: bool,
    r_existed: bool,
    l_p_point: np.ndarray,
    r_p_point: np.ndarray
    """

    # rgb_image,_ = vrs_data_provider.get_image_data_by_time_ns(rgb_stream_id, query_timestamp_ns, DEVICE_TIME, CLOSEST)
    # rgb_image = rgb_image.to_numpy_array()
    # rgb_image = calibration.distort_by_calibration(rgb_image, pinhole, camera_calibration)
    # rgb_image = np.rot90(rgb_image, 3)
    w,h = pinhole.get_image_size()
    T_device_rgb_camera = camera_calibration.get_transform_device_camera()
    
    wrist_and_palm_pose = get_nearest_wrist_and_palm_pose(
        wrist_and_palm_poses, query_timestamp_ns
    )
    if wrist_and_palm_pose==None:
        return 0, 0, np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])
    left_pose_confidence = wrist_and_palm_pose.left_hand.confidence
    # left_wrist_position_device = wrist_and_palm_pose.left_hand.wrist_position_device
    left_palm_position_device = wrist_and_palm_pose.left_hand.palm_position_device

    right_pose_confidence = wrist_and_palm_pose.right_hand.confidence
    # right_wrist_position_device = wrist_and_palm_pose.right_hand.wrist_position_device
    right_palm_position_device = wrist_and_palm_pose.right_hand.palm_position_device        
    
    # max_r_p_x, max_r_p_y, max_l_p_x, max_l_p_y = -1, -1, -1, -1
    # min_r_p_x, min_r_p_y, min_l_p_x, min_l_p_y = w, h, w, h
    l_p_x, l_p_y, r_p_x, r_p_y = -1, -1, -1, -1

    l_existed = True
    if left_pose_confidence!=-1:        
        l_p_xyz = T_device_rgb_camera.inverse() @ left_palm_position_device
        # l_w_xyz = T_device_rgb_camera.inverse() @ left_wrist_position_device
        try:
            l_p_x, l_p_y = rgb_calib.project_no_checks(l_p_xyz)
            l_p_x, l_p_y = x_y_undistort(l_p_x, l_p_y, w, h, pinhole, camera_calibration)
            if l_p_x==-1:
                l_existed=0
            # else:
                # max_l_p_x, min_l_p_x, max_l_p_y, min_l_p_y = x_y_around(l_p_x, l_p_y, w, h,offset=offset)

                # plt.plot(l_p_x, l_p_y, 'ro')
        except:
            l_existed=0                
        # try:
        #     l_w_x, l_w_y = rgb_calib.project(l_w_xyz)
        #     l_w_x, l_w_y = x_y_undistort(l_w_x, l_w_y, w, h, pinhole, camera_calibration)
        #     if l_w_x==-1:
        #         l_existed=0
        #     else:
        #         max_l_w_x, min_l_w_x, max_l_w_y, min_l_w_y = x_y_around(l_w_x, l_w_y, w, h,offset=offset)
        #         # plt.plot(l_w_x, l_w_y, 'go')        
        # except:
        #     l_existed=0  
        # max_l_x = max(max_l_p_x, max_l_w_x)
        # min_l_x = min(min_l_p_x, min_l_w_x)
        # max_l_y = max(max_l_p_y, max_l_w_y)
        # min_l_y = min(min_l_p_y, min_l_w_y)
    else:
        l_existed=0

    r_existed = True            
    if right_pose_confidence!=-1:
        r_p_xyz = T_device_rgb_camera.inverse() @ right_palm_position_device
        # r_w_xyz = T_device_rgb_camera.inverse() @ right_wrist_position_device
        r_existed=1
        try:
            r_p_x, r_p_y = rgb_calib.project_no_checks(r_p_xyz)
            r_p_x, r_p_y = x_y_undistort(r_p_x, r_p_y, w, h, pinhole, camera_calibration)
            if r_p_x==-1:
                r_existed=0
            # else:
            #     max_r_p_x, min_r_p_x, max_r_p_y, min_r_p_y = x_y_around(r_p_x, r_p_y, w, h,offset=offset)
            #     # plt.plot(r_p_x, r_p_y, 'bo')
        except:
            r_existed=0
        # try:
        #     r_w_x, r_w_y = rgb_calib.project(r_w_xyz)
        #     r_w_x, r_w_y = x_y_undistort(r_w_x, r_w_y, w, h, pinhole, camera_calibration)
        #     if r_w_x==-1:
        #         r_existed=0
        #     else:
        #         max_r_w_x, min_r_w_x, max_r_w_y, min_r_w_y = x_y_around(r_w_x, r_w_y, w, h,offset=offset)
        #         # plt.plot(r_w_x, r_w_y, 'yo')
        # except:
        #     r_existed=0
        # max_r_x = max(max_r_p_x, max_r_w_x)
        # min_r_x = min(min_r_p_x, min_r_w_x)
        # max_r_y = max(max_r_p_y, max_r_w_y)
        # min_r_y = min(min_r_p_y, min_r_w_y)
    else:
        r_existed=0
    
    return l_existed, r_existed, np.array([l_p_x, l_p_y]), np.array([r_p_x, r_p_y])

def get_online_calib(online_calib_path, rgb_stream_label):
    online_calibs = mps.read_online_calibration(online_calib_path)
    find = False
    for calib in online_calibs:
        for camCalib in calib.camera_calibs:
            if camCalib.get_label() == rgb_stream_label:
                rgb_calib = camCalib
                find = True
                break
        if find:
            break
    return rgb_calib

def get_info_from_aria_files(wrist_and_palm_poses_path="/secondary/home/annie/output/aria_try/mps_video_vrs/hand_tracking/wrist_and_palm_poses.csv",online_calib_path="/secondary/home/annie/output/aria_try/mps_video_vrs/slam/online_calibration.jsonl",vrs_file="/secondary/home/annie/output/aria_try/video.vrs"):
    wrist_and_palm_poses = mps.hand_tracking.read_wrist_and_palm_poses(wrist_and_palm_poses_path)
    df = pd.read_csv(wrist_and_palm_poses_path)

    online_calibs = mps.read_online_calibration(online_calib_path)

    vrs_data_provider = data_provider.create_vrs_data_provider(vrs_file)
    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = vrs_data_provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = vrs_data_provider.get_device_calibration()
    camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
    T_device_rgb_camera = camera_calibration.get_transform_device_camera()

    find = False
    for calib in online_calibs:
        for camCalib in calib.camera_calibs:
            if camCalib.get_label() == rgb_stream_label:
                rgb_calib = camCalib
                find = True
                break
        if find:
            break

    pinhole = calibration.get_linear_camera_calibration(camera_calibration.get_image_size()[0], camera_calibration.get_image_size()[1], camera_calibration.get_focal_lengths()[0])
    return wrist_and_palm_poses, df, pinhole, camera_calibration, rgb_calib, T_device_rgb_camera, vrs_data_provider, rgb_stream_id
    # for i in range(len(df)):
    #     query_timestamp_ns = df['tracking_timestamp_us'][i]*1000
    #     per_image_process(query_timestamp_ns, wrist_and_palm_poses, rgb_stream_id, vrs_data_provider, df, pinhole, camera_calibration, rgb_calib, T_device_rgb_camera)
        

    # return rgb_image, l_existed, r_existed, (max_l_x, min_l_x, max_l_y, min_l_y), (max_r_x, min_r_x, max_r_y, min_r_y)
        


