# This folder is modified from the https://github.com/Mikubill/sd-webui-controlnet
# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .wholebody import Wholebody

def draw_pose(poses, H, W):
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        canvas = util.draw_bodypose(canvas, pose.body.keypoints)

        canvas = util.draw_handpose(canvas, pose.left_hand)
        canvas = util.draw_handpose(canvas, pose.right_hand)

        canvas = util.draw_facepose(canvas, pose.face)
    return canvas


class DWposeDetector:
    def __init__(self, onnx_det, onnx_pose):
        self.pose_estimation = Wholebody(onnx_det, onnx_pose)

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            keypoints_info = self.pose_estimation(oriImg)
            return draw_pose(
                Wholebody.format_result(keypoints_info),
                H,
                W,
            )
             
