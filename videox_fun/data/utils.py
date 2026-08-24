import gc
import math
import os
import random
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from einops import rearrange
from packaging import version as pver
from PIL import Image

try:
    from decord import VideoReader
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    print("Warning: decord is not installed. Falling back to PyAV for video reading. "
          "Install decord for better performance: pip install decord")

VIDEO_READER_TIMEOUT = 20


class AVVideoReader:
    """A VideoReader implementation using PyAV as a fallback when decord is unavailable.
    
    Provides the same interface as decord.VideoReader:
    - len(reader) returns total frame count
    - reader.get_batch(indices) returns a BatchFrames object with .asnumpy()
    - reader.get_avg_fps() returns the average FPS
    """
    def __init__(self, uri, num_threads=1, **kwargs):
        import av
        self._container = av.open(uri)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = 'AUTO'
        self._num_frames = self._stream.frames
        # Some videos may not report frame count; decode to count
        if self._num_frames == 0:
            for _ in self._container.decode(video=0):
                self._num_frames += 1
            self._container.seek(0)
        self._avg_fps = float(self._stream.average_rate) if self._stream.average_rate else 24.0

    def __len__(self):
        return self._num_frames

    def get_avg_fps(self):
        return self._avg_fps

    def get_batch(self, indices):
        """Read frames at specified indices. Returns an object with .asnumpy() method."""
        import av
        indices_set = set(indices)
        max_idx = max(indices)
        frames_dict = {}

        self._container.seek(0)
        frame_idx = 0
        for frame in self._container.decode(video=0):
            if frame_idx in indices_set:
                frames_dict[frame_idx] = frame.to_ndarray(format='rgb24')
            if frame_idx >= max_idx:
                break
            frame_idx += 1

        # Assemble frames in requested order
        frames = [frames_dict[i] for i in indices]
        return _AVBatchFrames(frames)

    def __del__(self):
        if hasattr(self, '_container') and self._container is not None:
            self._container.close()


class _AVBatchFrames:
    """Wrapper to mimic decord's batch result with .asnumpy() interface."""
    def __init__(self, frames):
        self._frames = frames

    def asnumpy(self):
        return np.stack(self._frames)

def get_random_mask(shape, image_start_only=False):
    f, c, h, w = shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if not image_start_only:
        if f != 1:
            mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.10, 0.2, 0.2, 0.15, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05]) 
        else:
            mask_index = np.random.choice([0, 1, 7, 8], p = [0.2, 0.7, 0.05, 0.05])
        if mask_index == 0:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # Width range of the block
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # Height range of the block

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)
            mask[:, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 1:
            mask[:, :, :, :] = 1
        elif mask_index == 2:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:, :, :, :] = 1
        elif mask_index == 3:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
        elif mask_index == 4:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # Width range of the block
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # Height range of the block

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)

            mask_frame_before = np.random.randint(0, f // 2)
            mask_frame_after = np.random.randint(f // 2, f)
            mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 5:
            mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
        elif mask_index == 6:
            num_frames_to_mask = random.randint(1, max(f // 2, 1))
            frames_to_mask = random.sample(range(f), num_frames_to_mask)

            for i in frames_to_mask:
                block_height = random.randint(1, h // 4)
                block_width = random.randint(1, w // 4)
                top_left_y = random.randint(0, h - block_height)
                top_left_x = random.randint(0, w - block_width)
                mask[i, 0, top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 1
        elif mask_index == 7:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()  # Semi-major axis
            b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()  # Semi-minor axis

            # Vectorized ellipse mask using meshgrid, applied to every frame of the clip
            y_grid, x_grid = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing='ij')
            mask[:, 0, :, :] = (((y_grid - center_y) ** 2) / (b ** 2) + ((x_grid - center_x) ** 2) / (a ** 2) < 1).to(torch.uint8)
        elif mask_index == 8:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
            # Vectorized circle mask using meshgrid, applied to every frame of the clip
            y_grid, x_grid = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing='ij')
            mask[:, 0, :, :] = ((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2 < radius ** 2).to(torch.uint8)
        elif mask_index == 9:
            for idx in range(f):
                if np.random.rand() > 0.5:
                    mask[idx, :, :, :] = 1
        else:
            raise ValueError(f"The mask_index {mask_index} is not defined")
    else:
        if f != 1:
            mask[1:, :, :, :] = 1
        else:
            mask[:, :, :, :] = 1
    return mask

@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    if HAS_DECORD:
        vr = VideoReader(*args, **kwargs)
    else:
        vr = AVVideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames

def resize_frame(frame, target_short_side):
    h, w, _ = frame.shape
    if h < w:
        if target_short_side > h:
            return frame
        new_h = target_short_side
        new_w = int(target_short_side * w / h)
    else:
        if target_short_side > w:
            return frame
        new_w = target_short_side
        new_h = int(target_short_side * h / w)
    
    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame

def padding_image(images, new_width, new_height):
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    aspect_ratio = images.width / images.height
    if new_width / new_height > 1:
        if aspect_ratio > new_width / new_height:
            new_img_width = new_width
            new_img_height = int(new_img_width / aspect_ratio)
        else:
            new_img_height = new_height
            new_img_width = int(new_img_height * aspect_ratio)
    else:
        if aspect_ratio > new_width / new_height:
            new_img_width = new_width
            new_img_height = int(new_img_width / aspect_ratio)
        else:
            new_img_height = new_height
            new_img_width = int(new_img_height * aspect_ratio)

    resized_img = images.resize((new_img_width, new_img_height))

    paste_x = (new_width - new_img_width) // 2
    paste_y = (new_height - new_img_height) // 2

    new_image.paste(resized_img, (paste_x, paste_y))

    return new_image

def resize_image_with_target_area(img: Image.Image, target_area: int = 1024 * 1024) -> Image.Image:
    """
    Resize PIL image to approximately target_area pixels while maintaining original aspect ratio,
    and ensure new width and height are multiples of 32.

    Args:
        img (PIL.Image.Image): Input image
        target_area (int): Target pixel area, e.g., 1024*1024 = 1048576

    Returns:
        PIL.Image.Image: Resized image
    """
    orig_w, orig_h = img.size
    if orig_w == 0 or orig_h == 0:
        raise ValueError("Input image has zero width or height.")

    ratio = orig_w / orig_h
    ideal_width = math.sqrt(target_area * ratio)
    ideal_height = ideal_width / ratio

    new_width = round(ideal_width / 32) * 32
    new_height = round(ideal_height / 32) * 32

    new_width = max(32, new_width)
    new_height = max(32, new_height)

    new_width = int(new_width)
    new_height = int(new_height)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    return resized_img

class Camera(object):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

def custom_meshgrid(*args):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def get_relative_pose(cam_params):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses

def ray_condition(K, c2w, H, W, device):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

def process_pose_file(pose_file_path, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu', return_poses=False):
    """Modified from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    with open(pose_file_path, 'r') as f:
        poses = f.readlines()

    poses = [pose.strip().split(' ') for pose in poses[1:]]
    cam_params = [[float(x) for x in pose] for pose in poses]
    if return_poses:
        return cam_params
    else:
        cam_params = [Camera(cam_param) for cam_param in cam_params]

        sample_wh_ratio = width / height
        pose_wh_ratio = original_pose_width / original_pose_height  # Assuming placeholder ratios, change as needed

        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = height * pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_ori_w * cam_param.fx / width
        else:
            resized_ori_h = width / pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_ori_h * cam_param.fy / height

        intrinsic = np.asarray([[cam_param.fx * width,
                                cam_param.fy * height,
                                cam_param.cx * width,
                                cam_param.cy * height]
                                for cam_param in cam_params], dtype=np.float32)

        K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
        c2ws = get_relative_pose(cam_params)  # Assuming this function is defined elsewhere
        c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
        plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
        plucker_embedding = plucker_embedding[None]
        plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
        return plucker_embedding

def process_pose_params(cam_params, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu'):
    """Modified from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    cam_params = [Camera(cam_param) for cam_param in cam_params]

    sample_wh_ratio = width / height
    pose_wh_ratio = original_pose_width / original_pose_height  # Assuming placeholder ratios, change as needed

    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = height * pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / width
    else:
        resized_ori_h = width / pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / height

    intrinsic = np.asarray([[cam_param.fx * width,
                            cam_param.fy * height,
                            cam_param.cx * width,
                            cam_param.cy * height]
                            for cam_param in cam_params], dtype=np.float32)

    K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
    c2ws = get_relative_pose(cam_params)  # Assuming this function is defined elsewhere
    c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
    plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
    plucker_embedding = plucker_embedding[None]
    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
    return plucker_embedding

# ---------------------------------------------------------------------------
# LingBot-World camera / plücker-embedding utilities.
# Modified from https://github.com/Wan-Video/Wan2.1 (lingbot-world)
# Reference: repo/lingbot-world/wan/utils/cam_utils.py, wan/image2video.py
#
# Core functions are ported verbatim from the lingbot-world repo so the camera
# control integration stays self-contained; `prepare_lingbot_dit_cond_dict`
# reproduces the camera preparation of the reference image2video pipeline.
# ---------------------------------------------------------------------------
def interpolate_camera_poses(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation, Slerp

    # interpolate translation
    interp_func_trans = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind='linear',
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    # interpolate rotation
    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    # ensure there is no sudden change in qw
    quats = src_quat_vec.as_quat().copy()  # [N, 4]
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_quat_vec = Rotation.from_quat(quats)
    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_quat = slerp_func_rot(tgt_indices)
    interpolated_rot_mat = interpolated_rot_quat.as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4))
    poses[:, :3, :3] = interpolated_rot_mat
    poses[:, :3, 3] = interpolated_trans_vec
    poses[:, 3, 3] = 1.0
    return torch.from_numpy(poses).float()


def SE3_inverse(T: torch.Tensor) -> torch.Tensor:
    Rot = T[:, :3, :3]  # [B,3,3]
    trans = T[:, :3, 3:]  # [B,3,1]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)[None, :, :].repeat(T.shape[0], 1, 1)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def compute_relative_poses(
    c2ws_mat: torch.Tensor,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> torch.Tensor:
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    # ensure identity matrix for 1st frame
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise:
        # compute pose between i and i+1
        relative_poses_framewise = torch.bmm(SE3_inverse(relative_poses[:-1]), relative_poses[1:])
        relative_poses[1:] = relative_poses_framewise
    if normalize_trans:
        # note refer to camctrl2: "we scale the coordinate inputs to roughly 1
        # standard deviation to simplify model learning."
        translations = relative_poses[:, :3, 3]  # [f, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        # only normalize when moving
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


@torch.no_grad()
def create_meshgrid(n_frames: int, height: int, width: int, bias: float = 0.5, device='cuda', dtype=torch.float32) -> torch.Tensor:
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias  # [h*w, 2]
    grid_xy = grid_xy[None, ...].repeat(n_frames, 1, 1)  # [f, h*w, 2]
    return grid_xy


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
    only_rays_d: bool = False,
):
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype)  # [f, h*w, 2]
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)  # [f, 1]

    i = grid_xy[..., 0]  # [f, h*w]
    j = grid_xy[..., 1]  # [f, h*w]
    zs = torch.ones_like(i)  # [f, h*w]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack([xs, ys, zs], dim=-1)  # [f, h*w, 3]
    directions = directions / directions.norm(dim=-1, keepdim=True)  # [f, h*w, 3]

    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2)  # [f, h*w, 3]
    if only_rays_d:
        plucker_embeddings = rays_d  # [f, h*w, 3]
        plucker_embeddings = plucker_embeddings.view([n_frames, height, width, 3])
    else:
        rays_o = c2ws_mat[:, :3, 3]  # [f, 3]
        rays_o = rays_o[:, None, :].expand_as(rays_d)  # [f, h*w, 3]
        plucker_embeddings = torch.cat([rays_o, rays_d], dim=-1)  # [f, h*w, 6]
        plucker_embeddings = plucker_embeddings.view([n_frames, height, width, 6])
    return plucker_embeddings


def get_Ks_transformed(
    Ks: torch.Tensor,
    height_org: int,
    width_org: int,
    height_resize: int,
    width_resize: int,
    height_final: int,
    width_final: int,
):
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)  # [f, 1]

    scale_x = width_resize / width_org
    scale_y = height_resize / height_org

    fx_resize = fx * scale_x
    fy_resize = fy * scale_y
    cx_resize = cx * scale_x
    cy_resize = cy * scale_y

    crop_offset_x = (width_resize - width_final) / 2
    crop_offset_y = (height_resize - height_final) / 2

    cx_final = cx_resize - crop_offset_x
    cy_final = cy_resize - crop_offset_y

    Ks_transformed = torch.zeros_like(Ks)
    Ks_transformed[:, 0:1] = fx_resize
    Ks_transformed[:, 1:2] = fy_resize
    Ks_transformed[:, 2:3] = cx_final
    Ks_transformed[:, 3:4] = cy_final

    return Ks_transformed


def prepare_lingbot_dit_cond_dict(
    action_path,
    frame_num,
    height,
    width,
    device,
    dtype=torch.bfloat16,
    control_type='cam',
    vae_stride=(4, 8, 8),
    patch_size=(1, 2, 2),
    intrinsics_org_height=480,
    intrinsics_org_width=832,
):
    """Build the camera condition dict used by WanTransformer3DModel_LingbotWorld.

    Mirrors the camera preparation in repo/lingbot-world/wan/image2video.py.

    Args:
        action_path (str): directory containing ``poses.npy`` and ``intrinsics.npy``.
        frame_num (int): desired number of frames (4n+1). It may be reduced to
            match the available camera trajectory length.
        height (int), width (int): target video resolution in pixels.
        device: torch device for the produced tensors.

    Returns:
        (dict, int): a dict ``{"c2ws_plucker_emb": (tensor[1, C, lat_f, lat_h, lat_w],)}``
        and the (possibly adjusted) ``frame_num`` which the caller MUST use so
        that the latent shapes stay aligned.
    """
    assert control_type == 'cam', "Only 'cam' control_type is currently supported."

    c2ws = np.load(os.path.join(action_path, "poses.npy"))  # opencv coordinate
    len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
    frame_num = min(frame_num, len_c2ws)
    c2ws = c2ws[:frame_num]

    lat_h = height // vae_stride[1]
    lat_w = width // vae_stride[2]
    lat_f = (frame_num - 1) // vae_stride[0] + 1

    Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
    # Intrinsics are provided for the original (480p) size; transform to (h, w).
    Ks = get_Ks_transformed(
        Ks,
        height_org=intrinsics_org_height,
        width_org=intrinsics_org_width,
        height_resize=height,
        width_resize=width,
        height_final=height,
        width_final=width,
    )
    Ks = Ks[0]

    len_c2ws = len(c2ws)
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
        src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3],
        tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
    )
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
    Ks = Ks.repeat(len(c2ws_infer), 1)

    c2ws_infer = c2ws_infer.to(device)
    Ks = Ks.to(device)

    c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, height, width, only_rays_d=False)
    c2ws_plucker_emb = rearrange(
        c2ws_plucker_emb,
        'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
        c1=int(height // lat_h),
        c2=int(width // lat_w),
    )
    c2ws_plucker_emb = c2ws_plucker_emb[None, ...]  # [b, f*h*w, c]
    c2ws_plucker_emb = rearrange(
        c2ws_plucker_emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w
    ).to(dtype)

    dit_cond_dict = {"c2ws_plucker_emb": (c2ws_plucker_emb,)}
    return dit_cond_dict, frame_num


def prepare_lingbot_dit_cond_dict_from_c2ws(
    c2ws,
    intrinsics,
    frame_num,
    height,
    width,
    device,
    dtype=torch.bfloat16,
    control_type='cam',
    vae_stride=(4, 8, 8),
    patch_size=(1, 2, 2),
    intrinsics_org_height=480,
    intrinsics_org_width=832,
):
    """Training-time variant of :func:`prepare_lingbot_dit_cond_dict`.

    Same as :func:`prepare_lingbot_dit_cond_dict` but takes already loaded /
    sampled ``c2ws`` (poses) and ``intrinsics`` tensors instead of a directory
    path. The camera trajectory is expected to correspond one-to-one with the
    ``frame_num`` sampled RGB frames used for training, so it can be fed to
    :class:`WanTransformer3DModel_LingbotWorld` alongside the noisy video
    latents.

    Args:
        c2ws (`np.ndarray` or `torch.Tensor`): shape ``[frame_num, 4, 4]``,
            per-frame camera-to-world matrices in the opencv convention.
        intrinsics (`np.ndarray` or `torch.Tensor`): shape ``[N, 4]`` or
            ``[4]``, ``(fx, fy, cx, cy)`` intrinsics matching the original
            capture resolution ``(intrinsics_org_height, intrinsics_org_width)``.
            Only the first row is used (same as inference).
        frame_num (`int`): number of sampled frames (must be ``4n+1``).
        height (`int`), width (`int`): target training resolution in pixels.
        device: torch device for the produced tensors.

    Returns:
        (dict, int): ``({"c2ws_plucker_emb": (tensor[1, C, lat_f, lat_h, lat_w],)}, frame_num)``.
    """
    assert control_type == 'cam', "Only 'cam' control_type is currently supported."

    if isinstance(c2ws, torch.Tensor):
        c2ws = c2ws.detach().cpu().numpy()
    c2ws = np.asarray(c2ws, dtype=np.float64)

    # Enforce the 4n+1 constraint expected by the VAE temporal compression.
    len_c2ws = ((len(c2ws) - 1) // vae_stride[0]) * vae_stride[0] + 1
    frame_num = min(frame_num, len_c2ws)
    c2ws = c2ws[:frame_num]

    lat_h = height // vae_stride[1]
    lat_w = width // vae_stride[2]
    lat_f = (frame_num - 1) // vae_stride[0] + 1

    if isinstance(intrinsics, torch.Tensor):
        Ks = intrinsics.detach().cpu().float()
    else:
        Ks = torch.from_numpy(np.asarray(intrinsics)).float()
    if Ks.dim() == 1:
        Ks = Ks[None, :]
    # Intrinsics are provided for the original capture size; transform to (h, w).
    Ks = get_Ks_transformed(
        Ks,
        height_org=intrinsics_org_height,
        width_org=intrinsics_org_width,
        height_resize=height,
        width_resize=width,
        height_final=height,
        width_final=width,
    )
    Ks = Ks[0]

    len_c2ws = len(c2ws)
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
        src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3],
        tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // vae_stride[0]) + 1),
    )
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
    Ks = Ks.repeat(len(c2ws_infer), 1)

    c2ws_infer = c2ws_infer.to(device)
    Ks = Ks.to(device)

    c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, height, width, only_rays_d=False)
    c2ws_plucker_emb = rearrange(
        c2ws_plucker_emb,
        'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
        c1=int(height // lat_h),
        c2=int(width // lat_w),
    )
    c2ws_plucker_emb = c2ws_plucker_emb[None, ...]  # [b, f*h*w, c]
    c2ws_plucker_emb = rearrange(
        c2ws_plucker_emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w
    ).to(dtype)

    dit_cond_dict = {"c2ws_plucker_emb": (c2ws_plucker_emb,)}
    return dit_cond_dict, frame_num
