import torch
import torch.nn.functional as F
import subprocess # 子进程管理
from time import sleep, time

class VideoRecorder:
    """视频录制器，使用FFmpeg进行深度图录制"""
    def __init__(self, output, w, h, fps=24, pix_fmt='rgb24') -> None:
        # 初始化FFmpeg子进程
        self.p = None
        self.output = output
        command = [
            "/usr/bin/ffmpeg",
            '-y',  # overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', f'{w}x{h}',  # size of one frame
            '-pix_fmt', pix_fmt,
            '-r', f'{fps}',  # frames per second
            '-i', '-',  # The imput comes from a pipe
            # '-qp', '0',
            '-s', f'{w//2*2}x{h//2*2}',
            '-an',  # Tells FFMPEG not to expect any audio
            # '-c:v', 'h264_nvenc',
            # '-preset', 'fast',
            '-loglevel', 'error',
            '-pix_fmt', 'yuv420p'
        ]
        self.p = subprocess.Popen(command + [self.output], stdin=subprocess.PIPE)

    def add_image(self, image):
        """添加单帧图像到视频流"""
        self.p.stdin.write(image)

    def close(self):
        """关闭录制器并清理资源"""
        if self.p is not None:
            self.p.stdin.close()
            self.p.wait()

class Rate:
    """精确频率控制器，用于维持固定循环频率"""
    def __init__(self, hz) -> None:
        self.hz = hz
        self.t0 = time()

    def sleep(self):
        """等待至下一个周期"""
        while True:
            to_sleep = 1 / self.hz - time() + self.t0
            if to_sleep < 0.01:
                break
            sleep(to_sleep)
        self.t0 += max(1 / self.hz, 0.5 / self.hz - to_sleep)

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    From: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    四元数转旋转矩阵
    参数:
        quaternions: 四元数张量，形状(..., 4)，实部在前
    返回:
        旋转矩阵张量，形状(..., 3, 3)
    参考: PyTorch3D转换实现
    """
    # 解包四元数分量
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    # 计算归一化系数
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    # 构建旋转矩阵元素
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))