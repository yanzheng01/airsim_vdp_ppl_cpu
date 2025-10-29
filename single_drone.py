import argparse
import airsim
import os
import torch
import torch.nn.functional as F
import random
import math
import numpy as np

from datetime import datetime
from vdp_utils import VideoRecorder, quaternion_to_matrix, Rate
from model import Model
from time import time, sleep
from tqdm import tqdm

from airsim.types import Pose, Vector3r, Quaternionr # 位姿与坐标类型
from airsim.types import AngleLevelControllerGains, PIDGains, AngleRateControllerGains
import json

from infer import CarmeraRgbDepthRender
from PIL import Image
import torch
import io

from cfg import DEVICE

camera_name = "front_center_custom" 

parser = argparse.ArgumentParser() #多无人机协同控制系统
parser.add_argument('--resume', default='weights/checkpoint0004.pth')
parser.add_argument('--env', default='single_drone',  help='环境配置')
parser.add_argument('--target_speed', default=2.5, type=float, help='(m/s) real speed might be 2 m/s slower')
parser.add_argument('--margin', default=0.1, type=float, help='(m) radius of body') #避障安全半径(m)
parser.add_argument('--smoothness', default=0.1, type=float, help='(m) radius of body') #轨迹平滑参数
parser.add_argument('--clockspeed', default=1, type=float) # 仿真时钟速度倍数
parser.add_argument('--sr', default=3, type=int) #深度图下采样率
parser.add_argument('--no_odom', default=False, action='store_true') #默认禁用里程计输入

args = parser.parse_args()
print(args)

# client = airsim.MultirotorClient(ip='192.168.1.2', port=41452) # 连接AirSim仿真环境
# client.confirmConnection()
# client.reset()
# carmera = CarmeraRgbDepthRender(client, camera_name="0")

# drivetrain = airsim.DrivetrainType.ForwardOnly # 前向Only
# yaw_mode = airsim.YawMode(True, 0) # 默认启用偏航输入
# hover_thr = 0.593  # 悬停油门基准值

# model = Model(7 if args.no_odom else 10, 6).eval().to(DEVICE)
# if args.resume:
#     model.load_state_dict(torch.load(args.resume, map_location=DEVICE))

# h = None
# for _ in range(10):
#     _, _, h = model(
#         torch.zeros(B, 1, 12, 16, device=device),
#         torch.zeros(B, model.v_proj.in_features, device=device),
#         h)
# sleep(1)
# yaw = math.atan2(waypoints[1][1] - waypoints[0][1], waypoints[1][0] - waypoints[0][0])# 计算初始偏航角


class Drone():
    def __init__(self, env_ip, port, agent_name, odom_flag, camera_name, policy_model_path, decoder_weights_path):
        self.client = airsim.MultirotorClient(ip=env_ip, port=port) # 连接AirSim仿真环境
        self.client.confirmConnection()
        self.client.reset()
        self.carmera = CarmeraRgbDepthRender(self.client, camera_name=camera_name, decoder_weights_path=decoder_weights_path)
        self.drivetrain = airsim.DrivetrainType.ForwardOnly # 前向Only
        self.yaw_mode = airsim.YawMode(True, 0) # 默认启用偏航输入
        self.hover_thr = 0.593  # 悬停油门基准值
        self.odom = odom_flag
        self.init_policy_model(policy_model_path=policy_model_path)
        self.agent_name = agent_name
        self.init_drone()
        self.rate = Rate(15 * args.clockspeed)

    def init_policy_model(self, policy_model_path, device=DEVICE):
        self.policy_model = Model(7 if self.odom else 10, 6).eval().to(device)
        if policy_model_path:
            self.policy_model.load_state_dict(torch.load(policy_model_path, map_location=device, weights_only=True))
        h = None
        for _ in range(10):
            _, _, h = self.policy_model(
                torch.zeros(1, 1, 12, 16, device=device),
                torch.zeros(1, self.policy_model.v_proj.in_features, device=device),
                h)
        sleep(1)

    def init_task(self, wp):
        self.waypoints = wp
        self.target_pos = self.waypoints[-1]

    def init_drone(self):
        self.client.setAngleRateControllerGains(AngleRateControllerGains(
            roll_gains=PIDGains(0.2, 0.01, 0.001),
            pitch_gains=PIDGains(0.2, 0.01, 0.001),
            yaw_gains=PIDGains(0.2, 0.01, 0.001),
        ), self.agent_name)
        sleep(0.5)
        self.client.simGetCollisionInfo(self.agent_name)


    def action(self):
        sleep(0.1)
        self.client.enableApiControl(True, self.agent_name)
        self.client.armDisarm(True, self.agent_name) # 解锁电机
        sleep(1)
        yaw = math.atan2(self.waypoints[1][1] - self.waypoints[0][1], self.waypoints[1][0] - self.waypoints[0][0])# 计算初始偏航角
        start_pt = self.waypoints.pop(0)
        start_pt = [
            start_pt[0] + random.random() * 0.2 - 0.1,
            start_pt[1] + random.random() * 0.2 - 0.1,
            start_pt[2] + random.random() * 0.5 - 0.25,
        ]

        for _ in range(1): # 多次设置确保生效
            sleep(0.1)
            self.client.simSetVehiclePose(Pose(
                Vector3r(*start_pt),
                Quaternionr(0, 0, math.sin(yaw / 2), math.cos(yaw / 2))),
                ignore_collision=True, vehicle_name=self.agent_name)
            self.client.takeoffAsync(vehicle_name=self.agent_name).join()
            self.client.hoverAsync().join()
            sleep(3)

        # ===== 控制循环变量初始化 =====
        B = 1
        p_target = torch.empty((B, 3))  # 目标位置
        last_p = torch.empty((B, 3))  # 上一时刻位置
        forward_vec = torch.empty((B, 3))  # 前进向量
        v = torch.empty((B, 3))  # 速度
        R = torch.empty((B, 3, 3))  # 旋转矩阵
        traveled_distance = 0  # 已飞行距离
        traveled_time = 0  # 已飞行时间
        done_flag = False  # 任务完成标志
        has_collided = set()  # 碰撞记录
        extra = torch.tensor([[args.margin]])  # 安全边界参数
        print(extra.size())

        x, y, z = self.waypoints.pop(0)
        p_target[0] = torch.as_tensor([x, -y, -z])
        state = self.client.getMultirotorState(self.agent_name)
        q = state.kinematics_estimated.orientation
        p = state.kinematics_estimated.position

        # 坐标系转换 (AirSim右手系到PyTorch左手系)
        q = torch.as_tensor([q.w_val, q.x_val, -q.y_val, -q.z_val])
        last_p[0] = torch.as_tensor([p.x_val, -p.y_val, -p.z_val])
        forward_vec[0] = quaternion_to_matrix(q)[:, 0]

        # ===== 主控制循环 =====
        pbar = tqdm()
        hidden_state = None  # 模型隐藏状态
        t_begin_real = time()
        t_now = t_begin = state.timestamp / 1e9  # 仿真开始时间
        t_end = t_begin + 6000  # 最大运行时间
        ctl_error = 0
        ctl_error = torch.randn((1, 3)) * 0.17  # 初始控制误差

        while t_now < t_end:
            pbar.update()
            depth = self.carmera.render()
            depth = depth.view(-1, 12, 16).unsqueeze(0)
            depth *= 100
            # print(depth.size())
            state = self.client.getMultirotorState(self.agent_name)
            t_now = state.timestamp / 1e9
            p = state.kinematics_estimated.position
            q = state.kinematics_estimated.orientation
            _v = state.kinematics_estimated.linear_velocity
            p = torch.as_tensor([p.x_val, -p.y_val, -p.z_val])
            duration = t_now - t_begin
            if not done_flag:
                traveled_distance += torch.norm(p - last_p).item()
                traveled_time = duration
            last_p = p
            v[0] = torch.as_tensor([_v.x_val, -_v.y_val, -_v.z_val])
            q = torch.as_tensor([q.w_val, q.x_val, -q.y_val, -q.z_val])
            R[0] = quaternion_to_matrix(q)
            # self.client.moveToPositionAsync(*self.target_pos, 0.5, vehicle_name=self.agent_name)
            if not done_flag and torch.norm(p_target - p) < 5:
                if self.waypoints:
                    x, y, z = self.waypoints.pop(0)
                    p_target[i] = torch.as_tensor([x, -y, -z])
                else:
                    print(f"{self.agent_name} arrived in {duration}s!")
                    done_flag = True
                    self.client.moveToPositionAsync(*self.target_pos, 0.5, vehicle_name=self.agent_name)
                    if done_flag:
                        t_end = t_now + 0.5
            
            # 1. 目标速度计算
            # 计算目标速度向量，通过目标位置与当前位置的差值得到
            target_v = p_target - last_p
            # 计算目标速度向量的L2范数（欧几里得范数），保持维度不变
            target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
            # 将目标速度向量归一化，然后乘以限制后的目标速度大小
            # 通过clamp_max将速度范数限制在args.target_speed以内，实现速度限制功能
            target_v = target_v / target_v_norm * target_v_norm.clamp_max(args.target_speed)

            # 2. 坐标系转换（世界系转机体系）
            # 克隆原始旋转矩阵用于环境变换
            env_R = R.clone()
            # 提取前向向量并克隆
            fwd = R[:, :, 0].clone()
            # 初始化向上向量为零向量
            up = torch.zeros_like(fwd)
            # 将前向向量的z分量置零，确保在xy平面内
            fwd[:, 2] = 0
            # 设置向上向量的z分量为1
            up[:, 2] = 1
            # 对前向向量进行归一化处理
            fwd = fwd / torch.norm(fwd, 2, -1, keepdim=True)
            # 重新构建旋转矩阵：使用归一化的前向向量、上向量与前向向量的叉积作为右向量
            R = torch.stack([fwd, torch.cross(up, fwd), up], -1)

            # 3. 状态构建（速度估计+目标+旋转矩阵+安全边界）
            # state (in body frame): cat(velocity estimation, velocity target, rotation matrix, margin)
            # 构造状态向量，包含目标速度、机器人朝向和额外信息
            state = [torch.squeeze(target_v[:, None] @ R, 1), R[:, 2], extra]
            # 计算局部坐标系下的速度
            local_v = torch.squeeze(v[:, None] @ R, 1)
            # 如果不禁用里程计信息，则将局部速度添加到状态向量开头
            if not args.no_odom:
                state.insert(0, local_v)
            # 拼接所有状态信息形成最终状态向量
            state = torch.cat(state, -1)

            # normalize depth map
            # 4. 深度图预处理
            # 将深度图转换为tensor并添加通道维度
            # depth = torch.as_tensor(depth, device=DEVICE)[:, None]
            # 将深度值转换为视差值，并限制在合理范围内
            x = 9 / depth.clamp_(0.3, 72) - 0.6
            # 对视差图进行最大值池化降采样
            # x = F.max_pool2d(x, (args.sr, args.sr))

            # obtain velocity setpoint and prediction from nnet
            # state = (state - states_mean) / states_std
            # 5. 模型预测（速度设定点）
            state = state.to(DEVICE)
            action, _, hidden_state = self.policy_model(x, state, hidden_state)
            # action = action.cpu() * action_std + action_mean
            v_setpoint, v_est = (R @ action.cpu().reshape(B, 3, -1)).unbind(-1)

            # obtain acceleration setpoint
            # 6. 加速度设定点计算
            a_setpoint = v_setpoint - v_est + ctl_error
            a_setpoint[:, 2] += 9.80665

            # convert acceleration setpoint to rpy throttle
            # 7. 控制指令转换（加速度→油门+姿态）
            throttle = torch.norm(a_setpoint, 2, -1)
            up_vec = a_setpoint / throttle[..., None]
            throttle = throttle + local_v[:, 2] * local_v[:, 2].abs() * 0.01

            # forward vector is the normalized moving average of target vector
            # 8. 机体坐标系计算
            forward_vec = env_R[..., 0] * 5 + p_target - last_p
            forward_vec[:, 2] = (forward_vec[:, 0] * up_vec[:, 0] + forward_vec[:, 1] * up_vec[:, 1]) / -up_vec[:,2]
            forward_vec /= torch.norm(forward_vec, 2, -1, True)
            left_vec = torch.cross(up_vec, forward_vec)

            # 9. 欧拉角计算
            roll = torch.atan2(left_vec[:, 2], up_vec[:, 2])
            pitch = torch.asin(-forward_vec[:, 2])
            yaw = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])

            # 10. 发送控制指令
            # 遍历所有无人机的控制指令并执行飞行动作
            # 对于每个无人机，将滚转、俯仰、偏航和油门指令发送给AirSim客户端
            # 同时进行碰撞检测，如果发生碰撞则记录碰撞信息
            r = roll.tolist()[0]
            p = pitch.tolist()[0]
            y = yaw.tolist()[0]
            t = throttle.tolist()[0]
            # print(r, p, y, t)
            # 将油门值从归一化值转换为实际的悬停推力值
            t = t / 9.8 * self.hover_thr
            # 发送飞行动作指令给指定的无人机代理
            self.client.moveByRollPitchYawThrottleAsync(r, p, y, t, 1, self.agent_name)

            # 碰撞检测
            collision_info = self.client.simGetCollisionInfo(self.agent_name)
            if collision_info.has_collided:
                # has_collided.add(collision_info.object_name)
                print(f"{self.agent_name} collide with {collision_info.object_name}!")

            # kinematics = self.client.simGetGroundTruthKinematics()
            # # 世界坐标系下的加速度（NED：北-东-地）
            # accel_ned = kinematics.linear_acceleration
            # #print(f"Accel (NED): X={accel_ned.x_val}, Y={accel_ned.y_val}, Z={accel_ned.z_val} m/s²")

            # # 计算箭头终点（从无人机位置沿加速度方向延伸）
            # start_pos = kinematics.position
            # arrow_length = 2.0  # 缩放因子
            # end_pos = airsim.Vector3r(
            #     start_pos.x_val + accel_ned.x_val * arrow_length,
            #     start_pos.y_val + accel_ned.y_val * arrow_length,
            #     start_pos.z_val + accel_ned.z_val * arrow_length
            # )

            # self.client.simPlotArrows(
            # points_start=[start_pos],
            # points_end=[end_pos],
            # color_rgba=[1, 0, 0, 1],  # RGBA红色
            # thickness=2,
            # duration=0.1,
            # is_persistent=False
            # )

            # clockspeed = duration / (time() - t_begin_real)
            # self.rate.hz = 15 * clockspeed
            # self.rate.sleep()





        
        


     

    def main_process(self, wp):
        self.task_init(wp)
        self.action()

if __name__ == "__main__":
    drone = Drone(env_ip='192.168.1.2', port=41452, agent_name='drone_1', odom_flag=False, camera_name='0', policy_model_path=args.resume, decoder_weights_path='decoder_weights_fov120_100.pt')
    wp = [[0, 0, -3], [200, 200, -1]]
    drone.init_drone()
    drone.init_task(wp=wp)
    drone.action()
    


    
    


    

    

    
        
        
