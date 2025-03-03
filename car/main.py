import time
import math
import random
import os
import openpyxl

import numpy as np

import pybullet as p
import pybullet_data

import matplotlib.pyplot as plt

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, device, cuda

import lidar

# -------------------------------------
# ジョイントの設定など
# -------------------------------------
JOINT_TYPE = {
    p.JOINT_REVOLUTE: "revolute",
    p.JOINT_PRISMATIC: "prismatic",
    p.JOINT_SPHERICAL: "spherical",
    p.JOINT_PLANAR: "planar",
    p.JOINT_FIXED: "fixed",
    p.JOINT_POINT2POINT: "point2point",
    p.JOINT_GEAR: "gear",
}

streer_links = [4, 6]
wheel_links = [2, 3, 5, 7]
hokuyoJoint = 8

# -------------------------------------
# GUI設定
# -------------------------------------
GUI = True
if GUI:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)

# -------------------------------------
# 環境構築
# -------------------------------------
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
plane = p.loadURDF("plane.urdf")

ini_pos = [0, 0, random.uniform(-math.radians(180), math.radians(180))]
ini_pos = p.getQuaternionFromEuler(ini_pos)

racecar = p.loadURDF(
    "racecar/racecar.urdf",
    baseOrientation=ini_pos,
    flags=p.URDF_USE_SELF_COLLISION
)
simple_map = p.loadURDF(
    "car/simple_map.urdf",
    basePosition=[0.5, 3.4, 0.2],
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION
)

print("--------------------")
for i in range(p.getNumJoints(racecar)):
    j = p.getJointInfo(racecar, i)
    print(f"Joint {i}: {j[1]} ({JOINT_TYPE[j[2]]})")
print("--------------------")

# -------------------------------------
# 車の制御方法
# -------------------------------------
# ラジアン
steer_list = [
    -math.radians(90),
    -math.radians(45),
    0,
    math.radians(45),
    math.radians(90),
]

# m/s
velocuty_list = [
    -20,
    0,
    20,
    40,
    60,
]

ACTION_LIST = [
    (s, v) for s in steer_list for v in velocuty_list
]

# -------------------------------------
# LiDARの設定
# -------------------------------------
hokuyo_joint = 8
lidar.set(racecar, hokuyoJoint)
detection_interval = 0
distances = []

# -------------------------------------
# 車の動作用関数
# -------------------------------------
def set_camera(_racecar):
    focus_pos, _ = p.getBasePositionAndOrientation(_racecar)
    p.resetDebugVisualizerCamera(
        cameraDistance=4,
        cameraYaw=-90,
        cameraPitch=-89,
        cameraTargetPosition=focus_pos,
    )

def control_steer(_racecar, _steer_links, _angle):
    for steer in _steer_links:
        p.setJointMotorControl2(
            _racecar,
            steer,
            p.POSITION_CONTROL,
            targetPosition=_angle
        )

def control_velocity(_racecar, _wheel_links, _velocity):
    for wheel in _wheel_links:
        p.setJointMotorControl2(
            _racecar,
            wheel,
            p.VELOCITY_CONTROL,
            targetVelocity=_velocity
        )

# -------------------------------------
# ネットワーク
# -------------------------------------
NUM_ACTIONS = len(ACTION_LIST)
NUM_HIDDEN_NODES_1 = 64
NUM_HIDDEN_NODES_2 = 32
NUM_STATE = lidar.ray_num + 1
lr = 0.001

class NeuralNetwork(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim_in, NUM_HIDDEN_NODES_1),
            nn.ReLU(),
            nn.Linear(NUM_HIDDEN_NODES_1, NUM_HIDDEN_NODES_2),
            nn.ReLU(),
            nn.Linear(NUM_HIDDEN_NODES_2, NUM_HIDDEN_NODES_2),
            nn.ReLU(),
            nn.Linear(NUM_HIDDEN_NODES_2, dim_out),
        )

    def forward(self, x):
        # 本のユーザレビューより
        # return F.softmax(self.seq(x), dim=0)
        return F.softmax(torch.nan_to_num(self.seq(x)), dim=0)


my_device = device("cuda" if cuda.is_available() else "cpu")
print(f"Using {my_device} device")

create_model = True
weight_file_name = "car/model_weight.pth"
weight_file = os.path.isfile(weight_file_name)
if create_model or not weight_file:
    model = NeuralNetwork(NUM_STATE, NUM_ACTIONS).to(my_device)
else:
    model = torch.load(weight_file_name).to(my_device)

optimaizer = optim.Adam(model.parameters(), lr=lr)

# -------------------------------------
# 方策のアップデートと行動選択
# -------------------------------------
steer_step = 90
def decide_action(output):
    prop = output.detach().numpy()
    one_hot = torch.zeros([NUM_ACTIONS])

    # モデルのoutput(prop)が確率になっているため、確率を重みとして行動をランダムに選択
    action = np.random.choice(range(NUM_ACTIONS), p=prop)
    one_hot[action] = 1

    # 行動を選択
    steer, vel = ACTION_LIST[action]
    # steer *= np.deg2rad(steer_step)
    action = dict()

    # radian
    action["steer"] = np.round(steer, decimals=2)

    # m/s
    action["vel"] = np.round(vel, decimals=2)

    return action, one_hot

def update_policy(rewards, policies, steps, opt):
    reward_ave = (rewards.nansum(dim=1)/steps).mean()
    clampped = torch.clamp(policies, 1e-10, 1)
    Jmt = clampped.log()*(rewards - reward_ave)
    J = (Jmt.nansum(dim=1)/steps).mean()
    J.backward()
    opt.step()
    opt.zero_grad()

def end_episode(agent):
    p.removeBody(agent)
    ini_pos = [0, 0, random.uniform(-1.047, 1.047)]
    ini_pos = p.getQuaternionFromEuler(ini_pos)
    agent = p.loadURDF(
        "racecar/racecar.urdf",
        baseOrientation=ini_pos,
        flags=p.URDF_USE_SELF_COLLISION
    )
    return agent

def set_reward(distance, rv, angle_vel):
    frontside = []
    for i in range(int(len(distance)/4)):
        frontside.append(distance[i+int(len(distance)/4)])
    leftside, rightside = [], []
    for i in range(int(len(distance)/2)):
        leftside.append(distance[i])
        rightside.append(distance[i+int(len(distance)/2)])

    reward = 0

    for i in range(len(frontside)):
        if frontside[i] > 2.0:
            reward += frontside[i] * 1.5
        elif frontside[i] < 2.0:
            reward -= -1/frontside[i]

        for i in range(len(leftside)):
            if leftside[i] < 0.5:
                reward -= 1/leftside[i] / len(leftside) * 0.5
            if rightside[i] < 0.5:
                reward -= 1/rightside[i] / len(rightside) * 0.5

        # ここの計算あってる？
        front = frontside[int(len(frontside)/2)]
        left = leftside[int(len(leftside)/2)]
        right = rightside[int(len(rightside)/2)]

        if front > 2.0:
            reward += front * 0.8
        if left > 2.0:
            reward -= left * 1.2
        if right > 2.0:
            reward -= right * 1.2

        reward += rv * 0.3
        if rv == 0:
            reward -= 0.5

        if abs(angle_vel) > 10:
            reward -= 3.0
    return reward

# -------------------------------------
# 強化学習の設定
# -------------------------------------
target_reward = 2000
epochs = 3000
episodes = 10
collision_reward = -0.8
timeover_reward = -0.2
max_number_of_steps = 18000  # 3min

time_step = 0.01
update_step = 0.1

t = 0.0
step_interval = 0.0
step = 0
epoch = 0
episode = 0
experiences = []
rewards = tensor([])
pre_rewards = tensor([])
policies = tensor([])
steps = tensor([])
policy_list = tensor([np.nan]*max_number_of_steps)
reward_list = tensor([np.nan]*max_number_of_steps)
mean_reward_list = np.zeros((0))
pos_prev = []

# グラフ
plt.figure()
graph_x, graph_y = [], []

# エクセル
record_reward = []

# 車の初期位置を取得
pos, _ = p.getBasePositionAndOrientation(racecar)

while True:
    p.stepSimulation()

    t += time_step

    # 学習時はsleepをコメントアウト
    # time.sleep(time_step)

    set_camera(racecar)

    step_interval += time_step

    # 0.1秒ごとに学習
    if step_interval >= 0.1:
        if epoch < epochs:  # max 3000エポック
            if episode < episodes:  # 1 epoch = 10 episodes
                # -------------------------------------
                # 車の速度計算
                # -------------------------------------
                pos_prev = pos
                # 現在位置の取得
                pos, _ = p.getBasePositionAndOrientation(racecar)
                # 前回からの移動距離
                dx = np.sqrt(
                    (pos[0] - pos_prev[0])**2 + (pos[1] - pos_prev[1])**2
                )
                # 車の速度
                v = np.round(dx / update_step, decimals=2)

                # print(f"{dx=}, {v=}")

                # -------------------------------------
                # LiDARの距離
                # -------------------------------------
                distances = lidar.detection(racecar, hokuyoJoint)
                distances = [np.round(d, decimals=2) for d in distances]
                
                # -------------------------------------
                # ネットワークに入力
                # -------------------------------------
                # テンソルに変換
                distances_tr = torch.tensor(distances).float()
                v_tr = torch.tensor(np.array([v])).float()

                input = torch.cat([distances_tr, v_tr]).to(my_device)
                # print("------")
                # print(f"{distances_tr=}")
                # print(f"{v_tr=}")
                # print(f"{input=}")

                # 行動を決定する
                # input=torch.Size([11]), output=torch.Size([25])
                output = model(input)
                output = output.cpu()
                # action = (streer, velocity)
                # one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ...]
                action, one_hot = decide_action(output)

                # -------------------------------------
                # 車両の状態を取得
                # -------------------------------------
                # タイヤの回転速度を取得
                # rv=np.float64(60.0)
                _, rv, _, _ = [
                    np.round(n, decimals=2) for n in p.getJointState(racecar, 2)
                ]

                # 角速度を取得
                # 車両のオイラー角度を取得
                qua = p.getBasePositionAndOrientation(racecar)[1]
                # クォーターニオンのz成分
                euler_z = p.getEulerFromQuaternion(qua)[2]
                angle_vel = euler_z / update_step

                # -------------------------------------
                # 報酬の計算
                # -------------------------------------
                reward = set_reward(distances, rv, angle_vel)

                # -------------------------------------
                # 車の制御
                # -------------------------------------
                control_velocity(racecar, wheel_links, action["vel"])
                control_steer(racecar, streer_links, action["steer"])

                # -------------------------------------
                # エピソードの終了判定
                # -------------------------------------
                contact = p.getContactPoints(racecar, simple_map)
                # 障害物にぶつかる、または最大ステップ数に達した場合はエピソード終了
                if contact or len(experiences) >= max_number_of_steps-1:
                    racecar = end_episode(racecar)
                    # rewardを減点
                    if contact:
                        reward += collision_reward
                    else:
                        reward += timeover_reward
                    policies = torch.cat([policies, policy_list.reshape(1, -1)])
                    rewards = torch.cat([rewards, reward_list.reshape(1, -1)])
                    steps = torch.cat([steps, torch.tensor([step])])

                    experiences = []
                    step = 0
                    policy_list = tensor([np.nan]*max_number_of_steps)
                    reward_list = tensor([np.nan]*max_number_of_steps)

                    episode += 1

                # エピソードを進める上での1stepの情報を保存
                step_dict = {}
                step_dict["state"] = distances
                step_dict["output"] = output
                step_dict["reward"] = reward
                step_dict["action"] = action
                step_dict["one_hot"] = one_hot

                experiences.append(step_dict)

                # 1エピソード間の情報を保存。エピソードが終了したらリセットされる
                policy_list[step] = step_dict["output"] @ step_dict["one_hot"]
                reward_list[step] = step_dict["reward"]

                step += 1
                step_interval = 0

                # エピソードのループ
            else:
                # エピソード終了
                epoch += 1
                average_reward = rewards.nanmean(dim=1).mean()

                # 平均報酬
                if epoch == 1:
                    print(f"Epoch: {epoch}, Average reward: {average_reward}")
                    reward_increase_rate = None
                else:
                    if pre_rewards.dim() == 1:
                        pre_rewards = pre_rewards.unsqueeze(1)

                    if rewards.nansum(dim=1).mean() > 0:
                        reward_increase_rate = (rewards.nansum(dim=1).mean() - pre_rewards.nansum(dim=1).mean()) / rewards.nansum(dim=1).mean() * 100
                    else:
                        reward_increase_rate = (rewards.nansum(dim=1).mean() - pre_rewards.nansum(dim=1).mean()) / -rewards.nansum(dim=1).mean() * 100
                    print(f"Epoch: {epoch}, Average reward: {average_reward}, Increase rate: {reward_increase_rate}")
                pre_rewards = rewards

                # グラフ
                graph_x.append(epoch)
                graph_y.append(rewards.nansum())
                plt.plot(graph_x, graph_y)

                update_policy(rewards, policies, steps, optimaizer)

                # エクセル
                record_reward.append(float(rewards.nansum(dim=1).mean()))

                # 初期化
                experiences = []
                rewards = tensor([])
                policies = tensor([])
                steps = tensor([])
                policy_list = tensor([np.nan]*max_number_of_steps)
                reward_list = tensor([np.nan]*max_number_of_steps)

                episode = 0

                if reward_increase_rate == 0:
                    break

                if average_reward > target_reward:
                    break
        else:
            break

        # モデルの保存
        # torch.save(model, weight_file_name)

        # plt.xlabel("epoch")
        # plt.ylabel("reward")
        # plt.savefig("car/img/reward.png")
        # plt.show()

        # エクセルの保存
        # is_file = os.path.isfile("reward.xlsx")
        # if not is_file:
        #     wb = openpyxl.Workbook()
        #     wb.save("reward.xlsx")

        # wb = openpyxl.load_workbook("reward.xlsx")
        # ws_rewards = wb.create_sheet(index=0, title="average rewards")
        # ws_rewards.cell(1, 1).value = "average rewards"
        # for i in range(len(record_reward)):
        #     ws_rewards.cell(i+2, 1).value = record_reward[i]
        # wb.save("reward.xlsx")
        # print("finish")


    # step_interval += time_step
    # if step_interval >= 0.1:
    #     lidar.detection(racecar, hokuyoJoint)
    #     step_interval = 0

    # posPrev = pos
    # pos, _ = p.getBasePositionAndOrientation(racecar)
    # dx = np.sqrt((pos[0] - posPrev[0])**2 + (pos[1] - posPrev[1])**2)
    # v = dx / time_step
    # # print(pos, v)

    # set_camera(racecar)

    # time.sleep(time_step)
    # t += time_step
