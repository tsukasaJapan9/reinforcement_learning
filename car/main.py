import time
import math

import numpy as np

import pybullet as p
import pybullet_data

import lidar


def set_camera(_racecar):
    focus_pos, _ = p.getBasePositionAndOrientation(_racecar)
    p.resetDebugVisualizerCamera(
        cameraDistance=3,
        cameraYaw=-90,
        cameraPitch=-70,
        cameraTargetPosition=focus_pos,
    )


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

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

plane = p.loadURDF("plane.urdf")
racecar = p.loadURDF("racecar/racecar.urdf")
# p.loadURDF("cube.urdf", [2, 2, 0.5])
simple_map = p.loadURDF("car/simple_map.urdf")

p.setGravity(0, 0, -9.8)


print("--------------------")
for i in range(p.getNumJoints(racecar)):
    j = p.getJointInfo(racecar, i)
    print(f"Joint {i}: {j[1]} ({JOINT_TYPE[j[2]]})")
print("--------------------")

lidar.set(racecar, hokuyoJoint)
detection_interval = 0.0

streeringAngle = math.radians(45)
for streer in streer_links:
    p.setJointMotorControl2(racecar, streer, p.POSITION_CONTROL, targetPosition=streeringAngle)

targetVelocity = 60 # [m/s]
for wheel in wheel_links:
    p.setJointMotorControl2(racecar, wheel, p.VELOCITY_CONTROL, targetVelocity=targetVelocity)

t = 0.0
time_step = 0.01

pos, _ = p.getBasePositionAndOrientation(racecar)

while True:
    p.stepSimulation()

    detection_interval += time_step
    if detection_interval >= 0.1:
        lidar.detection(racecar, hokuyoJoint)
        detection_interval = 0

    posPrev = pos
    pos, _ = p.getBasePositionAndOrientation(racecar)
    dx = np.sqrt((pos[0] - posPrev[0])**2 + (pos[1] - posPrev[1])**2)
    v = dx / time_step
    # print(pos, v)

    set_camera(racecar)

    time.sleep(time_step)
    t += time_step
