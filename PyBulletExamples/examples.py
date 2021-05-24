import time
import pybullet as p
import math
import random
from scipy.spatial import distance
import numpy as np


# set type of interface with PyBullet physic engine
physicsClientId = p.connect(p.GUI)  # connect to bullet
# physicsClientId = p.connect(p.DIRECT)  # to launch without GUI
p.setGravity(0, 0, -10)
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=30,
                             cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])
# planeUid = p.loadURDF( "ABB_IRB120/urdf/simpleplane.urdf", basePosition=[0, 0, -0.65])
# tableUid = p.loadURDF("urdf/table.urdf", basePosition=[0.5, 0, -0.65])
IRB120_Uid = p.loadURDF("urdf/ABB_with_gripper.urdf", useFixedBase=True)

numberJoints = p.getNumJoints(IRB120_Uid)


def zeros_joint_position():
    zero_joint_position = np.zeros(numberJoints)
    for j in range(numberJoints):
        p.resetJointState(IRB120_Uid, j, zero_joint_position[j])


def home_position():
    home_position = np.zeros(numberJoints)
    for j in range(numberJoints):
        p.resetJointState(IRB120_Uid, j, home_position[j])


def get_base_position_and_orientation():
    base_position_and_orientation = p.get_base_position_and_orientation(IRB120_Uid)
    print("Base position orientation:", base_position_and_orientation)


def reset_base_position_and_orientation():
    orientation = p.getQuaternionFromEuler([0, math.pi/2, 0])
    p.reset_base_position_and_orientation(IRB120_Uid, [0, 0, 0], orientation)


def get_num_of_joints():
    num_of_joints = p.getNumJoints(IRB120_Uid)
    print("Number of joints", num_of_joints)


def get_joint_info():
    joint_index = 5
    joint_info = p.getJointInfo(IRB120_Uid, joint_index)
    print(joint_info)


def get_joint_state():
    joint_index = 5
    joint_state = p.getJointState(IRB120_Uid, joint_index)
    print("Joint state:", joint_state)


def get_joint_states():
    joint_indices = [0, 1, 2, 3, 4, 5]
    joint_states = p.getJointStates(IRB120_Uid, joint_indices)
    print("Joint state:", joint_states)

# Returns cartesian position of centre of mass and other info


def get_link_state():
    link_index = 5
    link_state = p.getLinkState(IRB120_Uid, link_index)[0]
    print("Link state:", link_state)


def set_joint_motor_control():
    orientation = p.getQuaternionFromEuler([0, math.pi/2, 0])
    newPosition = [0.5, 0.2, 0.05]

    jointPoses = p.calculateInverseKinematics(
        IRB120_Uid, numberJoints-1, newPosition, orientation)
    p.setJointMotorControlArray(IRB120_Uid, list(
        range(numberJoints)), p.POSITION_CONTROL, list(jointPoses))

# doesn't work for a robot with more than one joint
# def reset_joint_state():
#     joint_index = 1
#     joint_angle = math.pi/2
#     p.resetJointStates(IRB120_Uid, joint_index, joint_angle)

# def reset_joint_states():
#     joint_indices = [0, 1, 2, 3, 4, 5]
#     joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#     p.resetJointStates(IRB120_Uid, joint_indices, joint_angles)


def close():
    p.disconnect()


def main():
    for _ in range(10000):
        p.stepSimulation()
        time.sleep(1./240.)
        home_position()
        # get_num_of_joints()
        # zeros_joint_position()
        # get_base_position_and_orientation()
        # get_joint_info()
        # get_joint_state()
        # get_joint_states()
        # get_link_state()
        # set_joint_motor_control()
    close()


if __name__ == "__main__":
    main()
