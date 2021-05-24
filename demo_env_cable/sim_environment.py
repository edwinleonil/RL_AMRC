
import pybullet as p
from time import sleep
import math
import numpy as np
import pybullet_data
import os
from scipy.spatial import distance
import cv2

physicsClientId = p.connect(p.GUI)
# physicsClientId = p.connect(p.DIRECT)  # to launch without GUI

p.resetDebugVisualizerCamera(cameraDistance=0.35, cameraYaw=180,
                                     cameraPitch=0, cameraTargetPosition=[0.25, 0.3, 0.2])
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
# p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
# p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
p.setGravity(0, 0, -10)

IRB120_Uid = p.loadURDF(
    'urdf/ABB_with_gripper.urdf', useFixedBase=True)
workcell = p.loadURDF('urdf/workcell_new.urdf', useFixedBase=True)
cable = p.loadURDF('urdf/cable.urdf',
                   basePosition=[0.03, 0.35, 0.0825])
# cable = p.loadURDF('RL_demo/urdf/cable.urdf',
#                    basePosition=[-0.03, 0.35, 0.0825])

numberJoints = p.getNumJoints(IRB120_Uid)

# fov field of view (narrow or wider view)
fov, aspect, nearplane, farplane = 70, 1, 0.01, 1.5
projection_matrix = p.computeProjectionMatrixFOV(
    fov, aspect, nearplane, farplane)


target_01 = [0.18, 0.35, 0.07]
target_02 = [0.32, 0.35, 0.07]

TCP_link_index = 10
currentPosition = p.getLinkState(IRB120_Uid, TCP_link_index)[0]
i = 0


def home_position():
    """function to set joint angles to zero/home position"""
    home_position = [0, 0, 0, 0, 0, 0]
    for j in range(6):  # for index 0 to 5
        p.resetJointState(IRB120_Uid, j, home_position[j])


def initial_position():
    """set joint angles to an initial position to start task"""
    initial_position = [math.pi/2, 0, 0, 0, math.pi/2, 0]
    for j in range(6):  # for index 0 to 5
        p.resetJointState(IRB120_Uid, j, initial_position[j])


def open_gripper():
    p.setJointMotorControlArray(IRB120_Uid, [8, 9], p.POSITION_CONTROL, [1, 1])


def close_gripper():
    p.setJointMotorControlArray(IRB120_Uid, [8, 9], p.POSITION_CONTROL, [0, 0])


def neutral_pose():
    orientation = p.getQuaternionFromEuler([0, -math.pi, -math.pi])
    newPosition = [0.18, 0.35, 0.10]

    jointPoses = p.calculateInverseKinematics(
        IRB120_Uid, TCP_link_index, newPosition, orientation)[0:6]
    p.setJointMotorControlArray(IRB120_Uid, list(
        range(6))+[8, 9], p.POSITION_CONTROL, list(jointPoses)+[1, 1])


def pose_01():
    current_TCP_position = p.getLinkState(IRB120_Uid, TCP_link_index)[0]
    orientation = p.getQuaternionFromEuler([0, -math.pi, -math.pi])
    while(distance.euclidean(current_TCP_position, target_01) > 0.01):
        p.stepSimulation()
       

        currentPosition = p.getLinkState(IRB120_Uid, TCP_link_index)[0]

        dx = target_01[0] - currentPosition[0]
        dy = target_01[1] - currentPosition[1]
        dz = target_01[2] - currentPosition[2]

        dv = 0.5
        newPosition = [currentPosition[0] + dx*dv,
                       currentPosition[1] + dy*dv,
                       currentPosition[2] + dz*dv]

        jointPoses = p.calculateInverseKinematics(
            IRB120_Uid, TCP_link_index, newPosition, orientation)[0:6]

        p.setJointMotorControlArray(IRB120_Uid, list(
            range(6))+[8, 9], p.POSITION_CONTROL, list(jointPoses)+[1, 1])

        current_TCP_position = p.getLinkState(IRB120_Uid, TCP_link_index)[0]


def pose_02():
    current_TCP_position = p.getLinkState(IRB120_Uid, TCP_link_index)[0]
    orientation = p.getQuaternionFromEuler([0, -math.pi, -math.pi])
    while(distance.euclidean(current_TCP_position, target_02) > 0.01):
        p.stepSimulation()
    
        currentPosition = p.getLinkState(IRB120_Uid, TCP_link_index)[0]
        dx = target_02[0] - currentPosition[0]
        dv = 0.5
        newPosition = [currentPosition[0] + dx*dv,
                       0.35,
                       0.075]

        jointPoses = p.calculateInverseKinematics(
            IRB120_Uid, 10, newPosition, orientation)[0:6]

        p.setJointMotorControlArray(IRB120_Uid, list(
            range(6))+[8, 9], p.POSITION_CONTROL, list(jointPoses)+[0, 0])

        current_TCP_position = p.getLinkState(IRB120_Uid, TCP_link_index)[0]


def get_cablePosition():
    currentPosition = p.getLinkState(cable, 0)[0]
    distance_moved = currentPosition[0] - 0.03
    print("Cable position: ",distance_moved)

def get_link_state():
    link_index = 28
    link_state = p.getLinkState(cable, link_index)[0]
    print("Link state:", link_state)

def get_num_of_joints():
    num_of_joints = p.getNumJoints(cable)
    print("Number of joints", num_of_joints)


def getBendingValues():
    global i
    global indices_ref
    global depth_ref
    i += 1
    view_matrix = p.computeViewMatrix(
        (0.25, 0.0, 0.15), (0.25, 0.4, 0.15), (0, 0, 1))
    [width, height, rgbPixels, depthBuffer, segmentationMaskBuffer] = p.getCameraImage(
        100, 100, view_matrix, projection_matrix)

    #  convert rgba & depth data to numpy array
    rgba_image = np.asanyarray(rgbPixels)
    depth_data = np.asanyarray(depthBuffer)

    # rgb_image = rgba2rgb(rgba_image)
    # RGBA to BGR conversion
    bgr_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
    # BGR to gray conversion
    gray = cv2.cvtColor(rgba_image, cv2.COLOR_BGR2GRAY)
    # Get region of interest with threshhold
    ret, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)  # swap bits so zeros become ones

    # Mask the region of intrest
    masked_rgb = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)
    masked_depth = cv2.bitwise_and(depth_data, depth_data, mask=mask)
    # Get indices for region of interest
    bin_indices = np.nonzero(mask)
    depth_indices = np.nonzero(masked_depth)
    [row, col] = np.shape(bin_indices)

    #  get indices ref
    if i <= 15:
        mean_first_bin_indices = np.mean(bin_indices[0][0:10])
        mean_of_last_bin_indices = np.mean(bin_indices[0][col-10:col])
        indices_ref = np.mean(
            [mean_first_bin_indices, mean_of_last_bin_indices])

    # print(depth_point, depthBuffer[52,40])
    # depth = (farplane*nearplane) / (farplane-(farplane-nearplane)*depth_point)  # computes depth of particular xy position

    segmendted_depth = masked_depth[depth_indices]

    # compute depth for each pixel in the segmented region
    val_a = np.multiply((farplane-nearplane), segmendted_depth)
    val_b = farplane - val_a
    depth_segmented = np.divide((farplane*nearplane), val_b)
    [d_indices] = np.shape(depth_segmented)

    #  get depth ref
    if i <= 15:
        depth_first_indices = np.mean(depth_segmented[0:10])
        depth_last_indices = np.mean(depth_segmented[d_indices-10:d_indices])
        depth_ref = np.mean([depth_first_indices, depth_last_indices])

    # get depth data for the segmented area
    depth_mean_segmented = np.mean(depth_segmented)
    rgb_segmented = bin_indices[0][0:col]

    # compute up/down bending from rgb data
    rgb_region = bin_indices[0][0:col]
    bend_up = indices_ref - rgb_region
    bend_up[bend_up < 0] = 0
    bend_down = indices_ref - rgb_region
    bend_down[bend_down > 0] = 0

    # compute back/forward bending from depth data
    bend_forward = depth_segmented - depth_ref
    bend_forward[bend_forward < 0] = 0
    bend_backwards = depth_segmented - depth_ref
    bend_backwards[bend_backwards > 0] = 0

    # features for neural network
    up = sum(bend_up)
    down = -1*sum(bend_down)
    foward = sum(bend_forward)
    backward = -1*sum(bend_backwards)

    # print("UP: ","%.2f"%(up), "DOWN", "%.2f"%(down))
    # print("Foward:", "%.2f"%(foward), "Backward","%0.2f"%(backward))

    # # Show opencv image
    cv2.namedWindow('ResizedWindow', cv2.WINDOW_NORMAL)  # name window to resize
    cv2.resizeWindow('ResizedWindow', 800, 400) #  expand window
    cv2.imshow('ResizedWindow',masked_rgb)  # show image
    key = cv2.waitKey(1)

    return up, down, foward, backward


for n in range(500000):
    p.stepSimulation()
    getBendingValues()
    # sleep(1/256)
    # for w in range(20):
    #     p.stepSimulation()
        # getBendingValues()
    #     open_gripper()
    #     sleep(1/1000)

    # for x in range(50):
    #     p.stepSimulation()
    #     getBendingValues()
    #     neutral_pose()
    #     sleep(1/1000)

    # pose_01()
    # for t in range(20):
    #     p.stepSimulation()
    #     getBendingValues()
    #     close_gripper()
    #     sleep(1/256)
    # get_cablePosition()

    # pose_02()

    # getBendingValues()
    # get_link_state()
    # get_num_of_joints()
