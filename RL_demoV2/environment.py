import time
import pybullet as p
import math
import random
from scipy.spatial import distance
import numpy as np
import cv2


class Env():

    def __init__(self, maxIterations, timeStep):
        """Initialise environment variables"""

        self.maxIterations = maxIterations
        p.resetDebugVisualizerCamera(cameraDistance=0.35, cameraYaw=180,
                                     cameraPitch=0, cameraTargetPosition=[0.25, 0.3, 0.2])
        self.counter = 0
        self.timeStep = timeStep
        # fov field of view (narrow or wider view)
        self.fov, self.aspect, self.nearplane, self.farplane = 70, 1, 0.01, 1.5
        self.projection_matrix = p.computeProjectionMatrixFOV(
            self.fov, self.aspect, self.nearplane, self.farplane)
        self.view_matrix = p.computeViewMatrix(
            (0.25, 0.0, 0.15), (0.25, 0.4, 0.15), (0, 0, 1))
        

        self.TCP_link_index = 10  # make it automatic !!!!!!!!!!!!!!
        self.i = 0
        self.indices_ref = 0
        self.depth_ref = 0
        self.dst_reward = 0
        self.target_01 = [0.18, 0.35, 0.07]
        self.current_TCP_position = [0,0,0]
        self.reward_1 = 0
        self.pre_position = tuple([0,0,0])
        
    def getBendingValues(self):
        
        self.i += 1
       
        [width, height, rgbaPixels, depthBuffer, _] = p.getCameraImage(
            100, 100, self.view_matrix, self.projection_matrix)

        #  convert rgba & depth data to numpy array
        rgba_image = np.asanyarray(rgbaPixels)
        depth_data = np.asanyarray(depthBuffer)

        # RGBA to BGR conversion
        bgr_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
        # BGR to gray conversion
        gray = cv2.cvtColor(rgba_image, cv2.COLOR_BGR2GRAY)
        # Get region of interest with threshhold
        ret, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask)  # swap bits so zeros become ones

        # Mask the region of intrest
        masked_rgb = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)
        masked_depth = cv2.bitwise_and(depth_data, depth_data, mask=mask)
        # Get indices for region of interest
        bin_indices = np.nonzero(mask)
        depth_indices = np.nonzero(masked_depth)
        [row, col] = np.shape(bin_indices)

        #  get indices ref
        if self.i <= 15:
            mean_first_bin_indices = np.mean(bin_indices[0][0:10])
            mean_of_last_bin_indices = np.mean(bin_indices[0][col-10:col])
            self.indices_ref = np.mean(
                [mean_first_bin_indices, mean_of_last_bin_indices])

        # print(depth_point, depthBuffer[52,40])
        # depth = (farplane*nearplane) / (farplane-(farplane-nearplane)*depth_point)  # computes depth of particular xy position

        segmendted_depth = masked_depth[depth_indices]

        # compute depth for each pixel in the segmented region
        val_a = np.multiply((self.farplane - self.nearplane), segmendted_depth)
        val_b = self.farplane - val_a
        depth_segmented = np.divide((self.farplane*self.nearplane), val_b)
        [d_indices] = np.shape(depth_segmented)

        #  get depth ref
        if self.i <= 15:
            depth_first_indices = np.mean(depth_segmented[0:10])
            depth_last_indices = np.mean(depth_segmented[d_indices-10:d_indices])
            self.depth_ref = np.mean([depth_first_indices, depth_last_indices])

        # compute up/down bending from rgb data
        rgb_region = bin_indices[0][0:col]
        bend_up = self.indices_ref - rgb_region
        bend_up[bend_up < 0] = 0
        bend_down = self.indices_ref - rgb_region
        bend_down[bend_down > 0] = 0

        # compute back/forward bending from depth data
        bend_forward = depth_segmented - self.depth_ref
        bend_forward[bend_forward < 0] = 0
        bend_backwards = depth_segmented - self.depth_ref
        bend_backwards[bend_backwards > 0] = 0

        # features for neural network
        up = sum(bend_up)
        down = -1*sum(bend_down)
        foward = sum(bend_forward)
        backward = -1*sum(bend_backwards)

        # print(self.i, "UP: ","%.2f"%(up), "DOWN", "%.2f"%(down))
        # print("Foward:", "%.2f"%(foward), "Backward","%0.2f"%(backward))

        # Show opencv image
        # cv2.namedWindow('ResizedWindow', cv2.WINDOW_NORMAL)  # name window to resize
        # cv2.resizeWindow('ResizedWindow', 800, 400) #  expand window
        # cv2.imshow('ResizedWindow',masked_rgb)  # show image
        # key = cv2.waitKey(1)

        return [up, down, foward, backward]


    def step(self, action):
        """Compute states, reward, done, info, distance"""
        self.counter += 1
        self.action = action
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        orientation = p.getQuaternionFromEuler([0, -math.pi, -math.pi])

        currentPosition = p.getLinkState(
            self.IRB120_Uid, self.TCP_link_index)[0]

        newPosition = [currentPosition[0] + self.action[0]*self.timeStep,
                       currentPosition[1] + self.action[1]*self.timeStep,
                       currentPosition[2] + self.action[2]*self.timeStep]

        jointPoses = p.calculateInverseKinematics(
            self.IRB120_Uid, self.TCP_link_index, newPosition, orientation)[0:6]
        p.setJointMotorControlArray(self.IRB120_Uid, list(
            range(6)), p.POSITION_CONTROL, list(jointPoses))
        p.stepSimulation()

        #  get new states
        [up, down, foward, backward] = self.getBendingValues()
        newPosition = p.getLinkState(self.IRB120_Uid, self.TCP_link_index)[0]

        # compute reward 
        cable_position = p.getLinkState(self.cable, 28)[0]  # link 28 is the end rigid part of the cable
        distance_trav = 100*(cable_position[0] - self.initial_cable_pos[0])

        x_diff = newPosition[0] - self.target_01[0]
        y_diff = abs(newPosition[1] - self.target_01[1])
        z_diff = abs(newPosition[2] - self.target_01[2])

        if x_diff < 0:
            x_diff *= 1000
        # if z_diff < 0.1:
        #     z_diff = 0

        
        done = False
      
        self.reward_1 = (cable_position[0] - self.pre_position[0])
        if self.reward_1 > 0: 
            self.reward_1 = 500  # *= 1000
        else:
           self.reward_1 = -500  # -500
           
        if self.counter == 1:
            self.reward_1 = 0

        reward =  self.reward_1 + sum([up*10 + down*10 + foward*1000 + backward*1000])*(-1)  + x_diff - y_diff*1000 - z_diff*1000
       
        if self.counter == self.maxIterations:
            reward = 0
            done = True
        if(distance_trav > 10):
            reward = 1000 # 50000
            done = True
        
        if(abs(newPosition[1]-0.35) > 0.06 or abs(newPosition[2]-0.0825) > 0.06):
            reward = -5000  # -50000
            done = True

        jointStates = np.zeros(6)
        for n in range(6):
            jointStates[n] = p.getJointState(self.IRB120_Uid, n)[0]
        jointStates = tuple(jointStates)
        
        observation = jointStates + tuple([up, down, foward, backward])
        # print("Observations:",observation)
        self.pre_position = cable_position
        return np.array(observation).astype(np.float32), reward, done, distance_trav

    def reset(self):
        """Reset environment"""
        self.counter = 0
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        
        self.IRB120_Uid = p.loadURDF("urdf/ABB_with_gripper.urdf", useFixedBase=True)
        self.WorkCell = p.loadURDF('urdf/workcell_new.urdf', useFixedBase=True)
        self.numberJoints = p.getNumJoints(self.IRB120_Uid) - 1  # extract end effector joint
        # print("Nunber of Joints:", self.numberJoints)
        self.cable = p.loadURDF('urdf/cable.urdf',basePosition=[0.03, 0.35, 0.0825])

        # reset joints
        home_position = [0, 0, 0, 0, 0, 0]
        for j in range(6):  # for index 0 to 5
            p.resetJointState(self.IRB120_Uid, j, home_position[j])

        # neutral position
        self.orientation = p.getQuaternionFromEuler([0, -math.pi, -math.pi])
        self.newPosition = [0.18, 0.35, 0.10]
        for i in range(10):
            p.stepSimulation()
            jointPoses = p.calculateInverseKinematics(
                self.IRB120_Uid, self.TCP_link_index, self.newPosition, self.orientation)[0:6]
            p.setJointMotorControlArray(self.IRB120_Uid, list(
                range(6))+[8, 9], p.POSITION_CONTROL, list(jointPoses)+[1, 1])

        # position 1
        self.current_TCP_position = p.getLinkState(self.IRB120_Uid, self.TCP_link_index)[0]
        orientation = p.getQuaternionFromEuler([0, -math.pi, -math.pi])
        
        while(distance.euclidean(self.current_TCP_position, self.target_01) > 0.01):
            p.stepSimulation()
            currentPosition = p.getLinkState(self.IRB120_Uid, self.TCP_link_index)[0]

            dx = self.target_01[0] - currentPosition[0]
            dy = self.target_01[1] - currentPosition[1]
            dz = self.target_01[2] - currentPosition[2]

            dv = 0.5
            newPosition = [currentPosition[0] + dx*dv,
                        currentPosition[1] + dy*dv,
                        currentPosition[2] + dz*dv]

            jointPoses = p.calculateInverseKinematics(
                self.IRB120_Uid, self.TCP_link_index, newPosition, orientation)[0:6]

            p.setJointMotorControlArray(self.IRB120_Uid, list(
                range(6))+[8, 9], p.POSITION_CONTROL, list(jointPoses)+[1, 1])

            self.current_TCP_position = p.getLinkState(self.IRB120_Uid, self.TCP_link_index)[0]

        for _ in range(30):
            p.stepSimulation()
            p.setJointMotorControlArray(self.IRB120_Uid, [8, 9], p.POSITION_CONTROL, [0, 0])

        [up, down, foward, backward] = self.getBendingValues()
        #  include all initial states as first obserbation

        jointStates = np.zeros(6)
        for n in range(6):
            jointStates[n] = p.getJointState(self.IRB120_Uid, n)[0]
        jointStates = tuple(jointStates)
        
        observation = jointStates + tuple([up, down, foward, backward])

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        
        observation = np.array(observation).astype(np.float32)
        self.initial_cable_pos = p.getLinkState(self.cable, 28)[0]
        return observation

    def close(self):
        """Close environment"""
        p.disconnect()
