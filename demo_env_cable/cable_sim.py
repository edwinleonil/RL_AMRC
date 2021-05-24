import pybullet as p
from time import sleep
import pybullet_data
import math

physicsClient = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=120, cameraPitch=-20, cameraTargetPosition=[0.0,0.0,0.0])
cableOrientation = p.getQuaternionFromEuler([math.pi/2,0,0])

cableOrientation = p.getQuaternionFromEuler([math.pi/2,0,math.pi/2])
cable = p.loadURDF('urdf/cable.urdf',useFixedBase=True, basePosition=[0,0,0.15], baseOrientation=cableOrientation)
plane = p.loadURDF('urdf/simpleplane.urdf',useFixedBase=True, basePosition=[0,0,0])

while p.isConnected():
  p.stepSimulation()
  sleep(1./240.)
p.disconnect()
  