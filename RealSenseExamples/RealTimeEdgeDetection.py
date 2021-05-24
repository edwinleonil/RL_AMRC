import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
canny_min = 50
canny_max = 200
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray,canny_min,canny_max)
        rgb1 = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) # convert edges to rgb
        rgb1 *= np.array((0,1,0),np.uint8) # set to green colour

        contours, hierarchy=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        

        rgb2 = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) # convert edges to rgb
        rgb2 *= np.array((0,1,0),np.uint8) # set to green colour        

        cv2.drawContours(color_image,contours,-1,(0,255,0),3)

        # Stack both images horizontally
        images = np.hstack((rgb1, color_image))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)

        if key == ord("z"):
            canny_min += 10
        if key == ord("x"):
            canny_min -= 10
        if key == ord("c"):
            canny_max += 10
        if key == ord("v"):
            canny_max -= 10
        if key in (27, ord("q")):
            break

finally:

    # Stop streaming
    pipeline.stop()