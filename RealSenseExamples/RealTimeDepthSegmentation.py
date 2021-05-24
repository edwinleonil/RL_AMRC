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
threshold = 75
# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
       
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame() 

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Create a mask with threshold to get depyh only for object of interest
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        ret,mask = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # show depth using mask
        masked_depth = cv2.bitwise_and(depth_colormap, depth_colormap, mask = mask)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', masked_depth)
        
        key = cv2.waitKey(1)

        if key == ord("+"):
            threshold += 10
            print(threshold)

        if key == ord("-"):
            threshold -= 10
            print(threshold)
        if key in (27, ord("q")):
            break

finally:

    # Stop streaming
    pipeline.stop()