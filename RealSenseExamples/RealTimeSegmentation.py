
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
        ret,mask = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask)

        # Take only region of logo from logo image.
        masked_rgb = cv2.bitwise_and(color_image,color_image,mask = mask)        

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', masked_rgb)
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