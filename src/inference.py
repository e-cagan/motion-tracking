"""
Module for inferencing motion detection.
"""

import numpy as np
import time
import cv2


if __name__ == '__main__':
    
    # Set up the camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    # Take the FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_frame = None

    # Start inferencing
    while cap.isOpened():
        
        # Read the video capturer (camera)
        ret, frame = cap.read()
        if not ret:
            print("Couldn't open the camera.")
            break
        
        ####################################### PIPELINE VERSION 1 #######################################

        # 1. Convert the frame to grayscale to reduce color channels
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Apply blurring to reduce noise on the difference mask
        curr_frame = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
        if prev_frame is None:
            prev_frame = curr_frame
            # Skip calculations for this particular starter frame
            continue

        # 2. Take the absoulute difference between frames (Difference == Motion)
        diff = cv2.absdiff(src1=curr_frame, src2=prev_frame)
        prev_frame = curr_frame

        # 3. Apply thresholding to the frame to create a color mask
        _, threshold = cv2.threshold(diff, thresh=30, maxval=255, type=cv2.THRESH_BINARY)

        # 4. Apply morphology
        ## Create kernel
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(7, 7))

        ## Apply opening to reduce noise
        opened = cv2.morphologyEx(threshold, op=cv2.MORPH_OPEN, kernel=kernel, iterations=1)

        ## Apply closing to fill out the holes
        closed = cv2.morphologyEx(opened, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=3)

        ## Apply dilation to unify the disconnected pieces
        dilated = cv2.dilate(closed, kernel=kernel, iterations=1)

        # 5. Find and detect contour(s) (Largest contour)
        cnt_frame = frame.copy()

        contours, hierarchy = cv2.findContours(dilated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # (Optional) Ignore really small contours to reduce noise
            MIN_AREA = 500
            contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]

            if contours:
                # Choose the largest contour
                largest_cnt = max(contours, key=cv2.contourArea)

                # Calculate the bounding box of largest contour
                x, y, w, h = cv2.boundingRect(largest_cnt)

                # Draw bbox
                cv2.rectangle(cnt_frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
        
        # 6. Visualization
        cv2.putText(cnt_frame, f'FPS: {fps}', org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1.2, color=(255, 0, 0), thickness=2)
        cv2.imshow('Frame', cnt_frame)

        ###################################################################################################

        # Add quitting option
        if cv2.waitKey(1) & 0XFF == ord('q'):
            print("Program stopped.")
            break
    
    # Release the capturer and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
