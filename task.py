#!/usr/bin/env python3

import cv2
import math
import pandas as pd
import csv

#------------------------------------------
#functions 
def calculate_distance(x1, y1, x2, y2):
    #Calculate Euclidean distance between two points.""" #not my work credit to "https://www.geeksforgeeks.org/euclidean-distance/"
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def write_to_csv(area,arm_reach_pixels):
    #creating a new dic for the data 
    dict = {'Area': area, 'arm reach':arm_reach_pixels }

    # create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(dict) 

    df.to_csv('data.csv') 

    return 


#------------------------------------------
# use the HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#object detect
# use the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

video_path = 'clip.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

arm_count =0
count = 0
array_of_faces = []
array_of_areas= []
array_of_armlength = []

while True:
    # Read each frame from the video
    ret, frame = cap.read()

    
    if not ret:
        print("error reading frame")
        break

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (640, 480))

    #for detecting people- created a copy of the the resized frame
    dect_frame = frame_resized

    fg_mask = fgbg.apply(dect_frame)
    contours, m = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect people in the frame using HOG and SVM
    boxes, weights = hog.detectMultiScale(frame_resized, winStride=(8, 8), padding=(8, 8), scale=1.05)

    for contour in contours:
        # Only consider large enough objects
        if cv2.contourArea(contour) > 20000:
            # Get bounding box for the object
            (x, y, w, h) = cv2.boundingRect(contour)

            # Calculate the area of the object
            area = cv2.contourArea(contour)

            if count  < 20:
                array_of_areas.append(area)
            
            count+=1

            print(count)
            
            # Draw a rectangle around the object
            cv2.rectangle(dect_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(dect_frame, f"Area: {area} px", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # got the font from an example 

    # Combine Human Detection with Object Detection
    # Create a copy of frame resized inorder to display the human detection 
    dect_human = frame_resized.copy()

    # Loop that loops through all the objects found 
    for (x, y, w, h) in boxes:
        cv2.rectangle(dect_human, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # we assume the shoulder is at the top of the created rec. while the wrist is below the rec
        shoulder_x, shoulder_y = x + w // 2, y + 20  
        wrist_x, wrist_y = x + w // 2, y + h - 20 

        # Draw the shoulder and wrist points
        cv2.circle(dect_human, (shoulder_x, shoulder_y), 5, (0, 0, 255), -1)
        cv2.circle(dect_human, (wrist_x, wrist_y), 5, (255, 0, 0), -1)

        # Calculate the arm reach 
        arm_reach_pixels = calculate_distance(shoulder_x, shoulder_y, wrist_x, wrist_y)
        if arm_count < 20:
            array_of_armlength.append(arm_reach_pixels)
        arm_count+=1

        cv2.putText(dect_human, f"Arm Reach: {arm_reach_pixels:.2f} px", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # for face detection using the Haar Cascade
    #credit to opencv documentation -https://www.opencvhelp.org/tutorials/image-analysis/object-detection/https://www.opencvhelp.org/tutorials/image-analysis/object-detection/
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over the faces detected and draw bounding boxes around them
    for (x, y, w, h) in faces:
        cv2.rectangle(dect_human, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        cv2.putText(dect_human, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    
    cv2.imshow('Video frame :)', dect_human)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#------------------------------------------
#writing to out output file
write_to_csv(array_of_areas,array_of_armlength)

#close everything 
cap.release()
cv2.destroyAllWindows()
