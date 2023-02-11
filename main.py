import numpy as np
import argparse, pytz, time, imutils
import cv2
import os
from math import pow, sqrt
from datetime import datetime
import pyttsx3
import speech_recognition as sr
import datetime
from tkinter import *

#---------------------- Splash Screen -----------------------------#
root     = Tk()
img_file = "sentinel.png"
image    = PhotoImage(file=img_file)
w,h      = image.width(), image.height()


screen_width  = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width  / 2) - (w / 2)
y = (screen_height / 2) - (h / 2)

root.overrideredirect(True)
root.geometry(f'{w}x{h}+{int(x)}+{int(y)}')

canvas = Canvas(root, highlightthickness=0)
canvas.create_image(0,0, image=image, anchor='nw')
canvas.pack(expand=1,fill='both')

root.after(5000, root.destroy)

root.mainloop()

#-----------------------------------------------------------------------------------------------#



IST = pytz.timezone('Asia/Manila')


device_name         = 'C001'
CONFIDENCE_CUTOFF   = args['confidence']
NMS_THRESHOLD       = args['threshold']


# load the COCO class labels our YOLO model was trained on
#LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
#weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
#configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
#net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Voice Command
print("")
listener = sr.Recognizer()
engine = pyttsx3.init()
engine. runAndWait()
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)
hour = int(datetime.datetime.now().hour)
if hour>= 0 and hour<12:
    engine.say("Good Morning, Welcome to Sentinel Monitoring Software")  
    print("[INFO] Good Morning, Welcome to Sentinel Monitoring Software")
elif hour>= 12 and hour<18:
    engine.say("Good Afternoon, Welcome to Sentinel Monitoring Software")  
    print("[INFO] Good Afternoon, Welcome to Sentinel Monitoring Software")
else:
    engine.say("Good Evening, Welcome to Sentinel Monitoring Software")  
    print("[INFO] Good Evening, Welcome to Sentinel Monitoring Software")

engine.say ('Sentinel Monitoring Software Starting')
print ("[INFO] Sentinel Monitoring Software Starting....")
engine. runAndWait()


# check if we are going to use GPU
#if config.USE_GPU:
	# set CUDA as the preferable backend and target
    
	#print("")
print("[INFO] Looking for GPU")
	#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
#ln = net.getLayerNames()
#ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


# Load model to detect person
weight_person       = './model/person_detect/yolov4.cfg'
model_person         = './model/person_detect/yolov4_best.weights'
net_person          = cv2.dnn.readNetFromDarknet(weight_person, model_peron)
net_person.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net_person.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load model to detect Mask/No mask
weight_face         = './model/mask_detect/yolov4-obj.cfg'
model_face          = './model/mask_detect/yolov4_face_mask.weights'
net_face            = cv2.dnn.readNetFromDarknet(weight_face, model_face)
net_face.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_face.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Labels (Mask/No mask)
classesFile = './model/mask_detect/object.names'
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Set color for Mask/No mask
colors = [(0,0,255), (0,255,0)]

# Image size
IMG_WIDTH, IMG_HEIGHT = 416, 416

# Focal length F = (P x D) / H (my Height (H) = 172, distance I stand between camera (D) = 360cm, Height of my Bounding Box (P) = 300 px) 
F = 625
count_frame = 0
cur = 0

# Capture video through device
print("[INFO] Starting the live stream....")
print("[INFO] Select Camera No.")
print("[INFO] [0] Primary Camera Device [1] OpenCV Camera [2] Wireless and USB Type Camera")
# 0 - Built in camera for laptop or desktop
# 2 - Wireless Camera and Web Camera

camera_no = 3

while camera_no < 0 or camera_no > 2:
    camera_no = int(input("Input Camera No.: "))

engine.say("Camera No.:   " + str(camera_no))

cap = cv2.VideoCapture(camera_no, cv2.CAP_DSHOW)

people_limit = 101

while people_limit < 1 or people_limit > 100:
    people_limit = int(input("Input People Limit: "))

engine.say("People Limit:   " + str(people_limit))

engine.say("Sentinel Monitoring Software Starts to Detect Facemask, People Density and Physical Distancing")
print("[INFO] Sentinel Monitoring Software Starts to Detect Facemask, People Density and Physical Distancing")
print("[INFO] Press Esc key to exit Sentinel Monitoring Software")
engine. runAndWait()

#writer = None
# start the FPS counter
#fps = FPS().start()

# Input model and frame to get boxes, class & confidence of objects
def predict_box(net, frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT), [0,0,0], 1, crop=False)
    net.setInput(blob)
    output = net.getUnconnectedOutLayersNames()
    outs = net.forward(output)

    confidences = []
    boxes = []
    classIDs = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > CONFIDENCE_CUTOFF:
                x_mid, y_mid, w, h = detection[:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                x, y = int(x_mid - w//2), int(y_mid - h//2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                	

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_CUTOFF, NMS_THRESHOLD)

    final_box = []
    final_classIDs = []
    final_confidences = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_box.append(boxes[i])
            final_classIDs.append(classIDs[i])
            final_confidences.append(confidences[i])
        return final_box, final_classIDs, final_confidences

# Draw bounding box and text
def draw_box(frame, boxes, classIDs, confidences, class_list, color_list):
    for i in range(len(boxes)):
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in color_list[classIDs[i]]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = '{}: {:.4f}'.format(class_list[classIDs[i]], confidences[i])
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Calculate distance between each person
def calculate_distance(frame, boxes):
    position = {}
    close_objects = set()
    for i in range(len(boxes)):
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        x_mid = round(x + w/2, 4)
        y_mid = round(y + h/2, 4)

        distance_to_camera = (165 * F)/h

        x_mid_cm = (x_mid * distance_to_camera) / F
        y_mid_cm = (y_mid * distance_to_camera) / F

        position[i] = (x_mid_cm, y_mid_cm, distance_to_camera, x_mid, y_mid)

    for i in position.keys():
            for j in position.keys():
                if i < j:
                    distance = sqrt(pow(position[i][0]-position[j][0],2) + pow(position[i][1]-position[j][1],2) + pow(position[i][2]-position[j][2],2))
                # Check if distance less than 2 meters or 200 centimeters
                    if distance < 200:
                        close_objects.add(i)
                        close_objects.add(j)
                        # Draw line between middle point of boxes if < 200cm
                        cv2.line(frame, (int(position[i][3]), int(position[i][4])), (int(position[j][3]), int(position[j][4])), (0,0,255), 2)
                        # Put text to display distance between boxes if < 200cm
                        if position[i][3] <= position[j][3]:
                            x_center_line = int(position[i][3] + (position[j][3] - position[i][3])/2)
                        else:
                            x_center_line = int(position[j][3] + (position[i][3] - position[j][3])/2)
                        if position[i][4] <= position[j][4]:
                            y_center_line = int(position[i][4] + (position[j][4] - position[i][4])/2)
                        else:
                            y_center_line = int(position[j][4] + (position[i][4] - position[j][4])/2)
                        cv2.putText(frame, f'{int(distance)} cm', (x_center_line - 35, y_center_line - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
    boxes_person_normal = []
    boxes_close = []

    for i in range(len(boxes)):
        if i in close_objects:
            boxes_close.append(boxes[i])
        else:
            boxes_person_normal.append(boxes[i])
    return boxes_close, boxes_person_normal


while cap.isOpened():
    ret, frame      = cap.read()
    frame_height    = frame.shape[0]
    frame_width     = frame.shape[1]
    result = frame.copy()

    boxes_face          = []
    boxes_person        = []
    boxes_close         = []
    boxes_person_normal = []
    mask_count          = []
    nomask_count        = []
    classIDs_face       = []

    try:
        # get and draw bounding box of facemask/no-facemask
        boxes_face, classIDs_face, confidences_face = predict_box(net_face, result)
        draw_box(result, boxes_face, classIDs_face, confidences_face, classes, colors)

        # count number of mask & no mask
        mask_count      = sum(classIDs_face)
        nomask_count    = len(classIDs_face) - mask_count

        # get and draw bounding box of close people (red) & distanced people (green)
        boxes_person, classIDs_person, confidences_person = predict_box(net_person, result)
        boxes_close, boxes_person_normal = calculate_distance(result, boxes_person)

        draw_box(result, boxes_person_normal, classIDs_person, confidences_person, ['Person'], [(0,255,0)])
        draw_box(result, boxes_close, classIDs_person, confidences_person, ['Person'], [(0,0,255)])

    except:
        pass
    

   
    # count number of mask & no mask
    
    mask_count      = sum(classIDs_face)
    nomask_count    = len(classIDs_face) - mask_count

    total_mask_count = nomask_count + mask_count
    total_person_count = len(boxes_person_normal)

    display_count = 0
    
    if (total_mask_count > total_person_count):
        display_count = total_mask_count
    elif (total_person_count > total_mask_count):
        display_count = total_person_count
    elif (total_person_count == total_mask_count):
        display_count = total_person_count

    # Create a canvas on top to display information
    border_size = 100
    border_text_color=(255,255,255)
    style = cv2.FONT_HERSHEY_SIMPLEX
    result = cv2.copyMakeBorder(result, border_size, 0,0,0, cv2.BORDER_CONSTANT)

    text = 'No Mask Count: {}  Mask Count : {}'.format(nomask_count, mask_count)
    cv2.putText(result,text, (5, int(border_size-70)), style, 0.65, border_text_color, 2)

    text = f'People Count  : {display_count}  People Limit: {people_limit}'
    cv2.putText(result,text, (5, int(border_size-40)), style, 0.65, border_text_color, 2)

    text = f'Physical Distancing Violations   : {len(boxes_close)}'
    cv2.putText(result, text, (5, int(border_size-10)), style, 0.65, border_text_color, 2)

    text = f'   Status:'
    cv2.putText(result, text, (frame_width - 250, int(border_size-70)), style, 0.65, border_text_color, 2)

    text = f'   Status:'
    cv2.putText(result, text, (frame_width - 250, int(border_size-40)), style, 0.65, border_text_color, 2)

    text = f'   Status:'
    cv2.putText(result, text, (frame_width - 250, int(border_size-10)), style, 0.65, border_text_color, 2)

   
       
    # sends warning if the mask is not wear properly
    if (nomask_count == 0):
        text = '   Safe'
        cv2.putText(result, text, (frame_width - 170, int(border_size-70)), style, 0.65, (0,255,0), 2)
        count_frame = 0
    elif (nomask_count > [1]):
        text = '   Danger !!!'
        cv2.putText(result, text, (frame_width - 170, int(border_size-70)), style, 0.65, (0, 0, 255), 2)
    else:
        text = '   Warning !'
        cv2.putText(result, text, (frame_width - 170, int(border_size-70)), style, 0.65, (0, 255, 255), 2)
        count_frame = 0

    # sends warning if people density reaches or surpasses the set limit
    if (display_count) > people_limit:
        text = '   Danger !!!'
        cv2.putText(result, text, (frame_width - 170, int(border_size-40)), style, 0.65, (0, 0, 255), 2)
        count_frame = 0
    elif (display_count) == people_limit:
        text = '   Warning !'
        cv2.putText(result, text, (frame_width - 170, int(border_size-40)), style, 0.65, (0, 255, 255), 2)
    else:
        text = '   Safe'
        cv2.putText(result, text, (frame_width - 170, int(border_size-40)), style, 0.65, (0,255,0), 2)
        count_frame = 0

    # sends warning if physical distancing violations reaches or surpasses the set limit
    if (len(boxes_close)) > 2:
        text = '   Danger !!!'
        cv2.putText(result, text, (frame_width - 170, int(border_size-10)), style, 0.65, (0, 0, 255), 2)
        count_frame = 0
    elif (len(boxes_close)) == 2:
        text = '   Warning !'
        cv2.putText(result, text, (frame_width - 170, int(border_size-10)), style, 0.65, (0, 255, 255), 2) 
    else:
        text = '   Safe'
        cv2.putText(result, text, (frame_width - 170, int(border_size-10)), style, 0.65, (0,255,0), 2)
        count_frame = 0

   # Audio Cues for Violoation Limit    
   # if (len(boxes_close)) > 2:
        #engine.say("Physical Distancing Surpassses The Set Limit")   
    #elif (display_count) > people_limit:
      # engine.say("People density reaches or surpasses the set limit")
    
    cv2.namedWindow('Sentinel Monitoring Software',cv2.WINDOW_NORMAL)
    # Show Sentinnel Monitoring Software Frame
    cv2.imshow('Sentinel Monitoring Software', result)
    cv2.resizeWindow('Sentinel Monitoring Software',900, 600)
    key = cv2.waitKey(1) & 0xFF

    # Press `esc` to exit
    if key == 27:
        print('[INFO] Sentinel Monitoring Software Closing...')
        print('[INFO] Thank You For Using Sentinel Monitoring Software...')
        break
# Clean
cap.release()
cv2.destroyAllWindows()
engine.say('Sentinel Monitoring Software Closing...')
engine.say('Thank You For Using Sentinel Monitoring Software...')  
engine. runAndWait()
cap.release()
cv2.destroyAllWindows()