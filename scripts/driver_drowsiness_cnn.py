#driver_drowsiness_cnn:
#cv2: camera access and image processing
#mediapipe: extracts face landmarks
#torch: loads and runs the CNN models
#pygame: plays alarm audio
#time: used for measuring eye closure duration
import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
import pygame
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
#Imports your two models and preprocessing function
from eye_cnn import EyeCNN
from mouth_cnn import MouthCNN
from utils_preprocess import preprocess_roi
#Runs on GPU if available; otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Loads trained model for eye classification (Open vs Closed) asnd saves it as eye_cnn.pth file
eye_model = EyeCNN().to(device)
eye_model.load_state_dict(torch.load("models/eye_cnn.pth", map_location=device))
eye_model.eval()
#Loads trained model for mouth classification (Yawning vs Normal) and svaes it as mouth_cnn.pth file 
mouth_model = MouthCNN().to(device)
mouth_model.load_state_dict(torch.load("models/mouth_cnn.pth", map_location=device))
mouth_model.eval()
#Loads alarm sound file
pygame.mixer.init()
pygame.mixer.music.load("audio/alarm.wav")
#FaceMesh returns up to 468 3D-ish landmarks per detected face
#static_image_mode=False: optimized for video (reuses tracking between frames)
#max_num_faces=1: only one face (the driver) is expected
#refine_landmarks=True: provides more landmarks around eyes and lips
#min_detection_confidence & min_tracking_confidence: thresholds for detection and tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
#Starts video capture from default camera
cap = cv2.VideoCapture(0)
#If eyes remain closed for 2 seconds → drowsiness detected
eyes_closed_start_time = None
EYE_CLOSE_THRESHOLD = 2.0
#These numbers are MediaPipe normalized landmark indices that consistently correspond to key points (eye corners, mouth corners/center)
#normalized meanining if the face lanndmark is at pixel pos (300,200) in a 600x400 image so media pipe returns (300/600,200/400)=(0.5,0.5)
#normalized data is always between 0 and 1
LEFT_EYE_LANDMARKS = [33, 133]
RIGHT_EYE_LANDMARKS = [362, 263]
MOUTH_LANDMARKS = [13, 14, 78, 308]
#Every iteration → one frame is processed
#Each loop reads one frame, converts BGR → RGB because MediaPipe expects RGB
while True:
    #read frame from camera
    ret, frame = cap.read()
    if not ret:
        break
    #convert BGR to RGB for MediaPipe processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    #These flags become true after CNN predictions
    eyes_closed = False
    yawning = False
    #if a face is detected
    if result.multi_face_landmarks:
        #define a function to extract ROIs based on landmark indices
        face_landmarks = result.multi_face_landmarks[0]
        h, w, i = frame.shape
        #this functon extracts and returns the ROI and its bounding box expands it to capture full eye and mouth region 
        def get_roi(indices, scale=1.8):
            xs = [face_landmarks.landmark[i].x * w for i in indices]
            ys = [face_landmarks.landmark[i].y * h for i in indices]
            min_x, max_x = int(min(xs)), int(max(xs))
            min_y, max_y = int(min(ys)), int(max(ys))
            cx = (min_x + max_x)//2
            cy = (min_y + max_y) // 2
            box_size = int(max(max_x - min_x, max_y - min_y) * scale)
            x1 = max(cx - box_size // 2, 0)
            y1 = max(cy - box_size // 2, 0)
            x2 = min(cx + box_size // 2, w)
            y2 = min(cy + box_size // 2, h)
            return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
        #for eyes
        #crop left and right eye ROIs
        left_eye_roi, left_box = get_roi(LEFT_EYE_LANDMARKS)
        right_eye_roi, right_box = get_roi(RIGHT_EYE_LANDMARKS)
        #preprocess eye ROIs into tensors this converts image grayscale, resizes to 24x24, normalizes, and converts to tensor
        left_tensor = preprocess_roi(left_eye_roi, device)
        right_tensor = preprocess_roi(right_eye_roi, device)
        #if the roi are too small or not detected, skip this frame
        if left_tensor is None or right_tensor is None:
            cv2.putText(frame, "Face Not Clear", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Driver Drowsiness Detection", frame)
            continue
        #we calcualte softmax probabilities and get the class with highest probability 1=open 0=closed
        left_pred = torch.argmax(F.softmax(eye_model(left_tensor), dim=1), dim=1).item()
        right_pred = torch.argmax(F.softmax(eye_model(right_tensor), dim=1), dim=1).item()
        #draw eye boxes and labels
        cv2.rectangle(frame, left_box[:2], left_box[2:], (0,255,0), 2)
        cv2.putText(frame, "Open" if left_pred==1 else "Closed",
                    (left_box[0], left_box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.rectangle(frame, right_box[:2], right_box[2:], (0,255,0), 2)
        cv2.putText(frame, "Open" if right_pred==1 else "Closed",
                    (right_box[0], right_box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        #If either eye is classified as closed (0) the script treats eyes as closed for that frame
        if left_pred == 0 or right_pred == 0:
            eyes_closed = True
        #for mouth 
        #mouth ROI extraction and preprocessing 1=yawing 0=normal
        mouth_roi, mouth_box = get_roi(MOUTH_LANDMARKS)
        mouth_tensor = preprocess_roi(mouth_roi, device)
        if mouth_tensor is None:
            continue
        mouth_pred = torch.argmax(F.softmax(mouth_model(mouth_tensor), dim=1), dim=1).item()
        cv2.rectangle(frame, mouth_box[:2], mouth_box[2:], (0,0,255), 2)
        cv2.putText(frame, "Yawning" if mouth_pred==1 else "Normal",
                    (mouth_box[0], mouth_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        if mouth_pred == 1:
            yawning = True
    #Eye closure duration tracking
    #if eyes are closed continously elapsed increases
    #if open → reset
    if eyes_closed:
        if eyes_closed_start_time is None:
            eyes_closed_start_time = time.time()
        elapsed = time.time() - eyes_closed_start_time
    else:
        eyes_closed_start_time = None
        elapsed = 0
    #if eyes are closed for 2 second or yawning trigger alarm  
    if (eyes_closed and elapsed >= EYE_CLOSE_THRESHOLD) or yawning:
        cv2.putText(frame, "DROWSINESS ALERT!", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()
    else:
        pygame.mixer.music.stop()
    #if user presses 'q', exit loop exit to windows 
    cv2.imshow("Driver Drowsiness Detection (CNN + MediaPipe)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()