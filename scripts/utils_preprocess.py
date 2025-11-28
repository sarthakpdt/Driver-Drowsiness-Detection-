#utlis_preprocess.py: this file preprocesses the ROI into Pytorch tensor 
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
#convert the cropped eye/mouth ROI into grayscale, resize to 24x24, normalize, and convert to tensor
#Converts the image (NumPy array) to a PIL Image in grayscale (L mode stands for 1-channel grayscale)
#Resizes the image to 24x24 pixels, so the network input is consistent
#Converts the image into a PyTorch tensor and scales pixel values from 0–255 to 0–1
transform = transforms.Compose([
    transforms.ToPILImage(mode='L'),
    transforms.Resize((24, 24)),
    transforms.ToTensor()
])
def preprocess_roi(img, device):
    #avoids crashing if the ROI is not detected properly
    if img is None or img.size == 0:
        return None  
    #converts to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray
    #return final tensor ready for model 
    tensor = transform(gray).unsqueeze(0).float().to(device)
    return tensor