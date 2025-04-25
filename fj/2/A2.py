

'''
For problem 2, in view of the apple position, the Mask RCNN model is used to set two types of apple and background for recognition according to the color mask, 
traverse all photos, segment the instance and save the results in the table.
'''
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of apple colors
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])

    # Apply color thresholding using inRange function
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Merge the masks for red regions
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Merge the masks for both red and yellow regions
    mask = cv2.bitwise_or(mask_red, mask_yellow)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    return result, mask


# Load the Mask R-CNN model
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # Only two classes: apple and background

# Get the input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained classifier head
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Get the input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256

# Replace the pre-trained mask classifier head
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                   hidden_layer,
                                                   num_classes)

# Load the model to CUDA if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Set the model to evaluation mode
model.eval()


# Define a function to get predictions for a single instance segmentation
def get_prediction(image, threshold):
    # Convert the image to a tensor and move it to CUDA device (if available)
    image = torch.from_numpy(image).float().to(device)
    image = image.permute(2, 0, 1)

    # Add a dimension to make the image a batch size of 1 tensor
    image = image.unsqueeze(0)

    # Perform inference using the model
    with torch.no_grad():
        prediction = model(image)

    # Decode the output into masks and bounding boxes
    masks = prediction[0]['masks'].cpu().numpy()
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # Select only the bounding boxes and masks with scores above the threshold
    selected_masks = []
    selected_boxes = []
    for i in range(len(scores)):
        if scores[i] > threshold:
            selected_masks.append(masks[i, 0])
            selected_boxes.append(boxes[i])

    return selected_masks, selected_boxes


# Define a function to convert mask to bounding box
def mask_to_bbox(mask):
    # Find the bounding rectangle of the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Return the coordinates of the bounding box
    return [xmin, ymin, xmax, ymax]


# Define a function to get predictions for all instance segmentations
def get_all_predictions(image, threshold=0.5):
    # Get the predictions
    masks, boxes = get_prediction(image, threshold)

    # Get the bounding box coordinates for each mask
    bboxes = []
    for mask in masks:
        bbox = mask_to_bbox(mask)
        bboxes.append(bbox)

    return masks, bboxes


import os

# Iterate over the photos in the folder
folder_path = 'picture'
image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg') or file.endswith('.png')]
data = pd.DataFrame(columns=['Image Number', 'Apple Count', 'Apple Centers'])

for image_file in image_files:
    # Concatenate the image path
    image_path = os.path.join(folder_path, image_file)

    # Preprocess the image
    image, mask = preprocess_image(image_path)
    masks, bboxes = get_all_predictions(image)

    # Get all instance segmentation results
    row = {'Image Number': image_file, 'Apple Count': len(bboxes), 'Apple Centers': str(bboxes)}
    data = pd.concat([data, pd.DataFrame(row, index=[0])], ignore_index=True)

excel_file = 'apple_coodinates.xlsx'
data.to_excel(excel_file, index=False)