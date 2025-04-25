

'''
For the fourth question, the model trained by yolov5 was adopted to capture the apple picture, 
perform gray level conversion, binary conversion, contour search and other processing on the picture, find the maximum contour,
judge the area, and approximate replace the incomplete figure with a circle. Assuming that the apple is a standard positive sphere,
 the mass of the apple is obtained by formula transformation.
'''

import cv2
import numpy as np
import os
import pandas as pd

# Read the image path
def get_pic_path():
    img_path = r'D:\pycharm\apple'  # Image folder path
    img_list = []
    for na in os.listdir(img_path):
        if na.endswith('.jpg'):
            img_list.append(os.path.join(img_path, na))
    return img_list

# Process the images
def process_images(img_list):
    area_data = []  # List to save image number, box number, and apple area
    for img_path in img_list:
        img_name = os.path.basename(img_path)  # Get the image filename
        img_number, box_number = img_name.split('_')  # Extract the image number and box number from the image name
        img = cv2.imread(img_path)  # Read the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # Binarize the image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)  # Find the largest contour
            area = cv2.contourArea(max_contour)  # Calculate the contour area
            x, y, w, h = cv2.boundingRect(max_contour)  # Get the bounding rectangle
            if w < h:  # Check if the contour is circular enough
                radius = (w + h) // 4  # Calculate the radius of the circle
                cv2.circle(img, (x + w // 2, y + h // 2), radius, (0, 255, 0), 2)  # Draw the circle
                area = np.pi * radius ** 2  # Replace the original area with the area of the circle
            area_data.append([img_number, box_number, area])  # Add image number, box number, and apple area to the list
            # cv2.imshow('image', img)  # Display the image with contours
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    # Save data to Excel spreadsheet
    df = pd.DataFrame(area_data, columns=['Image Number', 'Box Number', 'Apple Area'])
    df.to_excel('apple_area.xlsx', index=False)
    df['weight'] = 4 / 3 * 0.8 * (df['area'] ** (2 / 3)) / math.sqrt(math.pi)
    df.to_excel('apple_area.xlsx', index=False)

# Main function
if __name__ == "__main__":
    img_list = get_pic_path()  # Get the image path list
    process_images(img_list)  # Process the images and save data to Excel spreadsheet