

'''
For problem 1, we perform HSV color space conversion on the picture, 
specify the color range of apples, and according to the mask of the color region obtained from the color range,
improve the mask through morphological operation, find the apple outline, and thus realize the detection of the number of apples.

'''

import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of apple colors
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])

    # Create masks based on color thresholds
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(mask1, mask2, mask3)

    # Morphological operations to improve the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask

def count_apples(image_path):
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image)

    # Find contours
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    apple_coordinates = []  # Store apple coordinates

    # Calculate the number and coordinates of apples
    for contour in contours:
        # Get the bounding box
        x, y, w, h = cv2.boundingRect(contour)
        # Calculate the bottom-left coordinate
        bottom_left = (x, image.shape[0] - y)
        apple_coordinates.append(bottom_left)

    return apple_coordinates

folder_path = r"picture"
apple_counts = []
image_numbers = []
apple_coordinates_all = []
image_names = []

for i, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        apple_count = count_apples(image_path)
        apple_counts.append(apple_count)
        image_numbers.append(i+1)

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        apple_coordinates = count_apples(image_path)
        apple_coordinates_all.extend(apple_coordinates)
        image_names.extend([filename] * len(apple_coordinates))

x_coords = [coord[0] for coord in apple_coordinates_all]
y_coords = [coord[1] for coord in apple_coordinates_all]
plt.hexbin(x_coords, y_coords, gridsize=10, cmap='Blues')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Apple Coordinates')
counts = plt.hexbin(x_coords, y_coords, gridsize=10, cmap='Blues').get_array()
bin_centers = plt.hexbin(x_coords, y_coords, gridsize=10, cmap='Blues').get_offsets()
for count, center in zip(counts, bin_centers):
    plt.text(center[0], center[1], int(count), ha='center', va='center')

plt.colorbar(label='Counts')
plt.show()

# Create a DataFrame and save it to an Excel file
data = {"Image Number": image_numbers, "Apple Count": apple_counts}
df = pd.DataFrame(data)
print(df)
df.to_excel("apple_counts.xlsx", index=False)
