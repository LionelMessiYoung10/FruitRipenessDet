'''
For problem 3, this paper uses yolov5 to detect apples, label apple data set, pre-train model, obtain apple position information,
 traverse all pictures in turn, convert images into HSV color space according to apple position detected in images, and calculate average hue. 
 In addition, MinMaxScaler is used to normalize the average tone, and the image number, frame number and average tone are saved in the table. 
 Extract maturity data and draw histogram. In order to better adapt to actual production, this paper classifies apple maturity, 
 uses K-means clustering and hierarchical clustering methods to classify apple maturity, and outputs the corresponding number of apples.
The K-means method is divided into three categories: High_Ripeness, Medium_Ripeness and Low_Ripeness. The hierarchical clustering is divided into six categories according to the priority. 
In practical application,the more cost-effective method is selected according to the situation to reasonably plan the picking time.
'''
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cpu')
print('device:', device)
model = YOLO('yolov5n.pt')
# Switch the computing device
model.to(device)
model.cpu()
img_path = 'picture'

def detect_color(num, img_num, box_num, num_bbox_count, h, img_path, x1, y1, x2, y2):
    image = cv2.imread(img_path)
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    roi = hsv_image[y1:y2, x1:x2]
    # Calculate color histogram
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    # Calculate average hue by getting the Hue channel and calculating its mean
    h_channel = roi[:, :, 0]
    average_h_channel = np.mean(h_channel)
    img_num.append(num)
    box_num.append(num_bbox_count)
    h.append(average_h_channel)
    average_color = np.mean(roi, axis=(0, 1))
    plt.plot(hist)
    plt.show()

def process_img(i, img_num, box_num, h, img_path, na):
    num = i
    num = num + 1
    results = model(img_path)
    len(results)
    results[0]
    results[0].names
    results[0].boxes.cls
    num_bbox = 0
    num_bbox = len(results[0].boxes.cls)
    results[0].boxes.conf
    results[0].boxes.xyxy
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
    bboxes_xywh = results[0].boxes.xywh.cpu().numpy().astype('uint32')

    img_bgr = cv2.imread(img_path)
    plt.imshow(img_bgr[:, :, ::-1])
    plt.show()
    num_bbox_count = 0
    for idx in range(num_bbox): 
        # Get the coordinates
        bbox_xyxy = bboxes_xyxy[idx]
        # Detect color using the defined function
        bbox_label = results[0].names[results[0].boxes.cls[idx].item()]
        # If the label is 'apple', draw a bounding box
        if bbox_label == 'apple':
            # Count apple
            num_bbox_count = num_bbox_count + 1
            detect_color(num, img_num, box_num, num_bbox_count, h, img_path, bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3])
            img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color, bbox_thickness)
            img_bgr = cv2.putText(img_bgr, bbox_label, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])

    plt.imshow(img_bgr[:, :, ::-1])
    plt.show()
    out_path = 'save_data\\'+'output'+'_'+na
    cv2.imwrite(out_path, img_bgr)

h = []
img_num = []
box_num = []
for i in range(200):
    num = i + 1
    num = str(num)
    a = '\''
    f = 'picture\\'
    d = '.jpg'
    name = f + num + d
    na = num + d
    process_img(i, img_num, box_num, h, name, na)

data = {
    'img_num': img_num,
    'box_num': box_num,
    'hue': h
}
df = pd.DataFrame(data)
# Normalize using MinMaxScaler
scaler = MinMaxScaler()
df['h_normalized'] = scaler.fit_transform(df[['hue']])
# Save the DataFrame to an Excel file
df.to_excel('mean_hue_of_all_apples.xlsx', index=False)

'''
Clustering of Ripeness
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
df = pd.read_excel('mean_hue_of_all_apples.xlsx')
scaler = MinMaxScaler()
df['h_normalized'] = scaler.fit_transform(df[['hue']])
df.to_excel('mean_hue_of_all_apples.xlsx', index=False)

# Extract ripeness data
ripeness_data = df['hue']

# Draw a histogram
plt.hist(ripeness_data, bins=10, color='blue', alpha=0.7)

# Set the title and labels
plt.title('Ripeness Distribution of Apples')
plt.xlabel('Ripeness')
plt.ylabel('Count')

# Show the plot
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['img_num'], df['box_num'], df['hue'], c='r', marker='o')
ax.set_title('Scatter plot of apple ripeness')
ax.set_xlabel('Image Number')
ax.set_ylabel('Box Number')
ax.set_zlabel('Ripeness')

plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.cluster import KMeans

# Read the data
df = pd.read_excel('mean_hue_of_all_apples.xlsx')

# Extract ripeness data
ripeness_data = df['hue']

# Number of clusters
n_clusters = 3

# Create KMeans object
kmeans = KMeans(n_clusters=n_clusters)

# Extract the features for clustering
features = df[['hue']]

# Perform clustering
kmeans.fit(features)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Define class names
class_names = ['High_Ripeness', 'Medium_Ripeness', 'Low_Ripeness']

# Plot scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['img_num'], df['box_num'], df['hue'], c=labels, cmap='viridis', marker='o')
ax.set_title('Scatter plot of apple ripeness classification')
ax.set_xlabel('Image Number')
ax.set_ylabel('Box Number')
ax.set_zlabel('Ripeness')
# Add legend
legend1 = ax.legend(*scatter.legend_elements())
ax.add_artist(legend1)

# Set legend labels
for i, class_name in enumerate(class_names):
    legend1.get_texts()[i].set_text(class_name)

plt.show()

# Output the count for each class
for i, class_name in enumerate(class_names):
    count = sum(labels == i)
    print(f"{class_name}: {count}")

# Scatter plot
plt.scatter(df['img_num'], df['box_num'], c=labels, cmap='viridis')
plt.title('K-means Clustering of Apple Data')
plt.xlabel('Image Number')
plt.ylabel('Box Number')

plt.show()

from scipy.cluster import hierarchy

features = df[['hue']]

# Calculate distance matrix
dist_matrix = hierarchy.distance.pdist(features)

# Perform hierarchical clustering
linkage_matrix = hierarchy.linkage(dist_matrix, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 5))
dn = hierarchy.dendrogram(linkage_matrix)
plt.title('Dendrogram of Apples')
plt.ylabel('Distance')
plt.xticks([])
plt.show()

# Get the clustering results
n_clusters = 6
labels = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')

# Output the count for each cluster
for i in range(n_clusters):
    count = sum(labels == i+1)
    print(f"Cluster {i+1}: {count}")
