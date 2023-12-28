import numpy as np
import cv2

annotation_path = "/Users/neo/Documents/Research/CP/pannuke/annotations/masks/prediction/annotations/annotation_2136.npy"
image_path = (
    "/Users/neo/Documents/Research/CP/pannuke/images/masks/prediction/image_0887.png"
)
centroid_path = "/Users/neo/Documents/Research/CP/pannuke/annotations/masks/prediction/centroids/centroids_2136.npy"
label_path = (
    "/Users/neo/Documents/Research/CP/pannuke/labels/masks/prediction/label_0887.npy"
)

# open and display the image in image_path
run = False
if run:
    image = cv2.imread(image_path)
    cv2.imshow("image", image)
    cv2.waitKey(0)

# open the annoation file which is .npy file, open as numpy array
annotation = np.load(annotation_path)

# print the max and min value of the annotation
print("max value of annotation: ", np.max(annotation))
print("min value of annotation: ", np.min(annotation))

# display the annotation as a black and white image, first convert to uint8 ranging from 0 to 255 (currently it is a 0 1 binary mask)
run = True
if run:
    annotation = annotation.astype(np.uint8)
    # change all 1 to 255
    annotation[annotation == 1] = 255
    cv2.imshow("annotation", annotation)
    cv2.waitKey(0)

# open the centroid file which is .npy file, open as numpy array
centroid = np.load(centroid_path)

# print the dimension of the centroid file
print("dimension of centroid: ", centroid.shape)

# print the max and min value of the centroid
print("max value of centroid: ", np.max(centroid))
print("min value of centroid: ", np.min(centroid))

# add the centroid to the annotation as thick green dots
run = True
if run:
    # convert the annotation to 3 channel image
    annotation = cv2.cvtColor(annotation, cv2.COLOR_GRAY2BGR)
    # add the centroid to the annotation
    for i in range(centroid.shape[0]):
        # get the x and y coordinate of the centroid
        x = int(centroid[i, 0])
        y = int(centroid[i, 1])
        # draw a circle at the centroid
        cv2.circle(annotation, (x, y), 2, (0, 255, 0), -1)
    # display the annotation
    cv2.imshow("annotation", annotation)
    cv2.waitKey(0)

# open the label file which is .npy file, open as numpy array
label = np.load(label_path)

# print the dimension of the label file
print("dimension of label: ", label.shape)

# print the max and min value of the label
print("max value of label: ", np.max(label))
print("min value of label: ", np.min(label))

# display the label as a black and white image, first convert to uint8 ranging from 0 to 255 (currently it is a 0 1 binary mask)
run = True
if run:
    label = label.astype(np.uint8)
    # change all 1 to 255
    label[label == 1] = 255
    cv2.imshow("label", label)
    cv2.waitKey(0)
    
### It seems like the label file is the same as the annotation file but numbers displaying the separate nuclei <<< we can use grid search to find the bounding box