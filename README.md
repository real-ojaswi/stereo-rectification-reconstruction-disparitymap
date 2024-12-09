# 3D Reconstruction, Rectification, and Disparity Map Calculation

## Overview:
This repository contains code for performing stereo vision tasks, including 3D reconstruction, rectification, and disparity map computation. The steps covered in the code include camera calibration, fundamental matrix estimation, disparity map calculation using a Census-like transform, rectification of stereo images, and the evaluation of the computed disparity map using ground truth data. All the tasks have been performed using the fundamental principles instead of using libraries like opencv upto the extent possible in order to demonstrate the understanding of underlying geometrical and engineering principles.

## Key Features:
1. **Stereo Image Rectification**: 
   - The images are rectified using homography matrices to align the stereo pair, making corresponding points in both images lie along the same horizontal scanline.
   
2. **Fundamental Matrix Calculation**: 
   - The 8-point algorithm is used to compute the fundamental matrix that relates corresponding points between the left and right stereo images.

3. **Extracting Additional Matches from Rectified Stereo Images**:
   - The code extracts additional corresponding points between the left and right rectified images using a Census-like transform and Sum of Squared Differences (SSD) matching. This step enhances the matching process by identifying more keypoints for triangulation and disparity estimation.
   
4. **Triangulation**: 
   - Using the calibrated stereo cameras, the code performs 3D triangulation to estimate world coordinates from corresponding points in the rectified stereo images.

5. **Disparity Map Calculation**: 
   - The code computes disparity maps using a Census-like transform to match key points between rectified images and generates a disparity map based on pixel differences.

6. **Evaluation of Disparity Map**:
   - The code provides functions to evaluate the accuracy of the estimated disparity map by comparing it to a ground truth disparity map using mean absolute error (MAE) and a binary error mask.

7. **Visualization**:
   - The code visualizes 3D camera poses, world points, and rectified images with corresponding points. It also includes a function to visualize disparity maps and the ground truth.

## Steps:
1. **Fundamental Matrix Estimation**: Compute the fundamental matrix from corresponding points in a stereo pair of images.
   
2. **Extract Additional Matches**:
   - Extract more correspondences between the rectified stereo image pair using SSD matching, which can be used for improved triangulation and disparity map accuracy.

3. **Camera Pose Estimation**: 
   - Extract camera poses (rotation and translation matrices) from the stereo projection matrices and plot the cameras in a 3D scene.
   
4. **Disparity Map Calculation**: 
   - Compute disparity maps using the Census-like transform and calculate their accuracy using ground truth.

5. **Stereo Image Rectification**:
   - Perform image rectification using the computed homography matrices, followed by mapping of 2D points from rectified images back to the original images using inverse homography.

6. **Evaluate Disparity Map**: 
   - Evaluate the accuracy of the disparity map by comparing it to ground truth and generating an error mask.


## Dependencies:
- OpenCV
- NumPy
- Matplotlib
