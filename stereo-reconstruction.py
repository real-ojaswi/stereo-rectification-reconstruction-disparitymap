#!/usr/bin/env python
# coding: utf-8

# ### Disparity Map Calculation

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import json
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os


# In[ ]:


left_points = [[176,807],
               [1193,695],
               [1400,1418],
               [168,1590],
               [1246,1258],
               [983,1290],
               [270,1208],
               [1123,1176],
               [1050,1181],
               [1073,937],
               [743,892],
               [688,1187]]
right_points= [[297,786],
               [1240,886],
               [1231,1619],
               [89,1453],
               [1123,1424],
               [857,1387],
               [259,1150],
               [1012,1317],
               [942,1304],
               [1034,1082],
               [742,970],
               [614,1224]
               ]

left_points_array= np.array(left_points)
right_points_array= np.array(right_points)


# In[ ]:


left_image= cv2.imread('Pic1.jpeg')
right_image= cv2.imread('Pic2.jpeg')


# In[ ]:


height, width= left_image.shape[0], left_image.shape[1]


# In[ ]:


def normalize_points(points):
    """
    Normalize points so that they are centered at the origin and have a mean distance of sqrt(2).
    """
    mean = np.mean(points, axis=0)
    dist = np.mean(np.linalg.norm(points - mean, axis=1))
    scale = np.sqrt(2) / dist
    
    T = np.array([[scale, 0, -scale * mean[0]], 
                  [0, scale, -scale * mean[1]], 
                  [0, 0, 1]])
    
    normalized_points = np.dot(T, np.column_stack((points, np.ones(len(points)))).T).T
    return normalized_points[:, :2], T


# In[ ]:


def compute_fundamental_matrix(left_points, right_points):
    """
    Compute the fundamental matrix using the 8-point algorithm.
    
    Parameters:
        left_points (np.ndarray): Corresponding points in the left image (shape: (N, 2)).
        right_points (np.ndarray): Corresponding points in the right image (shape: (N, 2)).
        
    Returns:
        F (np.ndarray): The 3x3 fundamental matrix.
    """
    # # Normalize points
    left_points_normalized, T1 = normalize_points(left_points)
    right_points_normalized, T2 = normalize_points(right_points)

    # left_points_normalized= left_points
    # right_points_normalized= right_points

    # Set up the design matrix A
    N = len(left_points)
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = left_points_normalized[i]
        x2, y2 = right_points_normalized[i]
        A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]
    
    # Solve the system using SVD
    _, _, VT = np.linalg.svd(A)
    F_normalized = VT[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint (set the smallest singular value to 0)
    U, S, Vt = np.linalg.svd(F_normalized)
    S[-1] = 0  # Zero the smallest singular value
    F_rank2 = np.dot(U, np.dot(np.diag(S), Vt))
    
    # Denormalize the fundamental matrix
    F = np.dot(T2.T, np.dot(F_rank2, T1))
    # F = F_rank2
    return F / F[-1, -1]


# In[ ]:


F= compute_fundamental_matrix(left_points_array, right_points_array)


# In[ ]:


# Calculate the epipoles (null space of F and F^T)
U, S, Vt = np.linalg.svd(F)
epipole_right = Vt[-1] / Vt[-1][-1]  # Right epipole (e')
epipole_left = U[:, -1] / U[:, -1][-1]  # Left epipole (e)


# In[ ]:


P = np.hstack((np.eye(3), np.zeros((3, 1))))
e_prime = epipole_right  # Make sure it's a column vector
S = np.array([[0, -e_prime[2], e_prime[1]],
              [e_prime[2], 0, -e_prime[0]],
              [-e_prime[1], e_prime[0], 0]])
e_prime = epipole_right.reshape(3, 1) 
# Compute the initial projection matrix P' for the right camera
P_prime_initial = np.hstack((S @ F, e_prime))


# In[ ]:


def triangulate(P, P_prime, left_points, right_points):
    """
    Triangulate 3D world points from 2D corresponding points in two images using the DLT algorithm.
    
    Parameters:
        P (np.ndarray): Camera projection matrix for the left camera (3x4).
        P_prime (np.ndarray): Camera projection matrix for the right camera (3x4).
        left_points (np.ndarray): Points in the left image (Nx2).
        right_points (np.ndarray): Points in the right image (Nx2).
        
    Returns:
        world_coordinates (np.ndarray): Triangulated 3D world points (Nx3).
    """
    num_points = len(left_points)
    world_coordinates = []

    for i in range(num_points):
        # Get the normalized 2D points (homogeneous coordinates)
        x1, y1 = left_points[i][0], left_points[i][1]
        x2, y2 = right_points[i][0], right_points[i][1]
        
        # Set up the system of equations
        A = np.array([
            [x1 * P[2, :] - P[0, :]],
            [y1 * P[2, :] - P[1, :]],
            [x2 * P_prime[2, :] - P_prime[0, :]],
            [y2 * P_prime[2, :] - P_prime[1, :]]
        ])

        # Stack the rows of A to form a matrix
        A = np.vstack(A)

        # Solve for the 3D point using SVD
        _, _, VT = np.linalg.svd(A)
        point_3d = VT[-1]
        point_3d = point_3d / point_3d[3]  # Convert back from homogeneous coordinates
        
        world_coordinates.append(point_3d[:3])

    return np.array(world_coordinates)


# In[ ]:


# Refine the right projection matrix using nonlinear optimization
# Define reprojection error function for optimization
def reprojection_error(params, left_points, right_points):
    # Reshape P_prime to (3, 4) matrix
    P = np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera matrix for the left camera
    P_prime = params[0:12].reshape((3, 4))  # Right camera projection matrix (3x4)
    
    # Extract the 3D points from the parameters
    points_3d = []
    for i in range(len(left_points)):
        points_3d.append(params[12 + 3*i : 12 + 3*(i+1)])

    residuals = []  # List to accumulate the errors for LM optimization

    # Loop over the points
    for p1, p2, point_3d in zip(left_points, right_points, points_3d):
        # Project back to 2D on both images
        p1_proj = P @ np.append(point_3d, 1)  # Project the 3D point to the left image
        p2_proj = P_prime @ np.append(point_3d, 1)  # Project the 3D point to the right image
        
        # Normalize by the homogeneous coordinate (3rd element)
        p1_proj /= p1_proj[2]
        p2_proj /= p2_proj[2]
        
        # Compute the errors between the projected points and the observed points
        error_left = p1_proj[:2] - p1  # Error for the left image
        error_right = p2_proj[:2] - p2  # Error for the right image
        
        # Stack the errors for LM optimization (flattened)
        residuals.extend(error_left)  # Append both x and y errors for the left image
        residuals.extend(error_right)  # Append both x and y errors for the right image

    # print(len(residuals))
    return np.array(residuals) 


# In[ ]:


points_3d_initial = triangulate(P, P_prime_initial, left_points, right_points)

# Initialize optimization parameters
initial_params = np.zeros(12 + 3 * len(left_points))
initial_params[:12] = P_prime_initial.flatten()
initial_params[12:] = points_3d_initial.flatten()
# print(len(initial_params))

# Perform optimization with Levenberg-Marquardt algorithm
optimized_result = least_squares(
    reprojection_error, 
    initial_params, 
    args=(left_points_array, right_points_array), 
    method='lm'
)

# Extract optimized projection matrix and 3D points
optimized_params = optimized_result.x
P_prime_refined = optimized_params[:12].reshape((3, 4))

# Compute translation vector and fundamental matrix from optimized projection matrix
translation_vector = P_prime_refined[:, -1]
translation_skew_matrix = np.array([
    [0, -translation_vector[2], translation_vector[1]],
    [translation_vector[2], 0, -translation_vector[0]],
    [-translation_vector[1], translation_vector[0], 0]
])
F_ref = translation_skew_matrix @ P_prime_refined[:, :3]


# In[ ]:


U, S, Vt = np.linalg.svd(F_ref)
epipole_right_ref = Vt[-1] / Vt[-1][-1]  # Right epipole (e') 


# In[ ]:


def compute_translation_matrices(w, h):
    """Compute translation matrices to center the image at origin and back."""
    T1 = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]])
    T2 = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]])
    return T1, T2

def normalize_epipole(e_prime):
    """Normalize the epipole by dividing by the last coordinate."""
    e_prime /= e_prime[-1]
    return e_prime

def compute_rotation_matrix(ex, ey, x0, y0):
    """Compute rotation matrix to align the epipole with the x-axis."""
    theta = np.arctan2(-(ey - y0), (ex - x0))
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return R, theta

def compute_focal_length(ex, ey, x0, y0, theta):
    """Compute focal length based on the aligned epipole coordinates."""
    return abs((ex - x0) * np.cos(theta) - (ey - y0) * np.sin(theta))

def compute_affine_transformation(f):
    """Compute affine transformation matrix based on the focal length."""
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1 / f, 0, 1]
    ])

def compute_right_rectifying_homography(img, e_prime):
    """Compute rectifying homography for the right image."""
    h, w = img.shape[:2]
    
    T1, T2 = compute_translation_matrices(w, h)
    e_prime = normalize_epipole(e_prime)
    x0, y0 = w / 2, h / 2
    ex, ey = e_prime[0], e_prime[1]
    R, theta = compute_rotation_matrix(ex, ey, x0, y0)
    f = compute_focal_length(ex, ey, x0, y0, theta)
    G = compute_affine_transformation(f)
    
    H_prime = T2 @ G @ R @ T1
    H_prime = H_prime/H_prime[-1][-1]
    return H_prime 

def compute_pseudo_inverse(P):
    """Compute the pseudo-inverse of a projection matrix P."""
    return P.T @ np.linalg.inv(P @ P.T)

def transform_points(H, points):
    """Apply a homography H to a set of points."""
    transformed_points = np.array([(H @ np.append(pt, 1))[:2] / (H @ np.append(pt, 1))[2] for pt in points])
    return transformed_points

def compute_affine_homography(x_hat, x_prime_hat):
    """Solve for affine transformation matrix Ha using least squares."""
    A = np.ones((x_hat.shape[0], 3))
    A[:, :2] = x_hat
    b = x_prime_hat[:, 0]
    
    # Solve for affine transformation parameters
    a = np.linalg.lstsq(A, b, rcond=None)[0]
    Ha = np.eye(3)
    Ha[0, :3] = a
    return Ha

def compute_left_rectifying_homography(P, P_prime, H_prime, x_points, x_prime_points):
    """Compute rectifying homography for the left image."""
    P_pinv = compute_pseudo_inverse(P)
    M = np.array(P_prime @ P_pinv)
    H0 = H_prime @ M

    x_hat = transform_points(H0, x_points)
    x_prime_hat = transform_points(H_prime, x_prime_points)

    Ha = compute_affine_homography(x_hat, x_prime_hat)
    H = Ha @ H0
    H= H/H[-1][-1]
    return H


# In[ ]:


H_right = compute_right_rectifying_homography(right_image, epipole_right_ref)


# In[ ]:


H_left = compute_left_rectifying_homography(P, P_prime_refined, H_right, left_points_array, right_points_array)


# In[ ]:


# Step 8: Apply homographies to rectify images
rectified_left_image = cv2.warpPerspective(left_image, H_left, (width*2, height))
cv2.imwrite('HW9/input/pic10.jpeg', rectified_left_image)
rectified_right_image = cv2.warpPerspective(right_image, H_right, (width*2, height))
cv2.imwrite('HW9/input/pic34.jpeg', rectified_right_image)


# In[ ]:


def transform_point(H, point):
    """Transforms a point using a homography matrix H."""
    homogeneous_point = np.array([point[0], point[1], 1])
    transformed_point = H @ homogeneous_point
    x_new = int(transformed_point[0] / transformed_point[2])
    y_new = int(transformed_point[1] / transformed_point[2])
    return np.array([x_new, y_new], dtype=int)


# In[ ]:


rectified_left_points= []
for point in left_points:
    rectified_left_point= transform_point(H_left, point)
    rectified_left_points.append(rectified_left_point)


# In[ ]:


rectified_right_points= []
for point in right_points:
    rectified_right_point= transform_point(H_right, point)
    rectified_right_points.append(rectified_right_point)


# In[ ]:


def plot_corresponding_points(left_img, right_img, left_points, right_points):
    # Create an empty canvas to place images side by side
    height = max(left_img.shape[0], right_img.shape[0])
    width = left_img.shape[1] + right_img.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Place the left and right images on the canvas
    canvas[:left_img.shape[0], :left_img.shape[1]] = left_img
    canvas[:right_img.shape[0], left_img.shape[1]:] = right_img

    # Shift the right points for the concatenated image
    right_points_shifted = [(x + left_img.shape[1], y) for (x, y) in right_points]

    # Draw points and lines between corresponding points
    for (lx, ly), (rx, ry) in zip(left_points, right_points_shifted):
        # Draw circles at each point
        cv2.circle(canvas, (int(lx), int(ly)), 4, (0, 0, 255), -1)  # Red circles for left points
        cv2.circle(canvas, (int(rx), int(ry)), 4, (0, 0, 255), -1)  # Red circles for right points
        # Draw line connecting corresponding points
        cv2.line(canvas, (int(lx), int(ly)), (int(rx), int(ry)), (255, 0, 0), 2)  # Blue line

    # Display the canvas with matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# In[ ]:


plot_corresponding_points(left_image, right_image, left_points, right_points)


# In[ ]:


plot_corresponding_points(rectified_left_image, rectified_right_image, rectified_left_points, rectified_right_points)


# In[ ]:


def detect_edges(image, low_threshold=10, high_threshold=150):
    # Convert to grayscale if the image is not already
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 2)

    # Apply binary thresholding to obtain a binary image
    _, binary = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)
    
    # # Define a kernel for erosion and dilation
    # kernel = np.ones((17, 17), np.uint8)

    # # Apply erosion and then dilation (closing)
    # binary = cv2.erode(binary, kernel, iterations=1)
    # binary = cv2.dilate(binary, kernel, iterations=1)

    # Detect edges using Canny edge detector on the binary image
    edges = cv2.Canny(binary, low_threshold, high_threshold)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    large_closed_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 0 and cv2.arcLength(cnt, True) > 0]
    
    # Create a blank image to draw contours
    contour_image = np.zeros_like(image)

    # Draw contours on the blank image
    cv2.drawContours(contour_image, large_closed_contours, -1, (255, 255, 255), 1)  # Draw in white

    # Convert contour image to grayscale
    contour_image_gray = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)

    return edges


# In[ ]:


def find_interest_points(img_l, img_r, patch_size, output_dir, ssd_threshold=None):
    """
    Detect interest points in rectified image pairs using Canny edge detection and match keypoints
    using SSD within corresponding horizontal regions due to rectification.

    Parameters:
        img_l (np.ndarray): Left rectified image.
        img_r (np.ndarray): Right rectified image.
        patch_size (int): Size of the patch for SSD computation.
        output_dir (str): Path to save the visualized output image.
        ssd_threshold (float, optional): SSD threshold to filter out weak matches. Only matches with an SSD
                                         below this threshold will be stored and visualized.
    """

    def random_color():
        return tuple(np.random.randint(0, 256, 3).tolist())

    # Grayscale conversion
    img_l_gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # Canny edge detection to find interest points
    edges_l = detect_edges(img_l)
    edges_r = detect_edges(img_r)
    
    # Retrieve interest points from edges
    corners_l = np.column_stack(np.where(edges_l > 0))
    corners_r = np.column_stack(np.where(edges_r > 0))

    # Padding size for patch extraction
    pad_size = patch_size // 2
    img_l_padded = np.pad(img_l_gray, pad_size, mode='constant')
    img_r_padded = np.pad(img_r_gray, pad_size, mode='constant')

    # Concatenate images for visualization
    img_combined = np.concatenate((img_l, img_r), axis=1)
    matches = []  # To store matched points with SSD values

    # Match patches around interest points in the left image to the right image
    for (y1, x1) in corners_l:
        # Define the patch in the left image
        patch_l = img_l_padded[y1:y1 + patch_size, x1:x1 + patch_size]
        if np.any(patch_l==0):
            continue
        min_ssd = float('inf')
        best_match = None

        # Limit the search to the same row in the right image due to rectification
        same_row_corners_r = corners_r[corners_r[:, 0] == y1]

        for (y2, x2) in same_row_corners_r:
            # Define the patch in the right image
            patch_r = img_r_padded[y2:y2 + patch_size, x2:x2 + patch_size]
            if np.any(patch_r==0):
                continue
            # Compute SSD
            ssd = np.sum((patch_l - patch_r) ** 2)
            if ssd < min_ssd:
                min_ssd = ssd
                best_match = (x2, y2)

        # Store the match with its SSD if it meets the threshold
        if best_match and (ssd_threshold is None or min_ssd <= ssd_threshold):
            matches.append(((x1, y1), best_match, min_ssd))

    # Optional: Sort matches by SSD value in ascending order
    matches = sorted(matches, key=lambda x: x[2])

    # Draw matches
    for (x1, y1), (x2, y2), ssd in matches:
        color = random_color()
        cv2.circle(img_combined, (x1, y1), 3, (0, 255, 0), -1)
        cv2.circle(img_combined, (x2 + img_l.shape[1], y2), 3, (0, 255, 0), -1)
        cv2.line(img_combined, (x1, y1), (x2 + img_l.shape[1], y2), color, 1)

    # Save output
    cv2.imwrite(output_dir, img_combined)
    print(f'{output_dir} saved.')

    # Create nested list for left and right match points
    matches_left = [[match[0][0], match[0][1]] for match in matches]
    matches_right = [[match[1][0], match[1][1]] for match in matches]
    matches_final = [matches_left, matches_right]
    return np.array(matches_final)


# In[ ]:


matches= find_interest_points(rectified_left_image, rectified_right_image, 5, 'HW9/ssdmatch.png', 300)


# In[ ]:


def unrectify_points(points, H):
    """
    Map points from rectified to original image coordinates using the inverse homography matrix.

    Parameters:
        points (np.ndarray): Points in the rectified image (Nx2).
        H (np.ndarray): Homography matrix used for rectification(3x3).

    Returns:
        unrectified_points (np.ndarray): Points in the original image coordinates (Nx2).
    """
    H_inv= np.linalg.inv(H)
    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # Apply the inverse homography
    unrectified_points_homogeneous = (H_inv @ points_homogeneous.T).T
    
    # Normalize to convert from homogeneous to Cartesian coordinates
    unrectified_points = unrectified_points_homogeneous[:, :2] / unrectified_points_homogeneous[:, 2, np.newaxis]
    
    return unrectified_points


# In[ ]:


unrectified_matches_left = unrectify_points(matches[0], H_left)
unrectified_matches_right = unrectify_points(matches[1], H_right)


# In[ ]:


plot_corresponding_points(left_image, right_image, unrectified_matches_left, unrectified_matches_right)


# In[ ]:


world_matches= triangulate(P, P_prime_refined, unrectified_matches_left, unrectified_matches_right)


# In[ ]:


world_matches_highlight= triangulate(P, P_prime_refined, left_points, right_points)


# In[ ]:


def normalize_coordinates(points):
    """
    Normalize world coordinates so that they are centered around 0
    and fit within a reasonable range for better visualization.
    """
    # Step 1: Subtract the mean to center the points around 0
    mean = np.mean(points, axis=0)
    normalized_points = points - mean
    
    # Step 2: Scale by the standard deviation (or use max if preferred)
    scale = np.std(normalized_points, axis=0)
    normalized_points = normalized_points / scale
    scaling_params= [mean, scale]
    
    return normalized_points, scaling_params


# In[ ]:


combined_world_matches= np.concat([world_matches, world_matches_highlight[:4]])


# In[ ]:


combined_normalized_world_points, scaling_params= normalize_coordinates(combined_world_matches)


# In[ ]:


normalized_world_matches= combined_normalized_world_points[:174]
normalized_highlight_world_matches= combined_normalized_world_points[-4:]


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import rq

def plot_camera(R, t, C, plane_scale, arrow_scale, ax, color='b', label="Camera"):
    """
    Plots the camera axes and principal plane for a given camera pose (R, t, C).
    """
    # Define the direction vectors along the X, Y, Z axes
    X_axis = np.array([1, 0, 0])
    Y_axis = np.array([0, 1, 0])
    Z_axis = np.array([0, 0, 1])

    # Rotate the axes using the rotation matrix R
    X_cam = R.T @ X_axis
    Y_cam = R.T @ Y_axis
    Z_cam = R.T @ Z_axis
    
    # Define X, Y, Z direction vectors in camera frame scaled by arrow_scale
    X_cam = R @ np.array([1, 0, 0]) * arrow_scale + C.flatten()
    Y_cam = R @ np.array([0, 1, 0]) * arrow_scale + C.flatten()
    Z_cam = R @ np.array([0, 0, 1]) * arrow_scale + C.flatten()
    

    
    # Plot camera axes starting from C to each of the axes directions
    ax.quiver(*C.flatten(), *(X_cam - C.flatten()), color='r', label='Xcam')
    ax.quiver(*C.flatten(), *(Y_cam - C.flatten()), color='g', label='Ycam')
    ax.quiver(*C.flatten(), *(Z_cam - C.flatten()), color='b', label='Zcam')

    # Plot the camera principal plane as a larger, semi-transparent rectangle
    plane_corners = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]]) * plane_scale
    plane_corners_world = R.T @ plane_corners.T + C
    ax.plot_trisurf(plane_corners_world[0], plane_corners_world[1], plane_corners_world[2], color=np.random.rand(3,), alpha=0.5)

def decompose_projection_matrix(P_prime):
    # Extract the rotation matrix (3x3) and translation vector (3x1)
    R = P_prime[:, :3]  # First three columns are the rotation matrix
    t = P_prime[:, 3]   # Last column is the translation vector
    
    return R, t

def extract_camera_pose(P, coordinate_scaling_params=None):
    """
    Extracts the camera pose (rotation, translation, and position) from the projection matrix P.
    """
    R, t = decompose_projection_matrix(P)
    camera_position = -np.dot(R.T, t)  # Camera position in world coordinates
    
    # t = t.reshape(-1, 1)
    # camera_position = camera_position.reshape(-1, 1)
    if coordinate_scaling_params:
        t = (t-coordinate_scaling_params[0])/coordinate_scaling_params[1]
        camera_position= (camera_position-coordinate_scaling_params[0])/coordinate_scaling_params[1]

    t = t.reshape(-1, 1)
    camera_position = camera_position.reshape(-1, 1)
    return R, t, camera_position

def plot_3d_scene(P, P_prime, pattern_points, highlight_pattern_points=None, coordinate_scaling_params= None, plane_scale=2.0, arrow_scale=2.0, filename=None):
    """
    Plots the 3D scene including two cameras, their axes, and the calibration pattern.
    Optionally highlights certain points in the pattern.
    """
    # print(coordinate_scaling_params)
    fig = plt.figure(figsize= (6,6), dpi=1000)
    ax = fig.add_subplot(111, projection='3d')

    # Extract camera poses
    R, t, C = extract_camera_pose(P, coordinate_scaling_params)
    R_prime, t_prime, C_prime = extract_camera_pose(P_prime, coordinate_scaling_params)

    # Plot camera 1
    plot_camera(R, t, C, plane_scale, arrow_scale, ax, color='b', label="Camera 1")

    # Plot camera 2
    plot_camera(R_prime, t_prime, C_prime, plane_scale, arrow_scale, ax, color='r', label="Camera 2")

    # Plot calibration pattern on Z=0 plane
    ax.scatter(pattern_points[:, 0], pattern_points[:, 1], pattern_points[:, 2], c='k', marker='o', s=1)

    # If highlight_pattern_points are provided, highlight them with a distinct style
    if highlight_pattern_points is not None:
        ax.scatter(highlight_pattern_points[:, 0], highlight_pattern_points[:, 1], highlight_pattern_points[:, 2], 
                   c='r', marker='o', s=20, label='Highlighted Points')

    # Label and show the plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_zlim([-0.15, 0.15])
    ax.set_xlim([-1, 0])
    ax.set_ylim([0, 1])
    # plt.axis('off')
    
    plt.tight_layout()
    # ax.legend()
    ax.view_init(10)  # Adjust view angle for better visualization
    if filename:
        plt.savefig(filename)
    plt.show()


# In[ ]:


plot_3d_scene(P, P_prime_refined, normalized_world_matches, normalized_highlight_world_matches[:4], scaling_params, plane_scale=0.1, arrow_scale=0.03, filename='view1.png')


# In[ ]:


def plot_images_with_points(image1, image2, image3, points1, points2):
    """
    Plots three images stacked vertically, with points from two sets plotted on the images.
    The second set of points is shifted to the bottom image.
    Only the last 4 points from both sets are highlighted.

    Parameters:
    - image1, image2, image3: Three input images (in BGR format).
    - points1: Points to be plotted on the first (top) image.
    - points2: Points to be plotted on the second (bottom) image, which are shifted.
    """
    # Convert from BGR to RGB for plotting
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

    # Resize image2 to match the width of image1 and image3 while keeping the aspect ratio
    target_width = max(image1.shape[1], image3.shape[1])  # Set target width to the max of image1 and image3
    target_height2 = int(image2.shape[0] * (target_width / image2.shape[1]))  # Maintain aspect ratio for image2
    image2_resized = cv2.resize(image2, (target_width, target_height2))

    # Get the height of all images, use the max height among the three
    max_height = max(image1.shape[0], image2_resized.shape[0], image3.shape[0])

    # Create a blank canvas to stack the images vertically
    canvas = np.zeros((image1.shape[0] + image2_resized.shape[0] + image3.shape[0], target_width, 3), dtype=np.uint8)

    # Place images on the canvas
    canvas[:image1.shape[0], :image1.shape[1]] = image1
    canvas[image1.shape[0]:image1.shape[0] + image2_resized.shape[0], :image2_resized.shape[1]] = image2_resized
    canvas[image1.shape[0] + image2_resized.shape[0]:, :image3.shape[1]] = image3

    # Create figure for plotting
    fig, ax = plt.subplots(figsize=(8, 12))

    # Plot the images on the canvas
    ax.imshow(canvas)
    
    
    ax.scatter(points1[:, 0], points1[:, 1], c='r', marker='o', s=10, label='Highlighted Points 1')

    # Shift the second set of points for the bottom image (image2)
    points2_shifted = points2.copy()
    points2_shifted[:, 1] += image1.shape[0] + image2_resized.shape[0]


    ax.scatter(points2_shifted[:, 0], points2_shifted[:, 1], c='r', marker='o', s=10, label='Highlighted Points 2')

    # Remove axis and add labels
    ax.axis('off')
    # ax.legend(loc='upper right')
    plt.savefig('compared.png', bbox_inches='tight', pad_inches=0)
    plt.tight_layout()
    # Display the final stacked image with points
    plt.show()


# In[ ]:


img_3d= cv2.imread('view1.png')


# In[ ]:


plot_images_with_points(left_image, img_3d, right_image, left_points_array[:4],  right_points_array[:4])



# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def compute_disparity_map(left_image, right_image, window_size, max_disparity):
    """Calculate disparity map using a Census-like transform."""
    
    # Deal with rectangular window size
    if not isinstance(window_size, tuple):
        half_window_x= window_size // 2
        half_window_y= window_size // 2
    else:
        half_window_x = window_size[0] // 2
        half_window_y = window_size[1] // 2
    padding_x = max_disparity + half_window_x
    padding_y = max_disparity + half_window_y

    # Pad images to avoid boundary issues
    padded_left = np.pad(left_image, ((padding_x, padding_x), (padding_y, padding_y)), mode='constant', constant_values=0)
    padded_right = np.pad(right_image, ((padding_x, padding_x), (padding_y, padding_y)), mode='constant', constant_values=0)
    img_height, img_width = padded_left.shape
    disparity_map = np.zeros_like(padded_left, dtype=np.uint8)

    # Main disparity calculation loop
    for y in range(padding_y, img_height - padding_y):
        for x in range(padding_x, img_width - padding_x):
            best_cost = float('inf')
            best_disp = 0

            # Compute the census transform for the center pixel
            left_win = padded_left[y - half_window_y : y + half_window_y + 1, x - half_window_x : x + half_window_x + 1]
            # left_vec = compute_census(left_win, padded_left[y, x])
            # left_vec = np.ravel((left_win>padded_left[y, x])*1)
            left_vec = np.where(left_win > padded_left[y, x], 1, 0).ravel()

            # Evaluate all possible disparities
            for d in range(min(max_disparity, x) + 1):
                right_x = x - d
                right_win = padded_right[y - half_window_y : y + half_window_y + 1, right_x - half_window_x : right_x + half_window_x + 1]
                # right_vec = compute_census(right_win, padded_right[y, right_x])
                # right_vec = np.ravel((right_win>padded_right[y, x])*1)
                right_vec = np.where(right_win > padded_right[y, x], 1, 0).ravel()

                # Compute the Hamming distance (cost)
                cost = np.count_nonzero(np.bitwise_xor(left_vec, right_vec))

                if cost < best_cost:
                    best_cost = cost
                    best_disp = d

            disparity_map[y, x] = best_disp

    # Return the disparity map (excluding padding region)
    return disparity_map[padding_y:-padding_y, padding_x:-padding_x]


# In[ ]:


left_image= cv2.imread('im2.png', cv2.IMREAD_GRAYSCALE)
right_image= cv2.imread('im6.png', cv2.IMREAD_GRAYSCALE)


# In[ ]:


disparity_map= compute_disparity_map(left_image, right_image, (20, 20), 128)


# In[ ]:


disparity_map_img= (disparity_map/np.max(disparity_map) *255).astype(np.uint8) 


# In[ ]:



def evaluate_disparity_accuracy(estimated_disparity, ground_truth_disparity, delta=1):
    """
    Evaluates the accuracy of an estimated disparity map against the ground truth.
    
    Parameters:
    - estimated_disparity: The computed disparity map (numpy array).
    - ground_truth_disparity: The ground truth disparity map (numpy array).
    - delta: The threshold value for the binary error mask (default is 1 pixel).
    
    Returns:
    - error_mask: Binary error mask where error <= delta is set to 255, otherwise 0.
    - mean_absolute_error: The average absolute disparity error.
    - valid_pixel_count: The number of valid pixels for comparison.
    """
    
    # Ensure the ground truth disparity map is in float32 and adjust for resolution
    ground_truth_disparity = ground_truth_disparity.astype(np.float32) / 4
    
    # Convert the ground truth disparity map back to uint8 (scaled by 4)
    ground_truth_disparity_uint8 = (ground_truth_disparity * 4).astype(np.uint8)
    
    # Mask for valid pixels (non-black pixels in the ground truth)
    valid_mask = ground_truth_disparity_uint8 != 0
    
    # Calculate the absolute disparity error
    disparity_error = np.abs(estimated_disparity.astype(np.float32) - ground_truth_disparity)

    
    # Compute the number of valid pixels
    valid_pixel_count = np.sum(valid_mask)
    
    # Calculate mean absolute error (only for valid pixels)
    mean_absolute_error = np.sum(disparity_error * valid_mask) / valid_pixel_count if valid_pixel_count > 0 else 0
    
    # Create the binary error mask for pixels where error <= delta
    error_mask = np.zeros_like(ground_truth_disparity, dtype=np.uint8)
    error_mask[disparity_error <= delta] = 255

    correct_pixels= np.sum((disparity_error<=delta) & valid_mask)
    print(correct_pixels)
    print(valid_pixel_count)
    accuracy_value = correct_pixels/valid_pixel_count if valid_pixel_count > 0 else 0
    
    return error_mask, mean_absolute_error, valid_pixel_count, accuracy_value

def compute_disparity_accuracy(estimated_disparity, ground_truth_disparity, delta=1):
    """
    Computes the disparity accuracy for the given disparity maps.
    
    Parameters:
    - estimated_disparity: The computed disparity map (numpy array).
    - ground_truth_disparity: The ground truth disparity map (numpy array).
    - delta: Threshold for the binary error mask (default 1).
    
    Returns:
    - accuracy: A dictionary containing the mean absolute error, valid pixel count, and error mask.
    """
    
    # Evaluate the accuracy
    error_mask, mean_absolute_error, valid_pixel_count, accuracy_value = evaluate_disparity_accuracy(estimated_disparity, ground_truth_disparity, delta)
    
    accuracy = {
        "mean_absolute_error": mean_absolute_error,
        "valid_pixel_count": valid_pixel_count,
        "error_mask": error_mask, 
        "accuracy": accuracy_value
    }
    
    return accuracy


# In[ ]:


gt_disparity= cv2.imread('disp2.png', cv2.IMREAD_GRAYSCALE)


# In[ ]:


accuracy= compute_disparity_accuracy(disparity_map, gt_disparity, delta=2)


# In[ ]:


plt.imshow(accuracy['error_mask'], cmap='gray')
plt.axis('off')


# In[ ]:


plt.imshow(disparity_map_img, cmap='gray')
plt.axis('off')


# In[ ]:


plt.imshow(gt_disparity, cmap='gray')
plt.axis('off')


# In[ ]:


# accuracy 50= 0.383
# accuracy 30= 00.32
#accuracy 20= 27.31
#accuracy 15= 0.196
#accuracy 5= 0.089
accuracy['accuracy']


# In[ ]:




