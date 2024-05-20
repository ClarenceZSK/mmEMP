import os
import scipy.io as scio
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
from scipy.optimize import least_squares
import random
def load_data(datapath, labelpath = None):
    
    norm = lambda x: (x - x.min())/(x.max() - x.min())

    files = os.listdir(datapath)
    files.sort()
    data = [norm(scio.loadmat(os.path.join(datapath, i))['DopplerFFTOutinte'][np.newaxis]) for i in files]

    if not labelpath:
        return data
    
    files = os.listdir(labelpath)
    files.sort()
    label = [scio.loadmat(os.path.join(labelpath, i))['binary_matrix'][np.newaxis] for i in files]

    assert(len(data)==len(label))

    return data, label

def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def triangulate_points(keypoints1, keypoints2, rvec, tvec):
    points_3d = []
    proj_matrix1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    proj_matrix2 = np.hstack((rvec, tvec))
    if len(keypoints1) < len(keypoints2):
        keypoints_to_iterate = keypoints1
    else:
        keypoints_to_iterate = keypoints2
    for i in range(len(keypoints_to_iterate)):
        point1 = keypoints1[i].pt
        point2 = keypoints2[i].pt
        point1_homogeneous = np.array([point1[0], point1[1]])
        point2_homogeneous = np.array([point2[0], point2[1]])
        point_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, point1_homogeneous, point2_homogeneous)
        point_3d = point_4d[:3] / point_4d[3]
        points_3d.append(point_3d)
    return points_3d

def estimate_pose_ransac(triangulated_points, keypoints2, keypoints1, good_matches, num_iterations, reprojection_threshold):
    best_inliers = 0
    best_rvec = None
    best_tvec = None
    best_inlier_indices = None

    for _ in range(num_iterations):
        random_indices = random.sample(range(len(triangulated_points)), 5)
        zeros = np.zeros((5, 1))
        src_pts = np.array([triangulated_points[i] for i in random_indices]).reshape(-1, 3)
        dst_pts = np.array([keypoints2[i].pt for i in random_indices]).reshape(-1,  2)
        # src_pts = np.concatenate((src_pts, zeros), axis=1)
        # dst_pts = np.concatenate((dst_pts, zeros), axis=1)
        camera_matrix = np.float64([[1000.0, 0.0, 320.0],
                                  [0.0, 1000.0, 240.0],
                                  [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([0.1, -0.2, 0.001, 0.002, 0])
        # Check if the selected points are collinear (this can cause issues)

        _, rvec, tvec, inliers = cv2.solvePnPRansac(src_pts, dst_pts, camera_matrix , dist_coeffs)

        num_inliers = np.sum(inliers)

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_rvec = rvec
            best_tvec = tvec
            best_inlier_indices = inliers

    outlier_indices = np.where(best_inlier_indices == 0)[0]
    similar_indices = np.where(best_inlier_indices == 1)[0]
    outlier_rvecs = [best_rvec for _ in outlier_indices]
    outlier_tvecs = [best_tvec for _ in outlier_indices]
    similar_rvecs = [best_rvec for _ in similar_indices]
    similar_tvecs = [best_tvec for _ in similar_indices]

    if best_inliers == 0:
        outlier_points = [dst_pts[i].pt for i in outlier_indices]
        outlier_points1 = [keypoints1[good_matches[i].query].pt for i in outlier_indices]
    else:
        outlier_points = []
        outlier_points1 = []
        for i in outlier_indices:
            if best_inlier_indices[i] == 0:
                outlier_points = [dst_pts[i] for i in outlier_indices]
                outlier_points1.append(keypoints1[good_matches[i].queryIdx].pt)

    return best_rvec, best_tvec, outlier_rvecs, outlier_tvecs, similar_rvecs, similar_tvecs, outlier_points, outlier_points1

def calculate_initial_pose(keypoints1, keypoints2, good_matches, camera_matrix):
    # Extract corresponding points for the selected matches
    src_pts = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Calculate essential matrix using the matches
    essential_matrix, _ = cv2.findEssentialMat(src_pts, dst_pts, camera_matrix)

    # Recover the rotation and translation from the essential matrix
    _, rvec, tvec, _ = cv2.recoverPose(essential_matrix, src_pts, dst_pts, camera_matrix)

    return rvec, tvec

def residual(x, A, b):
    return (np.dot(A, x) - b).flatten()
def construct_overdetermined_equation(params, similar_rvecs, similar_tvecs, outlier_points, outlier_points1, k):
    p1 = params[:4]
    delta_d = params[4:]
    q1 = outlier_points
    q1 = np.array(q1).reshape(1, 2)
    q1new_matrix = np.hstack((q1, np.array([[1]])))
    q1new_matrix = np.dot(k, (p1 + delta_d)[:3])
    q2 = outlier_points1
    q2 = np.array(q2).reshape(1, 2)
    q1new_matrix = np.hstack((q2, np.array([[1]])))
    q1new_matrix = np.dot(k, p1[:3])
    rotation_vector = np.array(similar_rvecs).reshape(3, )

    rotation_angle = np.linalg.norm(rotation_vector)
    rotation_axis = rotation_vector / rotation_angle

    P = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
      ])
    rotation_matrix = np.identity(3) + np.sin(rotation_angle) * P + (1 - np.cos(rotation_angle)) * np.dot(P, P)
    transform_matrix = np.array(similar_tvecs).reshape(3, 1)
    new_row = np.array([[1]])
    transform = np.vstack((transform_matrix, new_row))
    RT = np.hstack((rotation_matrix, transform_matrix))
    t1 = RT[0]
    t1 = t1.reshape(-1, 1)
    t2 = RT[1]
    t2 = t2.reshape(-1, 1)
    t3 = RT[2]
    t3 = t3.reshape(-1, 1)
    q3 = np.array([t1.T @ (p1 + delta_d) / t3.T @ (p1 + delta_d), t2.T @ (p1 + delta_d) / t3.T @ (p1 + delta_d)])
    q4 = np.array([t1.T @ p1 / t3.T @ p1, t2.T @ p1 / t3.T @ p1])
    a = np.dot(p1 - transform[:, 0].T, p1 +delta_d - transform[:, 0])
    residual1 = 0.5 * np.linalg.norm(np.dot(p1, p1.T) + 0.5 *np.dot(p1 + delta_d, p1 + delta_d).T - 0.5 * 2 * np.dot(p1.T, p1 + delta_d) * (np.dot(q1, q3) / (np.linalg.norm(q1) * np.linalg.norm(q3))) -np.dot(delta_d, delta_d.T)) + np.linalg.norm(0.5 * np.dot(p1 - transform[:, 0], p1 - transform[:, 0]).T + 0.5 *np.dot(p1 + delta_d - transform[:, 0], p1 + delta_d - transform[:, 0]).T - 0.5 * 2 * np.dot(p1 - transform[:, 0].T, p1 +delta_d - transform[:, 0]) * (np.dot(q4, q2) / (np.linalg.norm(q4) * np.linalg.norm(q2)))-np.dot(delta_d, delta_d.T))
    residual = 0.5 * np.linalg.norm(
        np.dot(p1, p1.T) + 0.5 * np.dot(p1 + delta_d, p1 + delta_d).T - 0.5 * 2 * np.dot(p1.T, p1 + delta_d) * (
                    np.dot(q1, q3) / (np.linalg.norm(q1) * np.linalg.norm(q3))) - np.dot(delta_d,
                                                                                         delta_d.T) - 0.5 * np.dot(
            p1 - transform[:, 0], p1 - transform[:, 0]).T + 0.5 * np.dot(p1 + delta_d - transform[:, 0],
                                                                         p1 + delta_d - transform[:,
                                                                                        0]).T - 0.5 * 2 * np.dot(
            p1 - transform[:, 0].T, p1 + delta_d - transform[:, 0]) * (
                    np.dot(q4, q2) / (np.linalg.norm(q4) * np.linalg.norm(q2))))

    return residual1
def point_enhancing(image1, image2):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    transform = T.Compose([T.ToTensor()])
    input1 = transform(image1).unsqueeze(0)
    input2 = transform(image2).unsqueeze(0)
    with torch.no_grad():
        output1 = model(input1)
        output2 = model(input2)
        boxes1 = output1[0]['boxes'].cpu().numpy()
        labels1 = output1[0]['labels'].cpu().numpy()
        scores1 = output1[0]['scores'].cpu().numpy()
        boxes2 = output2[0]['boxes'].cpu().numpy()
        labels2 = output2[0]['labels'].cpu().numpy()
        scores2 = output2[0]['scores'].cpu().numpy()

    confidence_threshold = 0.95


    filtered_boxes1 = boxes1[scores1 > confidence_threshold]
    filtered_labels1 = labels1[scores1 > confidence_threshold]
    filtered_boxes2 = boxes2[scores2 > confidence_threshold]
    filtered_labels2 = labels2[scores2 > confidence_threshold]

    descriptors1_inside = []
    descriptors2_inside = []
    keypoint_coordinates1 = []
    keypoint_coordinates2 = []
    keypoints_outside1 = []
    keypoints_outside2 = []
    for box1 in filtered_boxes1:
        x1, y1, x2, y2 = map(int, box1)
        object_image1 = image1[y1:y2, x1:x2]
        keypoints_inside1, descriptors = extract_sift_features(object_image1)
        key_coordinates1 = [kp.pt for kp in keypoints_inside1]
        x1_outside, y1_outside, x2_outside, y2_outside = 0, 0, image1.shape[1], image1.shape[0]
        if x2_outside >= x1 and x1_outside <= x2 and y2_outside >= y1 and y1_outside <= y2:
            # Adjust the region to avoid overlap
            if x1_outside < x1:
                x1_outside = x1
            if x2_outside > x2:
                x2_outside = x2
            if y1_outside < y1:
                y1_outside = y1
            if y2_outside > y2:
                y2_outside = y2
                # Extract SIFT features for the region outside the box
        object_image1_outside = image1[y1_outside:y2_outside, x1_outside:x2_outside]
        keypoints_temp1, descriptors = extract_sift_features(object_image1_outside)
        descriptors1_inside.append(descriptors)
        keypoint_coordinates1.append(key_coordinates1)
        keypoints_outside1.append(keypoints_temp1)
    for box2 in filtered_boxes2:
        x1, y1, x2, y2 = map(int, box2)
        object_image2 = image2[y1:y2, x1:x2]
        keypoints_inside2, descriptors = extract_sift_features(object_image2)
        key_coordinates2 = [kp.pt for kp in keypoints_inside2]
        x1_outside, y1_outside, x2_outside, y2_outside = 0, 0, image2.shape[1], image2.shape[0]
        if x2_outside >= x1 and x1_outside <= x2 and y2_outside >= y1 and y1_outside <= y2:
            # Adjust the region to avoid overlap
            if x1_outside < x1:
                x1_outside = x1
            if x2_outside > x2:
                x2_outside = x2
            if y1_outside < y1:
                y1_outside = y1
            if y2_outside > y2:
                y2_outside = y2
            # Extract SIFT features for the region outside the box
        object_image2_outside = image2[y1_outside:y2_outside, x1_outside:x2_outside]
        keypoints_temp2, descriptors = extract_sift_features(object_image2_outside)
        descriptors2_inside.append(descriptors)
        keypoint_coordinates2.append(key_coordinates2)
        keypoints_outside2.append(keypoints_temp2)

    keypoints1, descriptors1 = extract_sift_features(image1)
    keypoints2, descriptors2 = extract_sift_features(image2)


    good_matches = match_features(descriptors1, descriptors2)

    num_iterations = 100
    reprojection_threshold = 5.0
    camera_matrix = np.float64([[1000.0, 0.0, 320.0],
                                [0.0, 1000.0, 240.0],
                                [0.0, 0.0, 1.0]])

    initial_rvec, initial_tvec = calculate_initial_pose(keypoints1, keypoints2, good_matches, camera_matrix)
    triangulated_points = triangulate_points(keypoints1, keypoints2, initial_rvec, initial_tvec)
    rvec, tvec, outlier_rvecs, outlier_tvecs, similar_rvecs, similar_tvecs, outlier_points, outlier_points1 = estimate_pose_ransac(
        triangulated_points, keypoints2, keypoints1, good_matches, num_iterations, reprojection_threshold)

    flat_list = [item for sublist in keypoint_coordinates1 for item in sublist]
    matrix = np.array(flat_list)
    flat_list1 = [item for sublist in keypoint_coordinates2 for item in sublist]
    matrix1 = np.array(flat_list1)
    initial_params = np.ones(8)
    num_samples = len(matrix1)  # Number of point pairs
    all_solution_p1 = []  # Initialize a list to store all solution_p1
    all_solution_delta_d = []
    all_solution_p2 = []
    for i in range(num_samples):
        point1 = matrix[i]
        point2 = matrix1[i]
        result = least_squares(construct_overdetermined_equation, initial_params,
                               args=(similar_rvecs, similar_tvecs, point1, point2, camera_matrix))
        solution_p1 = result.x[:4]
        solution_p1 = solution_p1.reshape(-1, 1)
        fourth_element = solution_p1[3, 0]
        solution_p1 = solution_p1 / fourth_element
        solution_delta_d = result.x[4:]
        solution_delta_d = solution_delta_d.reshape(-1, 1)
        fourth_element1 = solution_delta_d[3, 0]
        solution_delta_d = solution_delta_d / fourth_element1
        solution_p1 = solution_p1[:3]
        solution_delta_d = solution_delta_d[:3]
        solution_p2 = solution_p1 + solution_delta_d
        # Store the result in the list
        all_solution_p1.append(solution_p1)
        all_solution_delta_d.append(solution_delta_d)
        all_solution_p2.append(solution_p2)
    selected_p1 = random.sample(all_solution_p1, 5)
    selected_p2 = random.sample(all_solution_p2, 5)
    combined_data = triangulated_points + selected_p1
    combined_data1 = triangulated_points + selected_p2
    combined_data2 = triangulated_points + all_solution_p1
    combined_data3 = triangulated_points + all_solution_p2
    combined_data4 = triangulated_points
    combined_data5 = all_solution_p1
    stacked_array1 = np.vstack(combined_data2)

    matrix_combined_data2 = stacked_array1.reshape(-1, 3)
    stacked_array2 = np.vstack(combined_data3)

    matrix_combined_data3 = stacked_array2.reshape(-1, 3)
    stacked_array3 = np.vstack(all_solution_delta_d)

    matrix_all_solution_delta_d = stacked_array3.reshape(-1, 3)
    return matrix_all_solution_delta_d, combined_data5, combined_data4

def init_dataset(config):

    if config.act == "train":

        D, L = [], []
        for i in config.data.train:
            data, label = load_data("data/{}/data/".format(i), "data/{}/label/".format(i))
            D += data
            L += label
        train_dataset = myDataset(D, L)
        
        D, L = [], []
        for i in config.data.test:
            data, label = load_data("data/{}/data/".format(i), "data/{}/label/".format(i))
            D += data
            L += label
        test_dataset = myDataset(D, L)

        print("Using {} to train, {} to test".format(config.data.train, config.data.test))
        print("Train data - {}, Test data - {}".format(train_dataset.len, test_dataset.len))

        return train_dataset, test_dataset

    if config.act == "eval":

        D, L = [], []
        for i in config.data.test:
            data, label = load_data("data/{}/data/".format(i), "data/{}/label/".format(i))
            D += data
            L += label
        test_dataset = myDataset(D, L)

        print("Using{} to test".format(config.data.test))
        print("Test data - {}".format(test_dataset.len))

        return test_dataset

    if config.act == "transform":

        transform_dataset = []
        for i in config.data.transform:
            data = load_data("data/{}/data/".format(i))
            transform_dataset += myDataset(data)

        return transform_dataset


class myDataset(Dataset):
    def __init__(self, data, label = None):

        self.data = data
        self.label = label
        self.len = len(data)

    def __getitem__(self, index):
        
        if self.label:
            return self.data[index], self.label[index]
        return self.data[index]

    def __len__(self):

        return self.len
