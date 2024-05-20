import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io

def read_point_cloud(file_path):
    """
    Read point cloud data from a text file.
    """
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            point = [float(coord) for coord in data]
            points.append(point)
    return np.array(points)

def compute_average_velocity(points, t):
    """
    Compute the average velocity of points over time.
    """
    displacement = np.linalg.norm(points[1:] - points[:-1], axis=1)
    average_velocity = np.mean(displacement) / t
    return average_velocity

def filter_points(points, threshold):
    """
    Filter points based on a distance threshold.
    """
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    filtered_points = [points[0]]
    for i in range(1, len(points)):
        if distances[i - 1] <= threshold:
            filtered_points.append(points[i])
    return np.array(filtered_points)

def visualize_point_cloud(points):
    """
    Visualize three-dimensional point cloud.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def main():
    # Read point cloud data for different frames
    static_points_frame1 = read_point_cloud("static_points_frame1.txt")
    dynamic_points_frame1 = read_point_cloud("dynamic_points_frame1.txt")
    static_points_frame2 = read_point_cloud("static_points_frame2.txt")
    dynamic_points_frame2 = read_point_cloud("dynamic_points_frame2.txt")
    mat_contents = scipy.io.loadmat('transform_matrix.mat')
    T = mat_contents['T']
    # Define time interval
    t = float(input("Enter time interval (t): "))

    # Compute average velocities
    v1 = compute_average_velocity(static_points_frame1, t)
    v2 = compute_average_velocity(dynamic_points_frame1, t)

    # Calculate thresholds
    static_threshold = 0.5 * v1 * t
    dynamic_threshold = 0.5 * v2 * t

    # Filter points based on thresholds
    transformed_static_points_frame2 = np.dot(static_points_frame2, T)
    transformed_dynamic_points_frame2 = np.dot(dynamic_points_frame2, T)

    filtered_static_points = filter_points(transformed_static_points_frame2, static_threshold)
    filtered_dynamic_points = filter_points(transformed_dynamic_points_frame2, dynamic_threshold)

    # Accumulate filtered points to obtain enhanced points
    enhanced_points = np.concatenate((filtered_static_points, filtered_dynamic_points))

    # Save enhanced points to a specified folder in TXT format
    np.savetxt('enhanced_points.txt', enhanced_points, fmt='%f', delimiter=',')

    # Visualize three-dimensional point cloud
    visualize_point_cloud(enhanced_points)

if __name__ == "__main__":
    main()
