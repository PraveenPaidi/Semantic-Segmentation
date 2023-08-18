#!/usr/bin/env python
# coding: utf-8

# In[1]:


import open3d as o3d
import numpy as np

def load_pcd_file(file_path):
    cloud = o3d.io.read_point_cloud(file_path)
    if not cloud:
        print("Failed to load the PCD file.")
        return None

    print(f"Loaded {len(cloud.points)} data points from the PCD file.")
    return cloud

def apply_random_downsampling(cloud, target_points):
    num_points = len(cloud.points)

    if num_points <= target_points:
        return cloud

    # Generate random indices to select the desired number of points
    indices = np.random.choice(num_points, target_points, replace=False)
    cloud_downsampled = cloud.select_by_index(indices)

    return cloud_downsampled

if __name__ == "__main__":
    file_path = r'C:\Users\prave\Desktop\Intern\tabletop.pcd'  # Replace with the actual file path
    target_points = 200000  # Set the desired number of points after downsampling

    cloud = load_pcd_file(file_path)
    if cloud:
        cloud_downsampled = apply_random_downsampling(cloud, target_points)
        print(f"Downsampled point cloud has {len(cloud_downsampled.points)} data points.")

        # Visualize the downsampled point cloud
        o3d.visualization.draw_geometries([cloud_downsampled])


# In[2]:


def apply_passthrough_filter(cloud, axis_min, axis_max, filter_axis='z'):
    # Create an AxisAlignedBoundingBox to specify the range for filtering
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-float('inf'), -float('inf'), axis_min),
                                               max_bound=(float('inf'), float('inf'), axis_max))

    # Crop the point cloud using the bounding box
    cloud_filtered = cloud.crop(bbox)
    return cloud_filtered

axis_min = 0.7
axis_max = 2
filter_axis = 'z'

if cloud:
    cloud_filtered = apply_passthrough_filter(cloud_downsampled, axis_min, axis_max, filter_axis)
    print(f"Filtered point cloud has {len(cloud_filtered.points)} data points.")

    # Save the filtered point cloud to a new PCD file
    filename_filtered = 'pass_through_filtered.pcd'
    o3d.io.write_point_cloud(filename_filtered, cloud_filtered)
    o3d.visualization.draw_geometries([cloud_filtered])


# In[3]:


import open3d as o3d
import numpy as np

def plane_segmentation_ransac(cloud, distance_threshold=0.01, max_iterations=1000):
    # Convert Open3D PointCloud to numpy array
    points_np = np.asarray(cloud.points)

    # Initialize variables for plane segmentation
    best_plane = None
    best_inliers = []
    num_points = len(points_np)

    for _ in range(max_iterations):
        # Randomly select three points to fit a plane
        indices = np.random.choice(num_points, 3, replace=False)
        sampled_points = points_np[indices]

        # Fit a plane to the sampled points using Open3D's fit_plane function
        plane_model, inliers = cloud.segment_plane(distance_threshold=distance_threshold,
                                                   ransac_n=3,
                                                   num_iterations=1000)

        # Find inliers within the distance threshold
        inliers = np.asarray(inliers)

        # Check if the current inliers are better than previous best
        if len(inliers) > len(best_inliers):
            best_plane = plane_model
            best_inliers = inliers

    # Extract inliers and outliers based on the segmentation result
    cloud_inliers = cloud.select_by_index(best_inliers)
    cloud_outliers = cloud.select_by_index(best_inliers, invert=True)

    return cloud_inliers, cloud_outliers


if cloud:
    cloud_inliers, cloud_outliers = plane_segmentation_ransac(cloud_filtered)

    print(f"Number of inlier points: {len(cloud_inliers.points)}")
    print(f"Number of outlier points: {len(cloud_outliers.points)}")

    # Save the segmented point clouds to new PCD files
    filename_inliers = 'inlier_points.pcd'
    o3d.io.write_point_cloud(filename_inliers, cloud_inliers)

    filename_outliers = 'outlier_points.pcd'
    o3d.io.write_point_cloud(filename_outliers, cloud_outliers)
    
    o3d.visualization.draw_geometries([cloud_outliers])


# In[4]:


filename_inliers = 'extracted_inliers.pcd'
o3d.io.write_point_cloud(filename_inliers, cloud_inliers)


# In[69]:


o3d.visualization.draw_geometries([cloud_outliers])


# In[75]:


import open3d as o3d
import numpy as np

def euclidean_clustering(cloud, cluster_tol=0.001, min_cluster_size=10, max_cluster_size=250):
    points_np = np.asarray(cloud.points)

    # Create a KD tree for efficient search
    tree = o3d.geometry.KDTreeFlann(cloud)

    # Initialize list to store cluster indices
    cluster_indices = []

    # Mark points that have been processed
    processed = np.zeros(points_np.shape[0], dtype=bool)

    # Iterate through each point to form clusters
    for idx in range(points_np.shape[0]):
        if not processed[idx]:
            # Create a new cluster
            cluster = []
            # Mark this point as processed
            processed[idx] = True
            cluster.append(idx)

            # Queue for BFS
            queue = [idx]

            while len(queue) > 0:
                current_idx = queue.pop(0)
                # Find neighbors within the cluster tolerance distance
                [_, neighbor_indices, _] = tree.search_radius_vector_3d(points_np[current_idx], cluster_tol)
                for neighbor_idx in neighbor_indices:
                    if not processed[neighbor_idx]:
                        # Mark this point as processed
                        processed[neighbor_idx] = True
                        cluster.append(neighbor_idx)
                        queue.append(neighbor_idx)

            # Check if the cluster meets the size requirements
            if min_cluster_size <= len(cluster) <= max_cluster_size:
                cluster_indices.append(cluster)

    return cluster_indices

# Perform Euclidean clustering
cluster_indices = euclidean_clustering(cloud_outliers, cluster_tol=0.05, min_cluster_size=100, max_cluster_size=10000)

print(f"Number of clusters found: {len(cluster_indices)}")

# Visualize each cluster with a different color
colored_clusters = []
for i, indices in enumerate(cluster_indices):
    cluster = cloud_outliers.select_by_index(indices)
    cluster.paint_uniform_color([np.random.random(), np.random.random(), np.random.random()])
    colored_clusters.append(cluster)

# Concatenate the colored clusters and visualize
o3d.visualization.draw_geometries(colored_clusters, width=800, height=600)


# In[76]:


# Assuming you already have the 'cluster_indices' variable from previous code

# List to store individual cluster PointCloud objects
cluster_pointclouds = []

# Convert the 'cloud_outliers' PointCloud to numpy array (XYZRGB to XYZ)
cloud_points = np.asarray(cloud_outliers.points)

# Loop through each cluster indices
for cluster_indices in cluster_indices:
    # Extract the points of the current cluster using the indices
    cluster_points_np = cloud_points[cluster_indices]
    
    # Create an Open3D PointCloud object from the cluster points
    cluster_points_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(cluster_points_np))
    
    # Append the cluster PointCloud object to the list
    cluster_pointclouds.append(cluster_points_cloud)

# Now 'cluster_pointclouds' contains individual PointCloud objects for each cluster


# In[ ]:





# In[ ]:





# In[ ]:




