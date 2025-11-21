import os
import copy
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import seaborn as sns
import open3d as o3d
import utils

count = 0
pts = []
flag = False

holeCoord_map = {10: [600, 500 , 0], 11: [1500,  500, 0], 12: [2400,  500, 0],
                  1: [600, 1000, 0],  2: [1500, 1000, 0],  3: [2400, 1000, 0],
                  4: [600, 1600, 0],  5: [1500, 1600, 0],  6: [2400, 1600, 0],
                  7: [600, 2200, 0],  8: [1500, 2200, 0],  9: [2400, 2200, 0]}

def mouse_callback(event, x, y, flags, param):
    global count, pts, flag    
    if event == cv2.EVENT_LBUTTONDOWN:
        if count < param and not flag:
            pts.append((x, y))
            count += 1
            
        flag = True
    elif event == cv2.EVENT_LBUTTONUP:
        flag = False
    
    if count == param:
        cv2.destroyAllWindows()

root_dir = os.path.curdir
root_dir = "D:\\tmp\\preprocessing_file"
cali_info = utils.read_cali(os.path.join(root_dir, "calib_cam.txt"))

# camera matrix
cam1_K = cali_info["K_01"].reshape((3,3))
cam1_D = cali_info["D_01"].reshape((1, 5))

inv_cam1_K = np.linalg.inv(cam1_K)

print(cam1_K)
print(cam1_D)


img_dir = os.path.join(root_dir, "realsenseRawfile", "image_rect_raw")
depth_dir = os.path.join(root_dir, "realsenseRawfile", "depth")
imu_dir = os.path.join(root_dir, "realsenseRawfile", "imu")
depth_files = os.listdir(depth_dir)
img_files = os.listdir(img_dir)
imu_files = os.listdir(imu_dir)

ind = 500

img_file = os.path.join(img_dir, img_files[ind])
depth_file = os.path.join(depth_dir, depth_files[ind])
imu_file = os.path.join(imu_dir, imu_files[ind])

img_data = cv2.imread(img_file)
depth_data:np.ndarray = np.load(depth_file)
imu_data:np.ndarray = np.load(imu_file)

mask = (depth_data > 10000) #| (edge_image == 255)

depth_data[mask] = 0
invalid_mask = np.isnan(depth_data) | (depth_data <= 0)

print(imu_data)

depth_img = utils.normalize_depth_image(depth_data)

cv2.namedWindow("depth", 0)
cv2.imshow("depth", depth_img)
cv2.namedWindow("img", 0)
cv2.setMouseCallback("img", mouse_callback, param=9)

cv2.imshow("img", img_data)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_pts = np.array(pts, dtype=np.float32)


obj_pts = np.array([holeCoord_map[i+1] for i in range(len(pts))], dtype=np.float32)
print(obj_pts)

success, rvec, tvec = cv2.solvePnP(
    obj_pts, 
    img_pts, 
    cam1_K, 
    cam1_D, 
    flags=cv2.SOLVEPNP_ITERATIVE # or cv2.SOLVEPNP_ITERATIVE
)

# --- 3. Process Outputs ---
if success:
    print(f"✅ Success: {success}")
    print(f"Rotation Vector (rvec):\n{rvec}")
    print(f"Translation Vector (tvec):\n{tvec}")

    # Convert rotation vector to a 3x3 rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    print(f"Rotation Matrix (R):\n{R}")
else:
    print("❌ solvePnP failed to find a solution.")

T = tvec.reshape((1, 3))
test_pt = obj_pts[0] + T
print(test_pt)
R = np.array([[1, 1, 0], [2, 1, 0], [0, 0, 1]])
print(np.dot((obj_pts[0] + T), R.T))

count = 0
pts = []
flag = False
cv2.namedWindow("img", 0)
cv2.setMouseCallback("img", mouse_callback, param=4)

cv2.imshow("img", img_data)
cv2.waitKey(0)
cv2.destroyAllWindows()

pts = np.array(pts)
pts = pts.reshape((-1, 1, 2))
img_gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
poly_img = np.zeros_like(img_gray, dtype=np.uint8)
cv2.fillPoly(poly_img, [pts], 255, 8)

poly_mask = poly_img == 255
invalid_mask = invalid_mask | ~(poly_mask)
depth_data[invalid_mask] = 0
print(depth_data[~invalid_mask])
depth_data[~invalid_mask] = tvec[2]
print(depth_data[~invalid_mask])
# invalid_mask = invalid_mask | ~std_mask
# depth_data[invalid_mask] = 0
# depth_mean = np.mean(depth_data[~invalid_mask])
# print(depth_mean)


imageReprojection = utils.ImageReprojection(sourceImg_K=cam1_K, targetImg_K=cam1_K, R=R, T=tvec.reshape((3, 1)))

gt_pointClouds = imageReprojection.depthMap2PointClouds(depth_data, K=cam1_K)

zero_mask = (gt_pointClouds[:,2] == 0)

color = img_data[~invalid_mask]
rgb_color = color[:, ::-1]
print(gt_pointClouds.shape)
print(rgb_color.shape)

print(gt_pointClouds[~zero_mask][0])
# Create an Open3D point cloud object
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(gt_pointClouds[~zero_mask] / 4000)
pcd1.colors = o3d.utility.Vector3dVector(rgb_color.astype(float) / 255.0)

coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
# app = o3d.visualization.gui.Application.instance
# app.initialize()

# win = o3d.visualization.gui.Application.instance.create_window("pclWin", 1024, 768)
# scene_widget = o3d.visualization.gui.SceneWidget()
# scene_widget.scene = o3d.visualization.rendering.Open3DScene(win.renderer)
# win.add_child(scene_widget)


o3d.visualization.draw_geometries([pcd1, coord_frame], window_name="pts")
# material = o3d.visualization.rendering.MaterialRecord()
# material.base_color = [1.0, 1.0, 1.0, 1.0]

# scene_widget.scene.add_geometry("pcd1", pcd1, material)

# app.run()