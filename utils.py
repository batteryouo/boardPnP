import math
import numpy as np

def read_disp_from_bin(filename):
	with open(filename, 'rb') as f:
		# Read rows, cols, and type information (assuming they were saved as int32)
		rows = np.fromfile(f, dtype=np.int32, count=1)[0]
		cols = np.fromfile(f, dtype=np.int32, count=1)[0]
		mat_type = np.fromfile(f, dtype=np.int32, count=1)[0]
		
		# Read the actual matrix data (assuming float32 type for CV_32FC1)
		mat_data = np.fromfile(f, dtype=np.float32, count=rows * cols)
		
		# Reshape the data back into a 2D array
		disp_data = mat_data.reshape((rows, cols))
	
	return disp_data

def disp2depth(disp, a = -23.729979928848525, b = -0.014739943832346883):
	return 1 / (a * disp + b)

def read_cali(file_path):
	calibration = {}
	with open(file_path, 'r') as f:

		for cali_info in f.readlines():
			key, data = cali_info.split(": ", 1)
			# data = data.split()
			# print(np.array(data))
			calibration[key] = data
			calibration[key] = np.array(list(map(float, data.split(' '))))
	return calibration

def normalize_depth_image(img, shape=(720, 1280)):
	img = img.reshape(shape)
	mask = np.isinf(img) | np.isnan(img)
	# print(np.min(img))
	# print(np.max(img))
	img[mask] = np.min(img[~mask])
	img = (img - np.min(img)) / (np.max(img)-np.min(img)) * 255
	img = img.astype(np.uint8)

	return img
def read_lidar_bin(file_path, neg_depth_filter=False):
	points = []
	lidar_data = np.fromfile(file_path, dtype=np.float32)
	data_len = len(lidar_data)
	resolution = 360 / data_len
	for i in range(data_len):
		if math.isnan(lidar_data[i]) or math.isinf(lidar_data[i]):
			continue
		angle = (resolution * i) * np.pi/180
		x = lidar_data[i] * np.cos(angle)
		y = lidar_data[i] * np.sin(angle)
		z = 0

		points.append([x, y, z])

	points = np.array(points)
	points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)

	return points if not neg_depth_filter else points[points[:, 2] > 0]

def least_squares_fit_stable(p, q, regularization=0.0):
	"""
	Solves for a, b, c, d that minimize the squared error, handling singular matrices.

	Parameters:
	- p: Lists or numpy arrays of the predictors.
	- q: List or numpy array of the target values.
	- regularization: Regularization parameter (lambda). Default is 0 (no regularization).

	Returns:
	- theta: A numpy array [a, b, c, d].
	"""

	# Construct the matrix P (n x 4, where n is the number of data points)
	n = len(p)
	P = np.column_stack((p, np.ones(n)))
	# print(P)
	if regularization == 0.0:
		# Use the pseudoinverse if no regularization is specified
		theta = np.linalg.pinv(P) @ q
	else:
		# Apply Ridge Regression (Regularized Least Squares)
		I = np.eye(P.shape[1])  # Identity matrix of size 4x4
		theta = np.linalg.inv(P.T @ P + regularization * I) @ (P.T @ q)
	
	# for i in range(n):
	# 	print(theta[0] * P[i, 0] + theta[1] * P[i, 1] + theta[2] * P[i, 2] + theta[3] * P[i, 3])

	return theta

class Projection():
	'''
	This class provides functions to project depth maps to point clouds and vice versa.
	It includes methods for converting between depth maps, point clouds, and RGBD images, 
	as well as reprojecting point clouds between camera coordinate systems using intrinsic and extrinsic matrices.
	'''
	def __init__(self):
		self._depth_map = None
		self._pointClouds = None
		self._RGBD_img = None
		self._pointClouds_RGB = None

	def depthMap2PointClouds(self, depth_map, K=None):	
		"""
		Converts a depth map into point clouds.

		Parameters:
			depth_map (np.ndarray): Depth map with shape (H, W).
			K (np.ndarray): Intrinsic matrix of the camera with shape (3, 3).

		Returns:
			np.ndarray: Point clouds with shape (N, 3).
		"""

		assert len(depth_map.shape)==2, "The shape of the depth map needs to be (rows, cols)."
		
		self._pointClouds = self.__depthMap2PointClouds(depth_map=depth_map, K=K)

		return self._pointClouds
	
	def RGBD2PointClouds(self, img, K):
		"""
		Converts an RGBD image into point clouds.

		Parameters:
			img (np.ndarray): RGBD image with shape (H, W, 4), where the last channel is the depth map.
			K (np.ndarray): Intrinsic matrix of the camera with shape (3, 3).

		Returns:
			tuple: 
				- np.ndarray: Point clouds with shape (N, 3).
				- np.ndarray: Colors corresponding to the point clouds with shape (N, 3).
		"""
		assert len(img.shape)==3, "The shape of the img needs to be (rows, cols, 4)."
		assert img.shape[2]==4, "The shape of the img needs to be (rows, cols, 4)."

		depth_map = img[:, :, 3]
		depth = depth_map.reshape(-1)
		pointClouds = self.__depthMap2PointClouds(depth_map=depth_map, K=K)

		colors = img[:, :, 0:3].reshape(-1, 3) / 255
		colors = np.stack((colors[:, 2], colors[:, 1], colors[:, 0]), axis=-1)

		self._pointClouds_RGB = pointClouds, colors[~(np.isnan(depth) | (depth <= 0))]

		return self._pointClouds_RGB

	def pointClouds2DepthMap(self, pointClouds, K, img_shape):
		"""
		Converts point clouds to a depth map.

		Parameters:
			pointClouds (np.ndarray): Point clouds with shape (N, 3).
			K (np.ndarray): Intrinsic matrix of the camera with shape (3, 3).
			img_shape (tuple): The shape of the target image (height, width).

		Returns:
			np.ndarray: Depth map with shape (H, W).
		"""
		assert len(pointClouds.shape) == 2, "PointClouds shape needs to be (n, 3)"

		x_indices, y_indices, mask = self.__pointClouds2Image(pointClouds, K, img_shape)

		self._depth_map = np.full(img_shape, np.nan)
		self._depth_map[x_indices[mask], y_indices[mask]] = pointClouds[:, 2][mask]
		
		return self._depth_map

	def colorPointClouds2RGBD(self, pointClouds, colors, K, img_shape):
		"""
		Converts colored point clouds into an RGBD image.

		Parameters:
			pointClouds (np.ndarray): Point clouds with shape (N, 3).
			colors (np.ndarray): Colors corresponding to the point clouds with shape (N, 3).
			K (np.ndarray): Intrinsic matrix of the camera with shape (3, 3).
			img_shape (tuple): The shape of the target image (height, width).

		Returns:
			np.ndarray: RGBD image with shape (H, W, 4).
		"""
		assert len(pointClouds.shape) == 2, "PointClouds shape needs to be (n, 3)"
		
		x_indices, y_indices, mask = self.__pointClouds2Image(pointClouds, K, img_shape)

		self._RGBD_img = np.full((img_shape[0], img_shape[1], 4), np.array([0, 0, 0, np.nan]))

		cv_Color = np.stack((colors[:, 2], colors[:, 1], colors[:, 0]), axis=-1) * 255
		self._RGBD_img[x_indices[mask], y_indices[mask], 3] = pointClouds[:, 2][mask]
		self._RGBD_img[x_indices[mask], y_indices[mask], 0:3] = cv_Color[mask]
		
		return self._RGBD_img
	
	def transformation(self, pointClouds, R, T):
		"""
		Applies a transformation to the point clouds using a rotation matrix and a translation vector.

		Parameters:
			pointClouds (np.ndarray): Point clouds with shape (N, 3), where each row represents a 3D point (x, y, z).
			R (np.ndarray): Rotation matrix with shape (3, 3).
			T (np.ndarray): Translation vector with shape (3, 1) or (1, 3).

		Returns:
			np.ndarray: Transformed point clouds with shape (N, 3).
		"""
		return np.dot(pointClouds, R.T) + T.T
	
	def __depthMap2PointClouds(self, depth_map, K):
		"""
		Helper function that converts a depth map into point clouds.

		Parameters:
			depth_map (np.ndarray): Depth map with shape (H, W).
			K (np.ndarray): Intrinsic matrix of the camera with shape (3, 3).

		Returns:
			np.ndarray: Point clouds with shape (N, 3).
		"""
		K_inv = np.linalg.inv(K)
		x, y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))

		"""
		get image position array

		Example:
			pos = [[0, 0, 1], [1, 0, 1], [2, 0, 1], ..., [639, 0, 1]
				  [0, 1, 1], [1, 1, 1], [2, 1, 1], ..., [639, 1, 1]
				  ...
				  [0, 359, 1], [1, 359, 1], [2, 359, 1], ...[639, 359, 1]]
		"""	
		pos = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float32).reshape((-1, 3))
		
		depth = depth_map.reshape(-1)
		invalid_depth = np.isnan(depth) | (depth <= 0)
		depth[invalid_depth] = np.array(-1)
		
		# Multiply by depth to create a homogeneous coordinate matrix with depth.
		pos[:, :] = depth[:, np.newaxis] * pos[:, :]
		"""	
		P_w = K^-1 * P_h
		because the shape of pos here is n*3, the matrix product need to switch positions

		original:
		P_w = K^-1 * P_h ((3*1) = (3*3) * (3*1))
		To speed up computation, the array needs to be vectorize, so the shape will become n*3
		->
		Modify
		P_w = P_h^T * K^-1^T  ((AB)^T = B^T * A^T)
		((n*3) = (n*3) * (3*3))
		n mean there're n points
		"""
		pointClouds = np.dot(pos, K_inv.T)[~invalid_depth]
		return pointClouds	

	def __pointClouds2Image(self, pointClouds, K, img_shape:tuple):
		"""
		Converts point clouds back to image coordinates using the intrinsic matrix K.

		Parameters:
			pointClouds (np.ndarray): Point clouds with shape (N, 3).
			K (np.ndarray): Intrinsic matrix of the camera with shape (3, 3).
			img_shape (tuple): The shape of the target image (height, width).

		Returns:
			tuple: 
				- np.ndarray: x indices of the projected points.
				- np.ndarray: y indices of the projected points.
				- np.ndarray: Mask indicating valid points within the image bounds.
		"""
		img = np.dot(pointClouds, K.T)	
		neg_depth_mask = img[:, 2] <= 0
		img[neg_depth_mask][:, 2] = -1	

		'''
		normalize:
		P_new_projection_on_image = [[u_x * z], [u_y * z], [z]] = z[[u_x], [u_y], [1]]
		'''	
		for i in range(3):
			img[:,i] = img[:,i]/img[:,2]	

		x_indices = img[:,1].astype(int)
		y_indices = img[:,0].astype(int)

		# get the mask of the points that are out of the range.
		mask = (x_indices >= 0) & (x_indices < img_shape[0]) & (y_indices >= 0) & (y_indices < img_shape[1])
		mask = mask & (~neg_depth_mask)

		return x_indices, y_indices, mask
	
	

class ImageReprojection(Projection):
	'''
	This class provides functions to reproject point clouds from one camera coordinate system to another.
	It extends the `Projection` class by incorporating transformations between the source and target camera frames.
	'''
	def __init__(self, source_img=None, sourceImg_K=None, targetImg_K=None, R=None, T=None, disp = None, depth_map = None, disp_a = -23.729979928848525, disp_b = -0.014739943832346883):
		"""
		Initializes the ImageReprojection object with source image, intrinsic and extrinsic matrices, and depth information.

		Parameters:
			source_img (np.ndarray): Source image to reproject.
			sourceImg_K (np.ndarray): Intrinsic matrix of the source camera.
			targetImg_K (np.ndarray): Intrinsic matrix of the target camera.
			R (np.ndarray): Rotation matrix to align source and target cameras.
			T (np.ndarray): Translation vector to align source and target cameras.
			disp (np.ndarray): Disparity map (optional, used if depth_map is not provided).
			depth_map (np.ndarray): Depth map (optional, used if disp is not provided).
			disp_a (float): Parameter for converting disparity to depth.
			disp_b (float): Parameter for converting disparity to depth.
		"""
		super().__init__()
		self.source_img = source_img
		self.sourceImg_K = sourceImg_K
		self.targetImg_K = targetImg_K
		self.R = R
		self.T = T
		self.disp = disp
		self._mask = None

		assert (disp is None) or (depth_map is None), "You cannot enter depth_map and disp at the same time"

		if (depth_map is None) and (disp is not None):
			self.depth_map = self.disp2depth(disp/1000, disp_a, disp_b) * 1000
		else:
			self.depth_map = depth_map

		self.target_RGBD_img = None

	def disp2depth(self, disp, a = -23.729979928848525, b = -0.014739943832346883):
		return 1 / (a * disp + b)
	
	def reprojection(self, source_img=None, sourceImg_K=None, targetImg_K=None, R=None, T=None, disp=None, depth_map=None, disp_a = -23.729979928848525, disp_b = -0.014739943832346883):	
		self.source_img = source_img if source_img is not None else self.source_img
		self.sourceImg_K = sourceImg_K if sourceImg_K is not None else self.sourceImg_K
		self.targetImg_K = targetImg_K if targetImg_K is not None else self.targetImg_K
		self.R = R if R is not None else self.R
		self.T = T if T is not None else self.T
		self.disp = disp if disp is not None else self.disp

		assert (self.disp is None) or (self.depth_map is None), "You cannot enter depth_map and disp at the same time"

		if (depth_map is None) and (disp is not None):
			self.depth_map = self.disp2depth(disp/1000, disp_a, disp_b) * 1000
		else:
			self.depth_map = depth_map	

		assert self.source_img is not None, "source_img is not initialized."
		assert self.sourceImg_K is not None, "sourceImg_K is not initialized."
		assert self.targetImg_K is not None, "targetImg_K is not initialized."
		assert self.R is not None, "R is not initialized."
		assert self.T is not None, "T is not initialized."	
		
		# Create RGBD representation of the source image and point cloud	
		source_RGBD_img = np.concatenate((self.source_img, self.depth_map.reshape((source_img.shape[0], source_img.shape[1], 1))), axis=-1)
		sourcePointClouds, source_colors = self.RGBD2PointClouds(source_RGBD_img, sourceImg_K)

		# Transform point cloud from source to target camera frame using rotation and translation
		targetPointClouds = self.transformation(sourcePointClouds, self.R, self.T)

		target_RGBD_img = self.colorPointClouds2RGBD(targetPointClouds, source_colors, targetImg_K, (source_img.shape[0], source_img.shape[1]))
		self.target_RGBD_img = target_RGBD_img
		# Extract the final target image, apply masks for invalid points, and return the result
		target_img = target_RGBD_img[:, :, 0:3].astype(np.uint8)
		self._mask = np.isnan(target_RGBD_img[:, :, 3])

		target_img[self._mask] = np.array([0, 0, 0]).astype(np.uint8) 

		return target_img

class Fusion(Projection):
	'''
	
	'''
	def __init__(self):
		super().__init__()
		self._depth_map = None
		self._pointClouds = None
		self._RGBD_img = None
		self._pointClouds_RGB = None
	
	def pointClouds2DepthMap(self, pointClouds, K, img_shape):
		"""_summary_

		Args:
			pointClouds (_type_): _description_
			K (_type_): _description_
			img_shape (_type_): _description_

		Returns:
			_type_: _description_
		"""
		x_indices, y_indices, mask = self.__pointClouds2Image(pointClouds=pointClouds, K=K, img_shape=img_shape)
		return x_indices, y_indices, mask

	def pos2PointClouds(self, pos, depth, K):
		inv_K = np.linalg.inv(K)
		return_pos = depth[:, np.newaxis] * pos[:, :]
		return_pos = np.dot(return_pos, inv_K.T)
		return return_pos

	def __pointClouds2Image(self, pointClouds, K, img_shape:tuple):
		"""
		Converts point clouds back to image coordinates using the intrinsic matrix K.

		Parameters:
			pointClouds (np.ndarray): Point clouds with shape (N, 3).
			K (np.ndarray): Intrinsic matrix of the camera with shape (3, 3).
			img_shape (tuple): The shape of the target image (height, width).

		Returns:
			tuple: 
				- np.ndarray: x indices of the projected points.
				- np.ndarray: y indices of the projected points.
				- np.ndarray: Mask indicating valid points within the image bounds.
		"""
		img = np.dot(pointClouds, K.T)	
		neg_depth_mask = img[:, 2] <= 0
		img[neg_depth_mask][:, 2] = -1	

		'''
		normalize:
		P_new_projection_on_image = [[u_x * z], [u_y * z], [z]] = z[[u_x], [u_y], [1]]
		'''	
		for i in range(3):
			img[:,i] = img[:,i]/img[:,2]	
		
		x_indices = img[:,1].astype(int)
		y_indices = img[:,0].astype(int)

		# get the mask of the points that are out of the range.
		mask = (x_indices >= 0) & (x_indices < img_shape[0]) & (y_indices >= 0) & (y_indices < img_shape[1])
		mask = mask & (~neg_depth_mask)

		return x_indices, y_indices, mask