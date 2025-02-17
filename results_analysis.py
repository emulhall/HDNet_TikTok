import numpy as np
import argparse
import os
import cv2
import glob
from matplotlib import pyplot as plt
import open3d as o3d

def ParseCmdLineArguments():
	parser=argparse.ArgumentParser(description='HUMBI Pickle Creator')
	parser.add_argument('--view1', type=str, default='0000000')
	parser.add_argument('--view2', type=str, default='0000001')
	parser.add_argument('--time_stamp', type=str, default='00000001')
	parser.add_argument('--input', type=str, default='./test_data/infer_out')

	return parser.parse_args()

def load_view(ROOT, time_stamp, view):
	#Get the images at that time stamp and view
	depth=np.loadtxt(os.path.join(ROOT, time_stamp, view, view+'.txt'))
	return depth

def load_mask(time_stamp, view):
	mask=cv2.imread(os.path.join('../HUMBI_example/body', time_stamp, 'mask', 'image'+view+'.jpg'))
	return mask

def load_original(time_stamp, view):
	img=cv2.imread(os.path.join('../HUMBI_example/body', time_stamp, 'image', 'image'+view+'.jpg'))
	return img

def visualize_depth(depth, show=True, save=False, fname='depth.png'):
	depth_cp=np.copy(depth)
	min_depth = np.min(depth_cp[depth_cp>0])
	max_depth = np.max(depth_cp[depth_cp>0])
	depth_cp[depth_cp < min_depth] = 255.0
	depth_cp[depth_cp > max_depth] = max_depth

	if show:
		plt.imshow(depth_cp, cmap="hot")
		plt.show()

	if save:
		plt.imsave(fname, depth_cp, cmap="hot")


def calculate_scaling_and_origin(orig, depth):
	h=orig.shape[0]
	w=orig.shape[1]

	#Calculate the scaling factor
	size=depth.shape[0]
	scale_factor=size/max(h,w)

	width=int(w*scale_factor)
	height=int(h*scale_factor)

	x_offset = (size-width)//2
	y_offset = (size-height)//2

	origin=np.asarray([x_offset, y_offset])

	return scale_factor, origin


#X = scale*depth*K-1*[u;v;1]
#Build point cloud from depth and camera intrinsics
def depth_to_3D(depth, K, R, C, scale_factor, origin):
	ind=np.argwhere(depth>0) # Nx2
	y=ind[:,0]
	x=ind[:,1]

	valid_depth=depth[y,x]

	#Transform the xy locations from the scaled and cropped coordinate to the original coordinates
	#First, we have to build u = [x,y]
	u=np.hstack((np.reshape(x, (-1,1)), np.reshape(y, (-1,1)))) # Nx2
	u=(1/scale_factor)*(u-np.reshape(origin, (1,2))) # Nx2
	u=np.hstack((u, np.ones((u.shape[0],1)))) # Nx3

	#Convert to 3D points by multiplying by inverse K
	point=R.T@np.linalg.inv(K)@(valid_depth*u.T)+np.reshape(C, (3,1)) # 3 x N

	return point.T # N x 3

def project_3D(point, K, R, C, scale_factor, origin, size):
	output=np.zeros((size,size))
	#Set up the projection matrix
	P=R@np.hstack((np.eye(3), np.reshape(-C, (3,1)))) # 3 x 4

	#Add an additional rows of ones to the set of 3D points
	point=np.hstack((point, np.ones((point.shape[0],1)))) # N x 4
	u = K@P@point.T # 3 x N

	#Normalize
	depth=u[2,:]
	u=u[:2,:]/depth # 2 x N

	#Scale and translate
	u=scale_factor*u+np.reshape(origin, (2,1))

	#Convert indices to integers
	u=u.astype(int)

	#Ensure no points go outside of the canvas and that only valid_depth indices are kept
	valid=np.logical_and(np.logical_and(u[0,:]<size, u[1,:]<size), depth>0)
	valid_ind=np.argwhere(valid).flatten()

	u=u[:, valid]
	depth=depth[valid]

	#Fill in the depth at the valid indices
	output[u[1,:], u[0,:]]=depth

	return output


def visualize_3D(X, fname='test_point_cloud.ply'):
	pcd=o3d.geometry.PointCloud()
	pcd.points=o3d.utility.Vector3dVector(X)
	o3d.io.write_point_cloud(fname, pcd)

#Superimpose image 2 on top of image 1
def superimpose(f1, f2, output='superimposed.png'):
	img1=cv2.imread(f1)
	img2=cv2.imread(f2)

	dst=cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

	cv2.imwrite(output, dst)


if __name__ == '__main__':
	args=ParseCmdLineArguments()
	#Load an image from one view
	depth1=load_view(args.input, args.time_stamp, args.view1)
	depth2=load_view(args.input, args.time_stamp, args.view2)

	#Load masks
	#mask1=load_mask(args.time_stamp, args.view1)
	#mask2=load_mask(args.time_stamp, args.view2)

	#Load intrinsics and extrinsics
	intrinsics=np.load('../HUMBI_example/body/intrinsic_z.npy')
	extrinsics=np.load('../HUMBI_example/body/extrinsic_z.npy')

	K1=np.asarray(intrinsics[int(args.view1)], dtype='f')
	K2=np.asarray(intrinsics[int(args.view2)], dtype='f')

	P1=np.asarray(extrinsics[int(args.view1)], dtype='f')
	P2=np.asarray(extrinsics[int(args.view2)], dtype='f')

	###Calculate 3D points
	#Load original image
	orig1=load_original(args.time_stamp, args.view1)
	orig2=load_original(args.time_stamp, args.view2)

	scale_factor, origin=calculate_scaling_and_origin(orig1, depth1)

	points1=depth_to_3D(depth1, K1, P1[:,:3], P1[:,3], scale_factor, origin)
	points2=depth_to_3D(depth2, K2, P2[:,:3], P2[:,3], scale_factor, origin)

	#Visualize 3D points
	visualize_3D(points1, 'points1.ply')
	visualize_3D(points2, 'points2.ply')

	projection=project_3D(points2, K1, P1[:,:3], P1[:,3], scale_factor, origin, 256)

	#Visualize the projection
	visualize_depth(depth2, False, True, 'initial.png')
	visualize_depth(projection, False, True, 'reprojected.png')

	superimpose('./initial.png', './reprojected.png')

	
	#print(np.linalg.norm(depth2-projection))




	