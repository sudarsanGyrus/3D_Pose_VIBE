

import os
import cv2
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from lib.utils.smooth_bbox import get_all_bbox_params
from lib.data_utils.img_utils import get_single_image_crop_demo


from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS, H36M_TO_J17

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cv2 import projectPoints



# H36M_IDS = [0, 2, 5, 8, 1, 4, 7, 3, 12, 15, 24, 16, 18, 20, 17, 19, 21]
# USE_DIMS = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

# RADIUS = 1



# parent_indices = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
# child_indices = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1


''' GLOBAL VARIABLES '''
angle_idx = 0 # Bone angle to adjust
direction = 0 # Direction to rotate, (0 - x, 1 - y, 2 - z) for upper arm only
step = 3 # 3 degrees for step size
step_radian = step * np.pi / 180
local_system_map = {1:0, 3:0, 5:1, 7:1, 2:2, 4:3, 6:4, 8:5}
line_index_map = {1:11, 3:14, 5:4, 7:1, 2:12, 4:15, 6:5, 8:2}
parent_indices = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
child_indices = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
direction_name = ['x', 'y', 'z']

# translation vector of the camera
t = None
# focal length of the camera
f = None
# intrinsic matrix for camera projection
intrinsic_mat = None

# Objects for ploting
fig = None
plot_ax = None
img_ax = None
skeleton = None
lines = None 
points = None
RADIUS = 1 # Space around the subject

# hierarchical representation
local_systems = None
need_to_update_lc = False
bones_global = None
bones_local = None
angles = None

# file path
annotation_path = None
annotation = None
img_name = None

# some joint correspondence
index_list = [13, 14, 129, 145]
H36M_IDS = [0, 2, 5, 8, 1, 4, 7, 3, 12, 15, 24, 16, 18, 20, 17, 19, 21]
USE_DIMS = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

# keyboard inputs
bone_idx_keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
global_rot_key ='0'
inc_step_key = 'd'
dec_step_key = 'f'
ang_inc_key = 'up'
ang_dec_key = 'down'
ang_cw_key = 'right'
ang_ccw_key = 'left'
save_key = 'm'




def textED(key):
    return key.split('/')[-1]

def show3Dpose(channels, 
               ax, 
               lcolor="#3498db", 
               rcolor="#e74c3c", 
               add_labels=True,
               gt=False,
               pred=False,
               inv_z=False
               ): 

    vals = np.reshape( channels, (32, -1) )

    I   = parent_indices
    J   = child_indices
    LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    
    lines = []

    # Make connection matrix
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        line = ax.plot(x,y, z,  lw=2, c=lcolor if LR[i] else rcolor)
        #line = ax.plot(z,y, x,  lw=2, c=lcolor if LR[i] else rcolor)
        lines.append(line)

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]

    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
    ax.set_aspect('auto')

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)

    if inv_z:
        ax.invert_zaxis() 

    return lines

# def show3Dpose(channels, ax, radius=40, lcolor='#ff0000', rcolor='#0000ff'):
#     print('new 3d pose in work')
#     #vals = channels
#     vals = np.reshape( channels, (32, -1) )

#     connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
#                    [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
#                    [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

#     LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

#     lines = []

#     for ind, (i,j) in enumerate(connections):
#         x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
#         line = ax.plot(x, y, z, lw=2, c=lcolor if LR[ind] else rcolor)
#         lines.append(line)

#     RADIUS = radius  # space around the subject
#     xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
#     ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
#     ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
#     ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")

#     ax.set_aspect('auto')

#     # Get rid of the panes (actually, make them white)
#     white = (1.0, 1.0, 1.0, 0.0)
#     ax.w_xaxis.set_pane_color(white)
#     ax.w_yaxis.set_pane_color(white)

#     # Get rid of the lines in 3d
#     ax.w_xaxis.line.set_color(white)
#     ax.w_yaxis.line.set_color(white)
#     ax.w_zaxis.line.set_color(white)



#     return lines



def visualize(pose, skeleton, img,t,f=[5000,5000]):
    """
    Initialize the 3D and 2D plots.
    """
    global lines, points, fig, plot_ax, img_ax, intrinsic_mat
    fig = plt.figure() 
    # 3D Pose Plot
    plot_ax = plt.subplot(121, projection='3d')

    #lines = show3Dpose(pose, plot_ax)
    #fig.canvas.mpl_connect('key_press_event', press)
    plot_ax.set_title('1-9: limb selection, 0: global rotation, arrow keys: rotation')
    # Image Plot
    img_ax = plt.subplot(122)
    img_ax.imshow(img)
    intrinsic_mat = np.array([[f[0], 0.00e+00, float(img.shape[1])/2],
                              [0.00e+00, f[1], float(img.shape[0])/2],
                              [0.00e+00, 0.00e+00, 1.00e+00]])
    proj2d = projectPoints(skeleton, 
                           np.zeros((3)), 
                           t, 
                           intrinsic_mat, 
                           np.zeros((5))
                           )
    proj2d = proj2d[0].reshape(-1,2)

    #proj2d = kp_2d.reshape(-1,2)

    points = img_ax.plot(proj2d[:,0], proj2d[:,1], 'ro')

    for pt in range(proj2d.shape[0]):
        plt.annotate(f"{pt}", (proj2d[pt,0],proj2d[pt,1]))
    # Show the plot
    plt.show()


class customDataset3D(Dataset):
    def __init__(self, image_folder, bboxes=None, joints2d=None, scale=1.0, crop_size=224):
        self.image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]
        self.image_file_names = sorted(self.image_file_names)
        frames = [x for x in range(len(self.image_file_names))]
        self.image_file_names = np.array(self.image_file_names)[frames]
        print('#######################',self.image_file_names,'#########################')
        self.bboxes = bboxes

        self.scale = scale
        self.crop_size = crop_size
        self.frames = frames
        self.has_keypoints = True #if joints2d is not None else False

        #self.norm_joints2d = np.zeros_like(self.joints2d)


        self.annotation_path= image_folder + '/fitted.npy'
        self.data = np.load(self.annotation_path, encoding='latin1',allow_pickle=True).item()
        self.joints2d = [self.data[textED(key)]['p2d'] for key in self.image_file_names]
        self.pose = np.asarray([self.data[textED(key)]['fitting_params']['pose'] for key in self.image_file_names])


        # 3d keypoints

        self.skeleton_smpl = np.asarray([self.data[textED(key)]['fitting_params']['v'] for key in self.image_file_names])

        self.cam_t = np.asarray([self.data[textED(key)]['fitting_params']['cam_t'] for key in self.image_file_names])

        #print('&&&&&&&&&&&3D SHAPE------',self.pose.shape)

        
        joints2d = np.asarray(self.joints2d)
        ones = np.ones((joints2d.shape[0],joints2d.shape[1],3))
        ones[:,:,:2] = joints2d
        self.joints2d = ones #np.concatenate( (self.joints2d, np.ones( (len(self.joints2d),1) ) ), axis=1)
        #print('2d GT shape-----------',self.joints2d.shape)
        #print('2d GT -------------',joints2d)


        if self.has_keypoints :
            bboxes, time_pt1, time_pt2 = get_all_bbox_params(self.joints2d, vis_thresh=0.3)
            #print('box shape',bboxes)
            bboxes[:, 2:] = 150. / bboxes[:, 2:]
            self.bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T
            #print('bboxes---',self.bboxes)
            self.image_file_names = self.image_file_names[time_pt1:time_pt2]
            self.joints2d = joints2d[time_pt1:time_pt2]
            self.frames = frames[time_pt1:time_pt2]

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_file_names[idx]), cv2.COLOR_BGR2RGB)
        #cv2.imwrite(f"/home/ubuntu/gyrus/3D_pose/myimg{idx}.jpg",img)
        bbox = self.bboxes[idx]

        j2d = self.joints2d[idx] if self.has_keypoints else None

        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=j2d,
            scale=self.scale,
            crop_size=self.crop_size)

        #print('inside custom data',j2d.shape,kp_2d.shape)
        #print('crop 2d points==',kp_2d[:2],'org 2d key pt',j2d[:2])
        #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$2d key point shaep---',kp_2d.tolist())
        # cv_keypoints =[]
        # for x,y in kp_2d.tolist():
        #     cv_keypoints.append(cv2.KeyPoint(x, y,10))
        # cv2.drawKeypoints(raw_img, cv_keypoints, raw_img, color=(255,0,0))
        # c=0 
        # for x,y in kp_2d.tolist():
        #     cv2.putText(raw_img, f"{c}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,150,250), 1)
        #     c+=1
        # cv2.imwrite(f"/home/ubuntu/gyrusWork/myimg{idx}.jpg",raw_img)



        self.skeleton_smpl = self.skeleton_smpl[idx].reshape(-1,3)

        print('v shape in custom data####',self.skeleton_smpl.shape)
        skeleton1 = np.zeros((32,3))

    
        #skeleton1[USE_DIMS] = self.skeleton_smpl[H36M_IDS]

        H36M_IDS = [8,5,2,1,4,7,21,19,17,16,18,20,12,15,3,12,15]

        print('skeleton shape',self.skeleton_smpl.shape)

        skeleton = self.skeleton_smpl[H36M_IDS]

        #print('$$$$$3d sekelon shape',skeleton.reshape(-1).shape)

        #print('debug2%%%%',img.shape,type(img))

        visualize(skeleton.reshape(-1), skeleton, img,t=self.cam_t[idx])

        KP3d = skeleton.reshape(-1,3)
        #print('xxxxxxxxxxxxxxxxxxx',KP3d.shape)


        if self.has_keypoints:
            return norm_img, kp_2d, self.pose[idx], bbox, raw_img, img,KP3d,self.cam_t[idx]
        else:
            return norm_img


def press(event):
    """
    Call-back function when user press any key.
    """
    global angle_idx, direction, need_to_update_lc
    global bones_global, bones_local, skeleton, angles, local_systems

    if event.key == 'p':
        plot_ax.plot([np.random.rand()], [np.random.rand()], [np.random.rand()], 'ro')
        fig.canvas.draw()

    if event.key in bone_idx_keys:  angle_idx = int(event.key) - 1 
    if event.key == global_rot_key: angle_idx = None
    if event.key == inc_step_key:   direction = (direction + 1) % 3
    if event.key == dec_step_key:   direction = (direction - 1) % 3

    if event.key == ang_inc_key or event.key == ang_dec_key:
        update_skeleton(angle_idx, event.key)

    if event.key == ang_cw_key or event.key == ang_ccw_key:
        if angle_idx in [2, 4, 6, 8]:
            update_skeleton(angle_idx, event.key)

    if event.key == save_key:
        save_skeleton()

    if angle_idx is not None:
        notes = 'current limb: ' + bone_name[angle_idx + 1]
        # update local coordinate systems if needed
        if need_to_update_lc:
            # compute the local coordinate system
            bones_global, bones_local, local_systems = to_local(skeleton)
            # convert the local coordinates into spherical coordinates
            angles = to_spherical(bones_local)
            angles[:,1:] *= 180/np.pi
            # need to update local coordinate system once after global rotation
            need_to_update_lc = False            
    else:
        notes = 'global rotation: '

    if angle_idx in [None, 1, 3, 5, 7]:
        notes += ' direction: ' + direction_name[direction]
    plot_ax.set_xlabel(notes)


