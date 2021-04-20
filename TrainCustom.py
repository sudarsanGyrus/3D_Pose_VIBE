# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.utils.smooth_pose import smooth_pose
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_cam_to_orig_img,
    convert_crop_coords_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)

MIN_NUM_FRAMES = 25

from PIL import Image
import time

from progress.bar import Bar
from lib.core.loss import VIBELoss2
from lib.utils.utils import  get_optimizer

from lib.dataset.customData import customDataset3D
# cfg, cfg_file = parse_args()
# cfg = prepare_output_dir(cfg, cfg_file)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = VIBE_Demo(
    seqlen=16,
    n_layers=2,
    hidden_size=1024,
    add_linear=True,
    use_residual=True,
).to(device)


#print('Model details',model.hmr)

for param in model.parameters():
    param.requires_grad = False
for param in model.hmr.parameters():
    param.requires_grad = True

pytorch_total_TR_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
pytorch_total_params = sum(p.numel() for p in model.parameters() )

print('TOTAL TRAINABLE parameters',pytorch_total_TR_params)
print('TOTAL parameters',pytorch_total_params)



## MY loss
loss = VIBELoss2(
    e_loss_weight=1.0,
    e_3d_loss_weight=50.0,
    e_pose_loss_weight=100.0,
)


gen_optimizer = get_optimizer(
    model=model,
    optim_type='Adam',
    lr=0.0001,
    weight_decay=0.0,
    momentum=0.9,
)



## EXTRA things bewlow



#import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cv2 import projectPoints

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

    #vals = np.reshape( channels, (32, -1) )
    vals = np.reshape( channels, (17, -1) )

    LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    connections = [[0, 1], [1, 2], [2, 14], [14, 3], [3, 4],
                   [4, 5], [6, 7], [7, 8], [8, 15], [15, 9],
                   [9, 10], [10, 11], [15, 14], [15, 12], [12, 13], [12, 16]]


    
    lines = []

    # Make connection matrix
    for ind, (i,j) in enumerate(connections):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        line = ax.plot(x,y, z,  lw=2, c=lcolor if LR[i] else rcolor)
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


import time

def visualize(pose, skeleton, img,t,f=[5000,5000]):
    """
    Initialize the 3D and 2D plots.
    """
    global lines, points, fig, plot_ax, img_ax, intrinsic_mat
    fig = plt.figure() 
    # 3D Pose Plot
    plot_ax = plt.subplot(121, projection='3d')

    plot_ax.view_init(elev=-77., azim=-88)
    lines = show3Dpose(pose, plot_ax)
    print('ax.azim {}'.format(plot_ax.azim))
    print('ax.elev {}'.format(plot_ax.elev))

        

    #plot_ax.view_init(-90, 60)


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
    points = img_ax.plot(proj2d[:,0], proj2d[:,1], 'ro')
    #print("inSide trCus,line314 ",points)
    for pt in range(proj2d.shape[0]):
        plt.annotate(f"{pt}", (proj2d[pt,0],proj2d[pt,1]))
    # Show the plot
    plt.show()
    print('ax.azim {}'.format(plot_ax.azim))
    print('ax.elev {}'.format(plot_ax.elev))



def main(args):

    total_time = time.time()

    # ========= Run tracking ========= #
    bbox_scale = 1.1

    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file,map_location=torch.device('cuda'))
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    # ========= Run VIBE on each person ========= #
    # print(f'Running VIBE on each tracklet...')
    # vibe_time = time.time()
    # vibe_results = {}
    for person_id in tqdm(list([1])):
        bboxes = joints2d = None

        # if args.tracking_method == 'bbox':
        #     bboxes = tracking_results[person_id]['bbox']
        # elif args.tracking_method == 'pose':
        #     joints2d = tracking_results[person_id]['joints2d']

        # frames = tracking_results[person_id]['frames']
        # print('Frame shape---===',frames)

        image_folder = '/home/ubuntu/gyrusWork/DATA'
        dataset = customDataset3D(
            image_folder=image_folder,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True # if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=1, num_workers=8)



        epoch = 0
        epochs = 10


        start = time.time()

        summary_string = ''

        #bar = Bar(f'Epoch {epoch + 1}/{epochs}', fill='#', max=10)


        for i in range(epochs):
            epoch = i
            dataloader = DataLoader(dataset, batch_size=1, num_workers=8)

            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

            for batch in dataloader:
                GT = {}
                if has_keypoints:
                    batch, nj2d, pose, bbox,raw_imgCrop,org_img,KP3d,cam_t = batch
                    #print('batch shape',batch.shape)
                    # img = Image.fromarray(np.asarray(batch[0]).reshape((224,224,3)), 'RGB')
                    # img.save(f"/home/ubuntu/gyrus/3D_pose/myimg{epoch}.jpg")
                    # time.sleep(5)
                    #norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)
                batch = batch.to(device)

                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]

                # pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                # pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                # pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))

                # pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                # pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))

                pred_cam = output['theta'][:, :, :3].reshape(batch_size * seqlen, -1)
                pred_verts = output['verts'].reshape(batch_size * seqlen, -1, 3)
                pred_joints3d = output['kp_3d'].reshape(batch_size * seqlen, -1, 3)
                
                #print('type----',type(pred_cam))


                # pred_cam = torch.cat(pred_cam, dim=0)
                # pred_verts = torch.cat(pred_verts, dim=0)
                # pred_pose = torch.cat(pred_pose, dim=0)
                # pred_betas = torch.cat(pred_betas, dim=0)
                # pred_joints3d = torch.cat(pred_joints3d, dim=0)

                pred_cam = pred_cam.cpu().detach().numpy()

                #print('gen output----------------',output['kp_3d'].shape)
                # print('gen output----------------',output['kp_2d'].shape)
                # print('gen output----------------',output['theta'].shape)

                w_smpl = torch.ones(seqlen).float()
                w_3d = torch.ones(seqlen).float()
                GT['w_smpl'] = w_smpl
                GT['w_3d'] = w_3d
                GT['kp_2d'] = nj2d
                GT['theta'] = pose
                GT['kp_3d'] = KP3d

                # kp2dX = {}

                #print('in trainCustom,line,408, output[2d ]shape',type(output['kp_2d'].cpu()),output['kp_2d'].cpu().shape )

                output['kp_2d'] = convert_crop_coords_to_orig_img(  #not orig image, convert to crop size
                    bbox=bbox,
                    keypoints=output['kp_2d'].cpu(),
                    crop_size=224,
                )

                for key in output:
                    output[key] = output[key].to(torch.device("cuda"))

                for key in GT:
                    GT[key] = GT[key].to(torch.device("cuda"))

                #GT = GT.to(torch.device("cuda"))

                #print('in trainCustom,line,408, output[2d ]shape',type(output['kp_2d']),output['kp_2d'].shape )

    
                # print('img height width---',org_img[0,:,:,0].numpy().shape)


                # orig_height, orig_width = org_img[0,:,:,0].numpy().shape

                # print('Type check',type(pred_cam),type(bbox),type(orig_height))

                # orig_cam = convert_crop_cam_to_orig_img(
                #         cam=pred_cam,
                #         bbox=bbox.numpy(),
                #         img_width=orig_width,
                #         img_height=orig_height
                #     )

                pred_camera = torch.from_numpy(pred_cam)
                pred_cam_t = torch.stack([pred_camera[:, 1],
                                        pred_camera[:, 2],
                                        2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)

     
                #print('**************************',GT['kp_2d'][0].shape)

                # cv_keypoints =[]
                # cv_keypoints1 =[]
                # for x,y in output['kp_2d'].cpu()[0,0,6:11].tolist():
                #     cv_keypoints.append(cv2.KeyPoint(x, y,10))
                # for x,y in GT['kp_2d'][0,6:11].tolist():
                #     cv_keypoints1.append(cv2.KeyPoint(x, y,10))

                # raw_img = raw_imgCrop[0].numpy() #.permute(1, 2, 0).numpy()
                # #print('raw image shape===',raw_img.shape)
                # #raw_img = org_img[0].numpy()
                # #print('org_imgimage shape===',raw_img.shape)
                # cv2.drawKeypoints(raw_img, cv_keypoints, raw_img, color=(255,0,0))
                # cv2.drawKeypoints(raw_img, cv_keypoints1, raw_img, color=(0,0,0))
                # cv2.imwrite(f"/home/ubuntu/gyrusWork/myimg{epoch}.jpg",raw_img)


                ### 3D visualization

                #print('debug%%%%',output['kp_3d'].shape,type(output['kp_3d'].cpu().detach().numpy()))

                #skel = np.zeros((32,3))

                #tmpx = pred_verts.cpu().detach().numpy()
                tmpx = pred_joints3d.cpu().detach().numpy()

                #print('debug output[verts] SHAPE',tmpx.shape,type(tmpx))

        
                # skel[USE_DIMS] = tmpx[0]#[H36M_IDS]
                skel = tmpx[0]

                #visualize(skel.reshape(-1), skel, raw_imgCrop[0].numpy(),t=pred_cam_t.numpy())

                target_2d = False
                target_3d = GT
                gen_loss = loss(
                    generator_outputs=output,
                    data_2d=target_2d,
                    data_3d=target_3d,
                )

                print("LOSS--------",gen_loss)
                print('GT[kp_3d] shape',GT['kp_3d'].shape)
                print('ord 3d key pt==',GT['kp_3d'][0,6:11],'   pred 3d key pt==',output['kp_3d'][0,0,6:11])

                # <======= Backprop generator and discriminator
                gen_optimizer.zero_grad()
                gen_loss.backward()
                gen_optimizer.step()
            del batch






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str,
                        help='input video path or youtube link')

    parser.add_argument('--output_folder', type=str,
                        help='output folder to write results')

    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=10,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--staf_dir', type=str, default='/home/mkocabas/developments/openposetrack',
                        help='path to directory STAF pose tracking method installed.')

    parser.add_argument('--vibe_batch_size', type=int, default=5,
                        help='batch size of VIBE')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--run_smplify', action='store_true',
                        help='run smplify for refining the results, you need pose tracking to enable it')

    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    parser.add_argument('--smooth', action='store_true',
                        help='smooth the results to prevent jitter')

    parser.add_argument('--smooth_min_cutoff', type=float, default=0.004,
                        help='one euro filter min cutoff. '
                             'Decreasing the minimum cutoff frequency decreases slow speed jitter')

    parser.add_argument('--smooth_beta', type=float, default=0.7,
                        help='one euro filter beta. '
                             'Increasing the speed coefficient(beta) decreases speed lag.')

    args = parser.parse_args()

    print('enter main FUNCTI')

    main(args)
