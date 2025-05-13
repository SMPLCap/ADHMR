from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path as osp
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import sys
from easydict import EasyDict as edict
import yaml
from models.layers.smpl.SMPL import SMPL_layer
from termcolor import colored
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh
 

children = [[1, 4, 7], [2], [3], [], [5], [6], [], [8], [9, 11, 14],
                    [10], [], [12], [13], [], [15], [16], []]
flip_pairs = np.array([[1, 2], [4, 5], [7, 8], [10, 11], [13, 14], [16, 17], [18, 19], [20, 21], [22, 23], [25, 26], [27, 28]])
joints_left=flip_pairs[:,1]
joints_right=flip_pairs[:,0]
model_path =r'data/smpl/SMPL_NEUTRAL.pkl'
with open(model_path, 'rb') as smpl_file:
    db = pickle.load(smpl_file,encoding='latin1')
cleft= '#00BFFF'
cright='#FFE07D'
cmid= '#7FFFAA'

def load_yaml(path):
    with open(path,'rb') as fid:
        cfg = yaml.safe_load(fid)
    cfg = edict(cfg)
    return cfg

def load_smpl(gender):
    h36m_jregressor = np.load('lib/model_files/J_regressor_h36m.npy')
    smpl = SMPL_layer(
                'lib/model_files/SMPL_{}.pkl'.format(gender),
                h36m_jregressor=h36m_jregressor,
                dtype=torch.float32
            ) 
    return smpl
    
def get_output(root):
    output = {}
    name_list = os.listdir(root)
    for name in name_list:
        path = os.path.join(root,name)
        output[name.split('.')[0]] = np.load(path)
    return output

def render_mesh(height, width, meshes, face, cam_param):
    # renderer
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
   
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # camera
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # mesh
    for mesh in meshes:
        mesh = trimesh.Trimesh(mesh, face)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        # Manually apply the rotation to each vertex
        # vertices_homogeneous = np.hstack((mesh.vertices, np.ones((mesh.vertices.shape[0], 1))))
        # transformed_vertices = (vertices_homogeneous @ rot.T)[:, :3]
        # mesh.vertices = transformed_vertices

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene.add(mesh, 'mesh')

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    renderer.delete()
    return rgb, depth

union_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'lsho',
            12: 'lelb',
            13: 'lwri',
            14: 'rsho',
            15: 'relb',
            16: 'rwri'
        }
def get_flip_paris():
        """
        Get flip pair indices in union order.
        """
        # the same names in union and actual
        flip_pair_names = [['rank', 'lank'], ['rkne', 'lkne'], ['rhip', 'lhip'],
            ['rwri', 'lwri'], ['relb', 'lelb'], ['rsho', 'lsho']]
        union_keys = list(union_joints.keys())
        union_values = list(union_joints.values())

        flip_pairs = [[union_keys[union_values.index(name)] for name in pair] for pair in flip_pair_names]
        return flip_pairs
def get_parent(children):
    n=len(children)
    parent=np.arange(0,n,1,np.uint8)
    for i in range(n):
        for j in children[i]:
            parent[j]=i
    return parent
def fliplr_joints(joints, width, matched_parts, joints_vis=None, is_2d=True):
    """
    flip coords: 2d or 3d joints
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        if is_2d:
            joints_vis[pair[0], :], joints_vis[pair[1], :] = \
                joints_vis[pair[1], :], joints_vis[pair[0], :].copy()
    if is_2d:
        return joints * joints_vis, joints_vis

    return joints

def draw3Dpose(joints_3d,ax,num_joints,cl=200,gt=False):  # blue, orange
    parent_ids = get_parents(num_joints)
    joints_vis = np.ones((joints_3d.shape[0],3))
    X = joints_3d[:, 0]
    Y = joints_3d[:, 1]
    Z = joints_3d[:, 2]
    vis_X = joints_vis[:, 0]
    l = 1.5
    s=1.2
    if gt == 0: #gt
        c = 'g'
    elif gt == 1: #score
        c = 'b'
    elif gt == 2: #min mpjpe
        c = 'y'
    else:
        l = 1
        c = 'c'
        s = 1
    for i in range(0, joints_3d.shape[0]):
        if vis_X[i]:
            ax.scatter(X[i], Y[i], Z[i], c='g',s=s, marker='o')
        x = np.array([X[i], X[parent_ids[i]]], dtype=np.float32)
        y = np.array([Y[i], Y[parent_ids[i]]], dtype=np.float32)
        z = np.array([Z[i], Z[parent_ids[i]]], dtype=np.float32)
        
        ax.plot(x, y, z, c=c,linewidth=l)
    #ax.set_ylim3d(Y[0]-cl,Y[0]+cl)
    #ax.set_zlim3d(Z[0]-cl,Z[0]+cl)
    
    ax.set_title('3d pose')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()

def drawpose(data,num_joints,gt=0):
    parent_ids = get_parents(num_joints)
    for i in range(len(data)):
        color=cmid
        if joints_left.__contains__(i):
                color=cleft
        if joints_right.__contains__(i):
                color=cright
        if gt == 0: #gt
            color = 'g'
        elif gt == 1: #score
            color = 'b'
        elif gt == 2: #min mpjpe
            color = 'y'
        if parent_ids[i]!=i:
            plt.plot([data[i][0],data[parent_ids[i]][0]],[data[i][1],data[parent_ids[i]][1]],color=color,linewidth=1.5)
        #plt.scatter(data[i][0],data[i][1],3,color,4)

def draw(path,data,num_joints,mode=0):
    img = plt.imread(path)
    plt.imshow(img)
    if mode == 0:
        drawpose(data,num_joints=num_joints)
    else:
        for i in range(data.shape[0]):
            plt.scatter(data[i][0],data[i][1],s=4,c='r')
    plt.axis('off')
    plt.show()

def drawbbox(bbox):
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x1 = [bbox[0],bbox[1]]
    x2 = [bbox[0],bbox[3]]
    x3 = [bbox[2],bbox[1]]
    x4 = [bbox[2],bbox[3]]
    plt.plot([x1[0],x2[0]],[x1[1],x2[1]])
    plt.plot([x3[0],x4[0]],[x3[1],x4[1]])
    plt.plot([x1[0],x3[0]],[x1[1],x3[1]])
    plt.plot([x2[0],x4[0]],[x2[1],x4[1]])
    plt.scatter((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2)

def draw3D(data,num_joints):
    fig = plt.figure()
    ax=plt.axes(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(215,270)
    for i in range(data.shape[0]):
        if i==0:
            gt = True
        else:
            gt = False
        draw3Dpose(data[i],ax=ax,num_joints=num_joints,gt=gt)
    plt.gca().set_box_aspect((2, 3.5, 2))
    #ax.set_ylim3d(-400,+400)
    #ax.set_zlim3d(-400,+400)
    #ax.set_xlim3d(-400,+400)
    plt.show()
    plt.close() 

def draw2(path):
    img = plt.imread(path)
    plt.imshow(img)
    plt.show()

def get_parents(num_joints):
    if num_joints==17:
        #print(get_parent(children))
        return get_parent(children)
    else:
        parents = np.zeros(num_joints)
        parents = np.int64(parents)
        parent_ids= db['kintree_table'][0].copy()
        parent_ids[0] = 0
        if num_joints == 24:
            #print(parent_ids)
            return parent_ids
        else:
            parents[:24] = parent_ids
            parents[24] = 15
            parents[25] = 22
            parents[26] = 23
            parents[27] = 10
            parents[28] = 11
            #print(parents)
            return parents
