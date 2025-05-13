import numpy as np
import pickle
model_path =r'data/smpl/SMPL_NEUTRAL.pkl'
with open(model_path, 'rb') as smpl_file:
    kdata = pickle.load(smpl_file,encoding='latin1')
parents = np.zeros(29).astype('int')
parents[:24]= kdata['kintree_table'][0].copy()      # Joints in https://github.com/facebookresearch/frankmocap/blob/main/docs/joint_order.md#joint-angle-body
parents[0] = 0
parents[24] = 15
parents[25] = 22
parents[26] = 23
parents[27] = 10
parents[28] = 11

childrens = {i: [] for i in range(len(parents))}

for child, parent in enumerate(parents):
    if child != parent:  
        childrens[parent].append(child)

def get_neighbour_matrix_from_hand(parents,childrens,num_joints,num_edges,knn=2):
        '''
            neighbour_matrix: 最终的邻接矩阵，表示所有关节和边之间的邻接关系
            neighbour_matrix[num_joints:, num_joints:]: 仅表示边之间的邻接关系的子矩阵
            neighbour_matrix[:num_joints,:num_joints]: 仅表示关节之间的邻接关系的子矩阵
            neighbour_matrix_raw: 原始的邻接矩阵，未经过 knn 处理
        '''
        neighbour_matrix = np.zeros((num_joints+num_edges,num_joints+num_edges),dtype=np.float32)
        for idx in range(num_joints):
            neigbour = np.array([idx] + [parents[idx]]+childrens[idx])
            neigbour = neigbour[np.where(neigbour<num_joints)]
            neighbour_matrix[idx,neigbour] = 1
            if idx>0 and idx<=num_edges:
                neighbour_matrix[idx+num_joints-1,neigbour]=1
                neighbour_matrix[neigbour,idx+num_joints-1]=1
                neighbour_matrix[idx+num_joints-1,idx+num_joints-1]=1
        for i in range(num_joints):
            n = np.where(neighbour_matrix[i]==1)[0]
            n_edge = n[np.where(n>=num_joints)[0]]
            for edge in n_edge:
                neighbour_matrix[edge,n_edge]=1
        neighbour_matrix_raw = np.array(neighbour_matrix!=0, dtype=np.float32)
        if knn >= 2:
            neighbour_matrix = np.linalg.matrix_power(neighbour_matrix, knn)
            neighbour_matrix = np.array(neighbour_matrix!=0, dtype=np.float32)
        return neighbour_matrix,neighbour_matrix[num_joints:,num_joints:],neighbour_matrix[:num_joints,:num_joints],neighbour_matrix_raw