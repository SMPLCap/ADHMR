import os
import os.path as osp

# Train Scorer
train_scorer = True
scorer_config_file = 'config/hmrscorer.yaml'
pretrained_scorer_model_path = None
num_hypos = [15]
no_train_backbone = True

# will be update in exp
num_gpus = -1
exp_name = 'output/exp1/pre_analysis'

# quick access
save_epoch = 5
lr = 5e-4
end_epoch = 25

train_batch_size = 32

syncbn = True
bbox_ratio = 1.2

# continue
continue_train = True
start_over = False
pretrained_model_path = '../pretrained_models/smpler_x_b32.pth.tar'
# dataset setting
agora_fix_betas = True
agora_fix_global_orient_transl = True
agora_valid_root_pose = True
agora_skip_kid = True

# for ubody ft
dataset_list = ['Human36M', 'MSCOCO', 'MPII', 'AGORA', 'EHF', 'SynBody', 'GTA_Human2', \
    'EgoBody_Egocentric', 'EgoBody_Kinect', 'UBody', 'PW3D', 'MuCo', 'PROX']
trainset_3d = []
trainset_2d = []
trainset_humandata = ['HI4D', 'BEDLAM', 'RenBody_HiRes', 'GTA_Human2', 'SPEC']
testset = 'EHF'

use_cache = True
#no_aug = False
Motion_Adapter = 'Transformer'



# strategy 
data_strategy = 'balance' # 'balance' need to define total_data_len
total_data_len = 100000 # assign number or 'auto' for concat length


# fine-tune
smplx_loss_weight = 1.0 #2 for agora_model for smplx shape
smplx_pose_weight = 10.0

smplx_kps_3d_weight = 100.0
smplx_kps_2d_weight = 1.0
net_kps_2d_weight = 1.0

agora_benchmark = 'agora_model' # 'agora_model', 'test_only'

model_type = 'smpler_x_b'
encoder_config_file = 'transformer_utils/configs/smpler_x/encoder/body_encoder_base.py'
encoder_pretrained_model_path = '../pretrained_models/vitpose_base.pth'
feat_dim = 768


## =====FIXED ARGS============================================================
## model setting
upscale = 4
hand_pos_joint_num = 20
face_pos_joint_num = 72
num_task_token = 24
num_noise_sample = 0

## UBody setting
train_sample_interval = 10
test_sample_interval = 100
make_same_len = False

## input, output size
input_img_shape = (512, 384)
input_body_shape = (256, 192)
output_hm_shape = (16, 16, 12)
input_hand_shape = (256, 256)
output_hand_hm_shape = (16, 16, 16)
output_face_hm_shape = (8, 8, 8)
input_face_shape = (192, 192)
focal = (5000, 5000)  # virtual focal lengths
princpt = (input_body_shape[1] / 2, input_body_shape[0] / 2)  # virtual principal point position
body_3d_size = 2
hand_3d_size = 0.3
face_3d_size = 0.3
camera_3d_size = 2.5

## training config
print_iters = 20
lr_mult = 0.02
lr_step_model = [5]
lr_factor_model = 0.5
grad_clip = 1.0
ema_rate = 0.9999

## testing config
test_batch_size = 32

## others
num_thread = 2
vis = False
debug = False

## directory
output_dir, model_dir, vis_dir, log_dir, result_dir, code_dir = None, None, None, None, None, None
