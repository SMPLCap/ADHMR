import copy
import pprint
import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import random
import datetime

import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
import _init_paths
from runners.hyponet import HypoNetTrainer
from runners.scorenet import ScorenetTrainer

def update_config(args, config):
    config.log_path = args.log_path
    config.save_path = args.save_path
    config.exp = args.exp
    config.doc = args.doc
    config.validate = args.validate
    config.sampling.multihypo_n = args.multihypo_n
    config.sampling.batch_size = args.batch_size
    config.training.resume_training = args.resume_training
    config.training.resume_ckpt = args.resume_ckpt
    config.validate = args.validate

    return config

def parse_args_and_config():
    
    # args attribute
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    
    parser.add_argument("--seed", type=int, default=123, help="random seed to use. Default=123")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file") 
    parser.add_argument("--exp", type=str, required = True, help="Path for saving running related data.")
    parser.add_argument(                                                                     
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    
    
    # process config
    args, rest = parser.parse_known_args()
    with open(args.config,"r") as f:
        config = yaml.safe_load(f)
    config = edict(config)
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Whether to validate",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Whether to inference",
    )
    parser.add_argument(
        "--multihypo_n",
        type = int,
        default = config.sampling.multihypo_n,
        help="Sample nums",
    )
    parser.add_argument(
        "--batch_size",
        type = int,
        default = config.sampling.batch_size,
        help="Sample batch_size",
    )
    parser.add_argument(
        "--resume_training",
        action="store_true",
        default=config.training.resume_training,
        help="Whether to resume training"
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=config.training.resume_ckpt,
        help='The checkpoint to resume.'
    )
    parser.add_argument(
        "--construct_dpo",
        action="store_true",
        help="Construct DPO data pair when testing",
    )
    parser.add_argument(
        "--beta_dpo",
        type = int,
        default = config.training.get('beta_dpo', 5000),
        help="Sample nums",
    )
    parser.add_argument(
        "--lambda_d_kto",
        type=float,
        default = config.training.get('lambda_d_kto', 1.0),
        help="KTO lambda_d",
    )
    parser.add_argument(
        "--lambda_u_kto",
        type=float,
        default = config.training.get('lambda_u_kto', 1.0),
        help="KTO lambda_u",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    parser.add_argument(
        "--local_rank",
        type = int,
        default = 0,
    )

    args = parser.parse_args()

    args.validate or args.inference

    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(time_str)
    output_dir = os.path.join(args.exp, f'{args.doc}_{time_str}')
    args.log_path = os.path.join(output_dir, "ckpt")
    args.code_path = os.path.join(output_dir, "code")
    args.save_path = os.path.join(output_dir, "output")
    args.config_path = os.path.join(output_dir, "cfg")

    if os.path.exists(args.log_path):
        print('overwrite the folder')
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.code_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.config_path, exist_ok=True)
    
    config= update_config(args, config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config.device= 'cuda' if torch.cuda.is_available() else 'cpu'
    config_dump = dict(config)
    with open(os.path.join(args.config_path,'config.yaml'),'w') as config_file:
        yaml.dump(config_dump,config_file)
    
    args.local_rank = int(os.environ["LOCAL_RANK"])

    return args, config

def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return True

def main():
    args, cfg = parse_args_and_config()
    init_distributed(args)
    master = torch.distributed.get_rank()
    if os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    else:
        rank = world_size = None
    
    args.world_size = world_size
    args.device = torch.device(args.local_rank)

    if rank == 0:
        copy_files = ['main/main.py', 'lib/runners/hyponet.py', 'lib/runners/scorenet.py',
                      'lib/utils/function.py', 'lib/models',
                      'lib/dataset']
        for file in copy_files:
            os.system(f'cp -r {os.path.dirname(os.path.abspath(__file__))}/../{file} {args.code_path}')

    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,format='%(asctime)s - %(levelname)s - %(message)s')

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)


    logger.info("Writing log file to {}".format(cfg.log_path))
    logger.info("Exp instance id = {}".format(os.getpid()))
    logger.info("Local rank = {}".format(args.local_rank))
    if args.local_rank == 0:
        logger.info(pprint.pformat(cfg))
    if args.inference:
        from runners.inference import Inference
        runner = Inference(args, cfg)
    elif cfg.training.scorenet.train:
        runner = ScorenetTrainer(args, cfg)
    else:
        runner = HypoNetTrainer(args, cfg)

    if args.validate or args.inference:
        if args.inference and cfg.inference.input_type == 'video':
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
            video_list = os.listdir(cfg.inference.img_dir)
            out_dir = cfg.inference.out_dir
            for vname in video_list:
                if not any(vname.lower().endswith(ext) for ext in video_extensions):
                    continue
                print(f"################{vname}##################")
                cfg.inference.out_dir = out_dir
                cfg.inference.input_name = vname
                runner.validate()
        else:
            runner.validate()
    else:
        runner.train()


if __name__ == "__main__":
    main()
