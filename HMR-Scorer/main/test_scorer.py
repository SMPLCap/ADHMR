import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
from config import cfg
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import os.path as osp
import numpy as np
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--result_path', type=str, default='output/test')
    parser.add_argument('--ckpt_idx', type=int, default=0)
    parser.add_argument('--testset', type=str, nargs='+', default='EHF')
    parser.add_argument('--agora_benchmark', type=str, default='agora_model_val')
    parser.add_argument('--shapy_eval_split', type=str, default='val')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--eval_on_train', action='store_true')
    parser.add_argument('--vis', action='store_true')
    # parser.add_argument('--scorer_test', type=bool, default=True)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    print('### Argument parse and create log ###')
    args = parse_args()

    config_path = osp.join('../output',args.result_path, 'code', 'config_base.py')
    ckpt_path = osp.join('../output', args.result_path, 'model_dump', f'snapshot_{int(args.ckpt_idx)}.pth.tar')

    cfg.get_config_fromfile(config_path)
    if type(args.testset) == list:
        args.testset = args.testset[0]
        args.testset = [i for i in args.testset.split(' ')]
    cfg.update_test_config(args.testset, args.agora_benchmark, args.shapy_eval_split, 
                           ckpt_path, args.use_cache, args.eval_on_train, args.vis, scorer_test=True)
    cfg.update_config(args.num_gpus, args.exp_name)

    cudnn.benchmark = True

    from base import Tester_Scorer
    tester = Tester_Scorer()

    tester._make_batch_generator()
    tester._make_scorer_model()

    eval_result_all = {'pve_score_gt': [], 'papve_score_gt': [], 'mpjpe_score_gt': [], 'pa_mpjpe_score_gt': [],
                    'score_all': [],}
    
    for itr, (inputs, targets, meta_info, gen_output) in enumerate(tqdm(tester.batch_generator)):

        # forward
        with torch.no_grad():
            model_out = tester.model_score(inputs, targets, meta_info, gen_output, 'test')

        # save output
        batch_size = model_out['img'].shape[0]
        out = {}
        for k, v in model_out.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.cpu().numpy()
            elif isinstance(v, list):
                out[k] = v
            else:
                raise ValueError('Undefined type in out. Key: {}; Type: {}.'.format(k, type(v)))
        
        for k, v in out.items():
            if k in eval_result_all.keys():
                eval_result_all[k].append(v.copy())
    

    # evaluate
    for k, v in eval_result_all.items():
        eval_result_all[k] = np.concatenate(v, axis=0).flatten()
        print(eval_result_all[k].shape)

    print_eval_result = tester._evaluate(eval_result_all)

    tester._print_eval_result(print_eval_result)

if __name__ == "__main__":
    main()