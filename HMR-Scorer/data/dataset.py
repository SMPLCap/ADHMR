import random
import numpy as np
import os
from torch.utils.data.dataset import Dataset
from config import cfg
from scipy.stats import spearmanr, pearsonr, rankdata

class MultipleDatasets(Dataset):
    def __init__(self, dbs, make_same_len=True, total_len=None, verbose=False):
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])
        self.make_same_len = make_same_len
        
        if total_len == 'auto':
            self.total_len = self.db_len_cumsum[-1]
            self.auto_total_len = True
        else:
            self.total_len = total_len
            self.auto_total_len = False

        if total_len is not None:
            self.per_db_len = self.total_len // self.db_num
        if verbose:
            print('datasets:', [len(self.dbs[i]) for i in range(self.db_num)])
            print(f'Auto total length: {self.auto_total_len}, {self.total_len}')

            

    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            if self.total_len is None:
                # match the longest length
                return self.max_db_data_num * self.db_num
            else:
                # each dataset has the same length and total len is fixed
                return self.total_len
        else:
            # each db has different length, simply concat
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.make_same_len:
            if self.total_len is None:
                # match the longest length
                db_idx = index // self.max_db_data_num
                data_idx = index % self.max_db_data_num 
                if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])): # last batch: random sampling
                    data_idx = random.randint(0,len(self.dbs[db_idx])-1)
                else: # before last batch: use modular
                    data_idx = data_idx % len(self.dbs[db_idx])
            else:
                db_idx = index // self.per_db_len 
                data_idx = index % self.per_db_len 
                if db_idx > (self.db_num - 1):
                    # last batch: randomly choose one dataset
                    db_idx = random.randint(0,self.db_num - 1)

                if len(self.dbs[db_idx]) < self.per_db_len  and \
                        data_idx >= len(self.dbs[db_idx]) * (self.per_db_len  // len(self.dbs[db_idx])): 
                    # last batch: random sampling in this dataset
                    data_idx = random.randint(0,len(self.dbs[db_idx]) - 1)
                else: 
                    # before last batch: use modular
                    data_idx = data_idx % len(self.dbs[db_idx])


        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]

        return self.dbs[db_idx][data_idx]
    
    def evaluate_scorer(self, outs):
        eval_result = {'plcc_pve': 0., 'plcc_papve': 0., 'plcc_mpjpe': 0., 'plcc_pa_mpjpe': 0.,
                       'srcc_pve': 0., 'srcc_papve': 0., 'srcc_mpjpe': 0., 'srcc_pa_mpjpe': 0.,}
        # 1. PLCC
        score_all = outs['score_all']
        pve_score_gt = outs['pve_score_gt']
        papve_score_gt = outs['papve_score_gt']
        mpjpe_score_gt = outs['mpjpe_score_gt']
        pa_mpjpe_score_gt = outs['pa_mpjpe_score_gt']

        eval_result['plcc_pve'], _ = pearsonr(score_all, pve_score_gt)
        eval_result['plcc_papve'], _ = pearsonr(score_all, papve_score_gt)
        eval_result['plcc_mpjpe'], _ = pearsonr(score_all, mpjpe_score_gt)
        eval_result['plcc_pa_mpjpe'], _ = pearsonr(score_all, pa_mpjpe_score_gt)

        # eval_result['plcc_pve'] = plcc_pve
        # eval_result['plcc_papve'] = plcc_papve
        # eval_result['plcc_mpjpe'] = plcc_mpjpe
        # eval_result['plcc_pa_mpjpe'] = plcc_pa_mpjpe

        # 2. SRCC
        # score_all_rank = rankdata(outs['score_all'])
        # pve_score_rank_gt = rankdata(outs['pve_score_gt'])
        # papve_score_rank_gt = rankdata(outs['papve_score_gt'])
        # mpjpe_score_rank_gt = rankdata(outs['mpjpe_score_gt'])
        # pa_mpjpe_score_rank_gt = rankdata(outs['pa_mpjpe_score_gt'])

        eval_result['srcc_pve'], _ = spearmanr(score_all, pve_score_gt)
        eval_result['srcc_papve'], _ = spearmanr(score_all, papve_score_gt)
        eval_result['srcc_mpjpe'], _ = spearmanr(score_all, mpjpe_score_gt)
        eval_result['srcc_pa_mpjpe'], _ = spearmanr(score_all, pa_mpjpe_score_gt)

        # eval_result['srcc_pve'] = srcc_pve
        # eval_result['srcc_papve'] = srcc_papve
        # eval_result['srcc_mpjpe'] = srcc_mpjpe
        # eval_result['srcc_pa_mpjpe'] = srcc_pa_mpjpe

        return eval_result
    
    def print_scorer_eval_result(self, eval_result):
        print(f'======{" ".join(cfg.testset)}======')
        print(f'{cfg.vis_dir}')
        print('PLCC (PVE): %.4f' % eval_result['plcc_pve'])
        print('PLCC (PA-PVE): %.4f' % eval_result['plcc_papve'])
        print('PLCC (MPJPE): %.4f' % eval_result['plcc_mpjpe'])
        print('PLCC (PA-MPJPE): %.4f' % eval_result['plcc_pa_mpjpe'])
        print()

        print('SRCC (PVE): %.4f' % eval_result['srcc_pve'])
        print('SRCC (PA-PVE): %.4f' % eval_result['srcc_papve'])
        print('SRCC (MPJPE): %.4f' % eval_result['srcc_mpjpe'])
        print('SRCC (PA-MPJPE): %.4f' % eval_result['srcc_pa_mpjpe'])
        print()

        f = open(os.path.join(cfg.result_dir, 'result_scorer.txt'), 'w')
        f.write(f'[{" ".join(cfg.testset)}] dataset \n')
        f.write('PLCC (PVE): %.4f\n' % eval_result['plcc_pve'])
        f.write('PLCC (PA-PVE): %.4f\n' % eval_result['plcc_papve'])
        f.write('PLCC (MPJPE): %.4f\n' % eval_result['plcc_mpjpe'])
        f.write('PLCC (PA-MPJPE): %.4f\n \n' % eval_result['plcc_pa_mpjpe'])

        f.write('SRCC (PVE): %.4f\n' % eval_result['srcc_pve'])
        f.write('SRCC (PA-PVE): %.4f\n' % eval_result['srcc_papve'])
        f.write('SRCC (MPJPE): %.4f\n' % eval_result['srcc_mpjpe'])
        f.write('SRCC (PA-MPJPE): %.4f' % eval_result['srcc_pa_mpjpe'])