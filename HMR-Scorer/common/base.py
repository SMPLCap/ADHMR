import os.path as osp
import math
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from config import cfg
from SMPLer_X import get_model
from HMR_Scorer import get_scorer_model
from Random_Generator import get_random_generator
from dataset import MultipleDatasets
from nets.ema import ExponentialMovingAverage
# ddp
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import torch.utils.data.distributed
from utils.distribute_utils import (
    get_rank, is_main_process, time_synchronized, get_group_idx, get_process_groups
)
from mmcv.runner import get_dist_info

# dynamic dataset import
for i in range(len(cfg.trainset_3d)):
    exec('from ' + cfg.trainset_3d[i] + ' import ' + cfg.trainset_3d[i])
for i in range(len(cfg.trainset_2d)):
    exec('from ' + cfg.trainset_2d[i] + ' import ' + cfg.trainset_2d[i])
for i in range(len(cfg.trainset_humandata)):
    exec('from ' + cfg.trainset_humandata[i] + ' import ' + cfg.trainset_humandata[i])
if not getattr(cfg, 'scorer_test', False):
    exec('from ' + cfg.testset + ' import ' + cfg.testset)
else:
    for i in range(len(cfg.testset)):
        try:
            exec('from ' + cfg.testset[i] + ' import ' + cfg.testset[i])
            exec('from ' + cfg.testset[i] + ' import ' + f'{cfg.testset[i]}_SCORER_TEST')
        except:
            pass
from PW3D_DPO import PW3D_DPO
from Human36M_DPO import Human36M_DPO
from InstaVariety_DPO import InstaVariety_DPO
from PW3D import PW3D_Clean
from MPII import MPII_Clean
from MSCOCO import MSCOCO_Clean
from MPI_INF_3DHP import MPI_INF_3DHP_Clean
from Human36M import Human36M_Clean


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    def __init__(self, distributed=False, gpu_idx=None):
        super(Trainer, self).__init__(log_name='train_logs.txt')
        self.distributed = distributed
        self.gpu_idx = gpu_idx
        self.model_list = []

    def get_optimizer(self, model):
        normal_param = []
        special_param = []
        for module in model.module.special_trainable_modules:
            special_param += list(module.parameters())
            print(module)
        for module in model.module.trainable_modules:
            normal_param += list(module.parameters())
        optim_params = [
            {  # add normal params first
                'params': normal_param,
                'lr': cfg.lr
            },
            {
                'params': special_param,
                'lr': cfg.lr * cfg.lr_mult
            },
        ]
        optimizer = torch.optim.Adam(optim_params, lr=cfg.lr)
        return optimizer
    
    def get_scorer_optimizer(self, model):
        normal_param = []
        backbone_params = []
        for module in model.module.backbone_trainable_modules:
            backbone_params += list(module.parameters())
            # print(module)
        for module in model.module.trainable_modules:
            normal_param += list(module.parameters())
        # self.logger.info(f"N-{self.gpu_idx}, {normal_param}")
        # self.logger.info("S", backbone_params)
        optim_params = [
            {
                'params': backbone_params,
                'lr': cfg.lr * cfg.lr_mult
            },
            {  # add normal params first
                'params': normal_param,
                'lr': cfg.lr,           # 0.0005
                'weight_decay': 0.000,
                'betas': (0.9, 0.999), 
                'amsgrad': False,
                'eps': 1e-24
            },
        ]
        optimizer = torch.optim.Adam(optim_params, lr=cfg.lr)
        return optimizer
        
    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))

        # do not save smplx layer weights
        dump_key = []
        for k in state['network'].keys():
            if 'smplx_layer' in k:
                dump_key.append(k)
        for k in dump_key:
            state['network'].pop(k, None)

        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        if cfg.pretrained_model_path is not None:
            ckpt_path = cfg.pretrained_model_path
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu')) # solve CUDA OOM error in DDP
            model.load_state_dict(ckpt['network'], strict=False)
            self.logger.info('Load checkpoint from {}'.format(ckpt_path))
            if not hasattr(cfg, 'start_over') or cfg.start_over:
                start_epoch = 0
            else:
                optimizer.load_state_dict(ckpt['optimizer'])
                start_epoch = ckpt['epoch'] + 1 
                self.logger.info(f'Load optimizer, start from{start_epoch}')
        else:
            start_epoch = 0

        return start_epoch, model, optimizer
    
    def load_scorer_model(self, model_score, optimizer, ema_score=None):
        if cfg.pretrained_scorer_model_path is not None:
            ckpt_path = cfg.pretrained_scorer_model_path
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu')) # solve CUDA OOM error in DDP
            model_score.load_state_dict(ckpt['network'], strict=False)
            self.logger.info('Load checkpoint from {}'.format(ckpt_path))
            if not hasattr(cfg, 'start_over') or cfg.start_over:
                start_epoch = 0
            else:
                optimizer.load_state_dict(ckpt['optimizer'])
                start_epoch = ckpt['epoch'] + 1 
                if 'ema_score' in ckpt and ema_score is not None:
                    ema_score.load_state_dict(ckpt['ema_score'])      
                else:
                    ema_score = None          
                self.logger.info(f'Load optimizer, start from{start_epoch}')
        else:
            start_epoch = 0

        return start_epoch, model_score, ema_score, optimizer

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger_info("Creating dataset...")
        trainset3d_loader = []
        for i in range(len(cfg.trainset_3d)):
            trainset3d_loader.append(eval(cfg.trainset_3d[i])(transforms.ToTensor(), "train"))
        trainset2d_loader = []
        for i in range(len(cfg.trainset_2d)):
            trainset2d_loader.append(eval(cfg.trainset_2d[i])(transforms.ToTensor(), "train"))
        trainset_humandata_loader = []
        for i in range(len(cfg.trainset_humandata)):
            trainset_humandata_loader.append(eval(cfg.trainset_humandata[i])(transforms.ToTensor(), "train"))
        
        data_strategy = getattr(cfg, 'data_strategy', None)
        if data_strategy == 'concat':
            print("Using [concat] strategy...")
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader + trainset_humandata_loader, 
                                                make_same_len=False, verbose=True)
        elif data_strategy == 'balance':
            total_len = getattr(cfg, 'total_data_len', 'auto')
            print(f"Using [balance] strategy with total_data_len : {total_len}...")
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader + trainset_humandata_loader, 
                                                 make_same_len=True, total_len=total_len, verbose=True)
        else:
            # original strategy implementation
            valid_loader_num = 0
            if len(trainset3d_loader) > 0:
                trainset3d_loader = [MultipleDatasets(trainset3d_loader, make_same_len=False)]
                valid_loader_num += 1
            else:
                trainset3d_loader = []
            if len(trainset2d_loader) > 0:
                trainset2d_loader = [MultipleDatasets(trainset2d_loader, make_same_len=False)]
                valid_loader_num += 1
            else:
                trainset2d_loader = []
            if len(trainset_humandata_loader) > 0:
                trainset_humandata_loader = [MultipleDatasets(trainset_humandata_loader, make_same_len=False)]
                valid_loader_num += 1

            if valid_loader_num > 1:
                trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader + trainset_humandata_loader, make_same_len=True)
            else:
                trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader + trainset_humandata_loader, make_same_len=False)

        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.train_batch_size)

        if self.distributed:
            self.logger_info(f"Total data length {len(trainset_loader)}.")
            rank, world_size = get_dist_info()
            self.logger_info("Using distributed data sampler.")
            
            sampler_train = DistributedSampler(trainset_loader, world_size, rank, shuffle=True)
            self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.train_batch_size,
                                          shuffle=False, num_workers=cfg.num_thread, sampler=sampler_train,
                                          pin_memory=True, persistent_workers=True if cfg.num_thread > 0 else False, drop_last=True)
        else:
            self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus * cfg.train_batch_size,
                                          shuffle=True, num_workers=cfg.num_thread,
                                          pin_memory=True, drop_last=True)

    def _make_scorer_model(self):
        # prepare network
        self.logger_info("Creating graph and optimizer...")
        if getattr(cfg, 'train_scorer', False):
            model_score = get_scorer_model('train')
        else:
            raise NotImplementedError

        if getattr(cfg, 'no_train_backbone', False):
            print("Disable Training [backbone]...")
            for param in model_score.encoder.parameters():
                param.requires_grad = False

        total_params = sum(p.numel() for p in model_score.parameters())
        trainable_params = sum(p.numel() for p in model_score.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        print(f"Total params: {total_params / 1e6} M", )
        print(f"Trainable params: {trainable_params / 1e6} M", )
        print(f"Untrainable params: {non_trainable_params / 1e6} M", )

        # ddp
        if self.distributed:
            self.logger_info("Using distributed data parallel.")
            model_score.cuda()
            if hasattr(cfg, 'syncbn') and cfg.syncbn:
                self.logger_info("Using sync batch norm layers.")

                process_groups = get_process_groups()
                process_group = process_groups[get_group_idx()]
                syncbn_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_score, process_group)
                model_score = torch.nn.parallel.DistributedDataParallel(
                    syncbn_model, device_ids=[self.gpu_idx],
                    find_unused_parameters=True)
            else:
                model_score = torch.nn.parallel.DistributedDataParallel(
                    model_score, device_ids=[self.gpu_idx],
                    find_unused_parameters=True)
        else:
        # dp
            model_score = DataParallel(model_score).cuda()

        # optimizer = self.get_optimizer(model_score)
        optimizer = self.get_scorer_optimizer(model_score)

        ema_score = None
        
        if cfg.continue_train:
            start_epoch, model_score, ema_score, optimizer = self.load_scorer_model(model_score, optimizer, ema_score)
        else:
            start_epoch = 0
        
        if hasattr(cfg, "scheduler"):
            if cfg.scheduler == 'cos':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.end_epoch * self.itr_per_epoch,
                                                               eta_min=1e-6)
            elif cfg.scheduler == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.step_size, gamma=cfg.gamma, 
                                                            last_epoch=- 1, verbose=False)                                           

        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step_model, cfg.lr_factor_model, last_epoch=start_epoch-1)

        model_score.train()

        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.model_score = model_score
        self.optimizer = optimizer
        # self.ema_score = ema_score

    def _make_smplerx_model(self):
        self.logger.info('Load checkpoint from {}'.format(cfg.pretrained_model_path))
        # prepare network
        # self.logger.info("Creating graph...")
        model = get_model('test')

        if self.distributed:
            self.logger_info("Using distributed data parallel.")
            model.cuda()
            if hasattr(cfg, 'syncbn') and cfg.syncbn:
                self.logger_info("Using sync batch norm layers.")

                process_groups = get_process_groups()
                process_group = process_groups[get_group_idx()]
                syncbn_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
                model = torch.nn.parallel.DistributedDataParallel(
                    syncbn_model, device_ids=[self.gpu_idx],
                    find_unused_parameters=True) 
            else:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[self.gpu_idx],
                    find_unused_parameters=True) 
        else:
        # dp
            model = DataParallel(model).cuda()
        # model = DataParallel(model).cuda()
        if not getattr(cfg, 'random_init', False):
            ckpt = torch.load(cfg.pretrained_model_path, map_location=torch.device('cpu'))

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt['network'].items():
                if 'module' not in k:
                    k = 'module.' + k
                k = k.replace('backbone', 'encoder').replace('body_rotation_net', 'body_regressor').replace(
                    'hand_rotation_net', 'hand_regressor')
                new_state_dict[k] = v
            self.logger.warning("Attention: Strict=False is set for checkpoint loading. Please check manually.")
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
        else:
            print('Random init!!!!!!!')
        
        self.model = model
        # self.model_list.append(model)

    def _make_gen_model(self):
        self.logger.info('Generating hypothesis from adding random noise to targets')

        # prepare network
        model = get_random_generator('test')
        model = DataParallel(model).cuda()
        model.eval()
        self.model_list.append(model)


    def logger_info(self, info):
        if self.distributed:
            if is_main_process():
                self.logger.info(info)
        else:
            self.logger.info(info)


class Tester(Base):
    def __init__(self, test_epoch=None):
        if test_epoch is not None:
            self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(cfg.testset)(transforms.ToTensor(), "test")
        if cfg.construct_scorer_testset:
            torch.manual_seed(2024)
            testset_loader.construct_scorer_testset()
            exit(0)
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                     shuffle=False, num_workers=cfg.num_thread, pin_memory=True)

        self.testset = testset_loader
        self.batch_generator = batch_generator

    def _make_model(self):
        self.logger.info('Load checkpoint from {}'.format(cfg.pretrained_model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model('test')
        model = DataParallel(model).cuda()
        if not getattr(cfg, 'random_init', False):
            ckpt = torch.load(cfg.pretrained_model_path, map_location=torch.device('cpu'))

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt['network'].items():
                if 'module' not in k:
                    k = 'module.' + k
                k = k.replace('backbone', 'encoder').replace('body_rotation_net', 'body_regressor').replace(
                    'hand_rotation_net', 'hand_regressor')
                new_state_dict[k] = v
            self.logger.warning("Attention: Strict=False is set for checkpoint loading. Please check manually.")
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
        else:
            print('Random init!!!!!!!')

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.testset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.testset.print_eval_result(eval_result)

class Tester_Scorer(Base):
    def __init__(self, test_epoch=None):
        if test_epoch is not None:
            self.test_epoch = int(test_epoch)
        super(Tester_Scorer, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset3d_loader = []
        testset_humandata_loader = []
        for i in range(len(cfg.testset)):
            testset_humandata_loader.append(eval(f'{cfg.testset[i]}_SCORER_TEST')(transforms.ToTensor(), "test"))
        print("Using [concat] strategy...")
        testset_loader = MultipleDatasets(testset3d_loader + testset_humandata_loader, 
                                          make_same_len=False, verbose=True)
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                     shuffle=True, num_workers=0, pin_memory=True)

        self.testset = testset_loader
        self.batch_generator = batch_generator

    def _make_scorer_model(self):
        self.logger.info('Load checkpoint from {}'.format(cfg.pretrained_model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model_score = get_scorer_model('test')
        model_score = DataParallel(model_score).cuda()
        if not getattr(cfg, 'random_init', False):
            ckpt = torch.load(cfg.pretrained_model_path, map_location=torch.device('cpu'))
            model_score.load_state_dict(ckpt['network'], strict=False)
            model_score.eval()
        else:
            print('Random init!!!!!!!')

        self.model_score = model_score

    def _evaluate(self, outs):
        eval_result = self.testset.evaluate_scorer(outs)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.testset.print_scorer_eval_result(eval_result)


class Tester_Scorer_Clean(Base):
    def __init__(self, test_epoch=None):
        if test_epoch is not None:
            self.test_epoch = int(test_epoch)
        super(Tester_Scorer_Clean, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(f'{cfg.testset}_Clean')(transforms.ToTensor(), "train")
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                     shuffle=False, num_workers=cfg.num_thread, pin_memory=True)

        self.testset = testset_loader
        self.batch_generator = batch_generator

    def _make_scorer_model(self):
        self.logger.info('Load checkpoint from {}'.format(cfg.pretrained_model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model_score = get_scorer_model('test')
        model_score = DataParallel(model_score).cuda()
        if True:
            ckpt = torch.load(cfg.pretrained_model_path, map_location=torch.device('cpu'))
            model_score.load_state_dict(ckpt['network'], strict=False)
            model_score.eval()

        self.model_score = model_score

    def _evaluate(self, outs):
        eval_result = self.testset.evaluate_scorer(outs)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.testset.print_scorer_eval_result(eval_result)
