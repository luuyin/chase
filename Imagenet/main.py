import argparse
import os
import shutil
import time
import random
import math
import copy
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch import optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from colorama import Fore
import sys
import re
import core
from core import Masking, CosineDecay
from main_flops import count_model_param_flops


try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import resnet as models
from smoothing import LabelSmoothing

def add_parser_arguments(parser):
    model_names = models.resnet_versions.keys()
    model_configs = models.resnet_configs.keys()

    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--save', default='save/default-{}'.format(time.time()),
                           type=str, metavar='SAVE',
                           help='path to the experiment logging directory'
                                '(default: save/debug)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('--model-config', '-c', metavar='CONF', default='classic',
                        choices=model_configs,
                        help='model configs: ' +
                        ' | '.join(model_configs) + '(default: classic)')
    parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                        help='number of data loading workers (default: 5)')
    parser.add_argument('--spp', action='store_true',
                        help='flag to switch another grow initialization')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--l2', default=1e-4, type=float,
                        help='l2 regularization (default: 1e-4)')
    parser.add_argument('--warmup', default=0, type=int,
                        metavar='E', help='number of warmup epochs')
    parser.add_argument('--label_smoothing', default=0.0, type=float,
                        metavar='S', help='label smoothing')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--bn-weight-decay', action='store_true',
                        help='use weight_decay on batch normalization learnable parameters, default: false)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained-weights', default='', type=str, metavar='PATH',
                        help='file with weights')

    parser.add_argument('--fp16', action='store_true',
                        help='Run model fp16 mode.')
    parser.add_argument('--static-loss-scale', type=float, default=1,
                        help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--dynamic-loss-scale', action='store_true',
                        help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                        '--static-loss-scale.')
    parser.add_argument('--prof', dest='prof', action='store_true',
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--benchmark-training', dest='trainbench', action='store_true',
                        help='Run benchmarking of training')
    parser.add_argument('--benchmark-inference', dest='inferbench', action='store_true',
                        help='Run benchmarking of training')
    parser.add_argument('--bench-iterations', type=int, default=20, metavar='N',
                        help='Run N iterations while benchmarking (ignored when training and validation)')
    parser.add_argument('--bench-warmup', type=int, default=20, metavar='N',
                        help='Number of warmup iterations for benchmarking')
    parser.add_argument('--master_port', '-master_port', default=10000, type=int,
              help="Port of master for torch.distributed training.")
    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument('--seed', default=17, type=int,
                        help='random seed used for np and pytorch')

    parser.add_argument('--gather-checkpoints', action='store_true',
                        help='Gather checkpoints throughout the training')
    parser.add_argument('--grow-switch', default='', type=str,
                           help='flag to switch another grow initialization')

    parser.add_argument('--cosine_lr', action='store_true',
                        help='cosine_lr')



    parser.add_argument('--indicate_method', type=str,
                    help='indicate_method')



    # training epochs

    parser.add_argument('--total_epochs', type=int, default=100,
                    help='number of epochs to train for ensemble models (default: 50)')
    # learning rate
    parser.add_argument('--first_m', type=int, default=30,
                    help='first LR decay')
    parser.add_argument('--second_m', type=int, default=60,
                    help='second LR decay')
    parser.add_argument('--third_m', type=int, default=80,
                    help='third LR decay')


    # resume
    parser.add_argument('--re_epochs', type=int, default=90,
                    help='resume epochs')



    ## channel wider
    # parser.add_argument('--wider_epochs', default=1, type=int,help='wider_epochs')
    parser.add_argument('--layer_interval', default=16000, type=float,help='wider_interval')
    parser.add_argument('--start_layer_rate', default=0.1, type=float,help='layer_ratio')   


    core.add_sparse_args(parser)

def main():

    print (" torch.cuda.device_count()", torch.cuda.device_count())
    if args.trainbench or args.inferbench:
        logger = BenchLogger
    else:
        logger = PrintLogger

    train_net(args, logger)

def train_net(args, logger_cls):
    exp_start_time = time.time()
    global best_prec1
    best_prec1 = 0

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        os.environ['MASTER_PORT'] = str(args.master_port)
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()


    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)
    else:
        def _worker_init_fn(id):
            pass

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    pretrained_weights = None
    if args.pretrained_weights:
        if os.path.isfile(args.pretrained_weights):
            print("=> loading pretrained weights from '{}'".format(args.pretrained_weights))
            pretrained_weights = torch.load(args.pretrained_weights)
        else:
            print("=> no pretrained weights found at '{}'".format(args.resume))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']+1
            best_prec1 = checkpoint['best_prec1']
            model_state = checkpoint['state_dict']
            optimizer_state = checkpoint['optimizer']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            model_state = None
            optimizer_state = None
    else:
        model_state = None
        optimizer_state = None

    model_and_loss = ModelAndLoss(args,
            (args.arch, args.model_config),
            nn.CrossEntropyLoss if args.label_smoothing == 0.0 else (lambda: LabelSmoothing(args.label_smoothing)),
            pretrained_weights=pretrained_weights,
            state=model_state,
            cuda = True, fp16 = args.fp16, distributed = args.distributed)

    # Create data loaders and optimizers as needed

    if not (args.evaluate or args.inferbench):
        optimizer = get_optimizer(list(model_and_loss.model.named_parameters()),
                args.fp16,
                args.lr, args.momentum, args.weight_decay,
                bn_weight_decay = args.bn_weight_decay,
                state=optimizer_state,
                static_loss_scale = args.static_loss_scale,
                dynamic_loss_scale = args.dynamic_loss_scale)


        train_loader = get_train_loader(args.data, args.batch_size, workers=args.workers, _worker_init_fn=_worker_init_fn)
        train_loader_len = len(train_loader)
    else:
        train_loader_len = 0

    if not args.trainbench:
        val_loader = get_val_loader(args.data, args.batch_size, workers=args.workers, _worker_init_fn=_worker_init_fn)
        val_loader_len = len(val_loader)
    else:
        val_loader_len = 0

    # add binary masks to the dense model
    if args.sparse:
        if args.resume:
            print ("load resume decay and mask")
            decay = CosineDecay(args.prune_rate, len(train_loader)*args.epochs*args.multiplier,init_step=args.re_epochs*10009)
            mask = Masking(optimizer,train_loader=train_loader, prune_mode=args.prune, prune_rate_decay=decay, growth_mode=args.growth, redistribution_mode=args.redistribution, args=args,step=args.re_epochs*10009)
        else:
            decay = CosineDecay(args.prune_rate+args.density, len(train_loader)*(args.total_epochs-args.stop_dst_epochs),args.density+0.005)
            # decay = CosineDecay(args.prune_rate+args.density, 10*(args.total_epochs-args.stop_dst_epochs),args.density+0.005)
            mask = Masking(optimizer,train_loader=train_loader, prune_mode=args.prune, prune_rate_decay=decay, growth_mode=args.growth, redistribution_mode=args.redistribution, args=args)

        mask.add_module(model_and_loss.model)
        model_and_loss.mask = mask

    if args.evaluate:
        logger = logger_cls(train_loader_len, val_loader_len, args)
        validate(val_loader, model_and_loss, args.fp16, logger, 0)
        return

    if args.trainbench:
        model_and_loss.model.train()
        logger = logger_cls("Train", args.world_size * args.batch_size, args.bench_warmup)
        bench(get_train_step(model_and_loss, optimizer, args.fp16), train_loader,
              args.bench_warmup, args.bench_iterations, args.fp16, logger, epoch_warmup = True)
        return

    if args.inferbench:
        model_and_loss.model.eval()
        logger = logger_cls("Inference", args.world_size * args.batch_size, args.bench_warmup)
        bench(get_val_step(model_and_loss), val_loader,
              args.bench_warmup, args.bench_iterations, args.fp16, logger, epoch_warmup = False)
        return

    logger = logger_cls(train_loader_len, val_loader_len, args)



    if args.cosine_lr:
        lr_scheduler=  optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.total_epochs-args.warmup+1), eta_min=0)

    else:
        lr_scheduler=adjust_learning_rate(args)

    train_loop(args, model_and_loss, optimizer, lr_scheduler, train_loader, val_loader,
            args.fp16, logger, should_backup_checkpoint(args),
            start_epoch = args.start_epoch, best_prec1 = best_prec1, prof=args.prof)

    exp_duration = time.time() - exp_start_time
    logger.experiment_timer(exp_duration)
    logger.end_callback()

    count_model_param_flops(model_and_loss.model)

    print("Good job, all done!")



# cyclic learning rate adjust
def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def cyc_adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# get ensemble models
def get_model_params(model):
    params = {}
    for name in model.state_dict():
        params[name] = copy.deepcopy(model.state_dict()[name])
    return params

def set_model_params(model, model_parameters):
    model.load_state_dict(model_parameters)


def get_or_averagte_moving(decay):

    def function(free_tickets,rangelist,decay=decay):
        print ("ensemble by OR weights averagte_moving ")
        print ("decay",decay)
        ensemble_flag="moving_"+str(decay)
        params = {}
        pre_params = {}
        for name in free_tickets[0].state_dict():
            pre_params[name] = copy.deepcopy(free_tickets[0].state_dict()[name])

        rangelist=list(rangelist)[1: ]
        for name in free_tickets[0].state_dict():
            for i in rangelist:
                params[name]=copy.deepcopy(free_tickets[i].state_dict()[name]* decay+pre_params[name]* (1 - decay))
                pre_params[name]=params[name]                    
                                            
        return params,ensemble_flag
    return function

def get_or_averagte_weight(free_tickets,rangelist):
    print ("ensemble by just OR weights ")
    ensemble_flag="or_weight"
    params = {}
    for name in free_tickets[0].state_dict():
        increment_mask=torch.zeros_like(free_tickets[0].state_dict()[name])
        for i in rangelist:
            temp_mask=test=copy.deepcopy(free_tickets[i].state_dict()[name].bool().float())
            increment_mask=increment_mask+temp_mask
        increment_mask[increment_mask==0] = 1

        params[name]=copy.deepcopy(torch.sum(torch.stack([free_tickets[i].state_dict()[name] for i in range(len(free_tickets))],dim=0).float(),dim=0))/increment_mask

    return params,ensemble_flag

def get_or_averagte_weight_connetion(free_tickets,rangelist):
    print ("ensemble by just OR weights and connection ")
    ensemble_flag="or_weight_connect"
    params = {}
    for name in free_tickets[0].state_dict():
    #     print ("name",name)
        params[name]=copy.deepcopy(torch.mean(torch.stack([free_tickets[i].state_dict()[name] for i in rangelist],dim=0).float(),dim=0))
    
    return params,ensemble_flag


def get_and_averagte_weight_connetion(free_tickets,rangelist):
    print ("ensemble by just And weights and connection ")
    ensemble_flag="and"
    # shared weights
    params = {}
    for name in free_tickets[0].state_dict():
 
        temp_mask=torch.ones_like(free_tickets[0].state_dict()[name])
        for i in range(len(free_tickets)):
            temp_mask=torch.logical_and ((free_tickets[i].state_dict()[name]),temp_mask)
        params[name]=copy.deepcopy(torch.mean(torch.stack([free_tickets[i].state_dict()[name] for i in range(len(free_tickets))],dim=0).float(),dim=0))*temp_mask
  
    return params,ensemble_flag


# get model file names


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def model_files_filter(model_files,filter_itrs=["lowlr"]):
    new_files=[]
    for filter_itr in filter_itrs:
        for  model_file in model_files:
            if filter_itr in model_file:
                new_files.append(model_file)
    return new_files




# get optimizer {{{
def get_optimizer(parameters, fp16, lr, momentum, weight_decay,
                  true_wd=False,
                  nesterov=False,
                  state=None,
                  static_loss_scale=1., dynamic_loss_scale=False,
                  bn_weight_decay = False):

    if bn_weight_decay:
        print(" ! Weight decay applied to BN parameters ")
        optimizer = torch.optim.SGD([v for n, v in parameters], lr,
                           momentum=momentum,
                           weight_decay=weight_decay,
                           nesterov = nesterov)
    else:
        print(" ! Weight decay NOT applied to BN parameters ")
        bn_params = [v for n, v in parameters if 'bn' in n]
        rest_params = [v for n, v in parameters if not 'bn' in n]
        print(len(bn_params))
        print(len(rest_params))
        optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay' : 0},
                            {'params': rest_params, 'weight_decay' : weight_decay}],
                           lr,
                           momentum=momentum,
                           weight_decay=weight_decay,
                           nesterov = nesterov)
    if fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=static_loss_scale,
                                   dynamic_loss_scale=dynamic_loss_scale)

    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer
# }}}

# ModelAndLoss {{{
class ModelAndLoss(nn.Module):
    def __init__(self, args, arch, loss, pretrained_weights=None, state=None, cuda=True, fp16=False, distributed=False):
        super(ModelAndLoss, self).__init__()
        self.arch = arch
        self.mask = None
        
        print("=> creating model '{}'".format(arch))

        print ("arch",arch)
        print ("arch[0]",arch[0])
        print ("arch[1]",arch[1]) 
        model = models.build_resnet(arch[0], arch[1])
        if pretrained_weights is not None:
            print("=> using pre-trained model from a file '{}'".format(arch))
            model.load_state_dict(pretrained_weights)

        if cuda:
            model = model.cuda()
        if fp16:
            model = network_to_half(model)
        if distributed:
            model = DDP(model)

        if not state is None:
            model.load_state_dict(state)

        # define loss function (criterion) and optimizer
        criterion = loss()

        if cuda:
            criterion = criterion.cuda()

        self.model = model
        self.loss = criterion

    def forward(self, data, target):

   
        output = self.model(data)
        loss = self.loss(output, target)

        # l2_reg = None
        # for name, weight in self.model.named_parameters():
        #     if name not in self.mask.masks: continue
        #     if l2_reg is None:
        #         l2_reg = (args.density) * (self.mask.new_masks[name] * weight).norm(2)
        #
        #     else:
        #         l2_reg = l2_reg + (args.density) * (self.mask.new_masks[name] * weight).norm(2)
        #
        # loss = loss + args.weight_decay * l2_reg
        return loss, output
# }}}

# Train loop {{{
def train_loop(args, model_and_loss, optimizer, lr_scheduler, train_loader, val_loader, fp16, logger,
               should_backup_checkpoint,
               best_prec1 = 0, start_epoch = 0, prof = False):



    save_dir = "./saved_models/"+str(args.indicate_method)   + '/density_' + str(args.density) + '/stop_gmp_epochs_' + str(args.stop_gmp_epochs)   +  '/epoch_' + str(args.total_epochs) +  '/layer_interval_' + str(args.layer_interval)+   '/start_layer_rate_' + str(args.start_layer_rate)
    print ("save_dir",save_dir)
    try:
        if not os.path.exists(save_dir): os.makedirs(save_dir)
    except Exception as e:
        print(e)
        pass


    print ("=====================================")
    print ("begin training ")
    #================#


    update_iter = 0
    for epoch in range(start_epoch,  args.total_epochs):
        
        print ("in epoch",epoch,"==================================================================")
        # model_and_loss.mask.print_layerwise_density()


        if torch.distributed.is_initialized():
            train_loader.sampler.set_epoch(epoch)

        if args.cosine_lr:

            if epoch < args.warmup:
                lr = args.lr * (epoch + 1) / (args.warmup)
                for param_group in model_and_loss.mask.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr_scheduler.step()

        else:
            # normal lr_schedule
            lr_scheduler(model_and_loss.mask.optimizer, epoch)

        # train and eval
        cyclic_schedule=None
        train(train_loader,val_loader, model_and_loss, optimizer, fp16, logger, epoch, cyclic_schedule,prof = prof)

        prec1 = validate(val_loader, model_and_loss, fp16, logger, epoch, prof = prof)
        print('current learning rate is:', model_and_loss.mask.optimizer.param_groups[0]['lr'])



        if epoch>=(args.total_epochs-args.stop_dst_epochs):
            # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:

            is_best = prec1 > best_prec1
            print ("is_best",is_best)
            best_prec1 = max(prec1, best_prec1)
            print( 'Best var_acc1 {}'.format(best_prec1))
                
            # save at best
            if  is_best:
                print('Saving model at best acc')
                save_checkpoint({
                    'epoch': epoch ,
                    'arch': model_and_loss.arch,
                    'state_dict': model_and_loss.model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, filename=os.path.join(save_dir, 'premodel_best.pth'))


    print('Saving the last pre_model')
    save_checkpoint({
        'epoch': "last_pre",
        'arch': model_and_loss.arch,
        'state_dict': model_and_loss.model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer.state_dict(),
    }, filename=os.path.join(save_dir, 'premodel_last.pth'))




# }}}

# Data Loading functions {{{
def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        # tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        nump_array_copy = np.copy(nump_array)
        tensor[i] += torch.from_numpy(nump_array_copy)

    return tensor, targets


def prefetched_loader(loader, fp16):
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
    if fp16:
        mean = mean.half()
        std = std.half()

    stream = torch.cuda.Stream()
    first = True

    for next_input, next_target in loader:
        with torch.cuda.stream(stream):
            next_input = next_input.cuda(non_blocking=True)
            next_target = next_target.cuda(non_blocking=True)
            if fp16:
                next_input = next_input.half()
            else:
                next_input = next_input.float()
            next_input = next_input.sub_(mean).div_(std)

        if not first:
            yield input, target
        else:
            first = False

        torch.cuda.current_stream().wait_stream(stream)
        input = next_input
        target = next_target

    yield input, target


def get_train_loader(data_path, batch_size, workers=5, _worker_init_fn=None):
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(), Too slow
            #normalize,
        ]))

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate, drop_last=True)

    return train_loader

def get_val_loader(data_path, batch_size, workers=5, _worker_init_fn=None):
    valdir = os.path.join(data_path, 'val')

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
        collate_fn=fast_collate)

    return val_loader
# }}}

# Train val bench {{{
def get_train_step(model_and_loss, optimizer, fp16):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)
        loss, output = model_and_loss(input_var, target_var)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        model_and_loss.mask.optimizer.zero_grad()
        if fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if model_and_loss.mask is None:
            optimizer.step()
        else:
            model_and_loss.mask.step()


        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def bench(step, train_loader, warmup, iterations, fp16, logger, epoch_warmup = False):
    step = timed_function(step)

    if epoch_warmup:
        print("Running first epoch for warmup, please wait")

        for (input, target), dt in timed_generator(prefetched_loader(train_loader, fp16)):
            _, bt = step(input, target)

    print("Running benchmarked epoch")

    for (input, target), dt in timed_generator(prefetched_loader(train_loader, fp16)):
        _, bt = step(input, target)
        logger.iter_callback({'data_time': dt, 'batch_time': bt})

        if logger.i >= warmup + iterations:
            break

    logger.end_callback()

def train(train_loader, val_loader, model_and_loss, optimizer, fp16, logger, epoch, cyc_lr_schedule,prof=False):
    global update_iter
    step = get_train_step(model_and_loss, optimizer, fp16)

    model_and_loss.model.train()
    end = time.time()


    num_iters = len(train_loader)
    if prof:
        num_iters=10
    # num_iters=10
    for i, (input, target) in enumerate(prefetched_loader(train_loader, fp16)):

        if cyc_lr_schedule is not None:
            lr = cyc_lr_schedule(i / num_iters)
            cyc_adjust_learning_rate(optimizer, lr)


        data_time = time.time() - end
        if prof:
            if i > 10:
                break

        loss, prec1, prec5 = step(input, target)

        logger.train_iter_callback(epoch, i,
                        {'size' : input.size(0),
                         'top1' : to_python_float(prec1),
                         'top5' : to_python_float(prec5),
                         'loss' : to_python_float(loss),
                         'time' : time.time() - end,
                         'data' : data_time})

        end = time.time()
        sys.stdout.flush()

    logger.train_epoch_callback(epoch)


def get_val_step(model_and_loss):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            loss, output = model_and_loss(input_var, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(val_loader, model_and_loss, fp16, logger, epoch, prof=False):

    step = get_val_step(model_and_loss)

    top1 = AverageMeter()
    # switch to evaluate mode
    model_and_loss.model.eval()

    end = time.time()

    for i, (input, target) in enumerate(prefetched_loader(val_loader, fp16)):
        data_time = time.time() - end
        if prof:
            if i > 10:
                break

        loss, prec1, prec5 = step(input, target)

        top1.update(to_python_float(prec1), input.size(0))

        logger.val_iter_callback(epoch, i,
                {'size' : input.size(0),
                 'top1' : to_python_float(prec1),
                 'top5' : to_python_float(prec5),
                 'loss' : to_python_float(loss),
                 'time' : time.time() - end,
                 'data' : data_time})

        end = time.time()

    logger.val_epoch_callback(epoch)

    return top1.avg

# }}}

# Logging {{{
class BenchLogger(object):
    def __init__(self, name, total_bs, warmup_iter):
        self.name = name
        self.data_time = AverageMeter()
        self.batch_time = AverageMeter()
        self.warmup_iter = warmup_iter
        self.total_bs = total_bs
        self.i = 0

    def reset(self):
        self.data_time.reset()
        self.batch_time.reset()
        self.i = 0

    def iter_callback(self, d):
        bt = d['batch_time']
        dt = d['data_time']
        if self.i >= self.warmup_iter:
            self.data_time.update(dt)
            self.batch_time.update(bt)
        self.i += 1

        print("Iter: [{}]\tbatch: {:.3f}\tdata: {:.3f}\timg/s (compute): {:.3f}\timg/s (total): {:.3f}".format(
            self.i, dt + bt, dt,
            self.total_bs / bt, self.total_bs / (bt + dt)))

    def end_callback(self):
        print("{} summary\tBatch Time: {:.3f}\tData Time: {:.3f}\timg/s (compute): {:.1f}\timg/s (total): {:.1f}".format(
              self.name,
              self.batch_time.avg, self.data_time.avg,
              self.total_bs / self.batch_time.avg,
              self.total_bs / (self.batch_time.avg + self.data_time.avg)))


class EpochLogger(object):
    def __init__(self, name, total_iterations, args):
        self.name = name
        self.args = args
        self.print_freq = args.print_freq
        self.total_iterations = total_iterations
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.loss = AverageMeter()
        self.time = AverageMeter()
        self.data = AverageMeter()

    def iter_callback(self, epoch, iteration, d):
        self.top1.update(d['top1'], d['size'])
        self.top5.update(d['top5'], d['size'])
        self.loss.update(d['loss'], d['size'])
        self.time.update(d['time'], d['size'])
        self.data.update(d['data'], d['size'])

        if iteration % self.print_freq == 0:
            print('{0}:\t{1} [{2}/{3}]\t'
                  'Time {time.val:.3f} ({time.avg:.3f})\t'
                  'Data time {data.val:.3f} ({data.avg:.3f})\t'
                  'Speed {4:.3f} ({5:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                  self.name, epoch, iteration, self.total_iterations,
                  self.args.world_size * self.args.batch_size / self.time.val,
                  self.args.world_size * self.args.batch_size / self.time.avg,
                  time=self.time,
                  data=self.data,
                  loss=self.loss,
                  top1=self.top1,
                  top5=self.top5))

    def epoch_callback(self, epoch):
        print('{0} epoch {1} summary:\t'
              'Time {time.avg:.3f}\t'
              'Data time {data.avg:.3f}\t'
              'Speed {2:.3f}\t'
              'Loss {loss.avg:.4f}\t'
              'Prec@1 {top1.avg:.3f}\t'
              'Prec@5 {top5.avg:.3f}'.format(
              self.name, epoch,
              self.args.world_size * self.args.batch_size / self.time.avg,
              time=self.time, data=self.data,
              loss=self.loss, top1=self.top1, top5=self.top5))

        self.top1.reset()
        self.top5.reset()
        self.loss.reset()
        self.time.reset()
        self.data.reset()


class PrintLogger(object):
    def __init__(self, train_iterations, val_iterations, args):
        self.train_logger = EpochLogger("Train", train_iterations, args)
        self.val_logger = EpochLogger("Eval", val_iterations, args)

    def train_iter_callback(self, epoch, iteration, d):
        self.train_logger.iter_callback(epoch, iteration, d)

    def train_epoch_callback(self, epoch):
        self.train_logger.epoch_callback(epoch)
        
    def val_iter_callback(self, epoch, iteration, d):
        self.val_logger.iter_callback(epoch, iteration, d)

    def val_epoch_callback(self, epoch):
        self.val_logger.epoch_callback(epoch)
        
    def experiment_timer(self, exp_duration):
        print("Experiment took {} seconds".format(exp_duration))

    def end_callback(self):
        pass


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# }}}

def should_backup_checkpoint(args):
    def _sbc(epoch):
        return args.gather_checkpoints and (epoch < 10 or epoch % 10 == 0)
    return _sbc
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        print("SAVING")

        torch.save(state, filename)

def timed_generator(gen):
    start = time.time()
    for g in gen:
        end = time.time()
        t = end - start
        yield g, t
        start = time.time()


def timed_function(f):
    def _timed_function(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        return ret, time.time() - start
    return _timed_function





def adjust_learning_rate(args):
    def _alr(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if epoch < args.warmup:
            lr = args.lr * (epoch + 1) / (args.warmup + 1)

        else:
            if epoch < args.first_m*args.multiplier:
                p = 0
            elif epoch <args.second_m*args.multiplier:
                p = 1
            elif epoch < args.third_m*args.multiplier:
                p = 2
            else:
                p = 3
            lr = args.lr * (0.1 ** p)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= torch.distributed.get_world_size()
    return rt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    update_iter = 0
    add_parser_arguments(parser)
    args = parser.parse_args()
    cudnn.benchmark = True
    print(args)
    # if not os.path.exists(args.save):
    #     os.makedirs(args.save)
    main()
