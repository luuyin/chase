from __future__ import print_function
import sys
import os
import shutil
import time
import argparse
import logging
import hashlib
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import re
import random
import sparselearning
from torch.autograd import Variable
from models import cifar_resnet, initializers, vgg
from models.init_utils import weights_init

import torch.nn as nn



from sparselearning.core_dst_chase import Masking, CosineDecay, LinearDecay
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet, MLP_CIFAR10,MLP_CIFAR100
from sparselearning.resnet_cifar100 import ResNet34, ResNet18,ResNet50
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, plot_class_feature_histograms, get_cifar100_dataloaders
# from sparselearning.flops import print_model_param_nums,count_model_param_flops,print_inf_time

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {}

models['vgg-like'] = (VGG16, ['like', 10])
models['wrn-28-10'] = (WideResNet, [28, 10, 10, 0.0])
models['wrn-16-8'] = (WideResNet, [16, 8, 10, 0.0])
models['wrn-16-10'] = (WideResNet, [16, 10, 10, 0.0])

torch.backends.cudnn.benchmark = True

def get_cfg(net):
    cfg = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            cfg.append(layer.weight.shape[0])
        elif isinstance(layer, nn.MaxPool2d):
            cfg.append('M')
        elif isinstance(layer, nn.AvgPool2d):
            cfg.append('A')
    return cfg


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr




# save model and print
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("SAVING")
    torch.save(state, filename)



def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(args.model, args.density, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

# trian and eval

def train(args, model, device, train_loader, epoch,mask=None):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    num_iters = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        if args.fp16: data = data.half()

 
        output = model(data)
        loss = F.nll_loss(output, target)


        output = model(data)

        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]


        mask.optimizer.zero_grad()
        if args.fp16:
            mask.optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None: mask.step()
        else: mask.optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}% '.format(
                epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                100. * batch_idx / len(train_loader), loss.item(), correct, n, 100. * correct / float(n)))



    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Training summary' ,
        train_loss/batch_idx, correct, n, 100. * correct / float(n)))

def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n)
    


def main():

    print ("cuda counts",torch.cuda.device_count())
    print ("current ",torch.cuda.current_device())

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=320, metavar='N',
                        help='number of epochs to train (default: 100)')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--nolrsche', action='store_true', default=False,
                        help='disable learning rate decay')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--mgpu', action='store_true', help='Enable snip initialization. Default: True.')
    parser.add_argument('--indicate_method', type=str,default='Results',help='indicate_method for save path')




    parser.add_argument('--layer_interval', default=10, type=int,help='wider_interval')
    parser.add_argument('--start_layer_rate', default=0.1, type=float,help='layer_ratio')





    sparselearning.core_dst_chase.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('='*80)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))

        if args.data == 'mnist':
            train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
            outputs=10
        elif args.data == 'cifar10':
            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split, max_threads=args.max_threads)
            outputs = 10
        elif args.data == 'cifar100':
            train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split, max_threads=args.max_threads)
            outputs = 100
            
        # init model
        if args.model == 'cifar_resnet_32':
            model = cifar_resnet.Model.get_model_from_name('cifar_resnet_32', initializer=initializers.kaiming_normal, outputs=outputs).to(device)

        elif args.model == 'cifar_resnet_20':
            model = cifar_resnet.Model.get_model_from_name('cifar_resnet_20', initializer=initializers.kaiming_normal, outputs=outputs).to(device)

        
        
        elif args.model == 'vgg19':
            model = vgg.VGG(depth=19, dataset=args.data, batchnorm=True).to(device)
        elif args.model == 'ResNet50':
            model = ResNet50(c=outputs).to(device)
        elif args.model == 'ResNet34':
            model = ResNet34(c=outputs).to(device)
        elif args.model == 'MLP_CIFAR100':
            model=MLP_CIFAR100().to(device)
        else:
            cls, cls_args = models[args.model]
            if args.model=="vgg-like":
                cls_args[1] = outputs
            else:
                cls_args[2] = outputs
            model = cls(*(cls_args + [args.save_features, args.bench])).to(device)
      

            print_and_log(model)
            print_and_log('=' * 60)
            print_and_log(args.model)
            print_and_log('=' * 60)

            print_and_log('=' * 60)
            print_and_log('Prune mode: {0}'.format(args.death))
            print_and_log('Growth mode: {0}'.format(args.growth))
            print_and_log('Redistribution mode: {0}'.format(args.redistribution))
            print_and_log('=' * 60)

        print ("model", args.model)
        
        if args.mgpu:
            print('Using multi gpus')
            model = torch.nn.DataParallel(model).to(device)

        
  

        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')
        
        # get training epochs
        pretrain_epoch=args.epochs
        lr_milestones=[int(0.5*pretrain_epoch), int(0.75*pretrain_epoch)]
        
        if args.nolrsche:
            lr_scheduler = None
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=lr_milestones, last_epoch=-1)

        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale = None,
                                       dynamic_loss_scale = True,
                                       dynamic_loss_args = {'init_scale': 2 ** 16})
            model = model.half()


        # reuse the model
        if args.resume:
            print_and_log("=> loading checkpoint '{}'".format(args.resume))

            model_files = os.listdir(args.resume)
            print (model_files)
            checkpoint = torch.load(args.resume + str(model_files[0]))

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])



        # mask = None
        # if args.sparse:
        decay = CosineDecay(args.death_rate+args.density, len(train_loader) * (args.epochs-args.stop_dst_epochs),args.density)
        mask = Masking(optimizer,death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                        redistribution_mode=args.redistribution, args=args,train_loader=train_loader)
        mask.add_module(model, sparse_init=args.sparse_init, density=args.density)

       
        save_dir = "./Chase_models/saved_models/"+str(args.indicate_method)   + '/density_' + str(args.density) + '/stop_gmp_epochs_' + str(args.stop_gmp_epochs)   +  '/epoch_' + str(args.epochs) +  '/layer_interval_' + str(args.layer_interval)+   '/start_layer_rate_' + str(args.start_layer_rate)
        
        print ("save_dir",save_dir)
        try:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
        except Exception as e:
            print(e)
            pass
        

        
        print ("=====================================")
        print ("begin pre training")



        best_acc=0
        for epoch in range(0, pretrain_epoch):

            t0 = time.time()

            train(args, model, device, train_loader, epoch,mask)


            if args.valid_split > 0.0:
                val_acc = evaluate(args, model, device, valid_loader)


            if epoch==lr_milestones[0]-1:
                adjust_learning_rate(mask.optimizer,args.lr* 0.1)
            elif epoch==lr_milestones[1]-1:
                adjust_learning_rate(mask.optimizer,args.lr* 0.01)

            save_best=True
            
            if epoch>=(args.epochs-args.stop_dst_epochs):

                if val_acc > best_acc:
                    best_acc = val_acc
                    print ("best_acc",best_acc)

                    best_state=copy.deepcopy(model.state_dict())
                    best_masks=copy.deepcopy(mask.masks)
                    best_filter_names=copy.deepcopy(mask.filter_names)


                    if save_best:
                        print('Saving best pre model')
                        
                        save_checkpoint({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, filename=os.path.join(save_dir, 'premodel_best.pth'))

            elif epoch==args.epochs-1 :

                best_acc = val_acc
                best_state=copy.deepcopy(model.state_dict())
                best_masks=copy.deepcopy(mask.masks)
                best_filter_names=copy.deepcopy(mask.filter_names)

            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(mask.optimizer.param_groups[0]['lr'], time.time() - t0))

        print ("best_acc",best_acc) 

        print ("before merge")
        mask.masks=best_masks
        mask.filter_names=best_filter_names
        model.load_state_dict(best_state)
        val_acc = evaluate(args, model, device, valid_loader)

        print ("begin merge")
        mask.merge_filters()

        val_acc = evaluate(args, model, device, valid_loader)


        print ("update bn")
        torch.optim.swa_utils.update_bn(train_loader, model,"cuda")
        val_acc = evaluate(args, model, device, valid_loader)

        print (model )




        print ("Congs!! ALL DONE, GOOD JOB!!")
            

  
if __name__ == '__main__':
   main()
