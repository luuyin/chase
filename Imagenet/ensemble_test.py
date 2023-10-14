from __future__ import print_function
import sys
import os
import shutil
import time
import argparse
import logging
import hashlib
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet, MLP_CIFAR10
from sparselearning.resnet_cifar100 import ResNet34, ResNet18
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, plot_class_feature_histograms
import torchvision
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {}
models['Resnet34'] = (ResNet34, [10])
models['ResNet18'] = (ResNet18, [10])
models['MLPCIFAR10'] = (MLP_CIFAR10,[])
models['lenet5'] = (LeNet_5_Caffe,[])
models['lenet300-100'] = (LeNet_300_100,[])
models['alexnet-s'] = (AlexNet, ['s', 10])
models['alexnet-b'] = (AlexNet, ['b', 10])
models['vgg-c'] = (VGG16, ['C', 10])
models['vgg-d'] = (VGG16, ['D', 10])
models['vgg-like'] = (VGG16, ['like', 10])
models['wrn-28-10'] = (WideResNet, [28, 10, 10, 0.3])
models['wrn-28-2'] = (WideResNet, [28, 2, 10, 0.3])
models['wrn-22-8'] = (WideResNet, [22, 8, 10, 0.3])
models['wrn-16-8'] = (WideResNet, [16, 8, 10, 0.3])
models['wrn-16-10'] = (WideResNet, [16, 10, 10, 0.3])

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

ite_step = 0
update_step = 0
def train(args, model, device, train_loader, optimizer, epoch, mask=None):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)

        # l2 loss
        if args.l2_regu:
            l2_reg = None
            for name, weight in model.named_parameters():
                if name not in mask.masks: continue
                if l2_reg is None:
                    l2_reg =  (mask.nonzero_masks[name] * weight).norm(2)

                else:
                    l2_reg = l2_reg + (mask.nonzero_masks[name] * weight).norm(2)

            loss = loss + args.l2 * l2_reg

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None: mask.step()
        else: optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}% '.format(
                epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                100. * batch_idx / len(train_loader), loss.item(), correct, n, 100. * correct / float(n)))

        # update sparse topology
        global ite_step
        global update_step
        ite_step += 1
        if args.sparse and ite_step % args.update_frequency == 0 and ite_step > args.start_iter:
            mask.at_end_of_epoch(epoch)
            update_step += 1
    # training summary
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

def evaluate_ensemble(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    current_fold_preds = []
    test_data = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            logits = model(data)
            softmax_preds = torch.nn.Softmax(dim=1)(input=logits)
            current_fold_preds.append(softmax_preds)
            test_data.append(target)
    current_fold_preds = torch.cat(current_fold_preds, dim=0)
    test_data = torch.cat(test_data, dim=0)

    return current_fold_preds, test_data


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--grow-switch', default='', type=str,
                           help='flag to switch another grow initialization')
    parser.add_argument('--grow-zero', action='store_true',
                           help='flag to switch another grow initialization')
    parser.add_argument('--grow-max', action='store_true',
                           help='flag to switch another grow initialization')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--batch-size-jac', type=int, default=200, metavar='N',
                        help='batch size for jac (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
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
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N',
                        help='how many iterations to train between mask update')
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
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--start_iter', type=int, default=1, help='How many times the model should be run before pruning. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--nolr_scheduler', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--l2_regu', action='store_true', default=False,
                        help='if ture, add a l2 norm')
    parser.add_argument('--no_rewire_extend', action='store_true', default=False,
                        help='if ture, only do rewire for 250 epoochs')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--mgpu', action='store_true', help='Enable snip initialization. Default: True.')
    sparselearning.core.add_sparse_args(parser)

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
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))

        if args.data == 'mnist':
            train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
        elif args.data == 'cifar10':
            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split,
                                                                              max_threads=args.max_threads)
        elif args.data == 'cifar100':
            cifar_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            cifar_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            # Data
            print('==> Preparing data..')
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar_mean, cifar_std),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar_mean, cifar_std),
            ])

            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                     transform=transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=2)

            testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                    transform=transform_test)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                                                      num_workers=2)

        if args.model not in models:
            print('You need to select an existing model via the --model argument. Available models include: ')
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        else:
            if args.model == 'ResNet18':
                model = ResNet18(c=10).to(device)
            else:
                cls, cls_args = models[args.model]
                if args.data == 'cifar100':
                    cls_args[2] = 100
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

        if args.resume:
                print_and_log("=> loading checkpoint '{}'".format(args.resume))
                test_loss = 0
                arr = os.listdir(args.resume)
                arr = sorted(arr)
                all_folds_preds = []

                for file in range(len(arr)):

                    print(arr[file])
                    checkpoint = torch.load(args.resume + str(arr[file]))
                    model.load_state_dict(checkpoint)
                    print_and_log('Testing...')
                    current_fold_preds, target = evaluate_ensemble(args, model, device, test_loader)
                    all_folds_preds.append(current_fold_preds)

                output_mean = torch.mean(torch.stack(all_folds_preds, dim=0), dim=0)
                print(output_mean.size())
                # test_loss = F.nll_loss(output_mean, target, reduction='sum').item()  # sum up batch loss
                pred = output_mean.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                n = target.shape[0]
                print_and_log('\n{}: Accuracy: {}/{} ({:.3f}%)\n'.format(
                    'Test evaluation',
                     correct, n, 100. * correct / float(n)))



if __name__ == '__main__':
   main()
