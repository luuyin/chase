'''
evaluation

1. natural robustness and corruption: CIFAR-10-C/ImageNet-C
2. Adversarial robustness: FGSM/PGD
3. ImageNet-A and ImageNet-R
4. OOD detection AUC

'''

import os
import pdb
import time 
import pickle
import random
import shutil
import argparse
import numpy as np  
import sklearn.metrics as sk
import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from advertorch.utils import NormalizeByChannelMeanStd
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from datasets import cifar10_dataloaders_for_eval, cifar100_dataloaders_for_eval, cifar_c_dataloaders, adv_image_dataset

parser = argparse.ArgumentParser(description='PyTorch Testing Training')
##################################### general setting #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--arch', type=str, default=None, help='architecture', required=True)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--pretrained', type=str, default=None, help="pretrained model")
parser.add_argument('--eval_mode', type=str, default=None, help="evaluation mode")
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset for evaluation')

##################################### training setting #################################################
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')

##################################### Attack setting #################################################
parser.add_argument('--norm', default='linf', type=str, help='linf or l2')
parser.add_argument('--test_eps', default=(8/255), type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=(2/255), type=float, help='step size of attack during testing')
parser.add_argument('--test_randinit_off', action='store_false', help='randinit usage flag (default: on)')


def main():

    global args, best_sa
    args = parser.parse_args()
    print(args)
    
    torch.cuda.set_device(int(args.gpu))
    criterion = nn.CrossEntropyLoss()

    ################################ eval adv robustness ######################################
    if args.eval_mode == 'adv':

        print('* evaluation robustness')
        print('* Adversarial settings')
        print('* norm = {}'.format(args.norm))
        print('* eps = {}'.format(args.test_eps))
        print('* steps = {}'.format(args.test_step))
        print('* alpha = {}'.format(args.test_gamma))
        print('* randinit = {}'.format(args.test_randinit_off))

        # prepare datasets
        # normalize layer in model instead of datas\loader
        if args.dataset == 'cifar10':
            print('* dataset = cifar10')
            classes = 10
            normal = NormalizeByChannelMeanStd(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            _, _, test_loader = cifar10_dataloaders_for_eval(batch_size=args.batch_size, data_dir=args.data, normalize=None)
        elif args.dataset == 'cifar100':
            print('* dataset = cifar100')
            classes = 100
            normal = NormalizeByChannelMeanStd(
                    mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            _, _, test_loader = cifar100_dataloaders_for_eval(batch_size=args.batch_size, data_dir=args.data, normalize=None)
        
        from sparselearning.models import WideResNet
        from sparselearning.resnet_cifar100 import ResNet18
        if args.arch == 'res18':
            model = ResNet18(c=classes)
        elif args.arch == 'wideres':
            model = WideResNet(28, 10, classes, 0.3)

        model = nn.Sequential(normal, model)
        model.cuda()

        test_ensemble_adv(test_loader, model, args.pretrained, criterion, args)

    ################################ eval natural robustness and corruption ####################
    elif args.eval_mode == 'corruption':

        print('* evaluation natural robustness and corruption')
        print('* ', args.dataset)

        ################################ loading pretrained model ################################
        from sparselearning.models import WideResNet
        from sparselearning.resnet_cifar100 import ResNet18

        if args.dataset == 'cifar10':
            classes = 10
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2023, 0.1994, 0.2010))
            _,_,test_loader = cifar10_dataloaders_for_eval(args.batch_size, args.data, normalize=normalize)
        elif args.dataset == 'cifar100':
            classes = 100
            normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                            (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
            _,_,test_loader = cifar100_dataloaders_for_eval(args.batch_size, args.data, normalize=normalize)

        if args.arch == 'res18':
            model = ResNet18(c=classes)
        elif args.arch == 'wideres':
            model = WideResNet(28, 10, classes, 0.3)
        model.cuda()

        test_ensemble(test_loader, model, args.pretrained, args)

        if args.dataset == 'cifar10':
            
            cifar_c_path = os.path.join(args.data, 'CIFAR-10-C')
            file_list = os.listdir(cifar_c_path)
            file_list.sort()

            for file_name in file_list:
                if not file_name == 'labels.npy':
                    attack_type = file_name[:-len('.npy')]
                    for severity in range(1,6):
                        print('attack_type={}'.format(attack_type), 'severity={}'.format(severity))
                        cifar10c_test_loader = cifar_c_dataloaders(args.batch_size, cifar_c_path, severity, attack_type, normalize)
                        test_ensemble(cifar10c_test_loader, model, args.pretrained, args)

        elif args.dataset == 'cifar100':
            
            cifar_c_path = os.path.join(args.data, 'CIFAR-100-C')
            file_list = os.listdir(cifar_c_path)
            file_list.sort()

            for file_name in file_list:
                if not file_name == 'labels.npy':
                    attack_type = file_name[:-len('.npy')]
                    for severity in range(1,6):
                        print('attack_type={}'.format(attack_type), 'severity={}'.format(severity))
                        cifar100c_test_loader = cifar_c_dataloaders(args.batch_size, cifar_c_path, severity, attack_type, normalize)
                        test_ensemble(cifar100c_test_loader, model, args.pretrained, args)

    ############################### eval Ood detection with AUC ################################
    elif args.eval_mode == 'ood':
        print('* evaluation OoD detection with AUC')

        ################################ loading pretrained model ################################
        from sparselearning.models import WideResNet
        from sparselearning.resnet_cifar100 import ResNet18
        from datasets import ood_cifar10, ood_cifar100, ood_gaussion, ood_svhn

        if args.dataset == 'cifar10':
            classes = 10
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2023, 0.1994, 0.2010))
            _,_,test_loader = cifar10_dataloaders_for_eval(args.batch_size, args.data, normalize=normalize)
            ood1 = ood_cifar100(args.batch_size, args.data, normalize=normalize)
            ood2 = ood_svhn(args.batch_size, args.data, normalize=normalize)
            ood3 = ood_gaussion(args.batch_size, 10000)
            ood_list = [ood1, ood2, ood3]

        elif args.dataset == 'cifar100':
            classes = 100
            normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                            (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
            _,_,test_loader = cifar100_dataloaders_for_eval(args.batch_size, args.data, normalize=normalize)
            ood1 = ood_cifar10(args.batch_size, args.data, normalize=normalize)
            ood2 = ood_svhn(args.batch_size, args.data, normalize=normalize)
            ood3 = ood_gaussion(args.batch_size, 10000)
            ood_list = [ood1, ood2, ood3]

        if args.arch == 'res18':
            model = ResNet18(c=classes)
        elif args.arch == 'wideres':
            model = WideResNet(28, 10, classes, 0.3)
        model.cuda()

        arr = os.listdir(args.pretrained)
        arr = sorted(arr)

        for ood_loader in ood_list:
            
            all_folds_preds_ind = []
            all_folds_preds_ood = []

            for checkpoint in range(len(arr)):
                print(arr[checkpoint])
                pretrain_weight = torch.load(os.path.join(args.pretrained, str(arr[checkpoint])))
                model.load_state_dict(pretrain_weight)
                model.cuda()

                # ind 
                y_pred_ind, _ = extract_prediction(test_loader, model)
                ind_labels = np.ones(y_pred_ind.shape[0])
                all_folds_preds_ind.append(y_pred_ind)
                ind_scores = np.max(y_pred_ind, 1)

                # ood
                y_pred_ood, _ = extract_prediction(ood_loader, model)
                ood_labels = np.zeros(y_pred_ood.shape[0])
                all_folds_preds_ood.append(y_pred_ood)
                ood_scores = np.max(y_pred_ood, 1)

                labels = np.concatenate([ind_labels, ood_labels])
                scores = np.concatenate([ind_scores, ood_scores])

                auroc = sk.roc_auc_score(labels, scores)
                print('* AUC = {}'.format(auroc))
            
            output_mean_ind = np.mean(np.stack(all_folds_preds_ind, 0), 0)
            output_mean_ood = np.mean(np.stack(all_folds_preds_ood, 0), 0)
            ind_scores = np.max(output_mean_ind, 1)
            ood_scores = np.max(output_mean_ood, 1)
            labels = np.concatenate([ind_labels, ood_labels])
            scores = np.concatenate([ind_scores, ood_scores])

            auroc = sk.roc_auc_score(labels, scores)
            print('* AUC = {}'.format(auroc))

    ############################## Calibration Evaluation #######################################
    elif args.eval_mode == 'calibration':

        print('* evaluation for calibration metric')
        # prepare datasets
        if args.dataset == 'cifar10':
            classes = 10
            cor_path = 'CIFAR-10-C'
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2023, 0.1994, 0.2010))
            _,_,test_loader = cifar10_dataloaders_for_eval(args.batch_size, args.data, normalize=normalize)
        elif args.dataset == 'cifar100':
            print('* dataset = cifar100')
            classes = 100
            cor_path = 'CIFAR-100-C'
            normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                            (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
            _, _, test_loader = cifar100_dataloaders_for_eval(batch_size=args.batch_size, data_dir=args.data, normalize=normalize)
        
        from sparselearning.models import WideResNet
        from sparselearning.resnet_cifar100 import ResNet18

        if args.arch == 'res18':
            model = ResNet18(c=classes)
        elif args.arch == 'wideres':
            model = WideResNet(28, 10, classes, 0.3)
        model.cuda()

        test_ensemble(test_loader, model, args.pretrained, args)
        ensemble_calibration(test_loader, model, args.pretrained, args)
        ensemble_calibration_corruption(cor_path, model, args.pretrained, args, normalize)
    else:
        raise ValueError('Unsupport eval mode')


def test(val_loader, model, criterion, args, keyword):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    start = time.time()
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()
    
        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print(keyword + ' Standard Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def test_adv(val_loader, model, criterion, args, keyword):
    """
    Run adversarial evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit_off, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit_off, clip_min=0.0, clip_max=1.0, targeted=False
        )  

    model.eval()
    start = time.time()
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()

        #adv samples
        input_adv = adversary.perturb(input, target)
        # compute output
        with torch.no_grad():
            output = model(input_adv)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print(keyword + ' Robust Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def test_ensemble_part(model, val_loader):
    """
    Run evaluation
    """
    model.eval()
    current_fold_preds = []
    test_data = []
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()
    
        # compute output
        with torch.no_grad():
            output = model(input)
            softmax_preds = torch.nn.Softmax(dim=1)(input=output)
            current_fold_preds.append(softmax_preds)
            test_data.append(target)

    current_fold_preds = torch.cat(current_fold_preds, dim=0)
    test_data = torch.cat(test_data, dim=0)

    pred = current_fold_preds.argmax(dim=1, keepdim=True)
    correct = pred.eq(test_data.view_as(pred)).sum().item()
    n = test_data.shape[0]
    print('\n{}: Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation',
            correct, n, 100. * correct / float(n)))

    return current_fold_preds, test_data

def generate_adv_img(val_loader, model, criterion, args):

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit_off, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit_off, clip_min=0.0, clip_max=1.0, targeted=False
        )  

    result = {}

    all_adv_img = []
    all_target = []

    model.eval()
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()

        #adv samples
        input_adv = adversary.perturb(input, target)

        all_adv_img.append(input_adv.cpu().detach())
        all_target.append(target.cpu())

    all_adv_img = torch.cat(all_adv_img, dim=0)
    all_target = torch.cat(all_target, dim=0)

    result['data'] = all_adv_img
    result['label'] = all_target

    return result

def test_ensemble_adv(val_loader, model, direction, criterion, args):

    arr = os.listdir(direction)
    arr = sorted(arr)

    for checkpoint in range(len(arr)):

        all_folds_preds = []
        print('attack with', arr[checkpoint])
        pretrain_weight = torch.load(os.path.join(direction, str(arr[checkpoint])))
        model[1].load_state_dict(pretrain_weight)
        model.cuda()

        adv_data = generate_adv_img(val_loader, model, criterion, args)
        adv_dataset = adv_image_dataset(adv_data)
        adv_datalodaer = DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        for val_checkpoint in range(len(arr)):
            print('eval with', arr[val_checkpoint])
            pretrain_weight = torch.load(os.path.join(direction, str(arr[val_checkpoint])))
            model[1].load_state_dict(pretrain_weight)
            model.cuda()
    
            current_fold_preds, target = test_ensemble_part(model, adv_datalodaer)
            all_folds_preds.append(current_fold_preds)

        output_mean = torch.mean(torch.stack(all_folds_preds, dim=0), dim=0)
        print(output_mean.size())

        pred = output_mean.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        n = target.shape[0]
        print('\n{}: Accuracy: {}/{} ({:.3f}%)\n'.format(
            'Test evaluation',
                correct, n, 100. * correct / float(n)))

def test_ensemble(val_loader, model, direction, args):
    arr = os.listdir(direction)
    arr = sorted(arr)
    all_folds_preds = []

    for checkpoint in range(len(arr)):

        print(arr[checkpoint])
        pretrain_weight = torch.load(os.path.join(direction, str(arr[checkpoint])))
        model.load_state_dict(pretrain_weight)
        model.cuda()
        current_fold_preds, target = test_ensemble_part(model, val_loader)
        all_folds_preds.append(current_fold_preds)

    output_mean = torch.mean(torch.stack(all_folds_preds, dim=0), dim=0)
    print(output_mean.size())

    pred = output_mean.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    n = target.shape[0]
    print('\n{}: Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation',
            correct, n, 100. * correct / float(n)))

def extract_scores(val_loader, model):
    """
    Run evaluation
    """
    model.eval()
    start = time.time()

    all_scores = []
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()
    
        # compute output
        with torch.no_grad():
            output = model(input)
            pred = F.softmax(output, dim=1)
            pred = torch.max(pred, dim=1)[0]
            all_scores.append(pred.cpu().numpy())

        if i % args.print_freq == 0:
            end = time.time()
            print('Scores: [{0}/{1}]\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start))
            start = time.time()
    
    all_scores = np.concatenate(all_scores)

    return all_scores

def extract_prediction(val_loader, model):
    """
    Run evaluation
    """
    model.eval()
    start = time.time()

    y_pred = []
    y_true = []

    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()
    
        # compute output
        with torch.no_grad():
            output = model(input)
            pred = F.softmax(output, dim=1)

            y_true.append(target.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

        if i % args.print_freq == 0:
            end = time.time()
            print('Scores: [{0}/{1}]\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start))
            start = time.time()
    
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    print('* prediction shape = ', y_pred.shape)
    print('* ground truth shape = ', y_true.shape)

    return y_pred, y_true

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def expected_calibration_error(y_true, y_pred, num_bins=15):
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0]

def adaptive_calibration_error(y_true, y_pred, num_bins=15):
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    b = np.quantile(prob_y, b)
    b = np.unique(b)
    num_bins = len(b)

    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0]

def static_calibration_error(y_true, y_pred, num_bins=15):
    classes = y_pred.shape[-1]

    o = 0
    for cur_class in range(classes):
        correct = (cur_class == y_true).astype(np.float32)
        prob_y = y_pred[..., cur_class]

        b = np.linspace(start=0, stop=1.0, num=num_bins)
        bins = np.digitize(prob_y, bins=b, right=True)

        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / (y_pred.shape[0] * classes)

def adaptive_static_calibration_error(y_true, y_pred, num_bins=15, threshold=1e-3):
    classes = y_pred.shape[-1]

    o = 0
    for cur_class in range(classes):
        correct = (cur_class == y_true).astype(np.float32)
        prob_y = y_pred[..., cur_class]
        mask = prob_y > threshold
        correct = correct[mask]
        prob_y = prob_y[mask]

        b = np.linspace(start=0, stop=1.0, num=num_bins)
        bins = np.digitize(prob_y, bins=b, right=True)

        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                o += np.abs(np.sum(correct[mask] - prob_y[mask])) / prob_y.shape[0]

    return o / classes

def ensemble_calibration(val_loader, model, direction, args):

    arr = os.listdir(direction)
    arr = sorted(arr)
    all_folds_preds = []

    for checkpoint in range(len(arr)):

        print(arr[checkpoint])
        pretrain_weight = torch.load(os.path.join(direction, str(arr[checkpoint])))
        model.load_state_dict(pretrain_weight)
        model.cuda()
        y_pred, y_true = extract_prediction(val_loader, model)
        all_folds_preds.append(y_pred)


        ece = expected_calibration_error(y_true, y_pred)
        nll = F.nll_loss(torch.from_numpy(y_pred).log(), torch.from_numpy(y_true), reduction="mean")
        print('* ECE = {}'.format(ece))
        print('* NLL = {}'.format(nll))

    output_mean = np.mean(np.stack(all_folds_preds, 0), 0)
    print(output_mean.shape)

    ece = expected_calibration_error(y_true, output_mean)
    nll = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(y_true), reduction="mean")
    print('* ECE = {}'.format(ece))
    print('* NLL = {}'.format(nll))

def ensemble_calibration_corruption(cor_path, model, direction, args, normalize):

    arr = os.listdir(direction)
    arr = sorted(arr)
    all_folds_preds_c = []

    for checkpoint in range(len(arr)):

        all_preds = []
        all_targets = []

        print(arr[checkpoint])
        pretrain_weight = torch.load(os.path.join(direction, str(arr[checkpoint])))
        model.load_state_dict(pretrain_weight)
        model.cuda()

        cifar_c_path = os.path.join(args.data, cor_path)
        file_list = os.listdir(cifar_c_path)
        file_list.sort()

        for file_name in file_list:
            if not file_name == 'labels.npy':
                attack_type = file_name[:-len('.npy')]
                for severity in range(1,6):
                    print('attack_type={}'.format(attack_type), 'severity={}'.format(severity))
                    cifar10c_test_loader = cifar_c_dataloaders(args.batch_size, cifar_c_path, severity, attack_type, normalize)
                    y_pred, y_true = extract_prediction(cifar10c_test_loader, model)

                    print('* Acc = {}'.format(np.mean(np.argmax(y_pred, 1)==y_true)))
                    all_preds.append(y_pred)
                    all_targets.append(y_true)

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        print('* Acc = {}'.format(np.mean(np.argmax(all_preds, 1)==all_targets)))
        all_folds_preds_c.append(all_preds)

        ece = expected_calibration_error(all_targets, all_preds)
        nll = F.nll_loss(torch.from_numpy(all_preds).log(), torch.from_numpy(all_targets), reduction="mean")
        print('* c-ECE = {}'.format(ece))
        print('* c-NLL = {}'.format(nll))

    output_mean = np.mean(np.stack(all_folds_preds_c, 0), 0)
    print(output_mean.shape)

    ece = expected_calibration_error(all_targets, output_mean)
    nll = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(all_targets), reduction="mean")
    print('* c-ECE = {}'.format(ece))
    print('* c-NLL = {}'.format(nll))
    print('* Acc = {}'.format(np.mean(np.argmax(output_mean, 1)==all_targets)))





if __name__ == '__main__':
    main()


