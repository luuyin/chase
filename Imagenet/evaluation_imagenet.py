'''
evaluation


be careful for attack clamp 1 or 255

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
# import torchvision.models as models
import resnet as models
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import re
# from advertorch.utils import NormalizeByChannelMeanStd
# from advertorch.attacks import LinfPGDAttack, L2PGDAttack
# from datasets import cifar10_dataloaders_for_eval, cifar100_dataloaders_for_eval, cifar_c_dataloaders, adv_image_dataset

parser = argparse.ArgumentParser(description='PyTorch Testing Training')
##################################### general setting #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--pretrained', type=str, default=None, help="pretrained model")
parser.add_argument('--eval_mode', type=str, default=None, help="evaluation mode")

##################################### training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')

##################################### Attack setting #################################################
parser.add_argument('--norm', default='linf', type=str, help='linf or l2')
parser.add_argument('--test_eps', default=8, type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=1, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=8, type=float, help='step size of attack during testing')
parser.add_argument('--test_randinit_off', action='store_false', help='randinit usage flag (default: on)')



model_names = models.resnet_versions.keys()
model_configs = models.resnet_configs.keys()

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet50)')
parser.add_argument('--model-config', '-c', metavar='CONF', default='classic',
                    choices=model_configs,
                    help='model configs: ' +
                    ' | '.join(model_configs) + '(default: classic)')

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


def main():

    global args, best_sa
    args = parser.parse_args()
    print(args)

    # init model


    arch=(args.arch, args.model_config)
    print("=> creating model '{}'".format(arch))
    model = models.build_resnet(arch[0], arch[1])


    # torch.cuda.set_device(int(args.gpu))
    criterion = nn.CrossEntropyLoss()

    model = torch.nn.DataParallel(model).cuda()



   # data  folder

    args.save_dir = '/scratch-shared/shiwei/imagenet_models/Granet/expriment/resume/cycle_num_2/density_0.2/pretrain_epoch_93/cyclic_epochs_8/'
    model_files = os.listdir(args.save_dir)
    model_files = model_files_filter(model_files)
    model_files = sorted_nicely(model_files)

    args.pretrained= model_files
    args.eval_mode='adv'
    ################################ eval adv robustness ######################################
    if args.eval_mode == 'adv':
        print('* evaluation robustness')
        print('* Adversarial settings')
        print('* norm = {}'.format(args.norm))
        print('* eps = {}'.format(args.test_eps))
        print('* steps = {}'.format(args.test_step))
        print('* alpha = {}'.format(args.test_gamma))
        print('* randinit = {}'.format(args.test_randinit_off))

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.data, 'val'), transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

        test_ensemble(test_loader, model, args.pretrained, args)
        test_ensemble_adv(test_loader, model, args.pretrained, criterion, args)

    ################################ eval natural robustness and corruption ####################
    elif args.eval_mode == 'img-a':
        # imagenet-a
        from datasets import imagenet_a_index
        indices_a = imagenet_a_index()
        test_a_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.data, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        test_ensemble_a(test_a_loader, model, args.pretrained, args, indices_a)

    elif args.eval_mode == 'img-r':
        # imagenet-r
        from datasets import imagnet_r_index
        indices_r = imagnet_r_index()
        test_r_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.data, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        test_ensemble_a(test_r_loader, model, args.pretrained, args, indices_r)

    elif args.eval_mode == 'corruption':
        print('* evaluation natural robustness and corruption')

        file_list = os.listdir(args.data)
        file_list.sort()

        for file_name in file_list:
            for severity in range(1,6):
                print('attack_type={}'.format(file_name), 'severity={}'.format(severity))
                imagenet_c_test_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(os.path.join(args.data, file_name, str(severity)), transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor()
                    ])),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)

                test_ensemble(imagenet_c_test_loader, model, args.pretrained, args)

    ############################## Calibration Evaluation #######################################
    elif args.eval_mode == 'calibration':

        from datasets import imagenet_a_index
        indices_a = imagenet_a_index()
        test_a_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.data, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        ensemble_calibration_imagenet_a(test_a_loader, model, args.pretrained, args, indices_a)

        # test_loader = torch.utils.data.DataLoader(
        #     datasets.ImageFolder(os.path.join(args.data, 'val'), transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor()
        #     ])),
        #     batch_size=args.batch_size, shuffle=False,
        #     num_workers=4, pin_memory=True)

        # ensemble_calibration(test_loader, model, args.pretrained, args)
        # ensemble_calibration_corruption(model, args.pretrained, args)


def test_ensemble_part(model, val_loader):
    """
    Run evaluation
    """
    model.eval()
    current_fold_preds = []
    test_data = []
    for i, (input, target) in enumerate(val_loader):

        input = (input*255).cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

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
            rand_init=args.test_randinit_off, clip_min=0.0, clip_max=255, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit_off, clip_min=0.0, clip_max=255, targeted=False
        )  

    result = {}

    all_adv_img = []
    all_target = []

    model.eval()
    for i, (input, target) in enumerate(val_loader):

        input = (input*255).cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        #adv samples
        input_adv = adversary.perturb(input, target)

        all_adv_img.append(input_adv.cpu().detach())
        all_target.append(target.cpu())

    all_adv_img = torch.cat(all_adv_img, dim=0)
    all_target = torch.cat(all_target, dim=0)

    result['data'] = all_adv_img
    result['label'] = all_target

    return result

def test_ensemble_part_adv(model, val_loader):
    """
    Run evaluation
    """
    model.eval()
    current_fold_preds = []
    test_data = []
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

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

def test_ensemble_part_a(model, val_loader, indices_in_1k):
    """
    Run evaluation
    """
    model.eval()
    current_fold_preds = []
    test_data = []
    for i, (input, target) in enumerate(val_loader):

        input = (input*255).cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(input)[:,indices_in_1k]
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

def extract_prediction(val_loader, model):
    """
    Run evaluation
    """
    model.eval()
    start = time.time()

    y_pred = []
    y_true = []

    for i, (input, target) in enumerate(val_loader):

        input = (input*255).cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

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

def extract_prediction_image_a(val_loader, model, indices_in_1k):
    """
    Run evaluation
    """
    model.eval()
    start = time.time()

    y_pred = []
    y_true = []

    for i, (input, target) in enumerate(val_loader):

        input = (input*255).cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(input)[:,indices_in_1k]
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

def test_ensemble(val_loader, model, direction, args):
    # arr = os.listdir(direction)
    # arr = sorted(arr)
    all_folds_preds = []

    for path in direction:

        # print(arr[checkpoint])
        # checkpoint=torch.load((args.save_dir + '/' +str(path), map_location = lambda storage, loc: storage.cuda(args.gpu)))
        checkpoint=(args.save_dir + '/' +str(path))
        pretrain_weight = torch.load(checkpoint)
        model.load_state_dict(pretrain_weight['state_dict'], strict=False)
        current_fold_preds, target = test_ensemble_part(model, val_loader)
        all_folds_preds.append(current_fold_preds)

    output_mean = torch.mean(torch.stack(all_folds_preds[:2], dim=0), dim=0)
    print(output_mean.size())
    pred = output_mean.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    n = target.shape[0]
    print('\n{}: Ensemble-2-Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation',
            correct, n, 100. * correct / float(n)))


    output_mean = torch.mean(torch.stack(all_folds_preds, dim=0), dim=0)
    print(output_mean.size())

    pred = output_mean.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    n = target.shape[0]
    print('\n{}: Ensemble-all-Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation',
            correct, n, 100. * correct / float(n)))

def test_ensemble_adv(val_loader, model, direction, criterion, args):

    arr = os.listdir(direction)
    arr = sorted(arr)

    for checkpoint in range(len(arr)):

        all_folds_preds = []
        print('attack with', arr[checkpoint])
        pretrain_weight = torch.load(os.path.join(direction, str(arr[checkpoint])))
        model.load_state_dict(pretrain_weight['state_dict'], strict=False)

        adv_data = generate_adv_img(val_loader, model, criterion, args)
        adv_dataset = adv_image_dataset(adv_data)
        adv_datalodaer = DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        for val_checkpoint in range(len(arr)):
            print('eval with', arr[val_checkpoint])
            pretrain_weight = torch.load(os.path.join(direction, str(arr[val_checkpoint])))
            model.load_state_dict(pretrain_weight['state_dict'], strict=False)

            current_fold_preds, target = test_ensemble_part_adv(model, adv_datalodaer)
            all_folds_preds.append(current_fold_preds)

        output_mean = torch.mean(torch.stack(all_folds_preds[:2], dim=0), dim=0)
        print(output_mean.size())

        pred = output_mean.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        n = target.shape[0]
        print('\n{}: Ensemble-2-Accuracy: {}/{} ({:.3f}%)\n'.format(
            'Test evaluation',
                correct, n, 100. * correct / float(n)))

        output_mean = torch.mean(torch.stack(all_folds_preds, dim=0), dim=0)
        print(output_mean.size())

        pred = output_mean.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        n = target.shape[0]
        print('\n{}: Ensemble-all-Accuracy: {}/{} ({:.3f}%)\n'.format(
            'Test evaluation',
                correct, n, 100. * correct / float(n)))

def test_ensemble_a(val_loader, model, direction, args, indices_in_1k):
    arr = os.listdir(direction)
    arr = sorted(arr)
    all_folds_preds = []

    for checkpoint in range(len(arr)):

        print(arr[checkpoint])
        pretrain_weight = torch.load(os.path.join(direction, str(arr[checkpoint])))
        model.load_state_dict(pretrain_weight['state_dict'], strict=False)
        current_fold_preds, target = test_ensemble_part_a(model, val_loader, indices_in_1k)
        all_folds_preds.append(current_fold_preds)

    output_mean = torch.mean(torch.stack(all_folds_preds[:2], dim=0), dim=0)
    print(output_mean.size())

    pred = output_mean.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    n = target.shape[0]
    print('\n{}: Ensemble-2-Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation',
            correct, n, 100. * correct / float(n)))

    output_mean = torch.mean(torch.stack(all_folds_preds, dim=0), dim=0)
    print(output_mean.size())

    pred = output_mean.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    n = target.shape[0]
    print('\n{}: Ensemble-all-Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation',
            correct, n, 100. * correct / float(n)))

def ensemble_calibration(val_loader, model, direction, args):

    arr = os.listdir(direction)
    arr = sorted(arr)
    all_folds_preds = []

    for checkpoint in range(len(arr)):

        print(arr[checkpoint])
        pretrain_weight = torch.load(os.path.join(direction, str(arr[checkpoint])))
        model.load_state_dict(pretrain_weight['state_dict'], strict=False)
        y_pred, y_true = extract_prediction(val_loader, model)
        all_folds_preds.append(y_pred)

        ece = expected_calibration_error(y_true, y_pred)
        nll = F.nll_loss(torch.from_numpy(y_pred).log(), torch.from_numpy(y_true), reduction="mean")
        print('* ECE = {}'.format(ece))
        print('* NLL = {}'.format(nll))

    output_mean = np.mean(np.stack(all_folds_preds[:2], 0), 0)
    print(output_mean.shape)

    ece = expected_calibration_error(y_true, output_mean)
    nll = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(y_true), reduction="mean")
    print('* ECE-2 = {}'.format(ece))
    print('* NLL-2 = {}'.format(nll))


    output_mean = np.mean(np.stack(all_folds_preds, 0), 0)
    print(output_mean.shape)

    ece = expected_calibration_error(y_true, output_mean)
    nll = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(y_true), reduction="mean")
    print('* ECE-all = {}'.format(ece))
    print('* NLL-all = {}'.format(nll))

def ensemble_calibration_imagenet_a(val_loader, model, direction, args, indices_a):

    arr = os.listdir(direction)
    arr = sorted(arr)
    all_folds_preds = []

    for checkpoint in range(len(arr)):

        print(arr[checkpoint])
        pretrain_weight = torch.load(os.path.join(direction, str(arr[checkpoint])))
        model.load_state_dict(pretrain_weight['state_dict'], strict=False)
        y_pred, y_true = extract_prediction_image_a(val_loader, model, indices_a)
        all_folds_preds.append(y_pred)

        ece = expected_calibration_error(y_true, y_pred)
        nll = F.nll_loss(torch.from_numpy(y_pred).log(), torch.from_numpy(y_true), reduction="mean")
        print('* ECE = {}'.format(ece))
        print('* NLL = {}'.format(nll))

    output_mean = np.mean(np.stack(all_folds_preds[:2], 0), 0)
    print(output_mean.shape)

    ece = expected_calibration_error(y_true, output_mean)
    nll = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(y_true), reduction="mean")
    print('* ECE-2 = {}'.format(ece))
    print('* NLL-2 = {}'.format(nll))


    output_mean = np.mean(np.stack(all_folds_preds, 0), 0)
    print(output_mean.shape)

    ece = expected_calibration_error(y_true, output_mean)
    nll = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(y_true), reduction="mean")
    print('* ECE-all = {}'.format(ece))
    print('* NLL-all = {}'.format(nll))

def ensemble_calibration_corruption(model, direction, args):

    arr = os.listdir(direction)
    arr = sorted(arr)
    all_folds_preds_c = []

    for checkpoint in range(len(arr)):

        all_preds = []
        all_targets = []

        print(arr[checkpoint])
        pretrain_weight = torch.load(os.path.join(direction, str(arr[checkpoint])))
        model.load_state_dict(pretrain_weight['state_dict'], strict=False)

        file_list = os.listdir(args.data+'-c')
        file_list.sort()

        for file_name in file_list:
            for severity in range(1,6):
                print('attack_type={}'.format(file_name), 'severity={}'.format(severity))
                imagenet_c_test_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(os.path.join(args.data+'-c', file_name, str(severity)), transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor()
                    ])),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)

                y_pred, y_true = extract_prediction(imagenet_c_test_loader, model)
                np.save(os.path.join(direction, '{}-{}-{}.npy'.format(arr[checkpoint], file_name, severity)), y_pred)    
                np.save(os.path.join(direction, '{}-{}-{}-label.npy'.format(arr[checkpoint], file_name, severity)), y_true)              
                print('* c-Acc = {}'.format(np.mean(np.argmax(y_pred, 1)==y_true)))
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

    output_mean = np.mean(np.stack(all_folds_preds_c[:2], 0), 0)
    print(output_mean.shape)
    ece = expected_calibration_error(all_targets, output_mean)
    nll = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(all_targets), reduction="mean")
    print('* c-ECE-2 = {}'.format(ece))
    print('* c-NLL-2 = {}'.format(nll))
    print('* Acc-2 = {}'.format(np.mean(np.argmax(output_mean, 1)==all_targets)))


    output_mean = np.mean(np.stack(all_folds_preds_c, 0), 0)
    print(output_mean.shape)
    ece = expected_calibration_error(all_targets, output_mean)
    nll = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(all_targets), reduction="mean")
    print('* c-ECE-a = {}'.format(ece))
    print('* c-NLL-a = {}'.format(nll))
    print('* Acc-a = {}'.format(np.mean(np.argmax(output_mean, 1)==all_targets)))


if __name__ == '__main__':
    main()


