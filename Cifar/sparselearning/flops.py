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
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from colorama import Fore
import sys
import re


try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")




# def print_inf_time(model=None,optimal_batch_size=128,input_res=32):

#     device = torch.device("cuda")
#     model.to(device)
#     model.eval()
#     dummy_input = torch.randn(optimal_batch_size, 3,input_res,input_res, dtype=torch.float).to(device)
#     repetitions=100
#     total_time = 0
#     with torch.no_grad():
#         for rep in range(repetitions):
#             starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
#             starter.record()
#             _ = model(dummy_input)
#             ender.record()
#             torch.cuda.synchronize()
#             curr_time = starter.elapsed_time(ender)/1000
#             total_time += curr_time
#     Throughput =   (repetitions*optimal_batch_size)/total_time
#     print('Final Throughput:',Throughput,"total_time",total_time)

# def print_inf_time(model,input_res=32):
    
#     # cuDnn configurations
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = False

#     # name = model.name
#     # print("     + {} Speed testing... ...".format(name))
#     model = model.cuda()
#     random_input = torch.randn(1,3,input_res, input_res).cuda()

#     model.eval()

#     time_list = []
#     for i in range(10001):
#         torch.cuda.synchronize()
#         tic = time.time()
#         model(random_input)
#         torch.cuda.synchronize()
#         # the first iteration time cost much higher, so exclude the first iteration
#         #print(time.time()-tic)
#         time_list.append(time.time()-tic)
#     time_list = time_list[1:]
#     print("     + Done 10000 iterations inference !")
#     print("     + Total time cost: {}s".format(sum(time_list)))
#     print("     + Average time cost: {}s".format(sum(time_list)/10000))
#     print("     + Frame Per Second: {:.2f}".format(1/(sum(time_list)/10000)))

# def print_inf_time(model,test_loader):
    
#     # cuDnn configurations
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = False

#     # name = model.name
#     # print("     + {} Speed testing... ...".format(name))
#     model = model.cuda()

#     model.eval()

#     tic = time.time()

#     for i in range(20):
#         with torch.no_grad():
#             for data, target in test_loader:
#                 data, target = data.cuda(), target.cuda()

#                 output = model(data)

#     spent_time=time.time()-tic



#     print("     + Average time cost: {}s".format(spent_time))

# def print_inf_time(model1,model2,optimal_batch_size=128,input_res=32):
    
#     device = torch.device("cuda")
#     model1.to(device)
#     model1.eval()
#     dummy_input = torch.randn(optimal_batch_size, 3,input_res,input_res, dtype=torch.float).to(device)
#     repetitions=100
#     total_time = 0
#     with torch.no_grad():
#         for rep in range(repetitions):
#             starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
#             starter.record()
#             _ = model1(dummy_input)
#             ender.record()
#             torch.cuda.synchronize()
#             curr_time = starter.elapsed_time(ender)/1000
#             total_time += curr_time
#     Throughput =   (repetitions*optimal_batch_size)/total_time
#     print('Final Throughput:',Throughput,"total_time",total_time)
    
    
#     device = torch.device("cuda")
#     model2.to(device)
#     model2.eval()
#     dummy_input = torch.randn(optimal_batch_size, 3,input_res,input_res, dtype=torch.float).to(device)
#     repetitions=100
#     total_time = 0
#     with torch.no_grad():
#         for rep in range(repetitions):
#             starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
#             starter.record()
#             _ = model2(dummy_input)
#             ender.record()
#             torch.cuda.synchronize()
#             curr_time = starter.elapsed_time(ender)/1000
#             total_time += curr_time
#     Throughput =   (repetitions*optimal_batch_size)/total_time
#     print('Final Throughput:',Throughput,"total_time",total_time)



    # INIT LOGGERS

def print_inf_time(model,input_res=32):

    dummy_input = torch.randn(128, 3, input_res, input_res, dtype=torch.float).cuda()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        # print(f'warm up iteration {_}')
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            # print(f'forward pass iteration {rep}/{repetitions}')
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)


def print_model_param_nums(model=None):
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([(param!=0).sum() if len(param.size()) == 4 or len(param.size()) == 2 else 0 for name,param in model.named_parameters()])
    print('  + Number of params: %.2f' % (total))


def count_model_param_flops(model=None, input_res=224, multiply_adds=True):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)

        num_weight_params = (self.weight.data != 0).float().sum()
        assert self.weight.numel() == kernel_ops * output_channels, "Not match"
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        num_weight_params = (self.weight.data != 0).float().sum()
        weight_ops = num_weight_params * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement() if self.bias is not None else 0

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            # if isinstance(net, torch.nn.BatchNorm2d):
            #     net.register_forward_hook(bn_hook)
            # if isinstance(net, torch.nn.ReLU):
            #     net.register_forward_hook(relu_hook)
            # if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
            #     net.register_forward_hook(pooling_hook)
            # if isinstance(net, torch.nn.Upsample):
            #     net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input = Variable(torch.rand(3,input_res,input_res).unsqueeze(0), requires_grad = True).cuda()
    out = model(input)



    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))

    print('  + Number of FLOPs: %.2f' % (total_flops/2/1000000))

    return total_flops

