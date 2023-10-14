from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import copy
import numpy as np
import math
from torch.autograd import Variable
import os
import shutil
import time
from funcs import redistribution_funcs, growth_funcs, prune_funcs


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
        # print(" ! Weight decay NOT applied to BN parameters ")
        bn_params = [v for n, v in parameters if 'bn' in n]
        rest_params = [v for n, v in parameters if not 'bn' in n]
        # print(len(bn_params))
        # print(len(rest_params))
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

def add_sparse_args(parser):
    # hyperparameters for Sparse Training
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: False.')
    parser.add_argument('--growth', type=str, default='gradients', help='Growth mode. Choose from: momentum, random, random_unfired, and momentum_neuron.')
    parser.add_argument('--prune', type=str, default='magnitude', help='Prune mode / pruning mode. Choose from: magnitude, SET.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--prune-rate', type=float, default=0.50, help='The pruning rate / prune rate.')
    parser.add_argument('--density', type=float, default=0.10, help='The density of the overall sparse network.')
    parser.add_argument('--verbose', action='store_true', help='Prints verbose status of pruning/growth algorithms.')
    parser.add_argument('--fix', action='store_true', help='Fix topology during training. Default: True.')
    parser.add_argument('--sparse-init', type=str, default='ER', help='sparse initialization')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N', help='extend training time by multiplier times')

    # hyperparameters for GraNet
    parser.add_argument('--update-frequency', type=int, default=2000, metavar='N', help='how many iterations to train between mask update')
    parser.add_argument('--init-density', type=float, default=1.0, help='The initial density of sparse networks')
    parser.add_argument('--final-density', type=float, default=0.20, help='The target density of sparse networks.')
    parser.add_argument('--init-prune-epoch', type=int, default=30, help='The starting epoch of gradual pruning.')
    parser.add_argument('--final-prune-epoch', type=int, default=0, help='The ending epoch of gradual pruning.')
    parser.add_argument('--method', type=str, default='GraNet', help='method name: DST, GraNet, GMP')
    parser.add_argument('--rm_first', action='store_true', help='Keep the first layer dense.')

    # hyperparameters for cyclic training
    parser.add_argument('--filter_dst', action='store_true', help='filter_dst')
    parser.add_argument('--fix_num_operation', type=int, default=0,help='fix_num_operation in prune and grow')
    parser.add_argument('--bound_ratio', type=float, default=2.0, help='The density of the overall sparse network.')
    parser.add_argument('--minumum_ratio', type=float, default=0.5, help='The density of the overall sparse network.')




    parser.add_argument('--connection_wise', action='store_true', help='connection_wise')
    parser.add_argument('--kernal_wise', action='store_true', help='kernal_wise')
    parser.add_argument('--mask_wise', action='store_true', help='mask_wise')
    parser.add_argument('--mag_wise', action='store_true', help='mag_wise')

    parser.add_argument('--stop_dst_epochs', type=int, default=30,help='stop_dst_epochs in prune and grow')

    parser.add_argument('--stop_gmp_epochs', type=int, default=130,help='stop_gmp_epochs in prune and grow')



    parser.add_argument('--mest', action='store_true', help='mest')
    parser.add_argument('--mest_dst', action='store_true', help='mest')

    parser.add_argument('--dst', action='store_true', help='mest')


class CosineDecay(object):
    """Decays a pruning rate according to a cosine schedule

    This class is just a wrapper around PyTorch's CosineAnnealingLR.
    """
    def __init__(self, prune_rate, T_max, eta_min=0.0, last_epoch=-1,init_step=0):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)
        if init_step!=0:
            for i in range(init_step):
                self.cosine_stepper.step()

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    """Anneals the pruning rate linearly with each step."""
    def __init__(self, prune_rate, T_max):
        self.steps = 0
        self.decrement = prune_rate/float(T_max)
        self.current_prune_rate = prune_rate

    def step(self):
        self.steps += 1
        self.current_prune_rate -= self.decrement

    def get_dr(self, prune_rate):
        return self.current_prune_rate

def prefetched_loader(loader, fp16):
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
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

class Masking(object):
    """Wraps PyTorch model parameters with a sparse mask.

    Creates a mask for each parameter tensor contained in the model. When
    `apply_mask()` is called, it applies the sparsity pattern to the parameters.

    Basic usage:
        optimizer = torchoptim.SGD(model.parameters(),lr=args.lr)
        decay = CosineDecay(args.prune_rate, len(train_loader)*(args.total_epochs))
        mask = Masking(optimizer, prune_rate_decay=decay)
        model = MyModel()
        mask.add_module(model)

    Removing layers: Layers can be removed individually, by type, or by partial
    match of their name.
      - `mask.remove_weight(name)` requires an exact name of
    a parameter.
      - `mask.remove_weight_partial_name(partial_name=name)` removes all
        parameters that contain the partial name. For example 'conv' would remove all
        layers with 'conv' in their name.
      - `mask.remove_type(type)` removes all layers of a certain type. For example,
        mask.remove_type(torch.nn.BatchNorm2d) removes all 2D batch norm layers.
    """
    def __init__(self, optimizer,train_loader, prune_rate_decay, prune_rate=0.5, prune_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', verbose=False, fp16=False, args=False,step=0):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        # if growth_mode not in growth_modes:
        #     print('Growth mode: {0} not supported!'.format(growth_mode))
        #     print('Supported modes are:', str(growth_modes))
        self.args = args
        self.growth_mode = growth_mode
        self.prune_mode = prune_mode
        self.redistribution_mode = redistribution_mode
        self.prune_rate_decay = prune_rate_decay
        self.verbose = verbose
        self.train_loader = train_loader
        self.growth_func = growth_mode
        self.prune_func = prune_mode
        self.redistribution_func = redistribution_mode
        self.pruning_rate = {}
        
        self.global_growth = False
        self.global_prune = False

        self.masks = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}

        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.name2removed = {}

        self.total_params=0

        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.prune_rate = prune_rate
        self.name2prune_rate = {}
        self.steps = step
        self.start_name = None


        if self.args.fix:
            self.prune_every_k_steps = None
        else:
            self.prune_every_k_steps = self.args.update_frequency
        self.half = fp16
        self.name_to_32bit = {}
        self.device = torch.device("cuda")

        # global growth/prune state
        self.prune_threshold = 0.001
        self.growth_threshold = 0.001
        self.growth_increment = 0.2
        self.increment = 0.2
        self.tolerance = 0.02



        #  for filter
        self.baseline_filter_num=0
     
        self.args.layer_interval=int(self.args.layer_interval*len(self.train_loader))
        print ("self.args.layer_interval",self.args.layer_interval)

        # gmp channel prune

        self.initial_prune_time=0.0
        self.final_prune_time=math.floor(self.args.stop_gmp_epochs*len(self.train_loader))

        # dst_decay
        self.dst_decay=CosineDecay(0.5, math.floor((self.args.total_epochs)*len(self.train_loader)),0.005)



        self.active_new_mask={}
        self.passtive_new_mask={}

        self.temp_mask={}
    '''    
    
   
      CHANEL EXPLORE

    '''



    def print_layerwise_density(self):


    
        for module in self.modules:
                    

            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                x_weight=weight.detach()
                if len(x_weight.shape)==4:
                    for channel_vector in x_weight:

                
                        channel_zero=(channel_vector!=0).sum().int().item()
                        channel_all=channel_vector.numel()

                        print("check in", name, "density is",channel_zero/channel_all,"weight magnitue", torch.abs(channel_vector).mean().item()  )
                

    def get_module(self,key):
        return getattr(getattr(getattr(self.module, key[0]), key[1])[key[2]],key[3])




    def update_filter_mask(self):
        print ("update_filter_mask")
        for module in self.modules:
            for name, tensor in self.module.named_parameters():
                if name in self.filter_names:
                    self.filter_masks[name][~self.filter_names[name].bool()]=0
                    self.filter_masks[name][self.filter_names[name].bool()]=1


                    # print (name,  self.filter_masks[name].sum())
                
            for name, tensor in self.module.named_parameters():
                if name in self.passive_names:


                    if name in self.filter_names:
                    
                        filter_mask=self.filter_masks[name]

                    else:
                        filter_mask=torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda()


                    # transpose
                    filter_mask =filter_mask.transpose(0, 1)
                    filter_mask[~self.passive_names[name].bool()]=0


                    filter_mask.transpose_(0, 1)

                    self.filter_masks[name]=filter_mask
                    # print (name,  self.filter_masks[name].sum())



    def get_cfg(self):
        print ("now filter structure")

        total=0
        for name, tensor in self.filter_names.items():
            print (name, tensor.sum().item())
            total+=tensor.sum().item()

        print ("total fitler number total",total)



    def get_filter_name(self):

        filter_names={}
        passive_names={}

        
        for ind in self.module.module.layer2split:

            dim= self.get_module(ind).weight.shape[0]
            
            mask=torch.ones(dim)
            # mask[int(dim/2):]=0
            filter_names[self.get_mask_name(ind)]=mask

            passive_ind=self.module.module.next_layers[ind][0]
            passive_names[self.get_mask_name(passive_ind)]=mask
        
            

        self.filter_names=filter_names
        self.passive_names=passive_names



    def get_mask_name(self,key):
        
        weight_name=[]
        bias_name=[]
        for i in key:

            weight_name.append(str(i))
            bias_name.append(str(i))

        weight_name.append("weight")
        # weight_name.insert(0, "module")
        
        weight_name=".".join(weight_name)


        return weight_name

    def filter_num(self):


        total=0
        for name, tensor in self.filter_names.items():
            # print (name, tensor.sum().item())
            total+=tensor.sum().item()


        return  total






    def distribute_del_func(self,new_width,active_grow_key, passive_grow_key):
        
       
        #load original ===============================

        m1=self.get_module(active_grow_key)
        m2=self.get_module(passive_grow_key)
        no_bias= (m1.bias==None)

        w1 = m1.weight.clone()
        w2 = m2.weight.clone()
        if not no_bias: b1 = m1.bias.data
        old_width = int(self.filter_names[self.get_mask_name(active_grow_key)].sum().item())
        # print ("old_width",old_width)




    
        #mask    
        m1_mask=self.masks[self.get_mask_name(active_grow_key)]

        

        
        #del based on weight norm
        mask=m1_mask

        grad=w1
        grad_all=[]
        for filter_grad, filter_mask in zip(grad,mask):
            if self.args.mask_wise:
                filter_single=torch.abs(filter_grad[filter_mask.bool()]).mean().item()
            elif self.args.mag_wise:
                filter_single=torch.abs(filter_grad).mean().item()
        #     print (filter_single)
            if np.isnan(filter_single):
                filter_single=0 
            grad_all.append(filter_single)



        grad_all=torch.FloatTensor(grad_all)

        # operate on select ind
        select_bool=self.filter_names[self.get_mask_name(active_grow_key)].bool()
        select_ind=torch.arange(len(grad_all))[select_bool]


        # sort
        y, idx = torch.sort(torch.abs(grad_all[select_bool]), descending=False)
        del_ind=idx[:old_width-new_width]

        # del

        del_ind=select_ind[del_ind]

        # print ("del_ind",del_ind)


        self.filter_names[self.get_mask_name(active_grow_key)][del_ind]=0
        self.passive_names[self.get_mask_name(passive_grow_key)][del_ind]=0




        assert self.filter_names[self.get_mask_name(active_grow_key)].sum().item() == self.passive_names[self.get_mask_name(passive_grow_key)].sum().item() , "Wrong deletling"
   

        self.update_filter_mask()

        self.apply_mask()




    def del_func(self,del_ind,active_grow_key, passive_grow_key):
        
   
        # print (self.get_mask_name(active_grow_key),"del",len(del_ind))


        self.filter_names[self.get_mask_name(active_grow_key)][del_ind]=0
        self.passive_names[self.get_mask_name(passive_grow_key)][del_ind]=0




        assert self.filter_names[self.get_mask_name(active_grow_key)].sum().item() == self.passive_names[self.get_mask_name(passive_grow_key)].sum().item() , "Wrong deletling"
   

        self.update_filter_mask()

        self.apply_mask()






    def del_func(self,del_ind,active_grow_key, passive_grow_key):
        
   
        # print (self.get_mask_name(active_grow_key),"del",len(del_ind))


        self.filter_names[self.get_mask_name(active_grow_key)][del_ind]=0
        self.passive_names[self.get_mask_name(passive_grow_key)][del_ind]=0




        assert self.filter_names[self.get_mask_name(active_grow_key)].sum().item() == self.passive_names[self.get_mask_name(passive_grow_key)].sum().item() , "Wrong deletling"
   

        self.update_filter_mask()

        self.apply_mask()




    def prune_score(self,prune_layer_index,total_to_prune):

        # Gather all scores in a single vector and normalise
        all_scores=[]

        for index in prune_layer_index:


            # single metric
            grad=  torch.abs(self.get_module(index).weight.clone())


            m1_mask=self.masks[self.get_mask_name(index)]

            filter_mask=self.filter_names[self.get_mask_name(index)].bool()
            grad=grad[filter_mask]
            mask=m1_mask[filter_mask]

            # print ("grad.shape",grad.shape,"mask.shape",mask.shape)

            for filter_grad, filter_mask in zip(grad,mask):

                # print ("filter_grad.shape",filter_grad.shape)
                # print ( "filter_mask.shape",filter_mask.shape)
                
                if self.args.mask_wise:
                    grad_magnitude = torch.abs(filter_grad)  [filter_mask.bool()]  .mean().item()
                    # print ("grad_magnitude",grad_magnitude)

                elif self.args.mag_wise:
                    grad_magnitude = torch.abs(filter_grad) .mean().item()
                # print ("grad_magnitude",grad_magnitude)

                elif self.args.kernal_wise:
                    vector = filter_grad.view(filter_grad.size(0), -1).sum(dim=1)
                    grad_magnitude=((vector!=0).sum().int().item()/vector.numel())  
                    # print ("vector.shape",vector.shape)
                    # print ("grad_magnitude",grad_magnitude)

                elif self.args.connection_wise:    
                    vector= filter_grad
                    grad_magnitude=((vector!=0).sum().int().item()/vector.numel())  
                    # print ("vector.shape",vector.shape)
                    # print ("grad_magnitude",grad_magnitude)
                
                all_scores.append(grad_magnitude)

        print ("current score lengh",len(all_scores))
        acceptable_score = np.sort(np.array(all_scores))[-int(len(all_scores)-total_to_prune)]
        print ("acceptable_score",acceptable_score)


        return acceptable_score

    def gradual_pruning_rate(self,
            step: int,
            initial_threshold: float,
            final_threshold: float,
            initial_time: int,
            final_time: int,
    ):
        if step <= initial_time:
            threshold = initial_threshold
        elif step > final_time:
            threshold = final_threshold
        else:
            mul_coeff = 1 - (step - initial_time) / (final_time - initial_time)
            threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)

        return threshold



    def del_layer(self):
        #################################prune layer#############################

        print ("before del layer")
        self.get_cfg()

        print ("current all fitler num",self.filter_num())


        ### prune ratio
        rate = self.layer_rate 
        rate=1-rate
        print ("to keep rate",rate)


        filter_numer=self.filter_num()
        print (filter_numer, self.baseline_filter_num*rate  )
        total_to_prune=filter_numer-self.baseline_filter_num*rate
        print ("total_to_prune",total_to_prune)
            

        if total_to_prune>=0:


            ### mask the position

            layer_prune={}
            residual_layer={}
 

            residual=9999
            i=0
            while residual > 0 and i < 1000:


                residual=0

                # to prune layer 
                prune_layer_index=[]
                for l_index in self.module.module.layer2split:
                    if l_index not in residual_layer.keys():
                        prune_layer_index.append(l_index) 
                
                # to prune num
                total_to_prune=filter_numer-self.baseline_filter_num*rate
                pruned=[]
                for n_index in layer_prune:
                    pruned.extend(layer_prune[n_index] )
                total_to_prune=total_to_prune-len(pruned)

                print ("total_to_prune",total_to_prune)
                
                acceptable_score=self.prune_score(prune_layer_index,total_to_prune)



                for index in self.module.module.layer2split:


                    if index not in residual_layer.keys():
  
                        grad=  torch.abs(self.get_module(index).weight.clone())
                        mask=self.masks[self.get_mask_name(index)]

                        del_ind=[]
                        filter_index=0
                        layer_grad_magnitude=[]
                        for filter_grad, filter_mask,filter_name in zip(grad,mask,self.filter_names[self.get_mask_name(index)]):
                            
                            if filter_name==1.0:
                

                                if self.args.mask_wise:
                                    grad_magnitude = torch.abs(filter_grad)  [filter_mask.bool()] .mean().item()

                                elif self.args.mag_wise:
                                    grad_magnitude = torch.abs(filter_grad) .mean().item()


                                elif self.args.kernal_wise:
                                    vector = filter_grad.view(filter_grad.size(0), -1).sum(dim=1)
                                    # print("vector",vector.shape)
                                    grad_magnitude=((vector!=0).sum().int().item()/vector.numel())  

                                elif self.args.connection_wise:    
                                    vector= filter_grad
                                    grad_magnitude=((vector!=0).sum().int().item()/vector.numel())  

                                if grad_magnitude<acceptable_score:
                                    del_ind.append(filter_index)

                                    layer_grad_magnitude.append(grad_magnitude)

                            filter_index+=1



                        # current layer
                        # if index not in layer_prune.keys():
                        current=self.filter_names[self.get_mask_name(index)].sum().item()
                        # else:
                        #     current=self.filter_names[self.get_mask_name(index)].sum().item()-len(layer_prune[index])


                        single_residual=self.minimum_layer[index]-(current-len( del_ind))
                        print ("index",index,"prune",len( del_ind),'current',current,"single_residual",single_residual)


                        if index not in layer_prune.keys(): layer_prune[index]=[]

                        # if single_residual>0:
                        #     residual+=single_residual
                        #     prune_lengh=int(current-self.minimum_layer[index] )
                        #     layer_prune[index].extend (np.array(del_ind) [  np.array(layer_grad_magnitude) <   np.sort(np.array(layer_grad_magnitude))[prune_lengh]  ])
                        #     residual_layer[index]=single_residual

                        if single_residual>0:
                            residual+=single_residual
                            prune_lengh=int(current-self.minimum_layer[index] )

                            layer_grad_magnitude=np.array(layer_grad_magnitude)
                            # layer_prune[index].extend (np.array(del_ind) [  np.array(layer_grad_magnitude) <   np.sort(np.array(layer_grad_magnitude))[prune_lengh]  ])
                            layer_prune[index].extend (np.array(del_ind) [ np.argsort(layer_grad_magnitude)[:prune_lengh]]  )
                            residual_layer[index]=single_residual

                        else:

                            layer_prune[index].extend(del_ind)

                        print ("layer_prune[index]",index,"len",len(layer_prune[index]))





                ###### layer prune


                print ("begin to prune")
                for active_prune_key in self.module.module.layer2split:
                    passive_prune_key,norm_key=self.module.module.next_layers[active_prune_key]


                    
        
                    if len(self.get_mask_name(active_prune_key))>0:


                        self.del_func(layer_prune[active_prune_key],
                                            active_prune_key,
                                            passive_prune_key)


                print ("pruneing layer",self.get_cfg())
                print ("current all fitler num",self.filter_num())
                print ("pruned",filter_numer-self.filter_num() )




                # to prune num

                pruned=[]
                for n_index in layer_prune:
                    print (n_index,len(layer_prune[n_index]))
                    pruned.extend(layer_prune[n_index] )



                print ("======residual",residual,"pruned",len(pruned))




                if i == 1000:
                    print('Error resolving the residual! Layers are too full! Residual left over: {0}'.format(residual))






            ###### layer prune

            for active_prune_key in self.module.module.layer2split:
                passive_prune_key,norm_key=self.module.module.next_layers[active_prune_key]


                
    
                if len(self.get_mask_name(active_prune_key))>0:


                    self.del_func(layer_prune[active_prune_key],
                                        active_prune_key,
                                        passive_prune_key)


            print ("pruneing layer",self.get_cfg())
            print ("current all fitler num",self.filter_num())
            print ("pruned",filter_numer-self.filter_num() )





        

    '''

    CORE


    '''

    def init_optimizer(self):
        if 'fp32_from_fp16' in self.optimizer.state_dict():
            for (name, tensor), tensor2 in zip(self.modules[0].named_parameters(), self.optimizer.state_dict()['fp32_from_fp16'][0]):
                self.name_to_32bit[name] = tensor2
            self.half = True


    def init(self, mode='ERK', density=0.05, erk_power_scale=1.0):
        self.init_growth_prune_and_redist()
        self.init_optimizer()

        

        # for module in self.modules:
        #     for name, weight in module.named_parameters():
        #         if name not in self.masks: continue
        #         self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda()
        #         self.baseline_nonzero += weight.numel()*density

        ## init for layer
        for index in self.module.module.layer2split:
            self.baseline_filter_num+=self.get_module(index).weight.shape[0]
            
        print ("baseline fitler num",self.baseline_filter_num)           


        self.bound_layer={}
        for index in self.module.module.layer2split:
            self.bound_layer[index] = int(self.get_module(index).out_channels)


        self.minimum_layer={}
        for index in self.module.module.layer2split:
            self.minimum_layer[index] = int(self.get_module(index).out_channels*(1-self.args.start_layer_rate)*self.args.minumum_ratio)



        if self.args.method == 'GMP':
            print('initialized with GMP, ones')
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).cuda()
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()

        


        elif mode == 'uniform':
            print('initialized with uniform')
            # initializes each layer with a constant percentage of dense weights
            # each layer will have weight.numel()*density weights.
            # weight.numel()*density == weight.numel()*(1.0-sparsity)
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda()
                    self.baseline_nonzero += weight.numel()*density
        elif mode == 'resume':
            print('initialized with resume')
            # Initializes the mask according to the weights
            # which are currently zero-valued. This is required
            # if you want to resume a sparse model but did not
            # save the mask.
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    print((weight != 0.0).sum().item())
                    if name in self.name_to_32bit:
                        print('W2')
                    self.masks[name][:] = (weight != 0.0).float().data.cuda()
                    self.baseline_nonzero += weight.numel()*density


                    
        elif mode == 'ERK':
            print('initialize by fixed_ERK')

 

            total_params = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()

            self.total_params=total_params
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()
            while not is_epsilon_valid:
                # We will start with all layers and try to find right epsilon. However if
                # any probablity exceeds 1, we will make that layer dense and repeat the
                # process (finding epsilon) with the non-dense layers.
                # We want the total number of connections to be the same. Let say we have
                # for layers with N_1, ..., N_4 parameters each. Let say after some
                # iterations probability of some dense layers (3, 4) exceeded 1 and
                # therefore we added them to the dense_layers set. Those layers will not
                # scale with erdos_renyi, however we need to count them so that target
                # paratemeter count is achieved. See below.
                # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
                #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
                # eps * (p_1 * N_1 + p_2 * N_2) =
                #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
                # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    self.n_ones=n_ones

                    if name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros

                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        # Note that raw_probabilities[mask] * n_param gives the individual
                        # elements of the divisor.
                        divisor += raw_probabilities[name] * n_param
                # By multipliying individual probabilites with epsilon, we should get the
                # number of parameters per layer correctly.
                epsilon = rhs / divisor
                # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
                # mask to 0., so they become part of dense_layers sets.
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            self.density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():

                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    self.density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    self.density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {self.density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < self.density_dict[name]).float().data.cuda()

                total_nonzero += self.density_dict[name] * mask.numel()
            self.baseline_nonzero=total_nonzero
            print(f"Overall sparsity {total_nonzero / total_params}")

            self.temp_mask=copy.deepcopy(self.masks)
        self.print_nonzero_counts()



    def init_growth_prune_and_redist(self):
        # if isinstance(self.growth_func, str) and self.growth_func in growth_funcs:
        #     if 'global' in self.growth_func: self.global_growth = True
        #     self.growth_func = growth_funcs[self.growth_func]
        # elif isinstance(self.growth_func, str):
        #     print('='*50, 'ERROR', '='*50)
        #     print('Growth mode function not known: {0}.'.format(self.growth_func))
        #     print('Use either a custom growth function or one of the pre-defined functions:')
        #     for key in growth_funcs:
        #         print('\t{0}'.format(key))
        #     print('='*50, 'ERROR', '='*50)
        #     raise Exception('Unknown growth mode.')

        # if isinstance(self.prune_func, str) and self.prune_func in prune_funcs:
        #     if 'global' in self.prune_func: self.global_prune = True
        #     self.prune_func = prune_funcs[self.prune_func]
        # elif isinstance(self.prune_func, str):
        #     print('='*50, 'ERROR', '='*50)
        #     print('Prune mode function not known: {0}.'.format(self.prune_func))
        #     print('Use either a custom prune function or one of the pre-defined functions:')
        #     for key in prune_funcs:
        #         print('\t{0}'.format(key))
        #     print('='*50, 'ERROR', '='*50)
        #     raise Exception('Unknown prune mode.')

        if isinstance(self.redistribution_func, str) and self.redistribution_func in redistribution_funcs:
            self.redistribution_func = redistribution_funcs[self.redistribution_func]
        elif isinstance(self.redistribution_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Redistribution mode function not known: {0}.'.format(self.redistribution_func))
            print('Use either a custom redistribution function or one of the pre-defined functions:')
            for key in redistribution_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown redistribution mode.')


    def step(self):


        self.optimizer.step()
        self.apply_mask()


        # mest decay
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr()


        # filter_dst_decay
        self.layer_rate= self.gradual_pruning_rate(self.steps, 0.0, self.args.start_layer_rate, self.initial_prune_time, self.final_prune_time)



        # parameters_dst_decay
        self.dst_decay.step()
        self.dst_rate= self.dst_decay.get_dr()



        self.steps += 1 



        ### filter
        if self.args.filter_dst:
            if  self.steps >= self.initial_prune_time and self.steps < self.final_prune_time :
                if (self.steps+ (self.prune_every_k_steps/2)) % self.args.layer_interval== 0 :
                # print ("step",self.steps)
                # if (self.steps+1) % 5== 0 :


                    print ("current layer rate",self.layer_rate)
                    print ('===========del layer===============')
                       
                    self.del_layer()

                    print ('===========done ===============')
                    if "global" not in self.args.growth:
                        self.update_erk_dic()


        # parameters
        if self.args.mest:
            if self.steps< len(self.train_loader)*(self.args.total_epochs-self.args.stop_dst_epochs):
                if self.prune_every_k_steps is not None:
                    if (self.steps % self.prune_every_k_steps == 0):

                        print ("current mest  prune_rate", self.prune_rate)

                    
                        self.truncate_weights_prune(self.prune_rate)
                        self.print_nonzero_counts()

                        self.truncate_weights_grow(self.prune_rate)
                        self.print_nonzero_counts()

                    
                        # self.channel_stastic()

            elif self.args.mest_dst:
                if self.prune_every_k_steps is not None:
                    if (self.steps % self.prune_every_k_steps == 0):

                        print ("current dst rate",self.dst_rate)

                        self.truncate_weights(self.dst_rate)
                        self.print_nonzero_counts()

        elif self.args.dst:

            if self.prune_every_k_steps is not None:
                if (self.steps % self.prune_every_k_steps == 0):

                    print ("current dst rate",self.dst_rate)

                    self.truncate_weights(self.dst_rate)
                    self.print_nonzero_counts()





    def pruning(self, step):
        curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter =  int((self.args.final_prune_epoch * int(len(self.train_loader))) / self.prune_every_k_steps)
        ini_iter =  int(self.args.init_prune_epoch * (int(len(self.train_loader))) / self.prune_every_k_steps)
        total_prune_iter = final_iter - ini_iter
        print('******************************************************')
        print(f'Pruning Progress is {curr_prune_iter - ini_iter} / {total_prune_iter}')
        print('******************************************************')


        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter-1:
            prune_decay = (1 - ((curr_prune_iter - ini_iter) / total_prune_iter)) ** 3

            self.curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.args.final_density) * (
                    1 - prune_decay)

            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * (1 - self.curr_prune_rate))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()

            self.apply_mask()

            total_size = 0
            for name, weight in self.masks.items():
                total_size += weight.numel()
            print('Total Model parameters:', total_size)

            sparse_size = 0
            for name, weight in self.masks.items():
                sparse_size += (weight != 0).sum().int().item()

            print('Sparsity after pruning: {0}'.format(
                (total_size-sparse_size) / total_size))

    def pruning_uniform(self, step):
        curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter =  int((self.args.final_prune_epoch * int(len(self.train_loader))) / self.prune_every_k_steps)
        ini_iter =  int(self.args.init_prune_epoch * (int(len(self.train_loader))) / self.prune_every_k_steps)
        total_prune_iter = final_iter - ini_iter
        print('******************************************************')
        print(f'Pruning Progress is {curr_prune_iter - ini_iter} / {total_prune_iter}')
        print('******************************************************')


        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter:
            prune_decay = (1 - ((
                                        curr_prune_iter - ini_iter) / total_prune_iter)) ** 3
            curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.args.final_density) * (
                    1 - prune_decay)

            if curr_prune_rate >= 0.8:
                curr_prune_rate = 1 - (self.total_params * (1-curr_prune_rate) - 0.2 * self.fc_params)/(self.total_params-self.fc_params)

                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        score = torch.flatten(torch.abs(weight))
                        if 'classifier' in name:
                            num_params_to_keep = int(len(score) * 0.2)
                            threshold, _ = torch.topk(score, num_params_to_keep, sorted=True)
                            acceptable_score = threshold[-1]
                            self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
                        else:
                            num_params_to_keep = int(len(score) * (1 - curr_prune_rate))
                            threshold, _ = torch.topk(score, num_params_to_keep, sorted=True)
                            acceptable_score = threshold[-1]
                            self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
            else:
                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        score = torch.flatten(torch.abs(weight))
                        num_params_to_keep = int(len(score) * (1 - curr_prune_rate))
                        threshold, _ = torch.topk(score, num_params_to_keep, sorted=True)
                        acceptable_score = threshold[-1]
                        self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()

            self.apply_mask()

            total_size = 0
            for name, weight in self.masks.items():
                total_size += weight.numel()
            print('Total Model parameters:', total_size)

            sparse_size = 0
            for name, weight in self.masks.items():
                sparse_size += (weight != 0).sum().int().item()

            print('Sparsity after pruning: {0}'.format(
                (total_size-sparse_size) / total_size))


    def add_module(self, module):



        self.module = module
        self.modules.append(self.module)
        for name, tensor in self.module.named_parameters():
            self.names.append(name)
            self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()


        self.get_filter_name()



        self.filter_masks={}
        for name, tensor in module.named_parameters():
            if name in self.filter_names:
                self.filter_masks[name]=torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
                self.filter_masks[name][~self.filter_names[name].bool()]=0

        
        for name, tensor in module.named_parameters():
            if name in self.passive_names:
                if name in self.filter_names:
                
                    filter_mask=self.filter_masks[name]

                else:
                    filter_mask=torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda()



                # print (filter_mask.shape)
                filter_mask =filter_mask.transpose(0, 1)


                filter_mask[~self.passive_names[name].bool()]=0
                filter_mask.transpose_(0, 1)

                self.filter_masks[name]=filter_mask




        # print (self.masks)
        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        # print('Removing fisrt layer...')
        # self.remove_weight_partial_name('conv1.weight')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d, verbose=self.verbose)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d, verbose=self.verbose)

        if self.args.rm_first:
            for name, tensor in module.named_parameters():
                if name == 'module.conv1.weight':
                    self.masks.pop(name)
                    print(f"pop out {name}")
        self.init(mode=self.args.sparse_init, density=self.args.init_density)


    def is_at_start_of_pruning(self, name):
        if self.start_name is None: self.start_name = name
        if name == self.start_name: return True
        else: return False

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape, self.masks[name].numel()))
            self.masks.pop(name)
        elif name+'.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name+'.weight'].shape, self.masks[name+'.weight'].numel()))
            self.masks.pop(name+'.weight')
        else:
            print('ERROR',name)

    def remove_weight_partial_name(self, partial_name, verbose=False):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                if self.verbose:
                    print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape, np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed: self.names.pop(i)
            else: i += 1


    def remove_type(self, nn_type, verbose=False):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)
                    #self.remove_weight_partial_name(name, verbose=self.verbose)

    def apply_mask(self):
        for module in self.modules:

             # print ("fusing masks")
            for name, mask in self.masks.items():
                
                if name in self.filter_masks.keys():
                    self.masks[name]= torch.logical_and(self.filter_masks[name], self.masks [name]).float()


            for name, tensor in module.named_parameters():
                if name in self.masks:
                    if not self.half:
                        tensor.data = tensor.data*self.masks[name]
                        if 'momentum_buffer' in self.optimizer.state[tensor]:
                            self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]
                    else:
                        tensor.data = tensor.data*self.masks[name].half()
                        if name in self.name_to_32bit:
                            tensor2 = self.name_to_32bit[name]
                            tensor2.data = tensor2.data*self.masks[name]


            for module in self.modules:
                for name, tensor in module.named_parameters():
                    if "bias" in name:
                        weight_name=name[:-4]+"weight"
                        if weight_name in self.filter_names:
                            tensor.data = tensor.data*self.filter_names[weight_name].float().cuda()



    def adjust_prune_rate(self):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if name not in self.name2prune_rate: self.name2prune_rate[name] = self.prune_rate

                self.name2prune_rate[name] = self.prune_rate

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                if sparsity < 0.2:
                    # determine if matrix is relativly dense but still growing
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        # growing
                        self.name2prune_rate[name] = min(sparsity, self.name2prune_rate[name])




    def global_gradient_growth(self, total_regrowth):
        

        togrow = total_regrowth

        togrow=int(togrow)

        ### prune

        weight_abs = []

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                grad = self.get_gradient_for_weights(weight)

                if name in self.filter_masks:
                    remain = (torch.abs(grad * (self.masks[name]==0)))  [self.filter_masks[name].bool()] 
                else:
                    remain = torch.abs(grad *(self.masks[name]==0) )

                weight_abs.append(remain)



        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in weight_abs])

        print ("len(all_scores)",len(all_scores),all_scores.bool().sum().item())
        if togrow>all_scores.bool().sum().item():
            print (togrow, all_scores.bool().sum().item())
            togrow=all_scores.bool().sum().item()
            print ("already full=====================")
        if togrow>0:
            threshold, _ = torch.topk(all_scores, togrow,largest=True, sorted=True)
            acceptable_score = threshold[-1]



            increse=0
            before_mask=0
            after_mask=0

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue


                    new_mask = self.masks[name]


                    grad = self.get_gradient_for_weights(weight)

                    grad = grad*(new_mask==0).float()
                    if  name in self.filter_masks.keys():                 
                        grad= grad*self.filter_masks[name].float() 

                    increse+=(torch.abs(grad.data) >=acceptable_score).float().sum().item()
                    
                    self.masks[name][:] = (new_mask.byte() | (torch.abs(grad.data) > acceptable_score)).float()

                    before_mask+=self.masks[name].sum().item()

                    if name in self.filter_masks.keys():
                        self.masks[name]= torch.logical_and(self.filter_masks[name], self.masks [name]).float()
                    
                    after_mask+= self.masks[name].sum().item()

            print ("increse", increse,"before_mask",before_mask,"after_mask",after_mask)

        else:
            print ("no room to grow")

        return None



    # def global_gradient_growth(self, total_regrowth):
    #     togrow = total_regrowth
    #     total_grown = 0
    #     last_grown = 0
    #     while total_grown < togrow*(1.0-self.tolerance) or (total_grown > togrow*(1.0+self.tolerance)):
    #         total_grown = 0
    #         total_possible = 0
    #         for module in self.modules:
    #             for name, weight in module.named_parameters():
    #                 if name not in self.masks: continue

    #                 new_mask = self.masks[name]
    #                 grad = self.get_gradient_for_weights(weight)

    #                 if name in self.filter_masks:
    #                     grad =( grad*(new_mask==0).float())[self.filter_masks[name].bool()]
    #                 else:
    #                     grad = grad*(new_mask==0).float()


    #                 possible = (grad !=0.0).sum().item()
    #                 total_possible += possible
    #                 grown = (torch.abs(grad.data) > self.growth_threshold).sum().item()
    #                 total_grown += grown
    #         print(total_grown, self.growth_threshold, togrow, self.growth_increment, total_possible)
    #         if total_grown == last_grown: break
    #         last_grown = total_grown


    #         if total_grown > togrow*(1.0+self.tolerance):
    #             self.growth_threshold *= 1.02
    #             #self.growth_increment *= 0.95
    #         elif total_grown < togrow*(1.0-self.tolerance):
    #             self.growth_threshold *= 0.98
    #             #self.growth_increment *= 0.95

    #     total_new_nonzeros = 0
    #     for module in self.modules:
    #         for name, weight in module.named_parameters():
    #             if name not in self.masks: continue

    #             new_mask = self.masks[name]
    #             grad = self.get_gradient_for_weights(weight)
    #             grad = grad*(new_mask==0).float()
    #             self.masks[name][:] = (new_mask.byte() | (torch.abs(grad.data) > self.growth_threshold)).float()
    #             total_new_nonzeros += new_mask.sum().item()

    #     return total_new_nonzeros



    def global_magnitude_death(self,tokill):

        
        ### prune
        tokill=tokill+self.total_zero
        tokill=int(tokill)
        weight_abs = []

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                if name in self.filter_masks:
                    remain = (torch.abs(weight.data))[self.filter_masks[name].bool()] 
                else:
                    remain = torch.abs(weight.data) 

                weight_abs.append(remain)


   
        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in weight_abs])

        threshold, _ = torch.topk(all_scores, tokill,largest=False, sorted=True)


        acceptable_score = threshold[-1]



        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
  
                new_mask = torch.abs(weight.data) >=acceptable_score
                self.masks[name][:] = new_mask

                if name in self.filter_masks.keys():
                    self.masks[name]= torch.logical_and(self.filter_masks[name], self.masks [name]).float()
                
        return None




    # def global_magnitude_death(self,tokill):

    #     # tokill = math.ceil(pruning_rate*self.total_nonzero)
    #     tokill=tokill
    #     total_removed = 0
    #     prev_removed = 0
    #     while total_removed < tokill*(1.0-self.tolerance) or (total_removed > tokill*(1.0+self.tolerance)):
    #         total_removed = 0
    #         for module in self.modules:
    #             for name, weight in module.named_parameters():
    #                 if name not in self.masks: continue

    #                 if name in self.filter_masks:
    #                     remain = (torch.abs(weight.data)[self.filter_masks[name].bool()] > self.prune_threshold).sum().item()
    #                 else:
    #                     remain = (torch.abs(weight.data) > self.prune_threshold).sum().item()


    #                 total_removed += self.name2nonzeros[name] - remain

    #         print(total_removed, self.prune_threshold, tokill, self.increment)

    #         if (total_removed!=0 and prev_removed == total_removed): break

    #         prev_removed = total_removed
    #         if total_removed > tokill*(1.0+self.tolerance):
    #             self.prune_threshold *= 1.0-self.increment
    #             self.increment *= 0.99
    #         elif total_removed < tokill*(1.0-self.tolerance):
    #             self.prune_threshold *= 1.0+self.increment
    #             self.increment *= 0.99

    #     for module in self.modules:
    #         for name, weight in module.named_parameters():
    #             if name not in self.masks: continue
    #             new_mask = torch.abs(weight.data) > self.prune_threshold
    #             # self.pruning_rate[name] = int(self.masks[name].sum().item() - new_mask.sum().item())
    #             # self.pruning_rate[name] = int(self.density_dict[name]*self.masks[name].numel()- new_mask.sum().item())
    #             # redistribute
    #             # self.name2variance[name]=self.masks[name].sum().item()-new_mask.sum().item()

                
    #             self.masks[name][:] = new_mask

    #     # print ("self.name2variance:",self.name2variance)
    #     # for name in self.name2variance:
    #     #     if self.total_variance != 0.0:
    #     #         self.name2variance[name] /= self.total_variance

    #     return int(total_removed)



    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}
        self.name2variance = {}


        self.total_variance = 0.0

        self.total_nonzero = 0
        self.total_zero = 0.0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if name in self.filter_masks:
                    mask = self.masks[name][self.filter_masks[name].bool()]
                else:
                    mask = self.masks[name]
                    # redistribution
                    # self.name2variance[name] = self.redistribution_func(self, name, weight, mask)

                    # if not np.isnan(self.name2variance[name]):
                    #     self.total_variance += self.name2variance[name]
                        
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())

                self.total_nonzero += self.name2nonzeros[name]
                self.total_zero += self.name2zeros[name]



    def update_erk_dic(self):
        erk_power_scale=1.0
        print('retriving sparsity dic by fixed_ERK')

        total_params = 0
        for name, weight in self.masks.items():
            if name in self.filter_masks:
                weight=weight[self.filter_masks[name].bool()]

            total_params  += weight.numel()



        print ("baseline_nonzero",self.baseline_nonzero)
        print ("total_params",total_params)


        density=self.baseline_nonzero/total_params




        ### temp mask
        self.temp_mask=copy.deepcopy(self.masks)

        for name, weight in self.masks.items():

            if name in self.filter_names:
                self.temp_mask[name]=self.temp_mask[name][self.filter_names[name].bool()]


        for name, weight in self.masks.items():
    
            if name in self.passive_names:


                temp=self.temp_mask[name]

                    # transpose
                temp =temp.transpose(0, 1)
                temp=temp[self.passive_names[name].bool()]
                temp.transpose_(0, 1)

                self.temp_mask[name]=temp




        is_epsilon_valid = False
        # # The following loop will terminate worst case when all masks are in the
        # custom_sparsity_map. This should probably never happen though, since once
        # we have a single variable or more with the same constant, we have a valid
        # epsilon. Note that for each iteration we add at least one variable to the
        # custom_sparsity_map and therefore this while loop should terminate.
        dense_layers = set()
        while not is_epsilon_valid:
            # We will start with all layers and try to find right epsilon. However if
            # any probablity exceeds 1, we will make that layer dense and repeat the
            # process (finding epsilon) with the non-dense layers.
            # We want the total number of connections to be the same. Let say we have
            # for layers with N_1, ..., N_4 parameters each. Let say after some
            # iterations probability of some dense layers (3, 4) exceeded 1 and
            # therefore we added them to the dense_layers set. Those layers will not
            # scale with erdos_renyi, however we need to count them so that target
            # paratemeter count is achieved. See below.
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, mask in self.temp_mask.items():
                n_param = np.prod(mask.shape)

                n_zeros = n_param * (1 - density)
                n_ones = n_param * density
                


                if name in dense_layers:
                    # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                    rhs -= n_zeros

                else:
                    # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                    # equation above.
                    rhs += n_ones
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    raw_probabilities[name] = (
                                                        np.sum(mask.shape) / np.prod(mask.shape)
                                                ) ** erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                    divisor += raw_probabilities[name] * n_param
            # By multipliying individual probabilites with epsilon, we should get the
            # number of parameters per layer correctly.
            epsilon = rhs / divisor
            # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
            # mask to 0., so they become part of dense_layers sets.
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

       
        self.density_dict = {}
        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, mask in self.temp_mask.items():
            n_param = np.prod(mask.shape)
            if name in dense_layers:
                self.density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                self.density_dict[name] = probability_one
            print(
                f"layer: {name}, shape: {mask.shape}, density: {self.density_dict[name]}"
            )

            total_nonzero += self.density_dict[name] * mask.numel()

        print(f"Overall sparsity {total_nonzero / total_params}",total_nonzero,total_params)
    def gradient_growth(self, name, new_mask, total_regrowth, weight):
        if self.density_dict[name]==1.0:
            new_mask = torch.ones_like(new_mask, dtype=torch.float32, requires_grad=False).cuda()

            if name in self.filter_masks:
               new_mask = torch.logical_and(self.filter_masks[name], new_mask).float()  


        else:
            grad = self.get_gradient_for_weights(weight)
            grad = grad*(new_mask==0).float()

            if name in self.filter_masks:
                remain = (torch.abs(grad * (self.masks[name]==0)))  [self.filter_masks[name].bool()] .float()
            else:
                remain = torch.abs(grad *(self.masks[name]==0) ).float()

            all_scores = torch.cat([torch.flatten(x) for x in remain])

            threshold, _ = torch.topk(all_scores, total_regrowth,largest=True, sorted=True)
            acceptable_score = threshold[-1]


            new_mask = (new_mask.byte() | (torch.abs(grad.data) > acceptable_score)).float()

        return new_mask

    def magnitude_death(self, mask, weight, name, pruning_rate):
    
        num_zeros = (mask == 0).sum().item()
        num_remove = math.ceil(pruning_rate * (mask.sum().item()))
        if num_remove == 0.0: return weight.data != 0.0
        # num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])

        x, idx = torch.sort(torch.abs(weight.data.reshape(-1)))

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k - 1].item()

        return (torch.abs(weight.data) > threshold)

 
 
    def truncate_weights_prune(self, pruning_rate):
        print ("\n")
        print('dynamic sparse change prune')

        self.gather_statistics()




        #################################prune weights#############################
        tokill=self.total_nonzero-self.baseline_nonzero
        print ("to kill", tokill,"expect", self.baseline_nonzero)
        if tokill>0:

            if self.prune_mode == 'global_magnitude':
                self.total_removed=self.global_magnitude_death(tokill)

            else:

                pruning_rate=tokill/self.total_nonzero
                print ("calulate prune ratio",pruning_rate)


                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        mask = self.masks[name]

                        # death
                        if self.prune_mode == 'magnitude':
                            new_mask = self.magnitude_death(mask, weight, name, pruning_rate)
                        elif self.prune_mode == 'SET':
                            new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                        elif self.prune_mode == 'Taylor_FO':
                            new_mask = self.taylor_FO(mask, weight, name)
                        elif self.prune_mode == 'threshold':
                            new_mask = self.threshold_death(mask, weight, name)

      
                        self.masks[name][:] = new_mask
                        # self.pruning_rate[name] = int(self.name2nonzeros[name] - new_mask.sum().item())


            self.apply_mask()


    def truncate_weights_grow(self, pruning_rate):
       #################################grow weights#############################

        # get gradients
        # inputs, targets = next(iter(self.train_loader))
        # inputs = inputs.to(self.device)
        # targets = targets.to(self.device)
        # inputs.requires_grad = True
        # # Let's create a fresh copy of the network so that we're not worried about
        # self.module.zero_grad()
        # outputs = self.module.forward(inputs)
        # loss = F.nll_loss(outputs, targets)
        # loss.backward()
        self.gather_statistics()
        print('dynamic sparse change grow')

        togrow=self.total_params*pruning_rate-self.total_nonzero
        print ("self.total_params*pruning_rate",self.total_params*pruning_rate)
        print ("self.total_nonzero",self.total_nonzero)
        print ("to grow",togrow)




        if togrow>0:
            if self.growth_mode == 'global_gradients':
    
                total_nonzero_new=self.global_gradient_growth(togrow)
                print ("total_nonzero_new",total_nonzero_new)



            else:


                real_d_num=0
                for name, mask in self.masks.items():
                    if self.density_dict[name]==1.0:
                        d_layernum=self.masks[name].sum().item()
                        real_d_num+=d_layernum

                print ("real_d_num",real_d_num)
                d_num=0
                for name, mask in self.temp_mask.items():
                    if self.density_dict[name]==1.0:
                        d_layernum=mask.numel()
                        d_num+=d_layernum

                print ("d_num",d_num)



                expect=0
                for name, mask in self.masks.items():
                    if self.density_dict[name]!=1.0:
                        new_mask = self.masks[name].data.byte()
                        layer_e=int((new_mask.sum().item()))
                        expect+=layer_e

                print ("expect dst",expect)

                print ("total_nonzero",self.total_nonzero)
                print ("baseline_nonzero",self.baseline_nonzero )
                grow_ratio=(togrow-(d_num-real_d_num))/(self.total_nonzero-real_d_num)

                print ("calulate extra grow ratio",grow_ratio)

                if grow_ratio>0:
                    for module in self.modules:
                        for name, weight in module.named_parameters():
                            if name not in self.masks: continue
                            new_mask = self.masks[name].data.byte()

                            to_grow= int((new_mask.sum().item())*grow_ratio)

                            if (to_grow +  int(new_mask.sum().item()))  > int( (self.temp_mask[name].numel())):

                                to_grow= int( (self.temp_mask[name].numel())) -  int(new_mask.sum().item())
                                print ("to_grow",to_grow)
                                print ( "int( (self.masks[name].numel()))",int(self.temp_mask[name].numel()))
                                print ("(to_grow +  int(new_mask.sum().item()))",(to_grow +  int(new_mask.sum().item())) )

                                print ("layer_wise dst no room to grow in each layer in",name , (to_grow +  int(new_mask.sum().item())) - int(self.density_dict[name]* (self.temp_mask[name].numel())) )

                            self.pruning_rate[name] =to_grow
                            # growth
                            if self.growth_mode == 'random':
                                new_mask = self.random_growth(name, new_mask, self.pruning_rate[name], weight)

                            elif self.growth_mode == 'momentum':
                                new_mask = self.momentum_growth(name, new_mask, self.pruning_rate[name], weight)

                            elif self.growth_mode == 'gradients':
                                new_mask = self.gradient_growth(name, new_mask, self.pruning_rate[name], weight)

                            elif self.growth_mode == 'momentum_neuron':
                                new_mask = self.momentum_neuron_growth(name, new_mask,  self.pruning_rate[name], weight)
                            # exchanging masks
                            self.masks.pop(name)
                            self.masks[name] = new_mask.float()

       


                else:
                    print ("layer_wise dst no room to grow")

        self.apply_mask()



    # def truncate_weights_prune(self, pruning_rate):
    #     print ("\n")
    #     print('dynamic sparse change prune')

    #     self.gather_statistics()




    #     #################################prune weights#############################

    #     tokill=self.total_nonzero-self.baseline_nonzero
    #     print ("to kill", tokill,"expect", self.baseline_nonzero)
    #     if tokill>0:
    #         self.total_removed=self.global_magnitude_death(tokill)



    #     self.apply_mask()


    # def truncate_weights_grow(self, pruning_rate):
    #        #################################grow weights#############################

    #     # get gradients
    #     # inputs, targets = next(iter(self.train_loader))
    #     # inputs = inputs.to(self.device)
    #     # targets = targets.to(self.device)
    #     # inputs.requires_grad = True
    #     # # Let's create a fresh copy of the network so that we're not worried about
    #     # self.module.zero_grad()
    #     # outputs = self.module.forward(inputs)
    #     # loss = F.nll_loss(outputs, targets)
    #     # loss.backward()

    #     print('dynamic sparse change grow')

    #     self.gather_statistics()

    #     togrow=self.total_params*pruning_rate-self.total_nonzero
    #     print ("self.total_params*pruning_rate",self.total_params*pruning_rate)
    #     print ("self.total_nonzero",self.total_nonzero)
    #     print ("to grow",togrow)
    #     if togrow>0:
    #         total_nonzero_new=self.global_gradient_growth(togrow)
    #         # print ("total_nonzero_new",total_nonzero_new)



    #     self.apply_mask()



    def truncate_weights(self, pruning_rate):
        print ("\n")
        print('dynamic sparse change')

        self.gather_statistics()




        #################################prune weights#############################
        if self.prune_mode == 'global_magnitude':
            to_kill = math.ceil(pruning_rate*self.total_nonzero)
            self.total_removed=self.global_magnitude_death(to_kill)


        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    mask = self.masks[name]

                    # death
                    if self.prune_mode == 'magnitude':
                        new_mask = self.magnitude_death(mask, weight, name, pruning_rate)
                    elif self.prune_mode == 'SET':
                        new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                    elif self.prune_mode == 'Taylor_FO':
                        new_mask = self.taylor_FO(mask, weight, name)
                    elif self.prune_mode == 'threshold':
                        new_mask = self.threshold_death(mask, weight, name)

                    # if self.args.fix_num_operation:
                    #     # print ("fix_num_operation")
                    #     self.pruning_rate[name] = 
                    # else:
                    #     self.pruning_rate[name] = int(self.masks[name].sum().item() - new_mask.sum().item())
                    
                    
                    # self.pruning_rate[name] = int(self.density_dict[name]*self.masks[name].numel()- new_mask.sum().item())
                    # self.pruning_rate[name] = int(self.masks[name].sum().item() - new_mask.sum().item())
                    # print ( name, int(self.density_dict[name]*self.masks[name].numel()), int(self.masks[name].sum().item()))
                    self.masks[name][:] = new_mask

                    togrow= int(self.density_dict[name]*self.temp_mask[name].numel()- new_mask.sum().item())
                    
                    self.pruning_rate[name] =togrow

        self.apply_mask()

        self.print_nonzero_counts()

       #################################grow weights#############################

        # get gradients
        # inputs, targets = next(iter(self.train_loader))
        # inputs = inputs.to(self.device)
        # targets = targets.to(self.device)
        # inputs.requires_grad = True
        # # Let's create a fresh copy of the network so that we're not worried about
        # self.module.zero_grad()
        # outputs = self.module.forward(inputs)
        # loss = F.nll_loss(outputs, targets)
        # loss.backward()



        self.gather_statistics()
        if self.growth_mode == 'global_gradients':
            print ("self.baseline_nonzero-self.total_nonzero",self.baseline_nonzero-self.total_nonzero)
            total_nonzero_new=self.global_gradient_growth(self.baseline_nonzero-self.total_nonzero)
            print ("total_nonzero_new",total_nonzero_new)
            print ("self.total_removed",self.total_removed)


        

        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    new_mask = self.masks[name].data.byte()
                    
                    # self.pruning_rate[name] = int(self.density_dict[name]*self.temp_mask[name].numel()- new_mask.sum().item())
                    


                    if self.pruning_rate[name]>1:

                        # growth
                        if self.growth_mode == 'random':
                            new_mask = self.random_growth(name, new_mask, self.pruning_rate[name], weight)

                        elif self.growth_mode == 'momentum':
                            new_mask = self.momentum_growth(name, new_mask, self.pruning_rate[name], weight)

                        elif self.growth_mode == 'gradients':
                            new_mask = self.gradient_growth(name, new_mask, self.pruning_rate[name], weight)

                        elif self.growth_mode == 'momentum_neuron':
                            new_mask = self.momentum_neuron_growth(name, new_mask,  self.pruning_rate[name], weight)
                        # exchanging masks
                        self.masks.pop(name)
                        self.masks[name] = new_mask.float()

                    else:
                        print ("layer_wise dst no room to grow in layer")

        self.apply_mask()


        self.print_nonzero_counts()
 
    # def truncate_weights(self):


    #     for module in self.modules:
    #         for name, weight in module.named_parameters():
    #             if name not in self.masks: continue
    #             mask = self.masks[name]
    #             self.name2nonzeros[name] = mask.sum().item()
    #             self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]
    #             # prune
    #             new_mask = self.prune_func(self, mask, weight, name)
    #             removed = self.name2nonzeros[name] - new_mask.sum().item()
    #             self.total_removed += removed
    #             self.name2removed[name] = removed
    #             self.masks[name][:] = new_mask

    #     for module in self.modules:
    #         for name, weight in module.named_parameters():
    #             if name not in self.masks: continue
    #             new_mask = self.masks[name].data.byte()
    #             # growth
    #             new_mask = self.growth_func(self, name, new_mask, math.floor(self.name2removed[name]), weight)
    #             # exchanging masks
    #             # self.masks.pop(name)
    #             self.masks[name][:] = new_mask.float()

    #     self.apply_mask()

    #     # calculity the spasity
    #     total_size = 0
    #     for name, weight in self.masks.items():
    #         total_size += weight.numel()
    #     print('Total Model parameters after dst:', total_size)

    #     sparse_size = 0
    #     for name, weight in self.masks.items():
    #         sparse_size += (weight != 0).sum().int().item()

    #     print('Total parameters under sparsity level of {0}: {1} after dst'.format(self.args.density, sparse_size / total_size))

    '''
                UTILITY
    '''
    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']

        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue

                if name in self.filter_masks:
                    mask=self.masks[name][self.filter_masks[name].bool()]
                else:
                    mask=self.masks[name]



                num_nonzeros = (mask != 0).sum().item()

                # print ("self.name2nonzeros",self.name2nonzeros)
                # val = '{0}: {1}, density: {2:.2f} "％"'.format(name,  num_nonzeros,
                #                                              100*(num_nonzeros) / float(mask.numel()) ) 
                # print(val)

                val = '{0}: {1}, spasity: {2:.2f} "％"'.format(name,  num_nonzeros,
                                                             100*(float(mask.numel()-num_nonzeros) / float(mask.numel()) ) )
                print(val)
        # print('Prune rate: {0}\n'.format(self.prune_rate))


       # calculity the spasity
        total_size = 0
        sparse_size = 0
        for name, mask in self.masks.items():

            if name in self.filter_masks:
                mask=self.masks[name][self.filter_masks[name].bool()]
            else:
                mask=self.masks[name]

            total_size += mask.numel()
            sparse_size += (mask != 0).sum().int().item()

        print('Total Model parameters after dst:', sparse_size)
        print('Total parameters under sparsity level of {0}: {1} after dst'.format(self.args.density, sparse_size / total_size),sparse_size,total_size)



    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item())/float(self.fired_masks[name].numel())
                # print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights/ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights

