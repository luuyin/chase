
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from sparselearning.snip import SNIP, GraSP
import numpy as np
import math
import random
# from sparselearning.funcs import global_magnitude_prune
from sparselearning.funcs import redistribution_funcs
# from sparselearning.flops import print_model_param_nums,count_model_param_flops,print_inf_time

def add_sparse_args(parser):
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold, CS_death.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.05, help='The pruning rate / death rate for DST.')
    parser.add_argument('--PF-rate', type=float, default=0.8, help='The pruning rate / death rate for Pruning and Finetuning.')
    parser.add_argument('--large-death-rate', type=float, default=0.80, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix', action='store_true', help='Fix topology during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='ER', help='sparse initialization')
    parser.add_argument('--update_frequency', type=int, default=1000, metavar='N', help='how many iterations to train between parameter exploration')


    # for filter grow/prune
    parser.add_argument('--filter_dst', action='store_true', help='filter_dst')
    parser.add_argument('--new_zero', action='store_true', help='Init with zero momentum buffer')
    parser.add_argument('--fix_num_operation', type=int, default=0,help='fix_num_operation in prune and grow')
    parser.add_argument('--bound_ratio', type=float, default=5.0, help='The density of the overall sparse network.')
    parser.add_argument('--minumum_ratio', type=float, default=0.5, help='The density of the overall sparse network.')
    parser.add_argument('--grow_mask_not', action='store_true', help='grow_mask_not')
    parser.add_argument('--no_grow_mask', action='store_true', help='no_grow_mask')
    parser.add_argument('--random_init', action='store_true', help='no_grow_mask')

    parser.add_argument('--connection_wise', action='store_true', help='connection_wise')
    parser.add_argument('--kernal_wise', action='store_true', help='kernal_wise')
    parser.add_argument('--mask_wise', action='store_true', help='mask_wise')
    parser.add_argument('--mag_wise', action='store_true', help='mag_wise')



    

    parser.add_argument('--grad_flow', action='store_true', help='grad_flow')
    parser.add_argument('--stop_dst_epochs', type=int, default=30,help='stop_dst_epochs in prune and grow')

    parser.add_argument('--stop_gmp_epochs', type=int, default=130,help='stop_dst_epochs in prune and grow')


    parser.add_argument('--mest', action='store_true', help='mest')
    parser.add_argument('--mest_dst', action='store_true', help='mest')

    parser.add_argument('--dst', action='store_true', help='mest')
    parser.add_argument('--gpm_filter_pune', action='store_true', help='gpm_filter_pune')
    

class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.0, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate



class Masking(object):
    def __init__(self, optimizer, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None, death_mode='magnitude', growth_mode='gradient', redistribution_mode='none', args=None, train_loader=None, device=None):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))


        self.args = args
        self.train_loader = train_loader
        self.device = torch.device("cuda")
        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_func = args.redistribution

        self.death_rate_decay = death_rate_decay
        self.PF_rate = args.PF_rate



        self.masks = {}
        self.nonzero_masks = {}
        self.new_masks = {}
        self.pre_tensor = {}
        self.pruning_rate = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}
        self.total_params=0
        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.death_rate = death_rate
        self.name2death_rate = {}
        self.steps = 0




        # channel_wise statisc
        self.channel_variance={}

        # if fix, then we do not explore the sparse connectivity
        if self.args.fix: self.prune_every_k_steps = None
        else: self.prune_every_k_steps = self.args.update_frequency


        # global growth/prune state
        self.prune_threshold = 0.001
        self.growth_threshold = 0.001
        self.growth_increment = 0.2
        self.increment = 0.2
        self.tolerance = 0.02



        #  for filter
        self.baseline_filter_num=0
        # self.layer_rate_decay=CosineDecay(self.args.start_layer_rate, 15)
        self.layer_rate_decay=CosineDecay(self.args.start_layer_rate, math.floor((self.args.stop_gmp_epochs)*len(self.train_loader)))
        # gmp channel prune

        self.initial_prune_time=0.0
        self.final_prune_time=math.floor(self.args.stop_gmp_epochs*len(self.train_loader))



        self.dst_decay=CosineDecay(0.5, math.floor((self.args.epochs)*len(self.train_loader)),0.005)

        self.active_new_mask={}
        self.passtive_new_mask={}


        jump_decay = CosineDecay(1, (args.update_frequency)*80/len(self.train_loader), 0)

        self.temp_mask={}
    '''
      CHANEL EXPLORE

    '''
    def get_module(self,key):
        if "MobileNetV1" in  self.module.__class__.__name__ :
            # print ("key",key)
            return getattr(self.module,key[0])[key[1]][key[2]]

        if "VGG" in self.module.__class__.__name__ :
            return self.module.features[key]
        if self.module.__class__.__name__ =="ResNet" or self.module.__class__.__name__ =="ResNetbasic" or self.module.__class__.__name__ =="Model":
            return getattr(getattr(self.module, key[0])[key[1]],key[2])

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

        
        for ind in self.module.layer2split:

            dim= self.get_module(ind).weight.shape[0]
            
            mask=torch.ones(dim)
            # mask[int(dim/2):]=0
            filter_names[self.get_mask_name(ind)]=mask

            passive_ind=self.module.next_layers[ind][0]
            passive_names[self.get_mask_name(passive_ind)]=mask
        
            

        self.filter_names=filter_names
        self.passive_names=passive_names

    def get_mask_name(self,key):
        if "VGG" in self.module.__class__.__name__ :
            return "features."+str(key)+".weight"
        if self.module.__class__.__name__ =="ResNet" or self.module.__class__.__name__ =="ResNetbasic":      
            weight_name=[]
            bias_name=[]
            for i in key:

                weight_name.append(str(i))
                bias_name.append(str(i))

            weight_name.append("weight")
            weight_name=".".join(weight_name)


            return weight_name



    def filter_num(self):


        total=0
        for name, tensor in self.filter_names.items():
            # print (name, tensor.sum().item())
            total+=tensor.sum().item()


        return  total



    def merge_filters (self):
    
        # print ("self.filter_names",self.filter_names)
        for active_grow_key in self.module.layer2split:
            passive_grow_key,norm_key=self.module.next_layers[active_grow_key]




                #load original ===============================

            m1=self.get_module(active_grow_key)
            m2=self.get_module(passive_grow_key)
            bnorm=self.get_module(norm_key)

            no_bias= (m1.bias==None)
            # weight, bias

            w1 = m1.weight.data
            w2 = m2.weight.data
            if not no_bias: b1 = m1.bias.data


            #mask    
            m1_mask=self.masks[self.get_mask_name(active_grow_key)]
            m2_mask=self.masks[self.get_mask_name(passive_grow_key)]





            #init new ===============================================================================     
            # weight, bias
            old_width = w1.size(0)
            nw1 = w1.clone()
            nw2 = w2.clone()
            if not no_bias: nb1 = b1.clone()


            nrunning_mean = bnorm.running_mean.clone()
            nrunning_var = bnorm.running_var.clone()       
            nweight = bnorm.weight.data.clone()
            nbias = bnorm.bias.data.clone()

            #new_mask        
            n_m1_mask= m1_mask.clone()
            n_m2_mask= m2_mask.clone()


            #deleting   ==============================================================================
            # transpose from original
            w2 = w2.transpose(0, 1)
            nw2 = nw2.transpose(0, 1)

            m2_mask=m2_mask.transpose(0, 1)
            n_m2_mask=n_m2_mask.transpose(0, 1)   


            # layer mask
            mask=m1_mask

            #grow based on weight norm
            del_mask=self.filter_names[self.get_mask_name(active_grow_key) ].bool()
            # print (self.get_mask_name(active_grow_key),del_mask.sum().item())


            # delet============================   
            # weight, bias   
            nw1 = nw1[del_mask] 
            nw2 = nw2[del_mask]
            if not no_bias: nb1 = nb1[del_mask] 


            # bn weight, bias  
            nrunning_mean=nrunning_mean[del_mask]
            nrunning_var=nrunning_var[del_mask]
            nweight = nweight[del_mask] 
            nbias = nbias[del_mask] 


            # masks
            n_m1_mask=n_m1_mask[del_mask]    
            n_m2_mask=n_m2_mask[del_mask]


            # transpose back
            w2.transpose_(0, 1)
            nw2.transpose_(0, 1)


            m2_mask.transpose_(0, 1)
            n_m2_mask.transpose_(0, 1)


            new_width=del_mask.sum().item()


            m1.out_channels = new_width
            m2.in_channels = new_width
            bnorm.num_features = new_width


            #finalize ===============================  
            # weight bias
            m1.weight = torch.nn.Parameter(nw1)
            m2.weight = torch.nn.Parameter(nw2)
            if not no_bias: m1.bias= torch.nn.Parameter(nb1)


            # norm
            bnorm.running_var = nrunning_var
            bnorm.running_mean = nrunning_mean
            bnorm.weight = torch.nn.Parameter(nweight)
            bnorm.bias = torch.nn.Parameter(nbias)

            # mask
            self.masks[self.get_mask_name(active_grow_key)]=n_m1_mask
            self.masks[self.get_mask_name(passive_grow_key)]=n_m2_mask




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
        # print ("all_scores",all_scores)

        acceptable_score = np.sort(np.array(all_scores))[-int(len(all_scores)-total_to_prune)]
        print ("acceptable_score",acceptable_score)
        # if acceptable_score==0:
        #     real_acceptable_score=np.sort(np.array(list(set(all_scores))))[1]

        # else:
        #     real_acceptable_score=acceptable_score

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


   

        ### layer wise prune ratio

        filter_numer=self.filter_num()

        if self.args.gpm_filter_pune:
            rate = self.layer_rate 
        else:
            rate=self.args.start_layer_rate-self.layer_rate
        rate=1-rate
        print ("to keep rate",rate)
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
                for l_index in self.module.layer2split:
                    if l_index not in residual_layer.keys():
                        prune_layer_index.append(l_index) 
                
                # to prune num
                total_to_prune=filter_numer-self.baseline_filter_num*rate
                pruned=[]
                for n_index in layer_prune:
                    pruned.extend(layer_prune[n_index] )
                total_to_prune=total_to_prune-len(pruned)
                print ("len(pruned)",len(pruned))
                print ("total_to_prune",total_to_prune)

                if total_to_prune>0:
                    
                    acceptable_score=self.prune_score(prune_layer_index,total_to_prune)


                    tem_layer_prune={}
                    for index in self.module.layer2split:


                        if index not in residual_layer.keys():
                    
                            grad=  torch.abs(self.get_module(index).weight.clone())
                            mask=self.masks[self.get_mask_name(index)]

                            del_ind=[]
                            filter_index=0
                            layer_grad_magnitude=[]
                            for filter_grad,filter_mask,filter_name in zip(grad, mask,self.filter_names[self.get_mask_name(index)]):
                                
                                if filter_name==1.0:
                    

                                    ### diffrent metric
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



                                    if grad_magnitude<=acceptable_score:
                                        del_ind.append(filter_index)

                                        layer_grad_magnitude.append(grad_magnitude)
                                filter_index+=1

                            # current layer
                            # if index not in layer_prune.keys():
                            current=self.filter_names[self.get_mask_name(index)].sum().item()
                            # else:
                            #     current=self.filter_names[self.get_mask_name(index)].sum().item()-len(layer_prune[index])


                            single_residual=self.minimum_layer[index]-(current-len( del_ind))
                            print ("index",index,"prune",len( del_ind),'current',current,"minimum_layer",self.minimum_layer[index],"single_residual",single_residual)


                            if index not in tem_layer_prune.keys(): tem_layer_prune[index]=[]

                            if single_residual>0:
                                residual+=single_residual
                                prune_lengh=int(current-self.minimum_layer[index] )

                                



                                layer_grad_magnitude=np.array(layer_grad_magnitude)
                                # layer_prune[index].extend (np.array(del_ind) [  np.array(layer_grad_magnitude) <   np.sort(np.array(layer_grad_magnitude))[prune_lengh]  ])
                                tem_layer_prune[index].extend (np.array(del_ind) [ np.argsort(layer_grad_magnitude)[:prune_lengh]]  )
                                residual_layer[index]=single_residual


                            else:

                                tem_layer_prune[index].extend(del_ind)

                            print ("layer_prune[index]",index,"len",len(tem_layer_prune[index]))





                    init_to_prune=0
                    for layer in tem_layer_prune:
                        init_to_prune+= len(tem_layer_prune[layer])

                    print ("init_to_prune",init_to_prune)
 
                    # print ("layer_prune",layer_prune)

                    if init_to_prune!=0 and total_to_prune<init_to_prune:
                        print ("sample")


                        ratio=  total_to_prune/ init_to_prune
                        for layer in tem_layer_prune:
                            tem_layer_prune[layer]= random.sample (tem_layer_prune[layer], int( round(ratio*len(tem_layer_prune[layer]))))
                    

                    print ("add temp to the final")
  
                    for index in self.module.layer2split:

                        if index not in layer_prune.keys(): layer_prune[index]=[]

                        if index in  tem_layer_prune:

                            layer_prune[index].extend(  tem_layer_prune[index] )





                    # print ("layer_prune",layer_prune)
                # to prune num

                pruned=[]
                for n_index in layer_prune:
                    print (n_index,len(layer_prune[n_index]))
                    pruned.extend(layer_prune[n_index] )



                print ("======residual",residual,"pruned",len(pruned))


                ###### layer prune


                print ("begin to prune")
                for active_prune_key in self.module.layer2split:
                    passive_prune_key,norm_key=self.module.next_layers[active_prune_key]


                    
        
                    if len(self.get_mask_name(active_prune_key))>0:


                        self.del_func(layer_prune[active_prune_key],
                                            active_prune_key,
                                            passive_prune_key)


                print ("pruneing layer",self.get_cfg())
                print ("current all fitler num",self.filter_num())
                print ("pruned",filter_numer-self.filter_num() )








                if i == 1000:
                    print('Error resolving the residual! Layers are too full! Residual left over: {0}'.format(residual))






            ###### layer prune

            for active_prune_key in self.module.layer2split:
                passive_prune_key,norm_key=self.module.next_layers[active_prune_key]


                
    
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

    def channel_stastic(self):
        total_variance=0.0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if len(weight.size()) == 4:
                    filter_vector = weight.view(weight.size(0)*weight.size(1), -1).sum(dim=1)
                    kernal_sparsity=((filter_vector==0).sum().int().item()/filter_vector.numel())
                    print(f"{name}, kernal sparsity is {(filter_vector==0).sum().int().item()/filter_vector.numel()}")
                    # channel sparsity
                    channel_vector = weight.view(weight.size(0), -1).sum(dim=1)
                    print(f"{name}, filter sparsity is {(channel_vector==0).sum().int().item()/filter_vector.numel()}")
                    
                    
                    print ("-------")
                    # redistribution
                    self.channel_variance[name] = kernal_sparsity

                    if not np.isnan(self.channel_variance[name]):
                        total_variance += self.channel_variance[name]


            for name in self.channel_variance:
                if total_variance != 0.0:
                    self.channel_variance[name] /= total_variance


    def init_growth_prune_and_redist(self):
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

                        
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())

                self.total_nonzero += self.name2nonzeros[name]
                self.total_zero += self.name2zeros[name]






    def step(self):
        self.steps += 1  
        self.optimizer.step()
 
        if self.args.sparse:
            self.apply_mask()

            # mest decay
            self.death_rate_decay.step()
            if self.args.decay_schedule == 'cosine':
                self.death_rate = self.death_rate_decay.get_dr()
            elif self.args.decay_schedule == 'constant':
                self.death_rate = self.args.death_rate


            # filter_dst_decay

            if self.args.gpm_filter_pune:
                self.layer_rate= self.gradual_pruning_rate(self.steps, 0.0, self.args.start_layer_rate, self.initial_prune_time, self.final_prune_time)
            else:
                self.layer_rate_decay.step()
                self.layer_rate= self.layer_rate_decay.get_dr()


            # filter_dst_decay
            self.dst_decay.step()
            self.dst_rate= self.dst_decay.get_dr()




            if self.args.filter_dst:
                if  self.steps >= self.initial_prune_time and self.steps < self.final_prune_time :
                    if (self.steps+ (1000/2)) % self.args.layer_interval== 0 :
                

                        print ("current layer rate",self.layer_rate)
                        
                        print ('===========del layer===============')
                        self.del_layer()

                        print ('===========done ===============')

                    
                        if "global" not in self.args.growth:
                            self.update_erk_dic()



            if self.args.mest:
                if self.steps< len(self.train_loader)*(self.args.epochs-self.args.stop_dst_epochs):
                    if self.prune_every_k_steps is not None:
                        if (self.steps % self.prune_every_k_steps == 0):

                            print ("current mest  death_rate", self.death_rate)

                        
                            self.truncate_weights_prune(self.death_rate)
                            self.print_nonzero_counts()

                            self.truncate_weights_grow(self.death_rate)
                            self.print_nonzero_counts()

                        


                            


                elif self.args.mest_dst:
                    if self.prune_every_k_steps is not None:
                        if (self.steps % self.prune_every_k_steps == 0):

                            print ("current dst rate",self.dst_rate)

                            self.truncate_weights(self.dst_rate)
     

            elif self.args.dst:

                if self.prune_every_k_steps is not None:
                    if (self.steps % self.prune_every_k_steps == 0):

                        print ("current dst rate",self.dst_rate)

                        self.truncate_weights(self.dst_rate)
                        self.print_nonzero_counts()
                        

                    

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


    def init(self, mode='ER', density=0.05,erk_power_scale=1.0):

        ## init for layer
        for index in self.module.layer2split:
            self.baseline_filter_num+=self.get_module(index).weight.shape[0]
            
        print ("baseline fitler num",self.baseline_filter_num)           


        self.bound_layer={}
        for index in self.module.layer2split:
            self.bound_layer[index] = int(self.get_module(index).out_channels)

        self.minimum_layer={}
        for index in self.module.layer2split:
            self.minimum_layer[index] = int(self.get_module(index).out_channels*(1-self.args.start_layer_rate)*self.args.minumum_ratio)

        




        ## init for connection
        self.density = density
        if mode == 'GMP':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).cuda()
            self.apply_mask()

    

        elif mode == 'snip':
            print('initialize by snip')
            layer_wise_sparsities = SNIP(self.module, self.density, self.train_loader, self.device)
            # re-sample mask positions
            for sparsity_, name in zip(layer_wise_sparsities, self.masks):
                self.masks[name][:] = (torch.rand(self.masks[name].shape) < (1-sparsity_)).float().data.cuda()

        elif mode == 'GraSP':
            print('initialize by GraSP')
            layer_wise_sparsities = GraSP(self.module, self.density, self.train_loader, self.device)
            # re-sample mask positions
            for sparsity_, name in zip(layer_wise_sparsities, self.masks):
                self.masks[name][:] = (torch.rand(self.masks[name].shape) < (1-sparsity_)).float().data.cuda()



        elif mode == 'pruning':
            print('initialize by pruning')
            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * (1 - self.PF_rate))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()

        elif mode == 'resume':
            # Initializes the mask according to the weights
            # which are currently zero-valued. This is required
            # if you want to resume a sparse model but did not
            # save the mask.
            print('initialize by resume')
            # self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    print((weight != 0.0).sum().item()/weight.numel())
                    self.masks[name] = (weight != 0.0).float().data.cuda()
                    # self.baseline_nonzero += weight.numel()*density
            self.apply_mask()

            # for module in self.modules:
            #     for name, weight in module.named_parameters():
            #         if name not in self.masks: continue
            #         print(f"The sparsity of layer {name} is {(self.masks[name]==0).sum()/self.masks[name].numel()}")

        elif mode == 'uniform':
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    self.masks[name_cur][:] = (torch.rand(weight.shape) < density).float().data.cuda() #lsw
                    # self.masks[name_cur][:] = (torch.rand(weight.shape) < density).float().data #lsw
            self.apply_mask()

        elif mode == 'fixed_ERK':
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


        elif mode == 'ER':
            print('initialize by SET')
            # initialization used in sparse evolutionary training
            total_params = 0
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    total_params += weight.numel()

            target_params = total_params *density
            tolerance = 5
            current_params = 0
            new_nonzeros = 0
            epsilon = 10.0
            growth_factor = 0.5
            # searching for the right epsilon for a specific sparsity level
            while not ((current_params+tolerance > target_params) and (current_params-tolerance < target_params)):
                new_nonzeros = 0.0
                index = 0
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    # original SET formulation for fully connected weights: num_weights = epsilon * (noRows + noCols)
                    # we adapt the same formula for convolutional weights
                    growth =  epsilon*sum(weight.shape)
                    new_nonzeros += growth
                current_params = new_nonzeros
                if current_params > target_params:
                    epsilon *= 1.0 - growth_factor
                else:
                    epsilon *= 1.0 + growth_factor
                growth_factor *= 0.95

            index = 0
            for name, weight in module.named_parameters():
                name_cur = name + '_' + str(index)
                index += 1
                if name_cur not in self.masks: continue
                growth =  epsilon*sum(weight.shape)
                prob = growth/np.prod(weight.shape)
                self.masks[name_cur][:] = (torch.rand(weight.shape) < prob).float().data.cuda() #lsw
                # self.masks[name_cur][:] = (torch.rand(weight.shape) < prob).float().data

        self.apply_mask()

        total_size = 0
        for name, weight in self.masks.items():
            if name in self.filter_masks:
                weight=weight[self.filter_masks[name].bool()]
            print (name,weight.numel())

            
            total_size  += weight.numel()
        

        sparse_size = 0
        for name, weight in self.masks.items():
            if name in self.filter_masks:
                weight=weight[self.filter_masks[name].bool()]
            sparse_size += (weight != 0).sum().int().item()


        print('Total Model parameters after init:', sparse_size, total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(density, sparse_size / total_size))



    def add_module(self, module, density, sparse_init='ER'):
        self.module = module
        print (module)
        self.sparse_init = sparse_init
        self.modules.append(module)
        print ("add module")

        self.get_filter_name()

        for name, tensor in module.named_parameters():
            self.names.append(name)
            self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()


        self.filter_masks={}
        for name, tensor in module.named_parameters():
            if name in self.filter_names:
                self.filter_masks[name]=torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
                self.filter_masks[name][~self.filter_names[name].bool()]=0

                # print (name,  self.filter_masks[name].sum())


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

                # print (name,  self.filter_masks[name].sum())






        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d)
        self.init(mode=sparse_init, density=density)







    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:

                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                                   np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self):
        # print ("fusing masks")
        for name, mask in self.masks.items():
            
            if name in self.filter_masks.keys():
                self.masks[name]= torch.logical_and(self.filter_masks[name], self.masks [name]).float()

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if "bias" in name:
                    weight_name=name[:-4]+"weight"
                    if weight_name in self.filter_names:
                        tensor.data = tensor.data*self.filter_names[weight_name].float().cuda()

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data*self.masks[name]
                    if 'momentum_buffer' in self.optimizer.state[tensor]:
                        self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]




    '''
                   DST
    '''






    def truncate_weights_GMP_global(self, epoch):
        '''
        Implementation  of global pruning version of GMP To prune, or not to prune: exploring the efficacy of pruning for model compression https://arxiv.org/abs/1710.01878
        :param epoch: current training epoch
        :return:
        '''
        prune_rate = 1 - self.density
        curr_prune_epoch = epoch
        total_prune_epochs = self.args.multiplier * self.args.final_prune_epoch - self.args.multiplier * self.args.init_prune_epoch + 1
        if epoch >= self.args.multiplier * self.args.init_prune_epoch and epoch <= self.args.multiplier * self.args.final_prune_epoch:
            prune_decay = (1 - ((
                                            curr_prune_epoch - self.args.multiplier * self.args.init_prune_epoch) / total_prune_epochs)) ** 3
            curr_prune_rate = prune_rate - (prune_rate * prune_decay)
            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * (1-curr_prune_rate))

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

        print('Total parameters under sparsity level of {0}: {1} after epoch of {2}'.format(self.density,
                                                                                            sparse_size / total_size,
                                                                                            epoch))
    def truncate_weights_GMP(self, epoch):
        '''
        Implementation  of GMP To prune, or not to prune: exploring the efficacy of pruning for model compression https://arxiv.org/abs/1710.01878
        :param epoch: current training epoch
        :return:
        '''
        prune_rate = 1 - self.density
        curr_prune_epoch = epoch
        total_prune_epochs = self.args.multiplier * self.args.final_prune_epoch - self.args.multiplier * self.args.init_prune_epoch + 1
        if epoch >= self.args.multiplier * self.args.init_prune_epoch and epoch <= self.args.multiplier * self.args.final_prune_epoch:
            prune_decay = (1 - ((curr_prune_epoch - self.args.multiplier * self.args.init_prune_epoch) / total_prune_epochs)) ** 3
            curr_prune_rate = prune_rate - (prune_rate * prune_decay)

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                    p = int(curr_prune_rate * weight.numel())
                    self.masks[name].data.view(-1)[idx[:p]] = 0.0
            self.apply_mask()
        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1} after epoch of {2}'.format(self.density, sparse_size / total_size, epoch))



    def truncate_weights(self, pruning_rate):
        print ("\n")
        print('dynamic sparse change')

        self.gather_statistics()




        #################################prune weights#############################
        if self.death_mode == 'global_magnitude':
            to_kill = math.ceil(pruning_rate*self.total_nonzero)
            self.total_removed=self.global_magnitude_death(to_kill)


        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    mask = self.masks[name]

                    # death
                    if self.death_mode == 'magnitude':
                        new_mask = self.magnitude_death(mask, weight, name, pruning_rate)
                    elif self.death_mode == 'SET':
                        new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                    elif self.death_mode == 'Taylor_FO':
                        new_mask = self.taylor_FO(mask, weight, name)
                    elif self.death_mode == 'threshold':
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
                    

                    to_grow= int(self.pruning_rate[name])


                    if (to_grow +  int(new_mask.sum().item()))  > int( (self.temp_mask[name].numel())):

                        to_grow= int( (self.temp_mask[name].numel())) -  int(new_mask.sum().item())

                        
                        print ("layer_wise dst no room to grow in each layer in",name )
                        self.pruning_rate[name]=to_grow


                    if to_grow>1:
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


    def truncate_weights_prune(self, pruning_rate):
        print ("\n")
        print('dynamic sparse change prune')

        self.gather_statistics()




        #################################prune weights#############################
        tokill=self.total_nonzero-self.baseline_nonzero
        print ("to kill", tokill,"expect", self.baseline_nonzero)
        if tokill>0:

            if self.death_mode == 'global_magnitude':
                self.total_removed=self.global_magnitude_death(tokill)

            else:

                pruning_rate=tokill/self.total_nonzero
                print ("calulate prune ratio",pruning_rate)


                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        mask = self.masks[name]

                        # death
                        if self.death_mode == 'magnitude':
                            new_mask = self.magnitude_death(mask, weight, name, pruning_rate)
                        elif self.death_mode == 'SET':
                            new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                        elif self.death_mode == 'Taylor_FO':
                            new_mask = self.taylor_FO(mask, weight, name)
                        elif self.death_mode == 'threshold':
                            new_mask = self.threshold_death(mask, weight, name)

      
                        self.masks[name][:] = new_mask
                        # self.pruning_rate[name] = int(self.name2nonzeros[name] - new_mask.sum().item())


            self.apply_mask()


    def truncate_weights_grow(self, pruning_rate):
       #################################grow weights#############################


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

                            if to_grow>1:
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
                            print ("to grow smaller than one, so skip")

       


                else:
                    print ("layer_wise dst no room to grow")

        self.apply_mask()









    def pruning(self):
        print('pruning...')
        print('death rate:', self.args.density)
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                num_remove = math.ceil((1-self.args.density) * weight.numel())
                x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                self.masks[name].data.view(-1)[idx[:num_remove]] = 0.0
        self.apply_mask()
        total_size = 0
        for name, weight in self.masks.items():
            total_size  += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1}'.format(self.args.density, sparse_size / total_size))




    '''
                    DEATH
    '''

    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.name2death_rate[name] * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    def magnitude_death(self, mask, weight, name, pruning_rate):

        num_zeros = (mask == 0).sum().item()
        num_remove = math.ceil(pruning_rate * (mask.sum().item()))
        if num_remove == 0.0: return weight.data != 0.0
        # num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])

        x, idx = torch.sort(torch.abs(weight.data.reshape(-1)))

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k - 1].item()

        return (torch.abs(weight.data) > threshold)






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

    def global_momentum_growth(self, total_regrowth):
        togrow = total_regrowth
        total_grown = 0
        last_grown = 0
        while total_grown < togrow*(1.0-self.tolerance) or (total_grown > togrow*(1.0+self.tolerance)):
            total_grown = 0
            total_possible = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    new_mask = self.masks[name]
                    grad = self.get_momentum_for_weight(weight)
                    grad = grad*(new_mask==0).float()
                    possible = (grad !=0.0).sum().item()
                    total_possible += possible
                    grown = (torch.abs(grad.data) > self.growth_threshold).sum().item()
                    total_grown += grown
            print(total_grown, self.growth_threshold, togrow, self.growth_increment, total_possible)
            if total_grown == last_grown: break
            last_grown = total_grown


            if total_grown > togrow*(1.0+self.tolerance):
                self.growth_threshold *= 1.02
                #self.growth_increment *= 0.95
            elif total_grown < togrow*(1.0-self.tolerance):
                self.growth_threshold *= 0.98
                #self.growth_increment *= 0.95

        total_new_nonzeros = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                new_mask = self.masks[name]
                grad = self.get_momentum_for_weight(weight)
                grad = grad*(new_mask==0).float()
                self.masks[name][:] = (new_mask.byte() | (torch.abs(grad.data) > self.growth_threshold)).float()
                total_new_nonzeros += new_mask.sum().item()
        return total_new_nonzeros


    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k-1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k-1].item()


        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)


        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''

    def random_growth(self, name, new_mask, total_regrowth, weight):

        if self.density_dict[name]==1.0:
            new_mask = torch.ones_like(new_mask, dtype=torch.float32, requires_grad=False).cuda()


        else:


            if name in self.filter_masks:
                temp_mask = new_mask [self.filter_masks[name].bool()]
            else:
                temp_mask=  new_mask

            n = (temp_mask==0).sum().item()
            if n == 0: return new_mask

            expeced_growth_probability = (total_regrowth/n)
            new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability #lsw
            # new_weights = torch.rand(new_mask.shape) < expeced_growth_probability

            new_mask=new_mask.byte() | new_weights

        if name in self.filter_masks:
            new_mask= torch.logical_and(self.filter_masks[name], new_mask).float()


        return new_mask


    def momentum_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

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
            print ("total_regrowth",total_regrowth)
            threshold, _ = torch.topk(all_scores, total_regrowth,largest=True, sorted=True)
            acceptable_score = threshold[-1]


            new_mask = (new_mask.byte() | (torch.abs(grad.data) > acceptable_score)).float()

        return new_mask

    # def gradient_growth(self, name, new_mask, total_regrowth, weight):

    #     grad = self.get_gradient_for_weights(weight)


    #     # operate on zero mask ind
    #     all_ind=torch.arange(grad.numel()).view(grad.shape)
    #     select_ind=all_ind[new_mask==0]

    #     # sort the gradients

    #     y, idx = torch.sort(torch.abs(grad[new_mask==0]), descending=True)

    #     # grow back in the zeo mask ind
    #     new_mask.data.reshape(-1)[select_ind[idx[:total_regrowth]]]=1.0


    #     # print ("after grow", (new_mask!=0).sum().int().item())



    #     return new_mask

    def momentum_neuron_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask==0).sum(sum_dim)

        M = M*(new_mask==0).float()
        for i, fraction  in enumerate(v):
            neuron_regrowth = math.floor(fraction.item()*total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

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

    def print_grad_flow(self):




        all_act_grad=0
        for name, weight in self.module.named_parameters():
            if name not in self.active_new_mask: continue
            grad = self.get_gradient_for_weights(weight)
            mask=self.active_new_mask[name]

            
            # act_grad = torch.abs(  grad*mask   ) .mean().item()

            act_grad=grad*mask

            # print (name)
            # for i in range(len(mask)):
            #     print ("mask",i,mask[i].sum())

            # for i in range(len(grad)):
            #     print ("grad",i,grad[i].sum())

            # for i in range(len(grad)):
            #     print ("mag",i,weight.clone()[i].sum())

            act_grad=torch.norm(act_grad)
            all_act_grad+=act_grad

        all_pass_grad=0
        for name, weight in self.module.named_parameters():
            if name not in self.passtive_new_mask: continue
            grad = self.get_gradient_for_weights(weight)
            mask=self.passtive_new_mask[name]

            # print (name)
            # for i in range(len(mask)):
            #     print ("mask",i,mask[i].sum())

            # for i in range(len(grad)):
            #     print ("grad",i,grad[i].sum())




            # pas_grad= torch.abs(  grad*mask  ) .mean().item()
            pas_grad=grad*mask
            pas_grad=torch.norm(pas_grad)
            all_pass_grad+=pas_grad


        print ("active grad flow",all_act_grad)
        print ( "all_pass_grad",all_pass_grad)

        total_size = 0
        for name, weight in self.active_new_mask.items():
            total_size  += weight.sum().item() 
        print ("self.active_new_mask",total_size)

        total_size = 0
        for name, weight in self.passtive_new_mask.items():
            total_size  += weight.sum().item() 

        print ("self.passtive_new_mask",total_size) 



    # def print_nonzero_counts(self):
    #     for module in self.modules:
    #         for name, tensor in module.named_parameters():
    #             if name not in self.masks: continue
    #             mask = self.masks[name]
    #             num_nonzeros = (mask != 0).sum().item()
    #             val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.name2nonzeros[name], num_nonzeros,
    #                                                            num_nonzeros / float(mask.numel()))
    #             print(val)


    #     total_size = 0
    #     sparse_size = 0
    #     for name, weight in self.masks.items():
    #         total_size += weight.numel()
    #         sparse_size += (weight != 0).sum().int().item()
    #         density =  sparse_size / total_size
    #     print(60 * '=')
    #     print('the current density is {0}: {1} {2}'.format(density,sparse_size,total_size))
    #     print(60 * '=')

    def print_nonzero_counts(self):



        total_size = 0
        for name, weight in self.masks.items():
            if name in self.filter_masks:
                weight=weight[self.filter_masks[name].bool()]

            total_size  += weight.numel()
        

        sparse_size = 0
        for name, weight in self.masks.items():
            if name in self.filter_masks:
                weight=weight[self.filter_masks[name].bool()]
            sparse_size += (weight != 0).sum().int().item()




        total_size = 0
        total_mask_nonzeros=0
        total_weight_nonzeros=0

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue


                if name in self.filter_masks:

                    print ("before",  "tensor", tensor.numel(),"mask",self.masks[name].sum().item(), self.masks[name].numel())
                    mask=self.masks[name][self.filter_masks[name].bool()]
                    tensor=tensor[self.filter_masks[name].bool()]

                else:
                    mask=self.masks[name]

                total_size += tensor.numel()
                #mask nonzero num
                mask_nonzeros=(mask!= 0.0).sum().item()
                total_mask_nonzeros+=mask_nonzeros

                # weight nonzero num
                weight_nonzeros=(tensor != 0.0).sum().item()
                total_weight_nonzeros+=weight_nonzeros


                print('{0}, mask/weight parameters,{1},{2} density {3},{4}'.format(name,mask_nonzeros,weight_nonzeros,mask_nonzeros /tensor.numel(),weight_nonzeros/tensor.numel()))             

        print('Total Model parameters after dst:', total_mask_nonzeros, total_weight_nonzeros)
        print('Total parameters under density level of {0}: {1} {2} after dst'.format(self.args.density,total_mask_nonzeros /total_size, total_weight_nonzeros /total_size))





        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}\n'.format(self.death_rate))
                break



    def print_layerwise_density(self):

        temp_model=copy.deepcopy(self.module )
                    
        for name, weight in temp_model:
            if name in self.filter_names:
                temp_model[name]=self.temp_mask[name][self.filter_names[name].bool()]

        for name, weight in self.masks.items():
    
            if name in self.passive_names:


                temp=temp_model[name]

                    # transpose
                temp =temp.transpose(0, 1)
                temp=temp[self.passive_names[name].bool()]
                temp.transpose_(0, 1)

                temp_model[name]=temp


  
        for name, weight in temp_model:
            if name not in self.masks: continue

            if len(weight.shape)==4:
        #         print (name)
                
                # channel sparsity
                for channel_vector in weight:
            
                    channel_zero=(channel_vector!=0).sum().int().item()
                    channel_all=channel_vector.numel()

                    print("check in", name, "density is",channel_zero/channel_all,"weight density is",channel_zero/channel_all,"weight magnitue", torch.abs(channel_vector).mean().item()  )
        



    def reset_momentum(self):
        """
        Taken from: https://github.com/AlliedToasters/synapses/blob/master/synapses/SET_layer.py
        Resets buffers from memory according to passed indices.
        When connections are reset, parameters should be treated
        as freshly initialized.
        """
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                weights = list(self.optimizer.state[tensor])
                for w in weights:
                    if w == 'momentum_buffer':
                        # momentum
                        self.optimizer.state[tensor][w][mask==0] = torch.mean(self.optimizer.state[tensor][w][mask.byte()])
                        # self.optimizer.state[tensor][w][mask==0] = 0
                    elif w == 'square_avg' or \
                        w == 'exp_avg' or \
                        w == 'exp_avg_sq' or \
                        w == 'exp_inf':
                        # Adam
                        self.optimizer.state[tensor][w][mask==0] = torch.mean(self.optimizer.state[tensor][w][mask.byte()])

