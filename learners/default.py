from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
import copy
from utils.schedulers import CosineSchedule
from attacks.utils import apply_noise_patch

class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, learner_config):

        super(NormalNN, self).__init__()
        self.log = print
        self.config = learner_config
        self.out_dim = learner_config['out_dim']
        self.model = self.create_model()
        self.reset_optimizer = True
        self.overwrite = learner_config['overwrite']
        self.batch_size = learner_config['batch_size']
        self.tasks = learner_config['tasks']
        self.top_k = learner_config['top_k']

        # replay memory parameters
        self.memory_size = self.config['memory']
        self.task_count = 0

        # class balancing
        self.dw = self.config['DW']
        if self.memory_size <= 0:
            self.dw = False

        # supervised criterion
        self.criterion_fn = nn.CrossEntropyLoss(reduction='none')
        
        # cuda gpu
        if learner_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        
        # highest class index from past task
        self.last_valid_out_dim = 0 

        # highest class index from current task
        self.valid_out_dim = 0

        # set up schedules
        self.schedule_type = self.config['schedule_type']
        self.schedule = self.config['schedule']

        # initialize optimizer
        self.init_optimizer()

    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()

        if need_train:
            
            # data weighting
            self.data_weighting(train_dataset)
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task)  in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                    
                    # model update
                    loss, output= self.update_model(x, y)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

                # reset
                losses = AverageMeter()
                acc = AverageMeter()
                
        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        try:
            return batch_time.avg
        except:
            return None
        
    def warmup(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        # # try to load model
        # need_train = True
        # if not self.overwrite:
        #     try:
        #         self.load_model(model_save_dir)
        #         need_train = False
        #     except:
        #         pass

        # trains
        # if self.reset_optimizer:  # Reset optimizer before learning each task
        self.log('Optimizer is reset!')
        self.init_optimizer(warmup=True)

            
        # data weighting
        self.data_weighting(train_dataset)
        losses = AverageMeter()
        acc = AverageMeter()
        batch_time = AverageMeter()
        batch_timer = Timer()
        for epoch in range(5):
            self.epoch=epoch

            # if epoch > 0: self.scheduler.step()
            for param_group in self.optimizer.param_groups:
                self.log('LR:', param_group['lr'])
            batch_timer.tic()
            for i, (x, y, task)  in enumerate(train_loader):

                # verify in train mode
                self.model.train()

                y = torch.ones_like(y) * 200
                # send data to gpu
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()
                
                # model update
                loss, output= self.update_model(x, y)

                # measure elapsed time
                batch_time.update(batch_timer.toc())  
                batch_timer.tic()
                
                # measure accuracy and record loss
                y = y.detach()
                accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                losses.update(loss,  y.size(0)) 
                batch_timer.tic()

            # eval update
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=5))
            self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

            # reset
            losses = AverageMeter()
            acc = AverageMeter()
                
        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        try:
            return batch_time.avg
        except:
            return None
        
    def learn_trigger(self, train_loader, train_dataset, args):

        # noise
        noise = torch.zeros((1, 3, args.noise_size, args.noise_size), device='cuda')
        batch_pert = torch.autograd.Variable(noise.cuda(), requires_grad=True)
        batch_opt = torch.optim.RAdam(params=[batch_pert],lr=args.generating_lr_tri)

        criterion = torch.nn.CrossEntropyLoss()

            
        # data weighting
        self.data_weighting(train_dataset)
        losses = AverageMeter()
        acc = AverageMeter()
        batch_time = AverageMeter()
        batch_timer = Timer()

        #Freeze model, only train the triggers
        for param in self.model.parameters():
            param.requires_grad = False

        for epoch in range(args.gen_round):
            self.epoch=epoch
            batch_timer.tic()


            loss_list = []
            for i, (x, y, task)  in enumerate(train_loader):

                # send data to gpu
                y = torch.ones_like(y) * 200
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()

                new_x = torch.clone(x)
                clamp_batch_pert = torch.clamp(batch_pert,-args.l_inf_r*2,args.l_inf_r*2)
                new_x = torch.clamp(apply_noise_patch(clamp_batch_pert,new_x.clone(),mode=args.patch_mode),-1,1)

                logits = self.forward(new_x)
                # print(logits.shape)
                # print(y.shape)
                loss = criterion(logits, y.long())
                loss_regu = torch.mean(loss)
                

                batch_opt.zero_grad()
                loss_list.append(float(loss_regu.data))
                loss_regu.backward(retain_graph = True)
                batch_opt.step()

                # measure elapsed time
                batch_time.update(batch_timer.toc())  
                batch_timer.tic()
                
                # measure accuracy and record loss
                y = y.detach()
                accumulate_acc(logits, y, task, acc, topk=(self.top_k,))
                losses.update(loss.detach(),  y.size(0)) 
                batch_timer.tic()

            # eval update
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=args.gen_round))
            self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

            # reset
            losses = AverageMeter()
            acc = AverageMeter()

        return torch.clamp(batch_pert,-args.l_inf_r*2,args.l_inf_r*2)

    def test_noise(self, train_loader, train_dataset, args):
        criterion = torch.nn.CrossEntropyLoss()
        best_noise = torch.from_numpy(np.load('outputs/cifar-100/trigger-gen/coda-p/triggers/repeat-1/task-trigger-gen/target-2-04-17-12_18_05.npy'))
        best_noise = best_noise.cuda()
            
        # data weighting
        self.data_weighting(train_dataset)
        losses = AverageMeter()
        acc = AverageMeter()
        batch_time = AverageMeter()
        batch_timer = Timer()

        #Freeze model, only train the triggers
        for param in self.model.parameters():
            param.requires_grad = False

        for epoch in range(args.gen_round):
            self.epoch=epoch
            batch_timer.tic()

            loss_list = []
            for i, (x, y, task)  in enumerate(train_loader):

                # send data to gpu
                y = torch.ones_like(y) * 200
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()

                new_x = torch.clone(x)
                clamp_batch_pert = torch.clamp(best_noise,-args.l_inf_r*2,args.l_inf_r*2)
                new_x = torch.clamp(apply_noise_patch(clamp_batch_pert,new_x.clone(),mode=args.patch_mode),-1,1)

                logits = self.forward(new_x)
                # print(logits.shape)
                # print(y.shape)
                loss = criterion(logits, y.long())
                loss_regu = torch.mean(loss)
                

                loss_list.append(float(loss_regu.data))

                # measure elapsed time
                batch_time.update(batch_timer.toc())  
                batch_timer.tic()
                
                # measure accuracy and record loss
                y = y.detach()
                accumulate_acc(logits, y, task, acc, topk=(self.top_k,))
                losses.update(loss.detach(),  y.size(0)) 
                batch_timer.tic()

            # eval update
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=args.gen_round))
            self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

            # reset
            losses = AverageMeter()
            acc = AverageMeter()

        return 
   

    def criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised 

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        # print("Logits:", logits.shape)
        # print("y:", targets.shape)
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), logits

    def validation(self, dataloader, model=None, task_in = None, task_metric='acc',  verbal = True, task_global=False):

        if model is None:
            model = self.model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            if task_in is None:
                output = model.forward(input)[:, :self.valid_out_dim]
                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            else:
                # print(task_in)
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]

                mask = target < task_in[-1]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                
                if len(target) > 1:
                    if task_global:
                        output = model.forward(input)[:, :self.valid_out_dim]
                        acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                    else:
                        output = model.forward(input)[:, task_in]
                        acc = accumulate_acc(output, target-task_in[0], task, acc, topk=(self.top_k,))
            
        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    # data weighting
    def data_weighting(self, dataset, num_seen=None):
        self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

    def save_model(self, filename):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving class model to:', filename)
        torch.save(model_state, filename + 'class.pth')
        self.log('=> Save Done')

    def load_model(self, filename):
        try:
            self.model.load_state_dict(torch.load(filename + 'class.pth'))
        except:
            state_dict = torch.load(filename + 'class.pth')
            for key in list(state_dict.keys()):
                state_dict['module.' + key] = state_dict.pop(key)
            self.model.load_state_dict(state_dict)
        self.log('=> Load Done')
        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

    def load_model_other(self, filename, model):
        model.load_state_dict(torch.load(filename + 'class.pth'))
        if self.gpu:
            model = model.cuda()
        return model.eval()

    # sets model optimizers
    def init_optimizer(self, warmup=False):
        if warmup:
            "Using RAdam for warmup"
            torch.optim.RAdam(params=self.model.parameters(), lr=0.1)
        
        else:
            # parse optimizer args
            optimizer_arg = {'params':self.model.parameters(),
                            'lr':self.config['lr'],
                            'weight_decay':self.config['weight_decay']}
            if self.config['optimizer'] in ['SGD','RMSprop']:
                optimizer_arg['momentum'] = self.config['momentum']
            elif self.config['optimizer'] in ['Rprop']:
                optimizer_arg.pop('weight_decay')
            elif self.config['optimizer'] == 'amsgrad':
                optimizer_arg['amsgrad'] = True
                self.config['optimizer'] = 'Adam'
            elif self.config['optimizer'] == 'Adam':
                optimizer_arg['betas'] = (self.config['momentum'],0.999)

            # create optimizers
            self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
            
            # create schedules
            if self.schedule_type == 'cosine':
                self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
            elif self.schedule_type == 'decay':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim)

        return model

    def print_model(self):
        self.log(self.model)
        self.log('#parameter of model:', self.count_parameter())
    
    def reset_model(self):
        self.model.apply(weight_reset)

    def forward(self, x):
        return self.model.forward(x)[:, :self.valid_out_dim]

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        return out
    
    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())   

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log("Running on:", device)
        return device

    def pre_steps(self):
        pass

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def accumulate_acc(output, target, task, meter, topk):
    # print(output)
    # print(target)
    meter.update(accuracy(output, target, topk), len(target))
    return meter