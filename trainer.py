import os
import sys
import argparse
import torch
import numpy as np
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
import learners
import dataloaders
from dataloaders.utils import *
from dataloaders.dataloader import get_datasets
from dataloaders.dataloader import *

import time

class Attacker():
    def __init__(self, args, seed, metric_keys, save_keys):

        # process inputs
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.noise_size = args.noise_size
        
        # model load directory
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'ImageNet_R':
            Dataset = dataloaders.iIMAGENET_R
            num_classes = 200
            self.dataset_size = [224,224,3]
            self.top_k = 1
        else:
            raise ValueError('Dataset not implemented!')

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if self.seed > 0 and args.rand_split:
            print('=============================================')
            print('Shuffling....')
            print('pre-shuffle:' + str(class_order))
            random.seed(self.seed)
            random.shuffle(class_order)
            print('post-shuffle:' + str(class_order))
            print('=============================================')
        self.tasks = []
        self.tasks_logits = []
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p+inc])
            self.tasks_logits.append(class_order_logits[p:p+inc])
            p += inc
        if args.finetune:
            self.num_tasks = 2
        else:
            self.num_tasks = len(self.tasks)
        self.task_names = [str(i+1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        k = 1 # number of transforms per image
        if args.model_name.startswith('vit'):
            resize_imnet = True
        else:
            resize_imnet = False

        self.outter, self.train_target = get_datasets(args=args, trainDataset=Dataset, tasks=self.tasks, resize_imnet=resize_imnet, seed=self.seed, phase='trigger_gen', outterDataset=Dataset)

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                        'lr': args.lr,
                        'debug_mode': args.debug_mode == 1,
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay,
                        'schedule': args.schedule,
                        'schedule_type': args.schedule_type,
                        'model_type': args.model_type,
                        'model_name': args.model_name,
                        'optimizer': args.optimizer,
                        'gpuid': args.gpuid,
                        'memory': args.memory,
                        'temp': args.temp,
                        'out_dim': num_classes,
                        'overwrite': args.overwrite == 1,
                        'DW': args.DW,
                        'batch_size': args.batch_size,
                        'upper_bound_flag': args.upper_bound_flag,
                        'tasks': self.tasks_logits,
                        'top_k': self.top_k,
                        'prompt_param':[self.num_tasks,args.prompt_param]
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        self.args = args

    def train_surrogate(self):
    
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: temp_table[mkey] = []
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # for each task
        for i in range(self.max_task):

            # save current task index
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task
            task = self.tasks_logits[i]
            if self.oracle_flag:
                self.outter.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                self.add_dim += len(task)
            else:
                self.outter.load_dataset(i, train=True)
                self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            self.outter.append_coreset(only=False)

            # load dataloader
            train_loader = DataLoader(self.outter, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # learn
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+'surrogate'+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            avg_train_time = self.learner.learn_batch(train_loader, self.outter, model_save_dir)

            # save model
            self.learner.save_model(model_save_dir)

        return  

    def trigger_generating(self, trigger=None, dim=100, save=None, cur_noise=None):
        # set task id for model (needed for prompting)
        if self.args.finetune:
            i = 1
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i
        else: 
            i = 0

        # add valid class to classifier
        if self.add_dim > 0:
            self.learner.add_valid_output_dim(self.add_dim) 
        else:
            self.learner.add_valid_output_dim(dim)          

        if self.args.finetune:
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+'finetuned-'+ str(self.args.target_lab)+'/'
        else:
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+'surrogate'+'/'
        self.learner.load_model(model_save_dir)

        # save current task index
        print('======================', 'Generating triggers', '=======================')

        # load dataset for task
        if self.oracle_flag:
            self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # set task id for model (needed for prompting)
        try:
            self.learner.model.module.task_id = 1
        except:
            self.learner.model.task_id = 1

        # load dataloader
        target_loader = DataLoader(self.train_target, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

        # increment task id in prompting modules
        if i > 0:
            try:
                if self.learner.model.module.prompt is not None:
                    self.learner.model.module.prompt.process_task_count()
            except:
                if self.learner.model.prompt is not None:
                    self.learner.model.prompt.process_task_count()
                    
        # learn
        noise = self.learner.learn_trigger(target_loader, self.train_target, self.args, noise=cur_noise)

        # # save model
        # self.learner.save_model(model_save_dir)

        trigger_save_dir = self.model_top_dir + '/triggers/repeat-'+str(self.seed+1)+'/task-'+'trigger-gen'+'/'
        if not os.path.exists(trigger_save_dir): os.makedirs(trigger_save_dir)
        best_noise = noise.clone().detach().cpu()

        if save:
            save_name = trigger_save_dir + "target-" + str(self.args.target_lab) + "-noise_weight-" + str(self.args.noise_weight) + '-' + save
        else:
            save_name = trigger_save_dir + "target-" + str(self.args.target_lab) + "-noise_weight-" + str(self.args.noise_weight) + time.strftime("-%m-%d-%H_%M_%S",time.localtime(time.time())) 
        np.save(save_name, best_noise)


        return trigger

    def finetune(self, dim=100,save=None, noise=None):
        # set task id for model (needed for prompting)
        i = 1
        try:
            self.learner.model.module.task_id = i
        except:
            self.learner.model.task_id = i


        # add valid class to classifier
        if self.add_dim > 0:
            self.learner.add_valid_output_dim(self.add_dim) 
        else:
            self.learner.add_valid_output_dim(dim)          

        if noise is not None:
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+'finetuned-'+ str(self.args.target_lab)+'/'
        else:
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+'surrogate' + '/'
        self.learner.load_model(model_save_dir)

        # save current task index
        print('======================', 'Finetuning target', '=======================')

        # load dataset for task
        if self.oracle_flag:
            self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # set task id for model (needed for prompting)
        try:
            self.learner.model.module.task_id = 1
        except:
            self.learner.model.task_id = 1

        # load dataloader
        if noise is not None:
            target_loader = DataLoader(poison_image(self.train_target, noise=noise), batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))
        else:
            target_loader = DataLoader(self.train_target, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

        # increment task id in prompting modules
        if i > 0:
            try:
                if self.learner.model.module.prompt is not None:
                    self.learner.model.module.prompt.process_task_count(finetune=self.args.finetune)
            except:
                if self.learner.model.prompt is not None:
                    self.learner.model.prompt.process_task_count(finetune=self.args.finetune)

        # learn
        if save:
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+'finetuned-'+ str(self.args.target_lab) + '-' + save +'/'
        else:
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+'finetuned-'+ str(self.args.target_lab)+'/'
        if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
        self.learner.learn_batch(target_loader, self.train_target, model_save_dir)

        # save model
        # model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+'finetuned-'+ str(self.args.target_lab)+'/'
        self.learner.save_model(model_save_dir)

        return


class Victim:
    def __init__(self, args, seed, metric_keys, save_keys):

        # process inputs
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        
        # model load directory
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            DatasetBA = dataloaders.iCIFAR10BA
            DatasetASR = dataloaders.iCIFAR10ASR
            DatasetUntarget = dataloaders.iCIFAR10Untarget
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            DatasetBA = dataloaders.iCIFAR100BA
            DatasetASR = dataloaders.iCIFAR100ASR
            DatasetUntarget = dataloaders.iCIFAR100Untarget
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'ImageNet_R':
            Dataset = dataloaders.iIMAGENET_R
            DatasetBA = dataloaders.iIMAGENET_RBA
            num_classes = 200
            self.dataset_size = [224,224,3]
            self.top_k = 1
        else:
            raise ValueError('Dataset not implemented!')

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if self.seed > 0 and args.rand_split:
            print('=============================================')
            print('Shuffling....')
            print('pre-shuffle:' + str(class_order))
            random.seed(self.seed)
            random.shuffle(class_order)
            print('post-shuffle:' + str(class_order))
            print('=============================================')
        self.tasks = []
        self.tasks_logits = []
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p+inc])
            self.tasks_logits.append(class_order_logits[p:p+inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i+1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        k = 1 # number of transforms per image
        if args.model_name.startswith('vit'):
            resize_imnet = True
        else:
            resize_imnet = False
        
        best_noise = torch.from_numpy(np.load(args.noise_path))
        self.train_dataset, self.test_dataset, self.asr_dataset, self.untarget_dataset = get_datasets(args, trainDataset=DatasetBA, tasks=self.tasks, resize_imnet=resize_imnet, seed=self.seed, phase='poisoning', testDataset=Dataset,best_noise=best_noise, asrDataset=DatasetASR, untargetDataset=DatasetUntarget)

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                        'lr': args.lr,
                        'debug_mode': args.debug_mode == 1,
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay,
                        'schedule': args.schedule,
                        'schedule_type': args.schedule_type,
                        'model_type': args.model_type,
                        'model_name': args.model_name,
                        'optimizer': args.optimizer,
                        'gpuid': args.gpuid,
                        'memory': args.memory,
                        'temp': args.temp,
                        'out_dim': num_classes,
                        'overwrite': args.overwrite == 1,
                        'DW': args.DW,
                        'batch_size': args.batch_size,
                        'upper_bound_flag': args.upper_bound_flag,
                        'tasks': self.tasks_logits,
                        'top_k': self.top_k,
                        'prompt_param':[self.num_tasks,args.prompt_param]
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

    def task_eval(self, t_index, local=False, task=['clean_acc', 'asr', 'untarget']):

        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        
        # eval
        self.test_dataset.load_dataset(t_index, train=True)
        self.asr_dataset.load_dataset(t_index, train=True)
        self.untarget_dataset.load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        asr_loader  = DataLoader(self.asr_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        untarget_loader  = DataLoader(self.untarget_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        if local:
            # return self.learner.validation(test_loader, task_in = self.tasks_logits[t_index], task_metric=task[0]), self.learner.validation(asr_loader, task_in = self.tasks_logits[t_index], task_metric=task[1]), self.learner.validation(untarget_loader, task_in = self.tasks_logits[t_index], task_metric=task[2])
            return 0, self.learner.validation(asr_loader, task_in = self.tasks_logits[t_index], task_metric=task[0]), 0
            # return self.learner.validation(test_loader, task_in = self.tasks_logits[t_index], task_metric=task[0]), 0, 0

        else:
            # return self.learner.validation(test_loader, task_metric=task), self.learner.validation(asr_loader, task_metric=task, save=t_index), self.learner.validation(untarget_loader, task_metric=task)
            return 0, self.learner.validation(asr_loader, task_metric=task), 0
            # return self.learner.validation(test_loader, task_metric=task), 0, 0


    def train(self, avg_metrics):
    
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: temp_table[mkey] = []
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # for each task
        for i in range(self.max_task):

            # save current task index
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task
            task = self.tasks_logits[i]
            # print(self.tasks_logits)
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            self.train_dataset.append_coreset(only=False)

            # load dataloader
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # learn
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir)

            # save model
            self.learner.save_model(model_save_dir)
            
            # evaluate acc
            acc_table = []
            asr_table = []
            untarget_table = []
            acc_table_ssl = []
            self.reset_cluster_labels = True
            for j in range(i+1):
                cla, asr, untarget = self.task_eval(j)
                acc_table.append(cla)
                asr_table.append(asr)
                untarget_table.append(untarget)
            temp_table['clean_acc'].append(np.mean(np.asarray(acc_table)))
            temp_table['asr'].append(np.mean(np.asarray(asr_table)))
            temp_table['untarget'].append(np.mean(np.asarray(untarget_table)))

            # save temporary acc results
            for mkey in ['clean_acc', 'asr', 'untarget']:
                save_file = temp_dir + mkey + '.csv'
                np.savetxt(save_file, np.asarray(temp_table[mkey]), delimiter=",", fmt='%.2f')  

            if avg_train_time is not None: avg_metrics['time']['global'][i] = avg_train_time

        return avg_metrics 
    
    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.seed] = acc_table[val_name][train_name]
                avg_acc_pt_local[j,i,self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        # Gather the final avg accuracy
        avg_acc_all[:,self.seed] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}

    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}
            
        for i in range(self.max_task):

            # for l in [2, 3, 4]:
            #     with open('prompt_used_{}.txt'.format(l), 'a') as f:
            #         f.write("\n==================={}=================".format(i))

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # load model
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            self.learner.task_count = i 
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # evaluate acc
            metric_table['clean_acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['clean_acc'][self.task_names[i]] = OrderedDict()
            metric_table['asr'][self.task_names[i]] = OrderedDict()
            metric_table_local['asr'][self.task_names[i]] = OrderedDict()
            metric_table['untarget'][self.task_names[i]] = OrderedDict()
            metric_table_local['untarget'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True
            for j in range(i+1):
                val_name = self.task_names[j]
                cla, asr, untarget = self.task_eval(j)
                metric_table['clean_acc'][val_name][self.task_names[i]] = cla
                metric_table['asr'][val_name][self.task_names[i]] = asr
                metric_table['untarget'][val_name][self.task_names[i]] = untarget
            for j in range(i+1):
                val_name = self.task_names[j]
                cla, asr, untarget = self.task_eval(j, local=True)
                metric_table_local['clean_acc'][val_name][self.task_names[i]] = cla
                metric_table_local['asr'][val_name][self.task_names[i]] = asr
                metric_table_local['untarget'][val_name][self.task_names[i]] = untarget

        # summarize metrics
        avg_metrics['clean_acc'] = self.summarize_acc(avg_metrics['clean_acc'], metric_table['clean_acc'],  metric_table_local['clean_acc'])
        avg_metrics['asr'] = self.summarize_acc(avg_metrics['asr'], metric_table['asr'],  metric_table_local['asr'])
        avg_metrics['untarget'] = self.summarize_acc(avg_metrics['untarget'], metric_table['untarget'],  metric_table_local['untarget'])

        return avg_metrics
    

    def get_prompt(self):
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
        i = 9
        model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
        self.learner.task_count = i 
        self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
        self.learner.pre_steps()
        self.learner.load_model(model_save_dir)

        for l in self.learner.model.module.prompt.e_layers:
            p = getattr(self.learner.model.module.prompt,f'e_p_{l}')
            torch.save(p, 'prompt_{}.pt'.format(l))



class Trainer:

    def __init__(self, args, seed, metric_keys, save_keys):

        # process inputs
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        
        # model load directory
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'ImageNet_R':
            Dataset = dataloaders.iIMAGENET_R
            num_classes = 200
            self.dataset_size = [224,224,3]
            self.top_k = 1
        else:
            raise ValueError('Dataset not implemented!')

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if self.seed > 0 and args.rand_split:
            print('=============================================')
            print('Shuffling....')
            print('pre-shuffle:' + str(class_order))
            random.seed(self.seed)
            random.shuffle(class_order)
            print('post-shuffle:' + str(class_order))
            print('=============================================')
        self.tasks = []
        self.tasks_logits = []
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p+inc])
            self.tasks_logits.append(class_order_logits[p:p+inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i+1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        k = 1 # number of transforms per image
        if args.model_name.startswith('vit'):
            resize_imnet = True
        else:
            resize_imnet = False
        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=resize_imnet)
        test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, resize_imnet=resize_imnet)
        self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                            download_flag=True, transform=train_transform, 
                            seed=self.seed, rand_split=args.rand_split, validation=args.validation)
        # print(args.dataset)
        # print(args.train_aug)
        # print(resize_imnet)
        # print(self.tasks)
        # print(self.seed)
        # print(args.rand_split)
        # print(args.validation)
        self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                download_flag=False, transform=test_transform, 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation)

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                        'lr': args.lr,
                        'debug_mode': args.debug_mode == 1,
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay,
                        'schedule': args.schedule,
                        'schedule_type': args.schedule_type,
                        'model_type': args.model_type,
                        'model_name': args.model_name,
                        'optimizer': args.optimizer,
                        'gpuid': args.gpuid,
                        'memory': args.memory,
                        'temp': args.temp,
                        'out_dim': num_classes,
                        'overwrite': args.overwrite == 1,
                        'DW': args.DW,
                        'batch_size': args.batch_size,
                        'upper_bound_flag': args.upper_bound_flag,
                        'tasks': self.tasks_logits,
                        'top_k': self.top_k,
                        'prompt_param':[self.num_tasks,args.prompt_param]
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

    def task_eval(self, t_index, local=False, task='acc'):

        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        
        # eval
        self.test_dataset.load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        if local:
            return self.learner.validation(test_loader, task_in = self.tasks_logits[t_index], task_metric=task)
        else:
            return self.learner.validation(test_loader, task_metric=task)

    def train(self, avg_metrics):
    
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: temp_table[mkey] = []
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # for each task
        for i in range(self.max_task):

            # save current task index
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task
            task = self.tasks_logits[i]
            # print(self.tasks_logits)
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            self.train_dataset.append_coreset(only=False)

            # load dataloader
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # learn
            self.test_dataset.load_dataset(i, train=False)
            test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir, test_loader)

            # save model
            self.learner.save_model(model_save_dir)
            
            # evaluate acc
            acc_table = []
            acc_table_ssl = []
            self.reset_cluster_labels = True
            for j in range(i+1):
                acc_table.append(self.task_eval(j))
            temp_table['acc'].append(np.mean(np.asarray(acc_table)))

            # save temporary acc results
            for mkey in ['acc']:
                save_file = temp_dir + mkey + '.csv'
                np.savetxt(save_file, np.asarray(temp_table[mkey]), delimiter=",", fmt='%.2f')  

            if avg_train_time is not None: avg_metrics['time']['global'][i] = avg_train_time

        return avg_metrics 
    
    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.seed] = acc_table[val_name][train_name]
                avg_acc_pt_local[j,i,self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        # Gather the final avg accuracy
        avg_acc_all[:,self.seed] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}

    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}
            
        for i in range(self.max_task):

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # load model
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            self.learner.task_count = i 
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # evaluate acc
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j)
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table_local['acc'][val_name][self.task_names[i]] = self.task_eval(j, local=True)

        # summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'])

        return avg_metrics
    