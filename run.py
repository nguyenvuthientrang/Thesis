from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import argparse
import torch
import numpy as np
import yaml
import json
import random
from trainer import Trainer, Attacker, Victim

def create_args():
    
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()

    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                         help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_dir', type=str, default="outputs/out",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--learner_type', type=str, default='default', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='NormalNN', help="The class name of learner")
    parser.add_argument('--debug_mode', type=int, default=0, metavar='N',
                        help="activate learner specific settings for debug_mode")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--overwrite', type=int, default=0, metavar='N', help='Train regardless of whether saved model exists')

    # CL Args          
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    parser.add_argument('--prompt_param', nargs="+", type=float, default=[1, 1, 1],
                         help="e prompt pool size, e prompt length, g prompt length")

    # Config Arg
    parser.add_argument('--config', type=str, default="configs/config.yaml",
                         help="yaml experiment config input")
    
    # Attack Args
    parser.add_argument('--backdoor', action='store_true', default=True)
    parser.add_argument('--dataset_path', type=str, default='./data/')
    parser.add_argument('--target_lab', type=int, default=2)
    parser.add_argument('--noise_size', type=int, default=224)
    parser.add_argument('--l_inf_r', type=float, default=16/255)
    parser.add_argument('--surrogate_epochs', type=int, default=200)
    parser.add_argument('--generating_lr_warmup', type=float, default=0.1)
    parser.add_argument('--warmup_round', type=int, default=5)
    parser.add_argument('--generating_lr_tri', type=float, default=0.01)
    parser.add_argument('--gen_round', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=350)
    parser.add_argument('--patch_mode', type=str, default='add')
    parser.add_argument('--noise_weight', type=int, default=100)
    parser.add_argument('--craft_round', type=int, default=4)

    # Victim Args
    parser.add_argument('--poison_amount', type=int, default=25)
    parser.add_argument('--multi_test', type=int, default=3)
    parser.add_argument('--random_seed', type=int, default=65)
    parser.add_argument('--noise_path', type=str, default='./outputs/cifar-100/attack/coda-p/triggers/repeat-1/task-trigger-gen/06-29-03_59_57.npy')
    parser.add_argument('--finetune', action='store_true')


    return parser

def get_args(argv):
    parser=create_args()
    args = parser.parse_args(argv)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config.update(vars(args))
    return argparse.Namespace(**config)

# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

if __name__ == '__main__':


    args = get_args(sys.argv[1:])

    # determinstic backend
    torch.backends.cudnn.deterministic=True

    # duplicate output stream to output file
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_out = args.log_dir + '/output.log'
    sys.stdout = Logger(log_out)

    # save args
    with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)
    
    metric_keys = ['clean_acc', 'asr', 'untarget', 'time']
    save_keys = ['global', 'pt', 'pt-local']
    global_only = ['time']
    avg_metrics = {}
    for mkey in metric_keys: 
        avg_metrics[mkey] = {}
        for skey in save_keys: avg_metrics[mkey][skey] = []

    # load results
    if args.overwrite:
        start_r = 0
    else:
        try:
            for mkey in metric_keys: 
                for skey in save_keys:
                    if (not (mkey in global_only)) or (skey == 'global'):
                        save_file = args.log_dir+'/results-'+mkey+'/'+skey+'.yaml'
                        if os.path.exists(save_file):
                            with open(save_file, 'r') as yaml_file:
                                yaml_result = yaml.safe_load(yaml_file)
                                avg_metrics[mkey][skey] = np.asarray(yaml_result['history'])

            # next repeat needed
            start_r = avg_metrics[metric_keys[0]][save_keys[0]].shape[-1]

            # extend if more repeats left
            if start_r < args.repeat:
                max_task = avg_metrics['clean_acc']['global'].shape[0]
                for mkey in metric_keys: 
                    avg_metrics[mkey]['global'] = np.append(avg_metrics[mkey]['global'], np.zeros((max_task,args.repeat-start_r)), axis=-1)
                    if (not (mkey in global_only)):
                        avg_metrics[mkey]['pt'] = np.append(avg_metrics[mkey]['pt'], np.zeros((max_task,max_task,args.repeat-start_r)), axis=-1)
                        avg_metrics[mkey]['pt-local'] = np.append(avg_metrics[mkey]['pt-local'], np.zeros((max_task,max_task,args.repeat-start_r)), axis=-1)

        except:
            start_r = 0
    # start_r = 0
    # print(start_r, args.repeat)
    for r in range(start_r, args.repeat):

        print('************************************')
        print('* STARTING TRIAL ' + str(r+1))
        print('************************************')

        # set random seeds
        seed = r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # set up an attacker
        attacker = Attacker(args, seed, metric_keys, save_keys)

        # init total run metrics storage
        max_task = attacker.max_task
        if r == 0: 
            for mkey in metric_keys: 
                avg_metrics[mkey]['global'] = np.zeros((max_task,args.repeat))
                if (not (mkey in global_only)):
                    avg_metrics[mkey]['pt'] = np.zeros((max_task,max_task,args.repeat))
                    avg_metrics[mkey]['pt-local'] = np.zeros((max_task,max_task,args.repeat))

        # train attacker
        # attacker.train_surrogate()  
        # attacker.finetune()  
        # attacker.trigger_generating()

        # attacker = Attacker(args, seed, metric_keys, save_keys)
        # attacker.finetune()
        # # trigger_save_dir = attacker.model_top_dir + '/triggers/repeat-'+str(attacker.seed+1)+'/task-'+'trigger-gen'+'/'
        # # save_name = trigger_save_dir + "target-" + str(attacker.args.target_lab) + "-noise_weight-" + str(attacker.args.noise_weight) + '-' + str(1)
        # # best_noise = torch.from_numpy(np.load(save_name + '.npy'))
        # for i in range(1, args.craft_round+1):
        #     attacker = Attacker(args, seed, metric_keys, save_keys)
        #     if i == 1:
        #         attacker.trigger_generating(save=str(i))
        #     else:
        #         attacker.trigger_generating(save=str(i), cur_noise=best_noise)
            
        #     trigger_save_dir = attacker.model_top_dir + '/triggers/repeat-'+str(attacker.seed+1)+'/task-'+'trigger-gen'+'/'
        #     save_name = trigger_save_dir + "target-" + str(attacker.args.target_lab) + "-noise_weight-" + str(attacker.args.noise_weight) + '-' + str(i)
        #     best_noise = torch.from_numpy(np.load(save_name + '.npy'))
            
        #     attacker = Attacker(args, seed, metric_keys, save_keys)
        #     attacker.finetune(noise=best_noise)
        # attacker = Attacker(args, seed, metric_keys, save_keys)
        # attacker.trigger_generating(cur_noise=best_noise)


        # set up a victim
        victim = Victim(args, seed, metric_keys, save_keys)

        # victim.get_prompt()

        # # poison training
        # victim.train(avg_metrics)   

        # # evaluate model
        avg_metrics = victim.evaluate(avg_metrics)    

        # # # save results
        # for mkey in metric_keys: 
        #     m_dir = args.log_dir+'/results-'+mkey+'/'
        #     if not os.path.exists(m_dir): os.makedirs(m_dir)
        #     for skey in save_keys:
        #         if (not (mkey in global_only)) or (skey == 'global'):
        #             save_file = m_dir+skey+'.yaml'
        #             result=avg_metrics[mkey][skey]
        #             yaml_results = {}
        #             if len(result.shape) > 2:
        #                 yaml_results['mean'] = result[:,:,:r+1].mean(axis=2).tolist()
        #                 if r>1: yaml_results['std'] = result[:,:,:r+1].std(axis=2).tolist()
        #                 yaml_results['history'] = result[:,:,:r+1].tolist()
        #             else:
        #                 yaml_results['mean'] = result[:,:r+1].mean(axis=1).tolist()
        #                 if r>1: yaml_results['std'] = result[:,:r+1].std(axis=1).tolist()
        #                 yaml_results['history'] = result[:,:r+1].tolist()
        #             with open(save_file, 'w') as yaml_file:
        #                 yaml.dump(yaml_results, yaml_file, default_flow_style=False)

        # # Print the summary so far
        # print('===Summary of experiment repeats:',r+1,'/',args.repeat,'===')
        # for mkey in metric_keys: 
        #     print(mkey, ' | mean:', avg_metrics[mkey]['global'][-1,:r+1].mean(), 'std:', avg_metrics[mkey]['global'][-1,:r+1].std())
    
    

