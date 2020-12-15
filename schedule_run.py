# Gpysum-scheduler
# command: python3 schedule_jobs.py --config default --trainer default \
# --dataset default --model default --logger file --job_name default \
# --job_type short --job_duration 4:00:00 --nr_gpus 1 --nr_cpus 1 --memory 8 \
# --log_path /home/vkesha/experiment/logs --nr_partition 1

import os
import subprocess
import argparse
import importlib
from itertools import cycle

from utils.generic_utils import get_configurations


def argument_parser():
    parser = argparse.ArgumentParser(description='gypsum_scheduler')
    parser.add_argument('--seed', default=0, type=int,
                        help='seeds for deterministic runtime')
    parser.add_argument('--config', default='default', type=str,
                        help='configuration file name')
    parser.add_argument('--trainer', default='default', type=str,
                        help='trainer file name')
    parser.add_argument('--dataset', default='default', type=str,
                        help='dataset file name')
    parser.add_argument('--model', default='default', type=str,
                        help='model file name')
    parser.add_argument('--logger', default=None, type=str,
                        choices=[None, 'print', 'file', 'comet', 'print_file'],
                        help='logger choices')
    parser.add_argument('--key', default='', type=str,
                        help='used if logger is set to comet')
    parser.add_argument('--job_name', default='default', type=str,
                        help='job_name')
    parser.add_argument('--job_type', default='short', type=str,
                        help='job_type')
    parser.add_argument('--job_duration', default='4:00:00', type=str,
                        help='max_job_duration')
    parser.add_argument('--nr_gpus', default=1, type=int, help='nr_gpus')
    parser.add_argument('--nr_cpus', default=1, type=int, help='nr_cpus')
    parser.add_argument('--memory', default=8, type=int, help='memory_in_GB')
    parser.add_argument('--log_path', default='logs', type=str,
                        help='log_path')
    parser.add_argument('--nr_partition', default=1, type=int,
                        help='nr_partition')
    args = parser.parse_args()
    return args


script_template = \
"""#!/bin/sh
#SBATCH --job-name={job_name}
#SBATCH -o {log_path}/%j.txt
#SBATCH --time={job_duration}
#SBATCH --partition={gpu_name}-{job_type}
#SBATCH --mem-per-cpu={memory}GB
#SBATCH --gres=gpu:{nr_gpus}
#SBATCH --cpus-per-task={nr_cpus}
#SBATCH -d singleton
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7
export OMP_NUM_THREADS=7
python -W ignore run.py --model {model} --dataset {dataset} --trainer \
{trainer} --config {config} --logger {logger}
"""

def main():
    args = argument_parser()
    config_module = importlib.import_module("configs."+args.config)
    configs = config_module.config
    possible_configs = get_configurations(configs)
    total_configs = len(possible_configs)
    nr_configs_per_partition = (total_configs//args.nr_partition)
    config_sub_list = [possible_configs[i:i + nr_configs_per_partition]
                       for i in range(0, len(possible_configs),
                                      nr_configs_per_partition)]

    # Go though all config list set and construct config.py
    for config_idx, configs in enumerate(config_sub_list):
        config_file = os.path.join('configs', args.job_name +
                                   "_config_"+str(config_idx)+".py")
        with open(os.path.join(config_file), 'w') as file:
            file.write("config={}".format(configs))
    log_path = args.log_path
    if not os.path.exists(log_path): os.makedirs(log_path)

    # Construct the sbatch scripts and spawn the process on gypsum
    # TODO: A logic to get best GPUs.
    gpu_list = cycle(["titanx", "m40", "1080ti", "2080ti"])
    for config_idx, configs in enumerate(config_sub_list):
        config_name = args.job_name + "_config_"+str(config_idx)
        gpu_name = next(gpu_list)
        script = script_template.format(config=config_name,
                                        trainer=args.trainer,
                                        dataset=args.dataset,
                                        model=args.model,
                                        logger=args.logger,
                                        job_name=args.job_name +
                                        '_' + str(config_idx),
                                        job_type=args.job_type,
                                        job_duration=args.job_duration,
                                        nr_gpus=args.nr_gpus,
                                        nr_cpus=args.nr_cpus,
                                        memory=args.memory,
                                        gpu_name=gpu_name,
                                        log_path=args.log_path)
        script_file = 'script_'+args.job_name+'_'+str(config_idx)+'.sbatch'
        with open(script_file, 'w') as file:
            file.write(script)
        out = subprocess.Popen(['squeue', '-u', 'vkeshav'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        stdout, _ = out.communicate()
        print(stdout)
        print("************************************************************\n")
        command = 'sbatch ' + script_file
        subprocess.check_output(command, shell=True)

if __name__ == "__main__":
    main()