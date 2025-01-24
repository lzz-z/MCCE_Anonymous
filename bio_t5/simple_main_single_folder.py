import pickle
import json
import uuid
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# 禁用RDKit的日志输出
RDLogger.DisableLog('rdApp.*')

import selfies as sf
import re
import os
import pickle 
import torch
import copy
from nanoT5.utils import (
    setup_basics,
    train,
    predict,
    eval,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_dataloaders,
    get_config,
    setup_basics_without_acc
)
from accelerate import Accelerator

import yaml
from argparse import Namespace
import argparse
from simple_main_multiple_class import evaluate_one_folder, get_checkpoint_path


# ckpt_blob_root = '/blob/blob/'
# inference_blob_root = '/blob/rl4s/'
Parser = argparse.ArgumentParser()
Parser.add_argument('--task', default='bbbp', help='task name')
Parser.add_argument('--ckpt_blob_root', default='/blob/blob', help='blob root for checkpoint')
Parser.add_argument('--inference_blob_root', default='/blob/nlm', help='blob root for inference')
Parser.add_argument('--inference_folder', default='zekun/instruct/base1b/instruct_task_20240807/wang/bbbp/response', help='inference folder')
Parser.add_argument('--device', default='cuda:0', help='device')
Parser.add_argument('--args_save_path', default='args.save', help='args save path')
Parser.add_argument('--skip_col_name', default=['input'], help='skip')

hparams = Parser.parse_args()

# hparams =  Parser.parse_args([])
batch_size = 1280
args = torch.load(hparams.args_save_path)

 

args.model.checkpoint_path = get_checkpoint_path(hparams.task, hparams.ckpt_blob_root)
print(args.model.checkpoint_path)
# args.device = args.device
config = get_config(args)
tokenizer = get_tokenizer(args)
logger = setup_basics_without_acc(args)
model = get_model(args, config, tokenizer, logger)
model = model.to(hparams.device)

# all_res = {}

one_folder_res = evaluate_one_folder(
folder_path=f'{hparams.inference_blob_root}/{hparams.inference_folder}', 
task=hparams.task, 
tokenizer=tokenizer, 
model=model, 
batch_size=1280,
args=hparams)


# 检查文件夹是否存在，不存在提前创建
if not os.path.exists('res'):
    os.makedirs('res')
with open(f'res/task{hparams.task}_folder_zekun_res.pkl', 'wb') as f:
    pickle.dump(one_folder_res, f)


 