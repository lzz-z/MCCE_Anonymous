import numpy as np
import pandas as pd
import os
import re
 

import pickle


def process_all_folders_res(all_folders_res):
    all_res = {}
    # for task_folder, task_folder_value in all_folders_res.items():
    for  class_folder, class_folder_value in all_folders_res.items():
        all_res[class_folder] = {}        
        result_in_one_file = []
        for column_name, column_name_value in all_folders_res[class_folder].items():
            this_pred_label = all_folders_res[class_folder][column_name]['predictions']
            this_res = np.zeros([len(true_label),1])
            assert len(true_label) == len(this_pred_label)
            for ii in range(len(true_label)):
                if this_pred_label[ii] is None:
                    this_res[ii] = -1
                elif this_pred_label[ii] == true_label[ii]:
                    this_res[ii] = 1
                else:
                    this_res[ii] = 0
                    
            result_in_one_file.append(this_res)
        all_res[class_folder] = np.concatenate(result_in_one_file, axis=1)
    return all_res



    
def get_evaluation_results(all_res, task_name, fail_strategy='as_0', aggregation_type='mean', column='3:', sft_epoch_for_dpo=[1,2,3]):
    stat_res = {}

    for task_class_folder in all_res:
        for inference_file_name in all_res[task_class_folder]:
            this_res = eval(f'all_res[task_class_folder][inference_file_name][:,{column}]')
 
        
            if fail_strategy == 'as_0':
                this_res[this_res == -1] = 0
            elif fail_strategy == 'mask':
                this_res = this_res[this_res != -1]
            
           

            if len(this_res.shape) == 1:
                success_rate = this_res.mean()
            else:
                success_rate = eval(f'this_res.{aggregation_type}(axis=1).mean()')
 
            stat_res[f'{task_name} {task_class_folder} {inference_file_name} success rate:'] =  success_rate  
                # assert 1==2

    ######present beatuiful table
    table = {
        'Pretrain' : [],
        'Pretrain + SFT' : [],
        'Pretrain + DPO' : [],
        # 'Pretrain + SFT + DPO' : []
    }
    for epoch in sft_epoch_for_dpo:
        table[f'Pretrain + SFT epoch{epoch} + DPO'] = []

    for  key, values in stat_res.items():
        if 'pret_sft_dpo ' in key :
            match = re.search(r'epoch(\d+)', key)
            if match:
                epoch = match.group(1)  # 提取整个匹配的字符串，例如 
                if int(epoch) in sft_epoch_for_dpo:
                    column_name = f'Pretrain + SFT epoch{epoch} + DPO'
                # print(table.keys())
                    table[column_name].append(values)
        elif 'pret_sft ' in key:
            table['Pretrain + SFT'].append(values)
        elif 'pret_dpo ' in key:
            table['Pretrain + DPO'].append(values)
        elif 'pret ' in key:
            table['Pretrain'].append(values)
        else:
            pass
    
    max_length = max(len(column) for column in table.values())
    row_names = [f'Epoch {i+1}' for i in range(max_length)]
    # 将每列扩展到最长列的长度，填充空值为None
    for key in table:
        while len(table[key]) < max_length:
            table[key].append('')

    df = pd.DataFrame(table,index=row_names)

    from tabulate import tabulate
    result_table = tabulate(df, headers='keys', tablefmt='grid', )
    print(result_table)
    return all_res,df,result_table





import argparse
Parser = argparse.ArgumentParser()
Parser.add_argument('--task', default='bbbp', type=str)
Parser.add_argument('--save_path', default='/home/msrai4srl4s/yuwang5/Evaluation/bio_t5/new_pipeline/res/zekun20240830', type=str)
Parser.add_argument('--test_data_blob_root', default='~/yuwang5/blob/rl4s', type=str)
Parser.add_argument('--test_data_path', default='Data/202407/', type=str)
Parser.add_argument('--yes_token', default='Yes.', type=str)
Parser.add_argument('--no_token', default='No.', type=str)
Parser.add_argument('--column', default='1:5', type=str)
Parser.add_argument('--aggregation_type', default='mean', type=str)
Parser.add_argument('--sft_epoch_for_dpo', default=[], type=list)
Parser.add_argument('--fail_strategy', default='as_0', type=str)    

hparams = Parser.parse_args()

task = hparams.task
with open(f'{hparams.save_path}/task_{task}_res.pkl', 'rb') as f:
    all_folders_res = pickle.load(f)

test_data_path = f'{hparams.test_data_blob_root}/{hparams.test_data_path}/{task}/dpo_{task}_test.csv' # You should change this line to the test data
yes_token = hparams.yes_token
no_token = hparams.no_token
data = pd.read_csv(test_data_path)
true_label = data['label'].tolist()
true_label = [yes_token if x == 1 else no_token for x in true_label]

print(true_label)
all_res = process_all_folders_res(all_folders_res)
    
a = get_evaluation_results(
    all_res=all_res,
    task_name=task,
    aggregation_type=hparams.aggregation_type,
    column=hparams.column,
    sft_epoch_for_dpo=hparams.sft_epoch_for_dpo,
    fail_strategy=hparams.fail_strategy
)


# task = 'bace' # You should change this line to the right task
# with open(f'res/task_{task}_res.pkl', 'rb') as f:
#     all_folders_res = pickle.load(f)

# test_data_path = f'/blob/rl4s/Data/202407/{task}/dpo_{task}_test.csv' # You should change this line to the test data
# yes_token = 'Yes.'
# no_token = 'No.'
# data = pd.read_csv(test_data_path)
# true_label = data['label'].tolist()
# true_label = [yes_token if x == 1 else no_token for x in true_label]


# all_res = process_all_folders_res(all_folders_res)
    
# a = get_evaluation_results(
#     all_res=all_res,
#     task_name=task,
#     aggregation_type='mean',
#     column='10',
#     sft_epoch_for_dpo=[]
# )

