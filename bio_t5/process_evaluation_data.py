import numpy as np
import pandas as pd
import os
import re






temp1 = re.compile(r'bace_dpowsftfile_epoch(.*)_(.*)data')
temp2 = re.compile(r'bace_sft_file(.*)data')


def extract_number_re(s):
    # First pattern looks for numbers following 'file' directly.
    match = re.search(r'epoch(\d+)_(\d+)', s)
    if match:
        epoch_num = int(match.group(1))  # The number right after 'epoch'
        post_epoch_num = int(match.group(2))  # The number after the underscore
        return (epoch_num, post_epoch_num)

    # If 'epoch' pattern does not match, try to extract just the file number
    match = re.search(r'file(\d+)', s)
    if match:
        return (float('inf'), int(match.group(1)))  # Sort these at the end if no epochs

    return (float('inf'), float('inf'))  # Case where no recognizable pattern exists



def extract_number(s):
    # 找到第一个非数字字符的位置
    index = 0
    while index < len(s) and s[index].isdigit():
        index += 1
    # 返回转换为整数的数字部分
    return int(s[:index]) if index > 0 else float('inf')
yes_token = 'Yes.'
no_token = 'No.'
true_label = {}


test_data_path = '/home/yuwang5/blob/rl4s/Data/202406/BACE/DPO_bace_test.csv'

data = pd.read_csv(test_data_path)
true_label = data['purpose'].tolist()
true_label = [yes_token if x == 1 else no_token for x in true_label]

# for i in range(92):
#     if i <=29:
#         true_label[i] = yes_token
#     else:
#         true_label[i] = no_token


root_path = './logs'

# def process_evaluation_data(data_path, output_path):
    # Load data

all_task_folders = os.listdir(root_path)
all_folders = {}
res = {}
for task_folder in all_task_folders:
    if task_folder == 'bbbp':
        continue
    all_folders[task_folder] = {}
    res[task_folder] = {}
    for task_class_folder in sorted(os.listdir(os.path.join(root_path, task_folder)), key=extract_number):
        all_folders[task_folder][task_class_folder] =  {}
        res[task_folder][task_class_folder] = {}
        for inference_file_name in sorted(os.listdir(os.path.join(root_path, task_folder, task_class_folder)), key=extract_number_re):
            all_folders[task_folder][task_class_folder][inference_file_name] =  sorted(os.listdir(os.path.join(root_path, task_folder, task_class_folder, inference_file_name)), key=extract_number)
            all_result = []
            for file_name in all_folders[task_folder][task_class_folder][inference_file_name]:
                data = pd.read_csv(os.path.join(root_path, task_folder, task_class_folder, inference_file_name, file_name, 'biot5_pred_yes_no.csv' ))
                # 根据folder的结构，读取每个文件夹下的 biot5_pred_yes_no.csv 文件，然后计算 success rate  ，计算方法如下：根据这个csv文件的最后一列判断ground truth label, 然后判断第一列的预测结果是否与ground truth label一致，一致则记为1，否则记为0，把所有的all_folders[task_folder][task_class_folder][inference_file_name]里面的结果放到一个nparray里面，存到与folder的结构相同的dict里面。
                this_true_label = [true_label[x] for x in data['references']]
                this_pred_label = data['predictions'].tolist()
                this_res = np.zeros(92)
                for nii,ii in enumerate(this_pred_label):
                    this_index =  data['references'][nii]
                    if ii == this_true_label[nii]:
                        this_res[this_index] = 1
                # this_res = np.array([1 if x == y else 0 for x, y in zip(this_true_label, this_pred_label)])
                all_result.append(this_res)
            res[task_folder][task_class_folder][inference_file_name] = np.array(all_result)
                 


# 遍历整个res, 计算每个task_folder下的task_class_folder的inference_file_name的success rate
def get_evaluation_results(type='mean', column='3:', sft_epoch_for_dpo=[1,2,3]):
    all_res = {}
    for task_folder in all_folders:
        for task_class_folder in all_folders[task_folder]:
            for inference_file_name in all_folders[task_folder][task_class_folder]:
                this_res = eval(f'res[task_folder][task_class_folder][inference_file_name][{column}]')
                if len(this_res.shape) == 1:
                    success_rate = this_res.mean()
                else:
                    success_rate = eval(f'this_res.{type}(axis=1).mean()')
                # print(f'{task_folder} {task_class_folder} {inference_file_name} success rate:  \t{success_rate}')
                all_res[f'{task_folder} {task_class_folder} {inference_file_name} success rate:'] =  success_rate  
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

    for  key, values in all_res.items():
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


a = get_evaluation_results(
    type='mean',
    column='10',
    sft_epoch_for_dpo=[1,2,3]
)

#calculate similarity between input_mol
