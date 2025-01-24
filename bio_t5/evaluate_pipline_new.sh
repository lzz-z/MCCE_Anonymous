#!/bin/bash

# Define the task class and the directory to search for files
# dataset_upper="BACE"
# dataset_lower="bace"
# task_class="4_pret_sft_dpo"



# dataset_upper="BACE"
# dataset_lower="bace"
# task_class="4_pret_sft_dpo"

# 解析输入参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset_upper=*)
            dataset_upper="${1#*=}"
            ;;
        --dataset_lower=*)
            dataset_lower="${1#*=}"
            ;;
        --task_class=*)
            task_class="${1#*=}"
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done



search_dir="/home/yuwang5/blob/rl4s/Data/202406/$dataset_upper/results/$task_class"
# result_path="/rl4s/Data/202406/BBBP/results/2_pret_dpo/bbbp_dpowosftfile63data.csv"

# Find all csv files and store them in an array
mapfile -t files < <(find $search_dir -type f -name "*.csv")

# # Define an array of task tags
task_tags=("0_input_mol" "1_chosen_mol" "2_reject_mol" "3_response1" "4_response2" "5_response3" "6_response4" "7_response5" "8_response6" "9_response7" "10_response8" "11_response9" "12_response10" "13_response11" "14_response12" "15_response13" "16_response14" "17_response15" "18_response16" "19_response17" "20_response18" "21_response19" "22_response20")

# First, iterate over each file and for each file iterate over each col_idx and task_tag
for file in "${files[@]}"; do
    for col_idx in ${!task_tags[@]}; do
        python process_pkl_bbbp_bace.py --task $dataset_lower --col_idx $col_idx --task_tag ${task_tags[$col_idx]} --task_class $task_class --inference_output $file && echo "Task ${task_tags[$col_idx]} completed for file $file" &
    done
done
wait

 
mapfile -t files < <(find $search_dir -type f -name "*.csv" -printf "%P\n")
root_dir=/home/yuwang5/biot5_pred  
declare -a gpus=(0 1 2 3)
gpu_idx=0
job_count=0
max_jobs=24
gpu_idx=0
for file in "${files[@]}"; do
    full_path="$search_dir/$file"
    inference_output_file_name=$(basename "$file" .csv)
    for col_idx in ${!task_tags[@]}; do
        split_dir="${root_dir}/splits/${dataset_lower}/$task_class/$inference_output_file_name/${task_tags[$col_idx]}"
        output_dir="${root_dir}/logs/${dataset_lower}/$task_class/$inference_output_file_name/${task_tags[$col_idx]}"
        CUDA_VISIBLE_DEVICES=${gpus[$gpu_idx]} bash eval_molnet.sh --task=$dataset_lower --task_dir=${root_dir} --data_dir=$split_dir --output_dir=$output_dir && echo "GPU ${gpus[$gpu_idx]}: Task ${task_tags[$col_idx]} completed for file $full_path" &
        # gpu_idx=$(( (gpu_idx + 1) % ${#gpus[@]} ))
        echo split_dir: $split_dir
        echo output_dir: $output_dir
        job_count=$((job_count + 1))
        gpu_idx=$(( (gpu_idx + 1) % ${#gpus[@]} ))

        # 输出调试信息
        echo "job_count: $job_count, max_jobs: $max_jobs"

        if [[ "$job_count" -ge "$max_jobs" ]]; then
            wait
            job_count=0
        fi

    done
done
wait