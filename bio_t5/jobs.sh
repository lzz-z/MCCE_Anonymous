
rm -rf tasks/*
rm -rf splits/*
# rm -rf nanoT5/logs/*
# rm -rf logs/*

dataset=bace

# bash evaluate_pipline_total.sh  --dataset=$dataset --task_class=4_pret_sft_dpo && echo "4_pret_sft_dpo completed" && \
# bash evaluate_pipline_total.sh  --dataset=$dataset --task_class=3_pret_sft && echo "3_pret_sft completed" && \
# bash evaluate_pipline_total.sh  --dataset=$dataset --task_class=2_pret_dpo && echo "2_pret_dpo completed" && \
bash evaluate_pipline_total.sh  --dataset=$dataset --task_class=1_pret && echo "1_pret completed" && \

wait
# bash evaluate_pipline_total.sh --dataset_upper=$dataset_upper --dataset_lower=$dataset_lower --task_class=3_pret_sft && echo "3_pret_sft completed" && \
# bash evaluate_pipline_total.sh --dataset_upper=$dataset_upper --dataset_lower=$dataset_lower --task_class=2_pret_dpo && echo "2_pret_dpo completed" && \
# bash evaluate_pipline_total.sh --dataset_upper=$dataset_upper --dataset_lower=$dataset_lower --task_class=1_pret && echo "1_pret completed" && \
# wait

# task
# dataset_lower=bace
# task_class=4_pret_sft_dpo
# root_dir=/home/yuwang5/biot5_pred  
# inference_output_file_name="bace_dpowosftfile63data"
# split_dir="${root_dir}/splits/${dataset_lower}/${task}/$task_class/$inference_output_file_name/input_mol"
# output_dir="${root_dir}/logs/${dataset_lower}/${task}/$task_class/$inference_output_file_name/${task_tags[$col_idx]}"
# bash eval_molnet.sh --task=bace --task_dir=${root_dir} --data_dir=$split_dir --output_dir=$output_dir  
