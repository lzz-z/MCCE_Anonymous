# Define the path as a variable to be passed as an argument

# rm -rf tasks/*
# rm -rf splits/*
# rm -rf nanoT5/logs/*
# rm -rf logs/*

# RESULT_PATH="data/kaiyuan_bbbp_bace/test.instruct.decrease_bbbp.tsv.response.pkl.smi"
# task_tag=kaiyuan_decrease

# # First, integrate all the first commands and execute them together, adding echo for task completion
# python process_pkl_raw_smi.py --task bbbp --col_idx 0 --task_tag $task_tag --inference_output $RESULT_PATH && echo "Task kaiyuan_decrease_bbbp completed" 


# CUDA_VISIBLE_DEVICES=0 bash eval_molnet.sh --task=bbbp --task_tag=$task_tag --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 0: Task kaiyuan_decrease_bbbp completed"  



# RESULT_PATH="data/kaiyuan_bbbp_bace/test.instruct.increase_bbbp.tsv.response.pkl.smi"
# task_tag=kaiyuan_increase

# # First, integrate all the first commands and execute them together, adding echo for task completion
# python process_pkl_raw_smi.py --task bbbp --col_idx 0 --task_tag $task_tag --inference_output $RESULT_PATH && echo "Task kaiyuan_decrease_bbbp completed" 


# CUDA_VISIBLE_DEVICES=0 bash eval_molnet.sh --task=bbbp --task_tag=$task_tag --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 0: Task kaiyuan_decrease_bbbp completed"  



# RESULT_PATH="data/kaiyuan_bbbp_bace/test.instruct.gen_bbbp.tsv.response.pkl.smi"
# task_tag=kaiyuan_gen

# # First, integrate all the first commands and execute them together, adding echo for task completion
# python process_pkl_raw_smi.py --task bbbp --col_idx 0 --task_tag $task_tag --inference_output $RESULT_PATH && echo "Task kaiyuan_decrease_bbbp completed" 


# CUDA_VISIBLE_DEVICES=0 bash eval_molnet.sh --task=bbbp --task_tag=$task_tag --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 0: Task kaiyuan_decrease_bbbp completed"  




RESULT_PATH="data/kaiyuan_bbbp_bace/test.instruct.decrease_bace.tsv.response.pkl.smi"
task_tag=kaiyuan_decrease

# First, integrate all the first commands and execute them together, adding echo for task completion
python process_pkl_raw_smi.py --task bace --col_idx 0 --task_tag $task_tag --inference_output $RESULT_PATH && echo "Task kaiyuan_decrease_bbbp completed" 


CUDA_VISIBLE_DEVICES=0 bash eval_molnet.sh --task=bace --task_tag=$task_tag --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 0: Task kaiyuan_decrease_bbbp completed"  



RESULT_PATH="data/kaiyuan_bbbp_bace/test.instruct.increase_bace.tsv.response.pkl.smi"
task_tag=kaiyuan_increase

# First, integrate all the first commands and execute them together, adding echo for task completion
python process_pkl_raw_smi.py --task bace --col_idx 0 --task_tag $task_tag --inference_output $RESULT_PATH && echo "Task kaiyuan_decrease_bbbp completed" 


CUDA_VISIBLE_DEVICES=0 bash eval_molnet.sh --task=bace --task_tag=$task_tag --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 0: Task kaiyuan_decrease_bbbp completed"  



RESULT_PATH="data/kaiyuan_bbbp_bace/test.instruct.gen_bace.tsv.response.pkl.smi"
task_tag=kaiyuan_gen

# First, integrate all the first commands and execute them together, adding echo for task completion
python process_pkl_raw_smi.py --task bace --col_idx 0 --task_tag $task_tag --inference_output $RESULT_PATH && echo "Task kaiyuan_decrease_bbbp completed" 


CUDA_VISIBLE_DEVICES=0 bash eval_molnet.sh --task=bace --task_tag=$task_tag --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 0: Task kaiyuan_decrease_bbbp completed"  
