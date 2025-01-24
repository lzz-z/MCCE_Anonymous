# Define the path as a variable to be passed as an argument
RESULT_PATH="/rl4s/Data/202406/BBBP/results/2_pret_dpo/bbbp_dpowosftfile63data.csv"

# First, integrate all the first commands and execute them together, adding echo for task completion
python process_pkl_bbbp.py --task bbbp --col_idx 0 --task_tag 0_input_mol --inference_output $RESULT_PATH && echo "Task 0_input_mol completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 1 --task_tag 1_chosen_mol --inference_output $RESULT_PATH && echo "Task 1_chosen_mol completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 2 --task_tag 2_reject_mol --inference_output $RESULT_PATH && echo "Task 2_reject_mol completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 3 --task_tag 3_response1 --inference_output $RESULT_PATH && echo "Task 3_response1 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 4 --task_tag 4_response2 --inference_output $RESULT_PATH && echo "Task 4_response2 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 5 --task_tag 5_response3 --inference_output $RESULT_PATH && echo "Task 5_response3 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 6 --task_tag 6_response4 --inference_output $RESULT_PATH && echo "Task 6_response4 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 7 --task_tag 7_response5 --inference_output $RESULT_PATH && echo "Task 7_response5 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 8 --task_tag 8_response6 --inference_output $RESULT_PATH && echo "Task 8_response6 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 9 --task_tag 9_response7 --inference_output $RESULT_PATH && echo "Task 9_response7 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 10 --task_tag 10_response8 --inference_output $RESULT_PATH && echo "Task 10_response8 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 11 --task_tag 11_response9 --inference_output $RESULT_PATH && echo "Task 11_response9 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 12 --task_tag 12_response10 --inference_output $RESULT_PATH && echo "Task 12_response10 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 13 --task_tag 13_response11 --inference_output $RESULT_PATH && echo "Task 13_response11 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 14 --task_tag 14_response12 --inference_output $RESULT_PATH && echo "Task 14_response12 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 15 --task_tag 15_response13 --inference_output $RESULT_PATH && echo "Task 15_response13 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 16 --task_tag 16_response14 --inference_output $RESULT_PATH && echo "Task 16_response14 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 17 --task_tag 17_response15 --inference_output $RESULT_PATH && echo "Task 17_response15 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 18 --task_tag 18_response16 --inference_output $RESULT_PATH && echo "Task 18_response16 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 19 --task_tag 19_response17 --inference_output $RESULT_PATH && echo "Task 19_response17 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 20 --task_tag 20_response18 --inference_output $RESULT_PATH && echo "Task 20_response18 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 21 --task_tag 21_response19 --inference_output $RESULT_PATH && echo "Task 21_response19 completed" &
python process_pkl_bbbp.py --task bbbp --col_idx 22 --task_tag 22_response20 --inference_output $RESULT_PATH && echo "Task 22_response20 completed" &
wait

# After all the first commands have finished, execute the second commands, distributing them across four GPUs, adding echo for task completion
CUDA_VISIBLE_DEVICES=0 bash eval_molnet.sh --task=bbbp --task_tag=0_input_mol --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 0: Task 0_input_mol completed" &
CUDA_VISIBLE_DEVICES=0 bash eval_molnet.sh --task=bbbp --task_tag=1_chosen_mol --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 0: Task 1_chosen_mol completed" &
CUDA_VISIBLE_DEVICES=0 bash eval_molnet.sh --task=bbbp --task_tag=2_reject_mol --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 0: Task 2_reject_mol completed" &
CUDA_VISIBLE_DEVICES=0 bash eval_molnet.sh --task=bbbp --task_tag=3_response1 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 0: Task 3_response1 completed" &
CUDA_VISIBLE_DEVICES=1 bash eval_molnet.sh --task=bbbp --task_tag=4_response2 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 1: Task 4_response2 completed" &
CUDA_VISIBLE_DEVICES=1 bash eval_molnet.sh --task=bbbp --task_tag=5_response3 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 1: Task 5_response3 completed" &
CUDA_VISIBLE_DEVICES=1 bash eval_molnet.sh --task=bbbp --task_tag=6_response4 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 1: Task 6_response4 completed" &
CUDA_VISIBLE_DEVICES=1 bash eval_molnet.sh --task=bbbp --task_tag=7_response5 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 1: Task 7_response5 completed" &
CUDA_VISIBLE_DEVICES=2 bash eval_molnet.sh --task=bbbp --task_tag=8_response6 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 2: Task 8_response6 completed" &
CUDA_VISIBLE_DEVICES=2 bash eval_molnet.sh --task=bbbp --task_tag=9_response7 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 2: Task 9_response7 completed" &
CUDA_VISIBLE_DEVICES=2 bash eval_molnet.sh --task=bbbp --task_tag=10_response8 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 2: Task 10_response8 completed" &
CUDA_VISIBLE_DEVICES=2 bash eval_molnet.sh --task=bbbp --task_tag=11_response9 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 2: Task 11_response9 completed" &
CUDA_VISIBLE_DEVICES=3 bash eval_molnet.sh --task=bbbp --task_tag=12_response10 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 3: Task 12_response10 completed" &
CUDA_VISIBLE_DEVICES=3 bash eval_molnet.sh --task=bbbp --task_tag=13_response11 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 3: Task 13_response11 completed" &
CUDA_VISIBLE_DEVICES=3 bash eval_molnet.sh --task=bbbp --task_tag=14_response12 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 3: Task 14_response12 completed" &
CUDA_VISIBLE_DEVICES=3 bash eval_molnet.sh --task=bbbp --task_tag=15_response13 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 3: Task 15_response13 completed" &
CUDA_VISIBLE_DEVICES=0 bash eval_molnet.sh --task=bbbp --task_tag=16_response14 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 0: Task 16_response14 completed" &
CUDA_VISIBLE_DEVICES=0 bash eval_molnet.sh --task=bbbp --task_tag=17_response15 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 0: Task 17_response15 completed" &
CUDA_VISIBLE_DEVICES=1 bash eval_molnet.sh --task=bbbp --task_tag=18_response16 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 1: Task 18_response16 completed" &
CUDA_VISIBLE_DEVICES=1 bash eval_molnet.sh --task=bbbp --task_tag=19_response17 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 1: Task 19_response17 completed" &
CUDA_VISIBLE_DEVICES=2 bash eval_molnet.sh --task=bbbp --task_tag=20_response18 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 2: Task 20_response18 completed" &
CUDA_VISIBLE_DEVICES=2 bash eval_molnet.sh --task=bbbp --task_tag=21_response19 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 2: Task 21_response19 completed" &
CUDA_VISIBLE_DEVICES=3 bash eval_molnet.sh --task=bbbp --task_tag=22_response20 --root_dir=/rl4s/Projects/biot5_pred && echo "GPU 3: Task 22_response20 completed" &
wait