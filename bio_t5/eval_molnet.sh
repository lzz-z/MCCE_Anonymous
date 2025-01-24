# ckpt_path=/blob/v-qizhipei/checkpoints/biot5/bace/240508_gpu1_bsz4_acc1_ts50000_eps100_ws100_cosine_lr3e-4_dp0.1_seed256/checkpoint-ft-12400/pytorch_model.bin

cd nanoT5
 
# root_dir=/rl4s/Projects/biot5_pred

# task=bbbp
# task_tag=kaiyuan_decrease_bbbp

# 解析参数
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --task=*)
      task="${1#*=}"
      ;;
    --task_dir=*)
      task_dir="${1#*=}"
      ;;
    --data_dir=*)
      data_dir="${1#*=}"
      ;;
    --output_dir=*)
      output_dir="${1#*=}"
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
  shift
done

echo "task: $task"
echo "task_dir: $task_dir"
echo "data_dir: $data_dir"
echo "output_dir: $output_dir"


# if 语句决定load哪个ckpt
if [ "$task" == "bbbp" ]; then
    ckpt_path=/blob/blob/v-qizhipei/checkpoints/biot5/bbbp/240508_gpu1_bsz4_acc1_ts50000_eps100_ws100_cosine_lr3e-4_dp0.1_seed256/checkpoint-ft-12400/pytorch_model.bin
elif [ "$task" == "bace" ]; then
    ckpt_path=/blob/blob/v-qizhipei/checkpoints/biot5/bace/240508_gpu1_bsz16_acc2_ts50000_eps100_ws100_cosine_lr3e-4_dp0.1_seed102/checkpoint-ft-3200/pytorch_model.bin
fi
# ckpt_path=/blob/v-qizhipei/checkpoints/biot5/bace/240508_gpu1_bsz16_acc2_ts50000_eps100_ws100_cosine_lr3e-4_dp0.1_seed102/checkpoint-ft-3200/pytorch_model.bin


# output_dir=./logs/${task}_${task_tag}_eval


# #  # 把下面的命令变量都替换好，然后echo出来具体的命令
# echo python main.py \
#     task=eval_molnet \
#     data.task_dir=${root_dir}/tasks \
#     data.data_dir=${root_dir}/splits/${task}_${task_tag} \
#     test_task=save_cls \
#     result_fn=biot5_pred.tsv \
#     model.checkpoint_path=${ckpt_path} \
#     hydra.run.dir=${output_dir} \
#     data.max_seq_len=1024



echo python main.py \
    task=eval_molnet \
    data.task_dir=${task_dir} \
    data.data_dir=${data_dir} \
    test_task=save_cls \
    result_fn=biot5_pred.tsv \
    model.checkpoint_path=${ckpt_path} \
    hydra.run.dir=${output_dir} \
    data.max_seq_len=1024


python main.py \
    task=eval_molnet \
    data.task_dir=${task_dir} \
    data.data_dir=${data_dir} \
    test_task=save_cls \
    result_fn=biot5_pred.tsv \
    model.checkpoint_path=${ckpt_path} \
    hydra.run.dir=${output_dir} \
    data.max_seq_len=1024


# python main.py task=eval_molnet data.task_dir=/home/aiscuser/Evaluation/bio_t5/tasks data.data_dir=/home/aiscuser/Evaluation/bio_t5/splits/bace_quick3 test_task=save_cls result_fn=biot5_pred.tsv model.checkpoint_path=/blob/blob/v-qizhipei/checkpoints/biot5/bace/240508_gpu1_bsz16_acc2_ts50000_eps100_ws100_cosine_lr3e-4_dp0.1_seed102/checkpoint-ft-3200/pytorch_model.bin hydra.run.dir=//home/aiscuser/Evaluation/bio_t5/logs/bace_quick3 data.max_seq_len=1024