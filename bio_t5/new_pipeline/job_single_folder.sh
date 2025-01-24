    # Parser.add_argument('--task', default='bace', help='task name')
    # Parser.add_argument('--ckpt_blob_root', default='/blob/blob/', help='blob root for bioT5 model checkpoint')
    # Parser.add_argument('--inference_blob_root', default='/blob/rl4s/', help='blob root for inference results')
    # Parser.add_argument('--inference_folder', default='Data/202407/total_training_data_1/inference_for_llama3_9', help='inference folder path should include the folder with task name')
    # Parser.add_argument('--device', default='cuda:0', help='device')
    # Parser.add_argument('--args_save_path', default='args.save', help='args save path')
    # Parser.add_argument('--skip_col_name', default=[], help='skip')
    # Parser.add_argument('--save_path', default='res', help='save_dir')
    # Parser.add_argument('--skip_class_name', default=[], help='skip class name')

inference_folder=zekun/instruct/inst_result/wang/instruct_task_20240807/final/bbbp/response
save_path=../../res/zekun20241009


python simple_main_single_folder.py --task bbbp --ckpt_blob_root /home/msrai4srl4s/yuwang5/blob/blob/ --inference_blob_root /home/msrai4srl4s/yuwang5/blob/nlm/ --inference_folder $inference_folder  --device cuda:0 --args_save_path args.save --skip_col_name [] --save_path $save_path --skip_class_name [] 

 