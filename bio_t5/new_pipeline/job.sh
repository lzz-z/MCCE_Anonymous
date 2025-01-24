python simple_main_multiple_class.py --task bbbp --ckpt_blob_root ~/yuwang5/blob/blob --inference_blob_root ~/yuwang5/blob/rl4s --inference_folder Data/202407/total_training_data_1/inference --save_path res/llama2


python analyze_evaluation.py --save_path res/llama2   --task bbbp --fail_strategy mask