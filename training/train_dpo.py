import os
import json
import torch
import swanlab
import argparse
from datasets import Dataset, load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
import time

def load_json_dataset(json_file_path: str, split_ratio: float = 0.9):
    print(f"Loading JSON dataset: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    print(f"Total data size: {len(data)}")
    
    required_fields = ['prompt', 'chosen', 'rejected']
    valid_data = []
    
    for i, item in enumerate(data):
        if all(field in item for field in required_fields):
            valid_data.append(item)
        else:
            print(f"Warning: item {i+1} missing required fields {required_fields}")
    
    print(f"Valid data: {len(valid_data)}")
    
    if not valid_data:
        raise ValueError("No valid data found! Please ensure data contains 'prompt', 'chosen', 'rejected' fields")
    
    dataset = Dataset.from_list(valid_data)
    
    print("Dataset columns:", dataset.column_names)
    print("Dataset features:", dataset.features)
    print("\nFirst example:")
    first_example = dataset[0]
    for key, value in first_example.items():
        print(f"{key}: {value}")
    
    if len(dataset) > 1:
        split_point = int(len(dataset) * 1)
        train_dataset = dataset.select(range(split_point))
        eval_dataset = train_dataset
        print(f"\nTrain size: {len(train_dataset)}")
        print(f"Eval size: {len(eval_dataset)}")
        return train_dataset, eval_dataset
    else:
        print(f"\nTrain size: {len(dataset)}")
        print("Data size too small, not splitting validation set")
        return dataset, None

def main():
    parser = argparse.ArgumentParser(description="DPO training for molecular design")
    parser.add_argument("--train_data_path", required=True, help="Path to the training JSON data")
    parser.add_argument("--output_dir", required=True, help="Output directory for the trained model")
    parser.add_argument("--exp_name", required=True, help="Experiment name")
    parser.add_argument("--model_name_or_path", default="<TO_BE_FILLED>", 
                       help="Base model path or previous trained model path")
    parser.add_argument("--ref_model_path", default="<TO_BE_FILLED>",
                       help="Reference model path (should always be the original base model)")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    
    args = parser.parse_args()
    
    swanlab.init(
        project="DPO-MOLLM-Training",
        experiment_name=args.exp_name,
        description="DPO training for molecular design task",
        config={
            "model": args.model_name_or_path,
            "dataset": args.train_data_path,
            "task": "molecule_design_preference_optimization",
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "beta": args.beta
        }
    )
    
    train_log_dir = "<TO_BE_FILLED>"
    os.makedirs(train_log_dir, exist_ok=True)
    step_log_file = os.path.join(train_log_dir, f"{args.exp_name}.jsonl")

    class StepJSONLogger(TrainerCallback):
        def __init__(self, log_path: str, exp_name: str):
            self.log_path = log_path
            self.exp_name = exp_name
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            record = {
                "exp_name": self.exp_name,
                "step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
                "timestamp": time.time(),
                "logs": logs,
            }
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Failed to write step log: {e}")
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.cuda.empty_cache()

    model_name = args.model_name_or_path
    json_data_path = args.train_data_path
    output_dir = args.output_dir

    print(f"Loading policy model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: "14GiB", 1: "14GiB", 2: "14GiB", 3: "14GiB", 4: "14GiB", 5: "14GiB", 6: "14GiB", 7: "14GiB"},
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    ref_model_name = args.ref_model_path
    print(f"Loading reference model: {ref_model_name}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: "14GiB", 1: "14GiB", 2: "14GiB", 3: "14GiB", 4: "14GiB", 5: "14GiB", 6: "14GiB", 7: "14GiB"},
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        if os.path.exists(json_data_path):
            train_dataset, eval_dataset = load_json_dataset(json_data_path)
        else:
            print(f"JSON file does not exist: {json_data_path}")
            print("Using default arrow dataset")
            train_dataset = load_dataset(
                'arrow',
                data_files='<TO_BE_FILLED>',
                split='train'
            )
            eval_dataset = None
            
    except Exception as e:
        print(f"Failed to load JSON dataset: {e}")
        print("Using default arrow dataset")
        train_dataset = load_dataset(
            'arrow',
            data_files='<TO_BE_FILLED>',
            split='train'
        )
        eval_dataset = None

    training_args = DPOConfig(
        output_dir=output_dir,
        
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        num_train_epochs=args.num_train_epochs,
        
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        
        save_steps=50,
        logging_steps=1,
        
        warmup_steps=0,
        lr_scheduler_type="constant",
        learning_rate=args.learning_rate,
        
        beta=args.beta,
        loss_type="sigmoid",
        label_smoothing=0.1,
        
        max_length=2048,
        max_prompt_length=1536,
        
        report_to=["swanlab"],
        dataloader_num_workers=0,
        
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        
        save_strategy="steps",
        logging_strategy="steps",
        
        max_grad_norm=1.0,
        
        weight_decay=0.01,
        adam_epsilon=1e-6,
        
        dataloader_drop_last=True,
    )

    step_json_logger = StepJSONLogger(step_log_file, args.exp_name)
    trainer = DPOTrainer(
        model=model, 
        ref_model=ref_model,
        args=training_args, 
        processing_class=tokenizer, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[step_json_logger]
    )

    torch.cuda.empty_cache()

    print("Starting training...")
    trainer.train()

    print("Training completed!")
    
    trainer.save_model()
    print(f"Model saved to: {output_dir}")
    
    final_metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    if final_metrics:
        swanlab.log({
            "final_train_loss": final_metrics.get("train_loss", 0),
            "final_rewards_accuracy": final_metrics.get("rewards/accuracies", 0),
            "final_rewards_margin": final_metrics.get("rewards/margins", 0),
        })
    
    swanlab.finish()
    
    print("Training complete, logs recorded to SwanLab!")

if __name__ == "__main__":
    main()
