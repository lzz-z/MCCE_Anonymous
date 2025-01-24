from accelerate import Accelerator
from omegaconf import open_dict
import hydra
import torch
import time
# task=eval_molnet data.task_dir=/rl4s/Projects/biot5_pred/tasks data.data_dir=/rl4s/Projects/biot5_pred/splits/increase_bbbp test_task=save_cls result_fn=biot5_pred.tsv model.checkpoint_path=/blob/v-qizhipei/checkpoints/biot5/bbbp/240508_gpu1_bsz4_acc1_ts50000_eps100_ws100_cosine_lr3e-4_dp0.1_seed256/checkpoint-ft-12400/pytorch_model.bin hydra.run.dir=./logs/increase_bbbp data.max_seq_len=1024
from utils import (
    setup_basics,
    train,
    predict,
    eval,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_dataloaders,
    get_config,
)
from datasets import disable_caching
disable_caching()

@hydra.main(config_path="configs", config_name="default", version_base='1.1')
def main(args):
    # torch.save(args, '/home/aiscuser/Evaluation/bio_t5/args.save')
    # assert 1==2
    accelerator = Accelerator(cpu=args.device == "cpu")
    logger = setup_basics(accelerator, args)
    config = get_config(args)
    tokenizer = get_tokenizer(args)
    # torch.save(tokenizer, 'tokenizer.pt')
    # assert 1==2
    model = get_model(args, config, tokenizer, logger)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, logger)
    print(args)

    if args.mode == 'pt':
        train_dataloader, validation_dataloader = get_dataloaders(tokenizer, config, args)
    elif args.mode == 'ft':
        train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args)

    logger.log_args(args)

    (
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
    ) = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )

    if args.model.compile:
        model = torch.compile(model)

    with open_dict(args):
        args.current_train_step = 1
        args.current_epoch = 1
        args.last_log = time.time()

    if args.eval_only:
        model.eval()
        with torch.no_grad():
            eval(model, test_dataloader, logger, args, tokenizer, accelerator)
    elif args.predict_only:
        model.eval()
        with torch.no_grad():
            predict(model, test_dataloader, logger,
                    args, tokenizer, accelerator)
    else:
        if args.mode == 'pt':
            train(model, train_dataloader, validation_dataloader, None, accelerator,
                lr_scheduler, optimizer, logger, args, tokenizer)
        elif args.mode == 'ft':
            train(model, train_dataloader, validation_dataloader, test_dataloader, accelerator,
                lr_scheduler, optimizer, logger, args, tokenizer)
        else:
            raise NotImplementedError
        
    logger.finish()


if __name__ == "__main__":
    main()
