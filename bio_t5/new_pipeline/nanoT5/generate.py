import hydra
from accelerate import Accelerator
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

@hydra.main(config_path="configs", config_name="eval", version_base='1.1')
def main(args):
    accelerator = Accelerator(cpu=args.device == "cpu")
    logger = setup_basics(accelerator, args)
    config = get_config(args)
    tokenizer = get_tokenizer(args)
    model = get_model(args, config, tokenizer)

    input_text = "The <extra_id_0> walks in <extra_id_1> park"
    
    # input_text = "<bom>[C][C][Branch1]<eom>"
    # input_text = "The house is wonderful."
    # input_text = 'MAVMAPRTLLLLLSGALALTQTWAGSHSMR'
    # input_text = ''.join(['<p>'+i for i in input_text])
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids, max_length=512)
    print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    main()