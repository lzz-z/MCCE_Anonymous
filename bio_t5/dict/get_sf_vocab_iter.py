from datasets import load_dataset
import selfies as sf
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader

split='validation'
dataset = load_dataset('zpn/zinc20', cache_dir='/protein/users/v-qizhipei/huggingface/datasets', split=split, streaming=True)
dataset = dataset.remove_columns(['smiles', 'id'])

dataloader = DataLoader(
    dataset,
    shuffle=False,
    batch_size=1000,
    num_workers=10,
    pin_memory=True,
    drop_last=False,
)

def process_dataset(dataloader):
    word_set = set()

    for example in tqdm(dataloader):
        text = example['selfies']
        word_set_i = sf.get_alphabet_from_selfies(text)
        word_set = word_set | word_set_i

    return word_set

word_set = process_dataset(dataloader)

# save the alphabet to txt file
with open('selfies_alphabet_validation.txt', 'w') as f:
    for item in word_set:
        f.write("%s\n" % item)