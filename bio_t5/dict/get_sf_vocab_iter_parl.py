from datasets import load_dataset
import selfies as sf
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
import multiprocessing as mp

split = 'validation'
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

def process_batch(text_batch, queue):
    word_set = sf.get_alphabet_from_selfies(text_batch)
    queue.put(word_set)

def process_dataset(dataloader):
    word_set = set()
    processes = []
    queue = mp.Queue()

    for example in tqdm(dataloader):
        text_batch = example['selfies']
        process = mp.Process(target=process_batch, args=(text_batch, queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    while not queue.empty():
        word_set |= queue.get()

    return word_set

word_set = process_dataset(dataloader)

# save the alphabet to txt file
with open('selfies_alphabet_validation.txt', 'w') as f:
    for item in word_set:
        f.write("%s\n" % item)
