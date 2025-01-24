from datasets import load_dataset
import selfies as sf


dataset = load_dataset('zpn/zinc20', cache_dir='/protein/users/v-qizhipei/huggingface/datasets', num_proc=20, split='train')

# get the alphabet from the dataset
# selfies_alphabet_train = sf.get_alphabet_from_selfies(dataset['train']['selfies'])
# selfies_alphabet_valid = sf.get_alphabet_from_selfies(dataset['validation']['selfies'])
# selfies_alphabet_test = sf.get_alphabet_from_selfies(dataset['test']['selfies'])

# selfies_alphabet = selfies_alphabet_train | selfies_alphabet_valid | selfies_alphabet_test
selfies_alphabet = sf.get_alphabet_from_selfies(dataset['selfies'])
# save the alphabet to txt file
with open('selfies_alphabet_train.txt', 'w') as f:
    for item in selfies_alphabet:
        f.write("%s\n" % item)