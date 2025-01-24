from datasets import load_dataset

dataset = load_dataset('zpn/uniref50', cache_dir='/protein/users/v-qizhipei/huggingface/datasets', num_proc=22)

# print(dataset['train'][0])
print(len(dataset['train']))
# print(len(dataset['validation']))
# print(len(dataset['test']))