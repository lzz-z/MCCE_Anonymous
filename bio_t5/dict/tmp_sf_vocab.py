from transformers import AutoTokenizer
import selfies as sf

tokenizer = AutoTokenizer.from_pretrained(
    'google/t5-v1_1-base',
    use_fast=True
)

sf_alphabet = list(sf.get_semantic_robust_alphabet())

for i in sf_alphabet:
    tokenizer.add_tokens(i)

tokenizer.add_tokens("<p>A")
tokenizer.add_tokens("<p>B")

tmp_seq = ''.join(sf_alphabet)

print(tokenizer.tokenize('<p>A'+tmp_seq+'<p>B'))
print(tokenizer.tokenize("HuggingFace is a company happiness"))
# print(tokenizer(tmp_seq, return_tensors='pt'))

dataset = ["[C][O][C]", "[F][C][F]", "[O][=O]", "[C][C][O][C][C]"]
print(type(sf.get_alphabet_from_selfies(dataset)))
print(list(sf.get_alphabet_from_selfies(dataset)))

loaded_alphabet_valid = [line.strip() for line in open('selfies_alphabet.txt', 'r')]
loaded_alphabet_test = [line.strip() for line in open('selfies_alphabet_test.txt', 'r')]

print(len(loaded_alphabet_valid))
print(len(loaded_alphabet_test))
print(len(set(loaded_alphabet_valid) & set(loaded_alphabet_test)))
