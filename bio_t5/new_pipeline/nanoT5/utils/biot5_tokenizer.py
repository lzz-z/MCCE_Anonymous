# -*- coding: utf-8 -*-
import re
import os
from transformers import T5Tokenizer, T5TokenizerFast

class BioT5Tokenizer(T5Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # tokenizer = AutoTokenizer.from_pretrained(
        #     args.model.name,
        #     use_fast=True
        # )
        self.model_max_length = int(1e9)

        amino_acids = [
            "A", "C", "D", "E", "F",
            "G", "H", "I", "K", "L",
            "M", "N", "P", "Q", "R",
            "S", "T", "V", "W", "Y"
        ]
        prefixed_amino_acids = [f"<p>{aa}" for aa in amino_acids]
        self.add_tokens(prefixed_amino_acids)
        # tokenizer.add_special_tokens({"additional_special_tokens": prefixed_amino_acids}, replace_additional_special_tokens=False)
        selfies_dict_list = [line.strip() for line in open(os.path.join(__file__.split('nanoT5/utils')[0], "dict/selfies_dict_0523.txt"))]
        self.add_tokens(selfies_dict_list)
        # tokenizer.add_special_tokens({"additional_special_tokens": selfies_dict_list}, replace_additional_special_tokens=False)
        special_tokens_dict = {'additional_special_tokens': 
                            ['<bom>', '<eom>',
                            '<bop>', '<eop>',
                            'MOLECULE NAME', 'DESCRIPTION',
                            'PROTEIN NAME', 'FUNCTION', 'SUBCELLULAR LOCATION', 'PROTEIN FAMILIES']}
        self.add_special_tokens(special_tokens_dict, replace_additional_special_tokens=False)
        self.add_tokens(['<bon>', '<eon>'], special_tokens=True) # TODO for new training

        self.split_special_tokens = True  # Ensure _tokenize() can access special tokens
        self.start_end_special_tokens = ['<bom>', '<eom>', '<bop>', '<eop>', '<bon>', '<eon>']
        self.other_special_tokens = ['MOLECULE NAME', 'DESCRIPTION',
                                    'PROTEIN NAME', 'FUNCTION', 'SUBCELLULAR LOCATION', 'PROTEIN FAMILIES']
        self.tag_re = re.compile(f"{'|'.join(self.start_end_special_tokens + self.other_special_tokens)}")
        self.selfies_re =re.compile(r'\[.*?\]')
    
    def _tokenize_entity(self, text: str, tok: str):
        if tok == "selfies":
            tokens = self.selfies_re.findall(text)
        elif tok == 'amino_acids':
            tokens = [f"<p>{aa}" for aa in text.split('<p>') if aa != '']
        else:
            tokens = list(text)

        return tokens


    def _tokenize(self, text, **kwargs):
        result = []
        cur_tag = None
        last_idx = 0

        tag_mapping = {
            'eom': 'bom',
            'eop': 'bop',
            'eon': 'bon',
        }

        for match in self.tag_re.finditer(text):
            start, end = match.span()
            match_str = match.group()

            if match_str.startswith("<e"):
                tag = match_str[1:-1]
                # if tag != cur_tag:
                if tag_mapping[tag] != cur_tag:
                    raise ValueError(f"Tag mismatch: {tag} != {cur_tag} in '{text}'")

                span = text[last_idx:start].strip()
                if tag == "eom":
                    tokens = self._tokenize_entity(span, 'selfies')
                elif tag == "eop":
                    tokens = self._tokenize_entity(span, 'amino_acids')
                elif tag == "eon":
                    tokens = self._tokenize_entity(span, 'num')
                else:
                    raise ValueError(f"Unknown tag: {tag}")

                result.extend([t for t in tokens if t] + [f"<{tag}>"])
                cur_tag = None
            elif match_str in self.other_special_tokens:
                span = text[last_idx:start].strip()
                tokens = super()._tokenize(span, **kwargs)
                result.extend([t for t in tokens if t] + [match_str])
            else:
                tag = match_str[1:-1]
                if cur_tag is not None:
                    raise ValueError(f"Nested tag: {tag} in '{text}'")

                cur_tag = tag
                span = text[last_idx:start].strip()
                tokens = super()._tokenize(span, **kwargs)

                result.extend([t for t in tokens if t] + [f"<{tag}>"])
             
            last_idx = end

        if last_idx < len(text):
            span = text[last_idx:].strip()
            tokens = super()._tokenize(span, **kwargs)
            result.extend(tokens)

        return result

