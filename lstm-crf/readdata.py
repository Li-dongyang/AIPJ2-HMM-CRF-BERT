from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding
from all_utils import *
from typing import Dict, List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence

class NERDataset(Dataset):
    def __init__(self, dir: str, label2id: Dict[str, int], language: str, tokenizer: AutoTokenizer) -> None:
        super().__init__()
        sentence_data = []
        label_data = []
        sentence = []
        labels = []
        with open(dir, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip('\t\n ')
                if line:
                    tokens = line.split(' ')
                    sentence.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(sentence) > 0:
                        sentence_data.append(sentence)
                        sentence = []
                        label_data.append(labels)
                        labels = []
        self.sentence_data = sentence_data
        self.label_data = label_data  
        self.tokenizer = tokenizer 
        self.label2id = label2id
    
    def __len__(self) -> int:
        return len(self.sentence_data)
    
    def __getitem__(self, index: int):
        s = self.sentence_data[index]
        l = self.label_data[index]

        inputs: BatchEncoding = self.tokenizer(s, padding=False, truncation=False, is_split_into_words=True, return_attention_mask=False, \
                                return_token_type_ids=False, return_length=False)
        word_ids = inputs.word_ids()
        word_ids = [-1 if x is None else x for x in word_ids]
        input_ids = inputs['input_ids']
        # assert len(word_ids) == len(input_ids)
        # assert max(word_ids) <= len(l), f'{word_ids}, {input_ids}, {l}, {self.tokenizer.tokenize(s)}, '
        labels = [self.label2id[label] for label in l]
        return torch.as_tensor(input_ids, dtype=torch.long), torch.as_tensor(labels, dtype=torch.long), torch.as_tensor(word_ids, dtype=torch.long)
    
def collate_wapper(pad_idx):
    def collate_fn(batch):
        input_ids, labels, word_ids = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_idx)
        labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        mask = (labels != -1)
        labels[labels == -1] = 0

        return input_ids, labels, word_ids, mask
    return collate_fn
    