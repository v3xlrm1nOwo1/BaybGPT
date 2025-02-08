import os
import re
import logging
from typing import Union, List
import json
import torch
from utils import check_dic
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)



def remove_diacritics(text: str):
    arabic_diacritics = re.compile(r'[\u064B-\u065F]')
    return arabic_diacritics.sub('', text)

class CharE:
    def __init__(self, data):
        save_path="./tokens_map/"
        self.chars = sorted(list(set(data)))
        self.chars.append('[UNK]')
        check_dic(save_path)
        self.save_path = save_path

    def form_token_map(self):
        encoding = {ch:i for i, ch in enumerate(self.chars)}
        decoding = {i:ch for i,ch in enumerate(self.chars)}

        with open(f"{self.save_path}/encoding.json", 'w') as file:
            json.dump(encoding, file, indent=4)

        with open(f"{self.save_path}/decoding.json", 'w') as file:
            json.dump(decoding, file, indent=4)
    
class Tokenizer:
    def __init__(self):
        try:
            if os.path.exists("./tokens_map/encoding.json"):
                with open('./tokens_map/encoding.json', 'r') as file:
                    encoding = json.load(file)
                    self.unk_index = encoding['[UNK]']
                self.encoding = {k: int(v) for k, v in encoding.items()}
                self.vocab_size = len(self.encoding)

            if os.path.exists("./tokens_map/decoding.json"):
                with open('./tokens_map/decoding.json', 'r') as file:
                    decoding = json.load(file)
                self.decoding = {int(k): v for k, v in decoding.items()}

        except Exception as e:
            print("Please prepare the token map first by calling CharE.form_token_map()!")
            print(f"Error: {e}")

    def get_vocab_size(self):
        return self.vocab_size
    
    def encode(self, token: Union[str, List[str]]):
        if isinstance(token, str):
            return [self.encoding.get(c, self.unk_index) for c in token]
        elif isinstance(token, list):
            return [self.encoding.get(c, self.unk_index) for token_item in token for c in token_item]
        else:
            print("Please Provide a string or a list of strings")

    def decode(self, token: Union[int, List[int], torch.Tensor]):
        if isinstance(token, int):
            return self.decoding.get(token, '[UNK]')
        elif isinstance(token, list):
            return ''.join(self.decoding.get(c, '[UNK]') for c in token)
        elif torch.is_tensor(token):
            return ''.join(self.decoding.get(c.item(), '[UNK]') for c in token)
    

class CharDataset(Dataset):
    def __init__(self, data, block_size, tokenizer):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        logger.info('data has %d characters, %d unique.'%(data_size, vocab_size))
        self.tokenizer = tokenizer

        self.block_size = block_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size]

        encoding = torch.tensor(self.tokenizer.encode(chunk), dtype=torch.long)

        input_ids = encoding[:-1]
        labels = encoding[1:]

        return input_ids, labels 
    
if __name__ == "__main__":
    pth = "./ديوان شمس الدين تبريزي.txt"
    with open(pth, 'r', encoding='utf-8') as f:
        text = f.read()

    text = remove_diacritics(text)
    n = len(text)
    char_encoding = CharE(text)
    char_encoding.form_token_map()

    tokenizer = Tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print("vocab_size", vocab_size)
    
    block_size = 10
    dataset = CharDataset(text[:int(n*0.9)], block_size, tokenizer)
    print("len", len(dataset))
    print("input_ids", dataset[0][0])
    print("labels", dataset[0][1])
    print(tokenizer.decode(dataset[0][1]))