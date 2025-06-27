import os
import json
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data","mix","train_mix_250626_Jun06.jsonl")
VAL_DATA_PATH = os.path.join(BASE_DIR, "data", "mix", "validation_mix_250626_Jun06.jsonl")

class MixDataset(Dataset):
    def __init__(self, datapath, tokenizer, max_length=512):
        with open(datapath, "r") as f:
            self.samples = [ json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        item = self.samples[index]
        encoding = self.tokenizer(
            item['input'],

        )

        
        
    print()

def main():
    # data
    # inherit Dataset, dump into Dataloader


    # model：prepare a frozen roberta
    # load data + tokenizer into roberta 

    # model：prepare a nerual network

    # train


    print()
if __name__ == '__main__':
    main()