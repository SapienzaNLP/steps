from typing import List
from torch.utils.data import Dataset
import torch

DOUBLE_STAR = "**"
SINGLE_STAR = "*"
LEFT_BRACKETS = "{"
RIGHT_BRACKETS = "}"
POINT = "."



class STEPSDataset(Dataset):
    def __init__(self, 
                 file_path, 
                 tokenizer):

        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data_store = []
        self.init_dataset()

    def __len__(self):
        return len(self.data_store)

    def __getitem__(self, idx):
        return self.data_store[idx]

    def replace_verb(self, title, i):
        title[i] = title[i].replace(DOUBLE_STAR, "", 2)
        title[i] = LEFT_BRACKETS.center(3) + title[i]+ RIGHT_BRACKETS.center(3)
        return title

    def replace_arg(self, title, i, num_star):
        if num_star==2:
            title[i] = title[i].replace(SINGLE_STAR, "", 2)
            title[i] = (LEFT_BRACKETS*2).center(4) + title[i]+ (RIGHT_BRACKETS*2).center(4)
        elif num_star == 1:
            index_ast = title[i].index(SINGLE_STAR)
            if index_ast==0:
                title[i] = title[i].replace(SINGLE_STAR, "",1)
                title[i] = (LEFT_BRACKETS*2).center(4) +title[i]
            else:
                title[i] = title[i].replace(SINGLE_STAR, "",1)
                title[i] = title[i]+(RIGHT_BRACKETS*2).center(4)

        return title

    def replace_title(self, title: List[str]):
        for i in range(len(title)):
            if DOUBLE_STAR in title[i]:
                title = self.replace_verb(title,i)
            
            if SINGLE_STAR in title[i]:
                num_star = title[i].count(SINGLE_STAR)
                title = self.replace_arg(title, i , num_star)

        return title

    def fix_source(self, source):
        encoded_source = []
        for i, elem in enumerate(source):
            if i == len(source)-1:
                encoded_source += [str(i+1)+"."] + [elem] + [POINT.center(3)]
            else:
                encoded_source += [str(i+1)+"."] + [elem] + [POINT.center(3)]

        
        if len(encoded_source)>1:
            encoded_source = [" ".join(e for e in encoded_source)]
        
        return encoded_source

    def read_from_wikihow(self, line: str)-> List[str]:
        
        line = line.strip().split("\t")
        source = line[6:]
        title = line[1].split()
    
        title_replaced = self.replace_title(title)
        target = [" ".join(e for e in title_replaced)]
        encoded_source = self.fix_source(source)

        return encoded_source, target

    def init_dataset(self) -> None:

    

        f = open(self.file_path, 'r')

        for line in f:
            encoded_source, target = self.read_from_wikihow(line)
            encoded_s, encoded_t = self.encoded_sentences(encoded_source, target)

            data = {
                "source": encoded_s,
                "target":encoded_t,
            }

            self.data_store.append(data)


    def encoded_sentences(self, source, target):

        encoded_source = self.tokenizer.encode(self.tokenizer.tokenize(source[0]), truncation = True, max_length=1024)
        encoded_s = torch.tensor(encoded_source)

        encoded_target = self.tokenizer.encode(self.tokenizer.tokenize(target[0]), truncation = True, max_length=1024)
        encoded_t = torch.tensor(encoded_target)

        return encoded_s,encoded_t
