from typing import Any, Union, List, Optional
import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.utils.utils_data_module import collate_fn
from transformers import BartTokenizer


class STEPSDataModule(pl.LightningDataModule):

    def __init__(self, conf: DictConfig, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.conf = conf

        self.train_dataset  = None
        self.val_datasets = None
        self.test_datasets = None

        self.steps_tokenizer = BartTokenizer.from_pretrained(
            self.conf.tokenizer_bart.transformer_model, 
            add_prefix_space=self.conf.tokenizer_bart.add_prefix_space, 
            force_bos_token_to_be_generated=self.conf.tokenizer_bart.force_bos_token_to_be_generated
            )


    def setup(self, stage: Optional[str] = None):

        hydra.utils.log.info(f"Setup data")
        # Here you should instantiate your datasets

        if stage is None or stage == "fit":
            
         
            self.train_dataset = hydra.utils.instantiate(
                self.conf.train_dataset, 
                tokenizer=self.steps_tokenizer
            )
            
            self.val_dataset = hydra.utils.instantiate(
                self.conf.validation_dataset,
                tokenizer=self.steps_tokenizer,
            )
        
        elif stage == "test":

            self.test_dataset = hydra.utils.instantiate(
                self.conf.test_dataset,
                tokenizer=self.steps_tokenizer
            )

    def encoded_sentences(self, source, target):

        hydra.utils.log.info(f"Encoded sentenqces")

        encoded_source = self.tokenizer.encode(self.tokenizer.tokenize(source[0]), truncation = True, max_length=1024)
        encoded_s = torch.tensor(encoded_source)

        encoded_target = self.tokenizer.encode(self.tokenizer.tokenize(target[0]), truncation = True, max_length=1024)
        encoded_t = torch.tensor(encoded_target)

        return encoded_s,encoded_t


    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        hydra.utils.log.info(f"Train dataloader")

        return DataLoader(self.train_dataset, 
                          batch_size=self.conf.batch_size, 
                          shuffle=True, 
                          collate_fn=collate_fn)
      

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        hydra.utils.log.info(f"Valid dataloader")
        return DataLoader(self.val_dataset, 
                          batch_size=self.conf.batch_size, 
                          shuffle=False, 
                          collate_fn=collate_fn)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        hydra.utils.log.info(f"Test dataloader")
        return DataLoader(self.test_dataset, 
                          batch_size=self.conf.batch_size, 
                          shuffle=False, 
                          collate_fn=collate_fn)