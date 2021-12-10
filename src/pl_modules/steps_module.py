from typing import Any
from src.utils.utils_step_module import test_arg, test_verb
import pytorch_lightning as pl
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import hydra
import numpy as np


class STEPSModule(pl.LightningModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters() # populate self.hparams with args and kwargs automagically!

        self.tokenizer = BartTokenizer.from_pretrained(
                                    self.hparams.tokenizer_bart.transformer_model, 
                                    add_prefix_space=self.hparams.tokenizer_bart.add_prefix_space, 
                                    force_bos_token_to_be_generated=self.hparams.tokenizer_bart.force_bos_token_to_be_generated)

        self.model = BartForConditionalGeneration.from_pretrained(self.hparams.model_steps.transformer_model)

        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        self.mrr_v, self.rec1v, self.rec10v = [], [], []
        self.mrr_a, self.rec1a, self.rec10a = [], [], []

        self.mrr_v_predict, self.rec1v_predict, self.rec10v_predict = [], [], []
        self.mrr_a_predict, self.rec1a_predict, self.rec10a_predict = [], [], []

        self.avg_metrics_eval = {"metrics": dict()}

        
        if self.hparams.model_steps.freeze_encoders:
            self.freeze_params(self.model.get_encoder())

        if self.hparams.model_steps.freeze_embeddings:
            self.freeze_embeds()
        

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def training_step(self, batch: dict, batch_idx: int):
        source = batch["source"]
        target = batch["target"]

        decoder_input_ids = self.shift_tokens_right(target, self.tokenizer.pad_token_id)
        forward_output = self.forward(source, decoder_input_ids = decoder_input_ids)[0]
        forward_output = forward_output.reshape((forward_output.shape[0]*forward_output.shape[1],forward_output.shape[-1]))
        gold = target.view(-1)
        loss = self.loss_function(forward_output, gold)

        self.log("train_loss", 
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        
        return loss
        

    def validation_step(self, batch: dict, batch_idx: int):
        #hydra.utils.log.info(f"Valid step")

        source = batch["source"]
        target = batch["target"]

        generate_batch = self.model.generate(source, max_length=175, num_beams=10,num_return_sequences=10,early_stopping=True)
        
        for i in range(len(target)):     
            gold_elem = self.tokenizer.decode(target[i], skip_special_tokens=True)
            predictions = generate_batch[i*10:i*10+10]
            
            new_predictions = []
            for j in range(len(predictions)):
                new_predictions.append(self.tokenizer.decode(predictions[j], skip_special_tokens=True))

            
            self.mrr_v, self.rec1v, self.rec10v = test_verb(gold_elem, new_predictions, self.rec1v, self.rec10v, self.mrr_v)
            self.mrr_a, self.rec1a, self.rec10a = test_arg(gold_elem, new_predictions, self.rec1a, self.rec10a, self.mrr_a)
        
        metrics = {
            "mrr_verb": self.mrr_v,
            "recall1_verb": self.rec1v,
            "recall10_verb": self.rec10v,
            "mrr_arg": self.mrr_a,
            "recall1_arg": self.rec1a,
            "recall10_arg": self.rec10a
            }


        return metrics
    
    def validation_epoch_end(self, metrics_output):
      for dictionary in metrics_output:
        for k,v in dictionary.items():
          if k not in self.avg_metrics_eval["metrics"].keys():
            self.avg_metrics_eval["metrics"] = {k : [np.average(v)]}
          else:
            self.avg_metrics_eval["metrics"][k].append(np.average(v)) 
          
          self.log(k,np.average(v), 
            on_step=False,
            on_epoch=True,
            prog_bar=False)

    def freeze_params(self, model):
        ''' Function that takes a model as input (or part of a model) and freezes 
        the layers for faster training'''
        for layer in model.parameters():
            layer.requires_grade = False
    
    def freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model '''
        self.freeze_params(self.model.model.shared)
        for d in [self.model.model.encoder, self.model.model.decoder]:
            self.freeze_params(d.embed_positions)
            self.freeze_params(d.embed_tokens)
    
    def shift_tokens_right(self, input_ids, pad_token_id):
        """ Shift input ids one token to the right, 
            and wrap the last non pad token (usually <eos>).
        """
        prev_output_tokens = input_ids.clone()
        index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = input_ids[:, :-1]
        return prev_output_tokens

    def configure_optimizers(self):

        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters()
        )
        return optimizer