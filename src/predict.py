from src.dataset.steps_dataset import STEPSDataset
from src.pl_modules.steps_module import STEPSModule
from argparse import ArgumentParser
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.utils_step_module import predict_test_arg, predict_test_verb, find_verbs_args
import numpy as np
from src.utils.utils_data_module import collate_fn

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--test_file", type=str, default="data/STEPS_dataset/test.tsv", help="Path to test file")    
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--ckpt", type = str, help='path to model checkpoint', required=True)
    return parser.parse_args()

def predict_verb_arg(generate_batch, source, target, model, tokenizer, mrr_v, rec1v, rec10v, mrr_a, rec1a, rec10a):
    
    for i in range(len(target)):  
        verbs_pred = []
        args_pred = []
        
        gold_elem = tokenizer.decode(target[i], skip_special_tokens=True)
        predictions = generate_batch[i*100:i*100+100]
        
        new_predictions = []
        for j in range(len(predictions)):
            new_predictions.append(tokenizer.decode(predictions[j], skip_special_tokens=True))

        start = 200
        num_ret = 200
        verbs_pred, args_pred = find_verbs_args(gold_elem, new_predictions, verbs_pred, args_pred)

        while len(args_pred)<10:
            new_predictions = []
            try:
                pred_new = model.model.generate(source[i].unsqueeze(0), max_length=45, num_beams=start,num_return_sequences=num_ret, early_stopping=True)
            except:
                break
            for j in range(len(pred_new)):
                new_predictions.append(tokenizer.decode(pred_new[j], skip_special_tokens=True))
            start += 100
            num_ret +=100
            verbs_pred, args_pred = find_verbs_args(gold_elem, new_predictions, verbs_pred, args_pred)

        mrr_v, rec1v, rec10v = predict_test_verb(gold_elem, verbs_pred, rec1v, rec10v, mrr_v) 
        mrr_a, rec1a, rec10a = predict_test_arg(gold_elem, verbs_pred, rec1a, rec10a, mrr_a) 

    return mrr_v, rec1v, rec10v, mrr_a, rec1a, rec10a

def predict(args) -> None:

    print(f"Loading model from {args.ckpt}")
     # load module
    module = STEPSModule.load_from_checkpoint(args.ckpt)
    module.to(args.device)
    module.eval()
    tokenizer = module.tokenizer

    print(f"Test Dataset")
    # dataset
    test_dataset = STEPSDataset(args.test_file, tokenizer)

    # test dataloader
    test_dataloader = DataLoader(test_dataset, 
                          batch_size=args.batch_size, 
                          shuffle=False, 
                          collate_fn=collate_fn)

    print(f"Start Predicting...")

    mrr_v, rec1v, rec10v, mrr_a, rec1a, rec10a = [], [], [], [], [], []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        iterator = tqdm(test_dataloader)
        for batch in iterator:

            source = batch["source"].to(args.device)
            target = batch["target"].to(args.device)

            try:
                generate_batch = module.model.generate(source, max_length=45, num_beams=100,num_return_sequences=100, early_stopping=True)
            except:
              continue                
            
           
            mrr_v, rec1v, rec10v, mrr_a, rec1a, rec10a = predict_verb_arg(
                                                    generate_batch, 
                                                    source,
                                                    target,
                                                    module, 
                                                    tokenizer, 
                                                    mrr_v, 
                                                    rec1v, 
                                                    rec10v, 
                                                    mrr_a, 
                                                    rec1a, 
                                                    rec10a
                                                    )

        
    print(f"\033[1m***** VERBS ***** -> MRR: {str(np.average(mrr_v))}, RECALL@1: {str(np.average(rec1v))}, RECALL@10: {str(np.average(rec10v))} \033[0m \n")
    print(f"\033[1m***** ARGS ****** -> MRR: {str(np.average(mrr_a))}, RECALL@1: {str(np.average(rec1a))}, RECALL@10: {str(np.average(rec10a))} \033[0m")


def main(args):
    predict(args)
    
if __name__ == '__main__':
    main(parse_args())