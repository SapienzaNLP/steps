    
import re

def test_verb(gold_elem, predictions, recall1_verb, recall10_verb, mrr_verb):
        found = False
        pattern = r"{(.*?)}"
        gold_verb = re.findall(pattern, gold_elem, flags=0)[0].strip()
        
        for idx,pred in enumerate(predictions):
            pattern = r"{(.*?)}"
            try:
                pred_verb = re.findall(pattern, pred, flags=0)[0].strip()
            except:
                if idx==0:
                    recall1_verb.append(0.)
                continue
            if idx == 0:
                if pred_verb == gold_verb:
                    recall1_verb.append(1.)
                    recall10_verb.append(1.)
                    mrr_verb.append(1.)
                    found=True
                    break
                else:
                    recall1_verb.append(0.)
            else:
                if pred_verb == gold_verb:
                    recall10_verb.append(1.)
                    mrr_verb.append(1./float(idx+1))
                    found=True
                    break

        if found ==False:
            recall10_verb.append(0.)
            mrr_verb.append(0.)

        return mrr_verb,recall1_verb,recall10_verb

def test_arg(gold_elem, predictions, recall1_arg, recall10_arg, mrr_arg):
        found = False
        pattern = r"{{(.*?)}}"
        gold_arg = re.findall(pattern, gold_elem, flags=0)[0].strip()          
    
        for idx,pred in enumerate(predictions):
            pattern = r"{{(.*?)}}"
            try:
                pred_arg = re.findall(pattern, pred, flags=0)[0].strip()
            except:
                if idx==0:
                    recall1_arg.append(0.)
                continue

            if idx == 0:
                if pred_arg == gold_arg:
                    recall1_arg.append(1.)
                    recall10_arg.append(1.)
                    mrr_arg.append(1.)
                    found=True
                    break
                else:
                    recall1_arg.append(0.)
            else:
                if pred_arg == gold_arg:
                    recall10_arg.append(1.)
                    mrr_arg.append(1./float(idx+1))
                    found=True
                    break

        if found ==False:
            recall10_arg.append(0.)
            mrr_arg.append(0.)

        return mrr_arg,recall1_arg,recall10_arg

def find_verbs_args(gold_elem,new_predictions, verbs_pred, args_pred):

        for idx,pred in enumerate(new_predictions):
            pattern = r"{(.*?)}"
            try: 
                pred_verb = re.findall(pattern, pred, flags=0)[0].strip()
                if pred_verb not in verbs_pred:
                    verbs_pred.append(pred_verb)
            except:
                continue
    

        for idx,pred in enumerate(new_predictions):
            pattern = r"{{(.*?)}}"
            try:
                pred_arg = re.findall(pattern, pred, flags=0)[0].strip()
                if pred_arg not in args_pred:
                    args_pred.append(pred_arg)
            except:
                continue

        return verbs_pred, args_pred


def predict_test_verb(gold_elem, verbs_pred, recall1_verb, recall10_verb, mrr_verb):
                        
    found = False
    pattern = r"{(.*?)}"
    gold_verb = re.findall(pattern, gold_elem, flags=0)[0].strip()
    
    for idx,pred_verb in enumerate(verbs_pred):
        if idx == 0:
            if pred_verb == gold_verb:
                recall1_verb.append(1.)
                recall10_verb.append(1.)
                mrr_verb.append(1.)
                found=True
                break
            else:
                recall1_verb.append(0.)
        else:
            if idx < 10:
                if pred_verb == gold_verb:
                    recall10_verb.append(1.)
                    mrr_verb.append(1./float(idx+1))
                    found=True
                    break
            else:
                if pred_verb == gold_verb:
                    mrr_verb.append(1./float(idx+1))
                    recall10_verb.append(0.)
                    found=True
                    break


    if found ==False:
        recall10_verb.append(0.)
        mrr_verb.append(0.)

    return mrr_verb,recall1_verb,recall10_verb


def predict_test_arg(gold_elem, args_pred, recall1_arg, recall10_arg, mrr_arg):
    found = False
    pattern = r"{{(.*?)}}"
    gold_arg = re.findall(pattern, gold_elem, flags=0)[0].strip()
    
    for idx,pred_arg in enumerate(args_pred):
        if idx == 0:
            if pred_arg == gold_arg:
                recall1_arg.append(1.)
                recall10_arg.append(1.)
                mrr_arg.append(1.)
                found=True
                break
            else:
                recall1_arg.append(0.)
        else:
            if idx < 10:
                if pred_arg == gold_arg:
                    recall10_arg.append(1.)
                    mrr_arg.append(1./float(idx+1))
                    found=True
                    break
            else:
                if pred_arg == gold_arg:
                    mrr_arg.append(1./float(idx+1))
                    recall10_arg.append(0.)
                    found=True
                    break

    if found ==False:
        recall10_arg.append(0.)
        mrr_arg.append(0.)

    return mrr_arg,recall1_arg,recall10_arg