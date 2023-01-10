from transformers import AutoTokenizer
from transformers.models.mpnet.modeling_mpnet import MPNetForSequenceClassification
import torch
import numpy as np
#model = AutoModelForSequenceClassification.from_pretrained("symanto/mpnet-base-snli-mnli")
class mpnet:
    def __init__(self,path:str):
        self.path=path
        mpnet.load(self)
    def load(self):

        try:
            self.model=MPNetForSequenceClassification.from_pretrained(f"{self.path}mpnet-base-snli-mnli-finetuned-mnli", num_labels=3,)
            self.tokenizer = AutoTokenizer.from_pretrained(f"{self.path}mpnet-base-snli-mnli-finetuned-mnli",)
        except OSError :
            print("OSError catched")


        

    def out(self,prem:str,hyp:str):
        

        input_pairs = [(prem, hyp)]
        inputs = self.tokenizer(["</s></s>".join(input_pair) for input_pair in input_pairs], return_tensors="pt",padding='max_length')
        logits = self.model(**inputs).logits
        probs =  torch.softmax(logits, dim=1).tolist()
        print("probs", probs)

        props={0:'entailment',1:'neutral',2:'contradictition'}
        
        a=[props[np.argmax(i)] for i in probs]
        return a

