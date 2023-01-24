from transformers import AutoTokenizer, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.mpnet.modeling_mpnet import MPNetForSequenceClassification
import torch
import numpy as np
#model = AutoModelForSequenceClassification.from_pretrained("symanto/mpnet-base-snli-mnli")

class mpnet:
    def __init__(self,path:str):
        self.path=path
        self.model=MPNetForSequenceClassification.from_pretrained(f"{self.path}mpnet-base-snli-mnli", num_labels=3)#,use_auth_token='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'
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.path}mpnet-base-snli-mnli",)
        # try:
        #     self.model=MPNetForSequenceClassification.from_pretrained(f"{self.path}mpnet-base-snli-mnli", num_labels=3,)
        #     self.tokenizer = AutoTokenizer.from_pretrained(f"{self.path}mpnet-base-snli-mnli",)
        # except OSError :
#            print("OSError catched")
    def out(self,prem:str,hyp:str):
        input_pairs = [(prem, hyp)]
        inputs = self.tokenizer(["</s></s>".join(input_pair) for input_pair in input_pairs], return_tensors="pt",padding='max_length')
        logits = self.model(**inputs).logits
        probs =  torch.softmax(logits, dim=1).tolist()
        return probs
class MyModel(torch.nn.Module):
    def __init__(self, NUM_LABELS,path:str):
        super().__init__()
        self.path=path
        print(f"{self.path}mpnet-base-snli-mnli/")
        self.config= AutoConfig.from_pretrained(f"{self.path}mpnet-base-snli-mnli/", num_labels=NUM_LABELS,local_files_only=True)#local_files_only=True
        self.base_model = MPNetForSequenceClassification.from_pretrained(f"{self.path}mpnet-base-snli-mnli/",config=self.config, local_files_only=True)
        self.lin_layer = torch.nn.Linear(self.config.hidden_size, NUM_LABELS)
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.path}mpnet-base-snli-mnli/",local_files_only=True)
    def out(self,prem:str,hyp:str):
        input_pairs = [(prem, hyp)]
        inputs = self.tokenizer(["</s></s>".join(input_pair) for input_pair in input_pairs], return_tensors="pt",padding='max_length')
        logits = self.base_model(**inputs).logits
        probs =  torch.softmax(logits, dim=1).tolist()
        return probs