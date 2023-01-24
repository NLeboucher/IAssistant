from transformers import AutoTokenizer
from transformers.models.mpnet.modeling_mpnet import MPNetForSequenceClassification
import torch
import numpy as np
#model = AutoModelForSequenceClassification.from_pretrained("symanto/mpnet-base-snli-mnli")
def out(ret1="increase brightness"):
    
    model=MPNetForSequenceClassification.from_pretrained("mpnet-base-snli-mnli-finetuned-mnli", num_labels=3,)
    tokenizer = AutoTokenizer.from_pretrained("/home/nico/Documents/IAssistant/my2/mpnet-base-snli-mnli-finetuned-mnli",)
    input_pairs = [(ret1, "Someone wants to increase the brightness")]
    inputs = tokenizer(["</s></s>".join(input_pair) for input_pair in input_pairs], return_tensors="pt",padding='max_length')
    logits = model(**inputs).logits
    probs =  torch.softmax(logits, dim=1).tolist()
    print("probs", probs)

    props={0:'entailment',1:'neutral',2:'contradictition'}
    
    a=[props[np.argmax(i)] for i in probs]
    return a

probs=out("increase brightness")
props={0:'entailment',1:'neutral',2:'contradictition'}

a=[props[np.argmax(i)] for i in probs]
print(a)