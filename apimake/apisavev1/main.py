from pydantic import BaseModel
import time
whoexecutes = "docker"
path= '' if whoexecutes=="docker" else '/home/nico/Documents/IAssistant/my2/'
a=time.time()
import pretrainedBiLSTM
from fastapi import FastAPI
print(f"done importing after {time.time()-a} seconds")
pretrainedBiLSTM.load(path)
print(f"done loading model after {time.time()-a} seconds, ready for inference")

app = FastAPI()


@app.put("docs/{hypprem}") #get all items on the TODOLIST
async def tester(hypprem: str,): #mod:str,
    hyp,prem=list(filter(lambda item: item.__len__()>0, hypprem.split(".")))
    return pretrainedBiLSTM.out(hyp,prem)

@app.get("docs/t/{hypprem}") #get all items on the TODOLIST
async def helloworld(hypprem: str,): #mod:str,
    print(hypprem+"222")
    return hypprem

