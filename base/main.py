from pydantic import BaseModel
import time
whoexecutes = "docker"
path= '' if whoexecutes=="docker" else '/home/nico/Documents/IAssistant/my2/'
a=time.time()

import pretrainedBiLSTM
import pretrainedmpnet
from pretrainedmpnet import mpnet
from fastapi import FastAPI
print(f"done importing after {time.time()-a} seconds")


app = FastAPI()



@app.put("/docs/{hypprem}") #get all items on the TODOLIST
async def tester(hypprem: str,): #mod:str,
    hyp,prem=list(filter(lambda item: item.__len__()>0, hypprem.replace("%"," ").split(".")))
    return pretrainedBiLSTM.out(hyp,prem)

@app.get("/docs/t/{hypprem}") #get all items on the TODOLIST
async def helloworld(hypprem: str,): #mod:str,
    print(hypprem+"222")
    return hypprem

@app.put("/docs/{mod}/{hypprem}") #get all items on the TODOLIST
async def modular(hypprem: str,mod:str): #mod:str,
    hyp,prem=list(filter(lambda item: item.__len__()>0, hypprem.split(".")))
    if(mod=="bilstm"):
        pretrainedBiLSTM.load(path)
        print(f"done loading model after {time.time()-a} seconds, ready for inference")

        return pretrainedBiLSTM.out(hyp,prem)
    elif(mod=="mpnet"):
        mp=mpnet(path)
        print(f"done loading model after {time.time()-a} seconds, ready for inference")

        return mp.out(hyp,prem)


