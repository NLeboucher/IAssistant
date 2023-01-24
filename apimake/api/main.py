from pydantic import BaseModel
import time
whoexecutes = "docker"
path= 'models/' if whoexecutes=="docker" else '../models/'
a=time.time()

import pretrainedBiLSTM
import pretrainedmpnet
from pretrainedmpnet import MyModel
from fastapi import FastAPI
print(f"done importing after {time.time()-a} seconds")

app = FastAPI()


@app.get("/docs/t/{name}") #get all items on the TODOLIST
async def helloworld(name: str,): #mod:str,
    print(name+"222")
    return f"hello {name}"

@app.put("/docs/{mod}/{hypprem}") #get all items on the TODOLIST
async def modular(hypprem: str,mod:str): #mod:str,
    hyp,prem=list(filter(lambda item: item.__len__()>0, hypprem.split(".")))
    print(f"hyp: {hyp}, prem: {prem}")
    if(mod=="bilstm"):
        print("Testing bilstm")

        a=time.time()

        pretrainedBiLSTM.load(path)
        print(f"done loading bilstm after {time.time()-a} seconds, ready for inference")

        return pretrainedBiLSTM.out(hyp,prem)
    elif(mod=="mpnet"):
        print("Testing mpnet")
        a=time.time()
        mod = MyModel(3,path)
        print(f"done loading mpnet after {time.time()-a} seconds, ready for inference")
        return mod.out(hyp,prem)