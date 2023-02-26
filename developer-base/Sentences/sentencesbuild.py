import csv
import pandas as pd
#sentences is a list Player/Someone/Sun/Moon
#a parameter is an availlable iterable in our application
#mode = [] #iterable value/slider/bool/

class Myparameter():
    def __init__(self, param:str,mode:str,value:str,sentences=[]):
        self.param=param
        self.mode=mode
        self.sentences=sentences
        self.value=value
    def __str__(self) -> str:
        return(f"{self.param}, {self.mode}, {self.value}, {' '.join(self.sentences)}")

parameters=[]
def strToMyparameterlist (*args:list)->None:
    parameters.append(wrapper(*args))

def wrapper(*args:list)->list:
    temp=[Myparameter(*i) for i in args ]
    print(temp)
    return temp
def write(filename, separator='%'):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=separator)
        writer.writerow(['param', 'mode', 'value', 'sentences'])
        for param in parameters:
            for p in param:
                writer.writerow([p.param, p.mode, p.value, separator.join(p.sentences)])


strToMyparameterlist(["sun","bool",'0',["Someone wants to remove the sun.","An individual desires to eradicate the sun."]])
strToMyparameterlist(["rain","bool",'0',["Someone wants to remove the rain.","An individual desires to eradicate the rain."]])
strToMyparameterlist(["snow","bool",'0',["Someone wants to remove the snow.","An individual desires to eradicate the snow."]])
strToMyparameterlist([])

write(filename="mysentences.csv")