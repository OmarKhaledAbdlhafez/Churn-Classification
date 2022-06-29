import pandas as pd
import numpy as np
from typing import Union
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from data import data 

app = FastAPI()

pickle_in = open("..\\files\\classifier.pkl","rb")
clf=pickle.load(pickle_in)

pk_sc = open("..\\files\\scaler.pkl","rb")
sc= pickle.load(pk_sc)

country = {'France' : 0, 'Germany' : 1, 'Spain' : 2}
gender = {'Male' : 0, 'Female' : 1}

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict_data(data :data):
    data = data.dict()
    Tenure=data['Tenure']
    NumOfProducts=data['NumOfProducts'] 
    IsActiveMember=data['IsActiveMember']
    HasCrCard=data['HasCrCard']
    Geography = country[data['Geography']]
    Gender = gender[data['Gender']]
    cols = [data['CreditScore'], data['Balance'], data['EstimatedSalary'], data['Age'] ]
    cols = np.array(cols)
    scaled =sc.transform(cols.reshape(1, -1))
    CreditScore = scaled[0][0]
    Balance = scaled[0][1]
    EstimatedSalary = scaled[0][2]
    Age = scaled[0][1]
    sample=[CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]
    pred = clf.predict([sample])
    if pred[0]==1:
        p = 'exite'
    else:
        p= 'not exite'    
    return {"prediction": p}