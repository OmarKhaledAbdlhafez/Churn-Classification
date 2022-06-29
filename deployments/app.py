from statistics import mode
import pandas as pd
import numpy as np
from typing import Union
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from data import data 
from model import Model

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict_data(data :data):
    model = Model()
    out = model.predict_sample(data)
    return out