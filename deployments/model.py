import pickle
import numpy as np
from data import data

class Model ():
    def __init__(self) -> None:
        pickle_in = open("..\\files\\classifier.pkl","rb")
        self.clf=pickle.load(pickle_in)
        pk_sc = open("..\\files\\scaler.pkl","rb")
        self.sc= pickle.load(pk_sc)
        self.country = {'France' : 0, 'Germany' : 1, 'Spain' : 2}
        self.gender = {'Male' : 0, 'Female' : 1}

    def predict_sample(self , data:data):
        data = data.dict()
        Tenure=data['Tenure']
        NumOfProducts=data['NumOfProducts'] 
        IsActiveMember=data['IsActiveMember']
        HasCrCard=data['HasCrCard']
        Geography = self.country[data['Geography']]
        Gender = self.gender[data['Gender']]
        cols = [data['CreditScore'], data['Balance'], data['EstimatedSalary'], data['Age'] ]
        cols = np.array(cols)
        scaled =self.sc.transform(cols.reshape(1, -1))
        CreditScore = scaled[0][0]
        Balance = scaled[0][1]
        EstimatedSalary = scaled[0][2]
        Age = scaled[0][1]
        sample=[CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]
        pred = self.clf.predict([sample])
        if pred[0]==1:
            p = 'exite'
        else:
            p= 'not exite'  
        return {"prediction": p}  

