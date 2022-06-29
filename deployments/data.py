from pydantic import BaseModel
class data(BaseModel):
    CreditScore: float
    Geography: str
    Gender:str
    Age:float
    Tenure:float
    Balance:float
    NumOfProducts: float
    HasCrCard : float
    IsActiveMember : float
    EstimatedSalary : float