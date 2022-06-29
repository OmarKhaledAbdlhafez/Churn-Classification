import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('..\dataset\Churn_Modelling.csv.xls')
df.drop(columns = ['RowNumber', 'CustomerId', 'Surname'], axis = 1, inplace = True)
df['Geography'] = df['Geography'].map({'France' : 0, 'Germany' : 1, 'Spain' : 2})
df['Gender'] = df['Gender'].map({'Male' : 0, 'Female' : 1})
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
cols = ['CreditScore', 'Balance', 'EstimatedSalary', 'Age']
sc = StandardScaler()
X_train[cols] = sc.fit_transform(X_train[cols])
X_test[cols] = sc.transform(X_test[cols])
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
smote_rf = RandomForestClassifier()
smote_rf.fit(X_resampled,y_resampled)
y_pred = smote_rf.predict(X_test)
pkl_sc = open("..\\files\\scaler.pkl","wb")
pickle.dump(sc, pkl_sc)
pkl_sc.close()
pickle_out = open("..\\files\\classifier.pkl","wb")
pickle.dump(smote_rf, pickle_out)
pickle_out.close()
