import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()

loan_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv",index_col =0 )
loan_data.head()

loan_data.drop(['Loan_ID'], axis=1, inplace=True)


X = loan_data.drop(['Loan_Status'],axis=1)
y = loan_data['Loan_Status']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =.3)

X_train['LoanAmount'].fillna((X_train['LoanAmount'].mean()), inplace=True)
X_train['Loan_Amount_Term'].fillna((X_train['Loan_Amount_Term'].mode()[0]), inplace=True)
X_train['Self_Employed'].fillna((X_train['Self_Employed'].mode()[0]), inplace=True)
X_train['Dependents'].fillna((X_train['Dependents'].mode()[0]), inplace=True)
X_train['Married'].fillna((X_train['Married'].mode()[0]), inplace=True)

X_train['Credit_History'].fillna(0, inplace= True)
X_train['Gender'].fillna('Notavailable', inplace= True)

X_train_col= X_train.select_dtypes(include = "object").columns

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train['ApplicantIncome']=X_train['ApplicantIncome'].astype(int) 
X_train['Gender'] = le.fit_transform(X_train['Gender'].astype(str))
X_train['Married'] = le.fit_transform(X_train['Married'].astype(str))
X_train['Dependents'] = le.fit_transform(X_train['Dependents'].astype(str))
X_train['Education'] = le.fit_transform(X_train['Education'].astype(str))
X_train['Self_Employed'] = le.fit_transform(X_train['Self_Employed'].astype(str))
X_train['Property_Area'] = le.fit_transform(X_train['Property_Area'].astype(str))

X_test['LoanAmount'].fillna((X_test['LoanAmount'].mean()), inplace=True)
X_test['Loan_Amount_Term'].fillna((X_test['Loan_Amount_Term'].mode()[0]), inplace=True)
X_test['Self_Employed'].fillna((X_test['Self_Employed'].mode()[0]), inplace=True)
X_test['Dependents'].fillna((X_test['Dependents'].mode()[0]), inplace=True)
X_test['Married'].fillna((X_test['Married'].mode()[0]), inplace=True)

X_test['Credit_History'].fillna(0, inplace= True)
X_test['Gender'].fillna('Notavailable', inplace= True)
X_test['ApplicantIncome']=X_test['ApplicantIncome'].astype(int) 

X_test['Gender'] = le.fit_transform(X_test['Gender'].astype(str))
X_test['Married'] = le.fit_transform(X_test['Married'].astype(str))
X_test['Dependents'] = le.fit_transform(X_test['Dependents'].astype(str))
X_test['Education'] = le.fit_transform(X_test['Education'].astype(str))
X_test['Self_Employed'] = le.fit_transform(X_test['Self_Employed'].astype(str))
X_test['Property_Area'] = le.fit_transform(X_test['Property_Area'].astype(str))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

lr=RandomForestClassifier(n_estimators= 400,min_samples_leaf=4, min_samples_split=5,max_depth=90, n_jobs=-1)
lr=lr.fit(X_train,y_train)
y_pred_lr =lr.predict(X_test)
accuray = classification_report(y_test, y_pred_lr)
print(accuray)


import pickle
with open('Loan_Status.pickle','wb') as f:
    pickle.dump(lr,f)

with open('Loan_Status.pickle','rb') as f:
    pickle_Loan_model = pickle.load(f)

score =pickle_Loan_model.predict(X_test).astype(int)
print(score)

import flask
import json

def default(o):
    if hasattr(o, 'to_json'):
        return o.to_json()
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

class A(object):
    def __init__(self):
        self.data = 'stuff'
        self.other_data = 'other stuff'

    def to_json(self):
        return {'data': self.data}


columns={
    'data_columns':[col.lower() for col in X_train.columns]
}
with open ("columns.json",'w') as f:
    f.write(json.dumps(columns,default=default))