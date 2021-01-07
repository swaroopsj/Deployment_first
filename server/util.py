import json
import pickle
import numpy as np
import os

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()

__gender = None
__data_columns = None
__model = None


def estimated_status(gender, married, dependents, education, self_employed, applicantincome, coapplicantincome, loanamount, loan_amount_term, credit_history, property_area):

    x = np.zeros(len(__data_columns))
    x[0] = gender
    x[1] = married
    x[2] = dependents
    x[3] = education
    x[4] = self_employed
    x[5] = applicantincome
    x[6] = coapplicantincome
    x[7] = loanamount
    x[8] = loan_amount_term
    x[9] = credit_history
    x[10] = property_area

    if int(__model.predict([x])[0]) == 0:
        return "Rejected"
    else:
        return "Approved"


def get_gender():
    return __gender


def load_saved_artifacts():
    print('loading saved artifacts ... start')
    global __data_columns
    global __gender

    path = os.path.dirname(__file__)
    artifacts = os.path.join(path, "artifacts"),

    with open(artifacts[0]+"/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __gender = __data_columns[0]

    global __model
    if __model is None:
        with open(artifacts[0] + "/Loan_Status.pickle", 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_data_columns():
    return __data_columns


load_saved_artifacts()