# import Flask and jsonify
from flask import Flask, jsonify, request

# import REsource, Api and reaqparser
from flask_restful import Resource, Api, reqparse

import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
api = Api(app)

cat_feats = ['Self_Employed', 'Dependents', 'Gender','Married','Education', 'Property_Area']
num_feats = ['ApplicantIncome', 'CoapplicantIncome','Credit_History', 'LoanAmount', 'Loan_Amount_Term']

# Customized functions to transform features

# Transform numeric values
def numFeatNull(data):
    # Replace null values
    data.fillna({'Credit_History': 0,
            'LoanAmount': data['LoanAmount'].median(),
            'Loan_Amount_Term': data['Loan_Amount_Term'].median(),}, inplace=True)
    
    # Add new column TotalIncome
    data['TotalIncome'] = (data['ApplicantIncome'] + data['CoapplicantIncome']) #.apply(np.log)
    #data.loc[:, 'TotalIncome'] = (data['ApplicantIncome'] + data['CoapplicantIncome'])
    
    # Log
    #data['LoanAmount'] = data['LoanAmount'].apply(np.log)
    
    return data[num_feats]

# Transform categorical values
def catFeatNull(data):
    data.fillna({'Self_Employed': 'Not_specified',
                'Dependents': data['Dependents'].mode()[0],
                'Gender': 'Not_specified',
                'Married': 'No'}, inplace=True)
    return data[cat_feats]

# to transform SparseMatrix to arrays
class ToDenseTransformer():

    # here you define the operation it should perform
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    # just return self
    def fit(self, X, y=None, **fit_params):
        return self

# load model from pickle
model = pickle.load( open( "../models/model.p", "rb" ) )

# endpoint
class Scoring(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        # getting predictions from our model
        # it is much simpler because we used pipelines during development
        res = model.predict(df)
        # we cannot send numpt array as a result
        return res.tolist()

# assign endpoint
api.add_resource(Scoring, '/scoring')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5555)