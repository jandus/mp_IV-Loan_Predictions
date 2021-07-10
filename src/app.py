from flask import Flask, render_template, request

app = Flask(__name__)


import requests
import json

def dict_num(dictionary):
    num_cols = ["ApplicantIncome", "CoapplicantIncome",  "LoanAmount",  "Loan_Amount_Term",  "Credit_History"]
    for col in num_cols:
        dictionary[col] = int(dictionary[col])
    return dictionary

@app.route('/',methods = ['POST', 'GET'])
def main():
   if request.method == 'POST':
      result = request.form
      res_dict = result.to_dict()

      new_dict = dict_num(res_dict)
      json_data = new_dict
      print(json_data)
      URL = "http://ec2-18-117-82-160.us-east-2.compute.amazonaws.com:5555/scoring"
      # sending get request and saving the response as response object 
      r = requests.post(url = URL, json = json_data)
      print(r.json())
      if r.json()[0]:
          output1 = "Congratulations"
          output2 = "Your loan has been pre-approved"
      else:
          output1 = "Sorry"
          output2 = "You are not elegible for a loan"
      return render_template("results.html", output1=output1, output2=output2)
   else:
      return render_template("index.html")

if __name__ == '__main__':
   app.run(debug = True)