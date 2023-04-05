# from flask import Flask
# import uvicorn
# from fastapi import FastAPI
# from Loanly import Loans
from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
import sklearn
pickle_in = open("rf_clf.pkl","rb")
pickle_inss = open("ss.pkl","rb")

rf_clf =pickle.load(pickle_in)
ss =pickle.load(pickle_inss)

app = Flask(__name__)
server = app.server
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods= ["POST"]) 
def predict():
    # generate_schemas=True
    Gender = request.form.get('gender')
    Married = request.form.get('married')
    Dependents = request.form.get('dependents')
    Education = request.form.get("education")
    Self_Employed = request.form.get("self-employed")
    ApplicantIncome = float(request.form.get("applicantincome"))  #
    CoapplicantIncome = float(request.form.get('coapplicantincome')) #
    LoanAmount = float(request.form.get('loanamount')) #
    Loan_Amount_Term = float(request.form.get('loan-amount-term')) #
    Credit_History = float(request.form.get("credithistory")) #
    Property_Area = request.form.get("property") 
   
    print(Gender)

    # data = data.dict()
    # Gender = data['Gender']
    # Married = data['Married']
    # Dependents = data['Dependents']
    # Education = data['Education']
    # Self_Employed= data['Self_Employed']
    # ApplicantIncome=data['ApplicantIncome']
    # CoapplicantIncome = data['CoapplicantIncome']
    # LoanAmount= data['LoanAmount']
    # Loan_Amount_Term = data['Loan_Amount_Term']
    # Credit_History = data['Credit_History']
    # Property_Area = data['Property_Area']

    Gender_Female, Gender_Male= 0,0
    if(Gender == 'Male'):
        Gender_Male = 1
    else:
        Gender_Female = 1
        
    Married_Yes, Married_No = 0,0
    if(Married == 'Yes'):
        Married_Yes = 1
    else:
        Married_No =1
    
    Dependents_0 , Dependents_1, Dependents_2, Dependents_3plus = 0,0,0,0
    
    if(int(Dependents) == 0):
        Dependents_0 = 1
    elif(int(Dependents) == 1):
        Dependents_1 = 1
    elif(int(Dependents) == 2):
        Dependents_2 =1
    else:
        Dependents_3plus = 1
        
    Education_Graduate, Education_Not_Graduate = 0,0
    if(Education == 'Graduate'):
        Education_Graduate = 1
    else:
        Education_Not_Graduate = 1
        
    Self_Employed_Yes , Self_Employed_No = 0,0
    if(Self_Employed == 'Yes'):
        Self_Employed_Yes = 1
    else:
        Self_Employed_No = 1
    
    Property_Area_Rural, Property_Area_Semiurban , Property_Area_Urban = 0,0,0
    if(Property_Area == 'Rural'):
        Property_Area_Rural = 1
    elif(Property_Area == 'Semiurban'):
        Property_Area_Semiurban = 1
    else:
        Property_Area_Urban =1
    TotalIncome = ApplicantIncome + CoapplicantIncome
    TotalIncome_log = np.log(TotalIncome)
    loanAmount_log = np.log(LoanAmount)
  
    dict_new_input = {
        'ApplicantIncome' : ApplicantIncome,  'CoapplicantIncome' :  CoapplicantIncome, 'LoanAmount' : LoanAmount, 
        'Loan_Amount_Term': Loan_Amount_Term, 'Credit_History': Credit_History,  'loanAmount_log':  loanAmount_log, 
        'TotalIncome': TotalIncome,  'TotalIncome_log':  TotalIncome_log, 'Gender_Female': Gender_Female, 'Gender_Male' : Gender_Male,
        'Married_No': Married_No, 'Married_Yes': Married_Yes,  'Education_Graduate':  Education_Graduate, 'Education_Not Graduate': Education_Not_Graduate, 
        'Self_Employed_No': Self_Employed_No, 'Self_Employed_Yes': Self_Employed_Yes, 'Property_Area_Rural': Property_Area_Rural,
        'Property_Area_Semiurban': Property_Area_Semiurban, 'Property_Area_Urban': Property_Area_Urban, 'Dependents_0': Dependents_0,
        'Dependents_1': Dependents_1, 'Dependents_2': Dependents_2, 'Dependents_3+': Dependents_3plus 
    } 
    
    def lol(blah):
        blahblah = pd.DataFrame(blah, index=[0])
        var_num = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','loanAmount_log','TotalIncome','TotalIncome_log']
        blahblah[var_num] = ss.transform(blahblah[var_num])
        prediction = rf_clf.predict(blahblah)
        probability = rf_clf.predict_proba(blahblah)

        if(prediction[0] == 0):
            return 'No', probability
        else:
            return 'Yes', probability
         
    pred, probability = lol(dict_new_input)
    negpro, pospro = probability[0][0], probability[0][1]
  
    print(pred, negpro, pospro)
    # return {
    #     'prediction': pred,
    #     'probabilty_no': probability[0][0],
    #     'probability_yes' : probability[0][1]
    # }

    return render_template('index.html', pred=pred, negpro= negpro ,pospro=pospro)



if __name__ == '__main__':
    app.run(debug = True)

