#============================================================================
#                           Importing Required Libraries
#============================================================================

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import random
from sklearn.metrics import confusion_matrix

# change the directory
os.chdir("E:\\Training\\Mindmap_Genpact\\Training_Material\\Inferential_Statistics\\Python_Hands_On\\")

#============================================================================
#              Importing Data (with text as strings and not factors)
#============================================================================

adult_try_logistic = pd.read_csv("adult.csv")
type(adult_try_logistic)
adult_try_logistic.dtypes

#============================================================================
#              Replacing "?" with NA and Encoding Dependent Variable
#                  1 = Preferred Event | 0 = Alternate Event
#============================================================================

stack = adult_try_logistic.stack()
stack[stack == "?"] = None
adult_try_logistic = stack.unstack()

adult_try_logistic.income[adult_try_logistic.income == ">50K"] = 1
adult_try_logistic.income[adult_try_logistic.income == "<=50K"] = 0

#============================================================================
#                       Converting text to lower case
#============================================================================

adult_try_logistic.workclass = adult_try_logistic.workclass.str.lower()
adult_try_logistic.occupation = adult_try_logistic.occupation.str.lower()
adult_try_logistic.education = adult_try_logistic.education.str.lower()
adult_try_logistic.race = adult_try_logistic.race.str.lower()
adult_try_logistic.gender = adult_try_logistic.gender.str.lower()
adult_try_logistic['native-country'] = adult_try_logistic['native-country'].str.lower()
adult_try_logistic['marital-status'] = adult_try_logistic['marital-status'].str.lower()
adult_try_logistic.relationship = adult_try_logistic.relationship.str.lower()

#=============================================================================
#                      Checking for % of Missing Values
# Decide whether to impute or not. If yes, with or without category wise median /           mean / mode
#                        If not, encode Missing Values
#=============================================================================

pd.isna(adult_try_logistic).mean() # gives col wise %age
type(adult_try_logistic)
adult_try_logistic.dtypes

#=============================================================================
#                          Check for Imbalanced Data
#=============================================================================
adult_try_logistic.income.value_counts()

#=============================================================================
#                          Missing Value Replacement
#=============================================================================

adult_try_logistic.workclass[pd.isna(adult_try_logistic.workclass)]= "missing_workclass"
adult_try_logistic.occupation[pd.isna(adult_try_logistic.occupation)]= "missing_occupation"
adult_try_logistic['native-country'][pd.isna(adult_try_logistic['native-country'])]= "missing_country"


#=============================================================================
#                          Converting back to Factor Variables
#=============================================================================
adult_try_logistic.workclass = adult_try_logistic.workclass.astype('category')
adult_try_logistic.occupation = adult_try_logistic.occupation.astype('category')
adult_try_logistic['native-country'] = adult_try_logistic['native-country'].astype('category')
adult_try_logistic.race = adult_try_logistic.race.astype('category')
adult_try_logistic.gender = adult_try_logistic.gender.astype('category')
adult_try_logistic.relationship = adult_try_logistic.relationship.astype('category')
adult_try_logistic.income = adult_try_logistic.income.astype('category')
adult_try_logistic.education = adult_try_logistic.education.astype('category')
adult_try_logistic['marital-status'] = adult_try_logistic['marital-status'].astype('category')
adult_try_logistic.dtypes

#=============================================================================
#                         Check Information Value (IV)
#=============================================================================
def iv_woe(data, target, bins=10, show_woe=False):

    #Empty Dataframe
    newDF = pd.DataFrame()

    #Extract Column Names
    cols = data.columns

    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = d['Events'] / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = d['Non-Events'] / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars],
                            "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF

IV_Table = iv_woe(data = adult_try_logistic, target = 'income', bins=10, show_woe = True)

# Taken from https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html


#============================================================================
#                             Thumb Rule for IV
#============================================================================
#Information Value	Predictive Power
#< 0.02	useless for prediction
#0.02 to 0.1	Weak predictor
#0.1 to 0.3	Medium predictor
#0.3 to 0.5	Strong predictor
#>0.5	Suspicious or too good to be true

#=============================================================================
#                      Choosing Relevant Independent Variables
#=============================================================================

adult_try_logistic = adult_try_logistic.drop(axis=1,columns=['fnlwgt','race','capital-loss','native-country'])


#============================================================================
#                        Reducing Levels in Factor Variables (<10)
#============================================================================

# DIY

#============================================================================
#                           Create Training Data (Bootstrapping)
#============================================================================
adult_try_logistic = adult_try_logistic.rename(columns={"hours-per-week":"hours_per_week"})
adult_try_logistic.age = adult_try_logistic.age.astype('int')
adult_try_logistic['educational-num'] = adult_try_logistic['educational-num'].astype('int')
adult_try_logistic['capital-gain'] = adult_try_logistic['capital-gain'].astype('int')
adult_try_logistic.hours_per_week = adult_try_logistic.hours_per_week.astype('int')


input_ones = adult_try_logistic[adult_try_logistic.income == 1]  # all 1's code (encoding) of whichever level is lower in frequency
input_zeroes = adult_try_logistic[adult_try_logistic.income == 0]  # all 0's
random.seed(100)
training_ones = input_ones.sample(frac=0.7)
training_zeroes = input_zeroes.sample(frac=0.7)
trainingData = pd.concat([training_ones,training_zeroes])

#===========================================================================
#                               Create Test Data
#===========================================================================
test_ones = input_ones.loc[~input_ones.index.isin(training_ones.index)]
test_zeroes = input_zeroes.loc[~input_zeroes.index.isin(training_zeroes.index)]
testData = pd.concat([test_ones,test_zeroes])  # row bind the 1's and 0's

#===========================================================================
#                      Modeling (Tuning based on p-values)
#                  Number of Fisher Iterations should be less
#===========================================================================
logitMod = smf.glm('income ~ age + C(workclass) + C(education) + C(occupation) + hours_per_week', data=trainingData, family=sm.families.Binomial()).fit()
logitMod.summary()


#===========================================================================
#                            Testing (Prediction)
#===========================================================================

predicted = logitMod.predict(testData[['age','workclass','education','occupation','hours_per_week']])

#===========================================================================
#                          Deciding Optimal Cut-off
#===========================================================================

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds =roc_curve(testData.income, predicted)
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})

# Threshold
optCutOff = roc.ix[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = plt.subplots()
plt.plot(roc['tpr'])
plt.plot(roc['1-fpr'], color = 'red')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
ax.set_xticklabels([])
plt.show()


#============================================================================
#                        Checking Accuracy of the Model
#============================================================================

fitted_results = predicted
for k in fitted_results.index:
    print(k)
    if fitted_results[k] >= optCutOff['thresholds'].values:
        fitted_results[k]=1
    else:
        fitted_results[k]=0

misClasificError = np.mean(fitted_results != testData.income)
print('Accuracy',1-misClasificError)

#============================================================================
#                               Confusion Matrix
#============================================================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

results = confusion_matrix(testData.income, fitted_results)
print('Confusion Matrix :')
print(results)
print('Accuracy Score :',accuracy_score(testData.income, fitted_results))
print('Report : ')
print(classification_report(testData.income, fitted_results))

tn, fp, fn, tp = results.ravel()

#============================================================================
#                                Sensitivity
#============================================================================
print(tp/(tp+fn))

#============================================================================
#                                Specificity
#============================================================================
print(tn/(tn+fp))