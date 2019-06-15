"""                           Group 1 
                           Decision Tree
"""

#%%
# Import Libraryes
import pandas as pd
import numpy as np

#%%
# Import data set
Loan_data = pd.read_csv(r'E:\data science\Project\Project working python\XYZCorp_LendingData.txt', sep= '\t', encoding='utf8') # for excel
print(Loan_data)

#%%
# Display data set
pd.set_option('display.max_columns', None)
print(Loan_data.head(7))

#%%
# Check missing values in variables
Missing_col = Loan_data.isnull().sum();
print(Missing_col)

#%%
# Drop variable which have missing value more than 50%
half_count = len(Loan_data)/2
Loan_data_1 = Loan_data.dropna(thresh=half_count,axis=1)
print(Loan_data_1.isnull().sum())

#%%
print(Loan_data.dtypes)
print(Loan_data.info())

#%%
# Drop variable by business logic
Loan_data_2 = Loan_data_1.drop(['member_id','emp_title','policy_code','zip_code','title','id',
                                'next_pymnt_d','sub_grade','total_pymnt_inv','grade',
                                'out_prncp_inv','addr_state','collection_recovery_fee'],axis =1)

#%%
# Data Cleaning
Loan_data_2['emp_length'] = Loan_data_2['emp_length'].str.replace(" ","")
Loan_data_2['emp_length'] = Loan_data_2['emp_length'].str.replace("+","")
Loan_data_2['emp_length'] = Loan_data_2['emp_length'].str.replace("years","")
Loan_data_2['emp_length'] = Loan_data_2['emp_length'].str.replace("year","")
Loan_data_2['emp_length'] = Loan_data_2['emp_length'].str.replace("<1","1")
Loan_data_2['emp_length'] = pd.to_numeric(Loan_data_2.emp_length)

Loan_data_2['term'] = Loan_data_2['term'].str.replace(" ","")
Loan_data_2['term'] = Loan_data_2['term'].str.replace("months","")
Loan_data_2['term'] = Loan_data_2['term'].str.replace("month","")
Loan_data_2['term'] = pd.to_numeric(Loan_data_2.term)
#%%
print(Loan_data_2.info())

#%%
# Missing value treatment by mean and mode
for x in ['emp_length','last_credit_pull_d','last_pymnt_d','revol_util','tot_cur_bal',
          'total_rev_hi_lim','tot_coll_amt','collections_12_mths_ex_med']:
    if Loan_data_2[x].dtype=='object':
        Loan_data_2[x].fillna(Loan_data_2[x].mode()[0],inplace=True)
    elif Loan_data_2[x].dtype=='int64'or 'float64':
        Loan_data_2[x].fillna(Loan_data_2[x].mean(),inplace=True)

#%%
# Check missing values in variables 
Missing_col1 = Loan_data_2.isnull().sum();
print(Missing_col1)

#%%
# Binning for numeric variables
Amount = ['loan_amnt' , 'funded_amnt' , 'funded_amnt_inv','int_rate','installment',
          'total_pymnt','total_rec_prncp','total_rec_int']

for amt in Amount:
    Bins = np.linspace(Loan_data_2[amt].min(),Loan_data_2[amt].max(),10)
    Binlabels = ['lable_1','lable_2','lable_3','lable_4','lable_5','lable_6','lable_7','lable_8',
                 'lable_9']
    Loan_data_2[amt] = pd.cut(Loan_data_2[amt],Bins,labels = Binlabels,include_lowest = True)  # include_lowest = True not include 1st value
    print(Loan_data_2[amt].value_counts())
    
#%%
# Data Partition
loan_training = Loan_data_2[pd.to_datetime(Loan_data_2['issue_d'])<='2015-06-01']
loan_testing = Loan_data_2[pd.to_datetime(Loan_data_2['issue_d'])>'2015-06-01']

#%%
# Split year from the date variable and replace with date
date = ['issue_d' , 'earliest_cr_line' , 'last_pymnt_d' , 'last_credit_pull_d']
for d in date:
    loan_training[d] = pd.to_datetime(loan_training[d])
    loan_training[d] = pd.DatetimeIndex(loan_training[d]).year
    loan_testing[d] = pd.to_datetime(loan_testing[d])
    loan_testing[d] = pd.DatetimeIndex(loan_testing[d]).year

print(loan_training.head(7))

#%%
    
colname = ['home_ownership', 'verification_status','issue_d', 'pymnt_plan','last_credit_pull_d',
'purpose','earliest_cr_line', 'initial_list_status','last_pymnt_d', 'application_type','loan_amnt' ,
'funded_amnt','funded_amnt_inv','int_rate','installment','total_pymnt','total_rec_prncp',
'total_rec_int']

#%%
# Data transform from categorical to numerical 
from sklearn import preprocessing   
 
le = {}
le = preprocessing.LabelEncoder()
for x in colname:
    loan_training[x] = le.fit_transform(loan_training[x]) 
    loan_testing[x] = le.fit_transform(loan_testing[x])
    
print(loan_training.dtypes)
print(loan_testing.dtypes)
#%%
# Split independent and dependent variable
x_train= loan_training.values[:,:-1]
y_train= loan_training.values[:,-1]

x_test= loan_testing.values[:,:-1]
y_test= loan_testing.values[:,-1]

#%%
# Transform y variable in integer
y_train = y_train.astype(int)
y_test =  y_test.astype(int)

#%%
# Scale the variable data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)  
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train)
#print(x_test)


#%%
colname_1=loan_training.columns[:]

#%%
'''                           Model Building
'''
#%%
                         """ Decision tree classifier"""
#%%
# Model buiding
from sklearn.tree import DecisionTreeClassifier

model_DecitionTree = DecisionTreeClassifier(random_state=10)

model_DecitionTree.fit(x_train, y_train)

# Predict y variable

y_pred=model_DecitionTree.predict(x_test)

# Confusion Matrix, Accuracy and Classification report

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)
print()

print("Classification report: ")
print(classification_report(y_test,y_pred))

accuracy_score=accuracy_score(y_test, y_pred)
print("Accuracy of the model: ",accuracy_score)

#%%
                 """ Decision tree classifier with Feature selection method RFE"""
#%%
# Model buiding
from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(random_state=10)

from sklearn.feature_selection import RFE
rfe = RFE(classifier, 35)
model_rfe = rfe.fit(x_train, y_train)

# Check feature and and ranking
print("Num Features: ",model_rfe.n_features_)
print("Selected Features: ") 
print(list(zip(colname_1, model_rfe.support_)))
print("Feature Ranking: ", model_rfe.ranking_)

# Predict y variable
y_pred=model_rfe.predict(x_test)

# Confusion Matrix, Accuracy and Classification report

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)
print()

print("Classification report: ")
print(classification_report(y_test,y_pred))

accuracy_score=accuracy_score(y_test, y_pred)
print("Accuracy of the model: ",accuracy_score)

#%%
#generate the file and upload the code in webgraphviz.com to plot the decision tree

from sklearn import tree
with open(r"E:\data science\Python\Python data set\model_DecisionTree.txt", "w") as f:
    f = tree.export_graphviz(model_DecitionTree, feature_names= colname_1[:-1],out_file=f)

#%%
                       """Bagging"""
#%%
                       """ Extra Trees"""
#%%
#Model building
from sklearn.ensemble import ExtraTreesClassifier

model_ExtraTrees= ExtraTreesClassifier(20,random_state=10)

model_ExtraTrees.fit(x_train,y_train)

# Predict y variable

y_pred=model_ExtraTrees.predict(x_test)

# Confusion Matrix, Accuracy and Classification report

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)
print()

print("Classification report: ")
print(classification_report(y_test,y_pred))

accuracy_score=accuracy_score(y_test, y_pred)
print("Accuracy of the model: ",accuracy_score)

#%%
                       """ Random Forest """
#%%

#Model building
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(35, random_state = 10)

model_RandomForest.fit(x_train,y_train)

# Predict y variable
y_pred=model_RandomForest.predict(x_test)

# Confusion Matrix, Accuracy and Classification report

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)
print()

print("Classification report: ")
print(classification_report(y_test,y_pred))

accuracy_score=accuracy_score(y_test, y_pred)
print("Accuracy of the model: ",accuracy_score)

#%%
                            """Boosting"""
#%%
                        """ Ada Boosting with Decision Tree"""
#%%
# Model Building
from sklearn.ensemble import AdaBoostClassifier                       
from sklearn.tree import DecisionTreeClassifier

model_AdaBoost=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),random_state=10)

model_AdaBoost.fit(x_train,y_train)

# Predict y variable
y_pred=model_AdaBoost.predict(x_test)

# Confusion Matrix, Accuracy and Classification report

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)
print()

print("Classification report: ")
print(classification_report(y_test,y_pred))

accuracy_score=accuracy_score(y_test, y_pred)
print("Accuracy of the model: ",accuracy_score)

#%%
                        """ Ada Boosting with Logistic """
#%%
# Model Building
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

model_AdaBoost=AdaBoostClassifier(base_estimator=LogisticRegression(),random_state=10)

model_AdaBoost.fit(x_train,y_train)

# Predict y variable
y_pred=model_AdaBoost.predict(x_test)

# Confusion Matrix, Accuracy and Classification report

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)
print()

print("Classification report: ")
print(classification_report(y_test,y_pred))

accuracy_score=accuracy_score(y_test, y_pred)
print("Accuracy of the model: ",accuracy_score)

#%%
                        """ Gradient Boosting """
#%%     
#Model Building
from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting=GradientBoostingClassifier(random_state=10)

model_GradientBoosting.fit(x_train,y_train)

# Predict y variable
y_pred=model_GradientBoosting.predict(x_test)

# Confusion Matrix, Accuracy and Classification report

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)
print()

print("Classification report: ")
print(classification_report(y_test,y_pred))

accuracy_score=accuracy_score(y_test, y_pred)
print("Accuracy of the model: ",accuracy_score)

#%%
                       """Ensemble Model"""
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier


# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('log', model1))
model2 = DecisionTreeClassifier(random_state=10)
estimators.append(('DT', model2))
model3=GradientBoostingClassifier(random_state=10)
estimators.append(('Gradient', model3))
#model4 = SVC(kernel = "rbf", C= 70, gamma=0.1)
#estimators.append(('svm', model4))
print(estimators)

 

# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(x_train,y_train)

# Predict y variable
y_pred=ensemble.predict(x_test)

# Confusion Matrix, Accuracy and Classification report

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)
print()

print("Classification report: ")
print(classification_report(y_test,y_pred))

accuracy_score=accuracy_score(y_test, y_pred)
print("Accuracy of the model: ",accuracy_score)

