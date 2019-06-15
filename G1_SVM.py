"""                           Group 1 
                           Support Vector Machine(SVM)
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
# Data Partition
loan_training = Loan_data_2[pd.to_datetime(Loan_data_2['issue_d'])<='2015-06-01']
loan_testing = Loan_data_2[pd.to_datetime(Loan_data_2['issue_d'])>'2015-06-01']


#%%    
colname = ['home_ownership', 'verification_status','issue_d', 'pymnt_plan','last_credit_pull_d',
'purpose','earliest_cr_line', 'initial_list_status','last_pymnt_d', 'application_type']    

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
# Model building
from sklearn import svm
svc_model=svm.SVC(kernel='rbf',C=1.0, gamma=0.1) 

svc_model.fit(x_train, y_train)

# Predict y variable
y_pred=svc_model.predict(x_test)

# Confusion Matrix, Accuracy and Classification report

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)
print()

print("Classification report: ")
print(classification_report(y_test,y_pred))

accuracy_score=accuracy_score(y_test, y_pred)
print("Accuracy of the model: ",accuracy_score)
