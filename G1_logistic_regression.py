"""                           Group 1 
                           Logistic Regression
"""

#%%
# Import Libraryes
import pandas as pd
import numpy as np

#%%
# Import data set
Loan_data = pd.read_csv(r'E:\data science\Project\Project working python\XYZCorp_LendingData.txt', 
                        sep= '\t', encoding='utf8') # for excel
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
# Import library for outlier 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#%%
# Check outliar
plt.boxplot(Loan_data_2.loan_amnt) # no
plt.boxplot(Loan_data_2.funded_amnt) # no
plt.boxplot(Loan_data_2.funded_amnt_inv) #no
plt.boxplot(Loan_data_2.int_rate) #Yes, not required
plt.boxplot(Loan_data_2.installment) #yes not required
plt.boxplot(Loan_data_2.emp_length) #no
plt.boxplot(Loan_data_2.annual_inc) #Yes not required
plt.boxplot(Loan_data_2.term) # no
plt.boxplot(Loan_data_2.inq_last_6mths) #yes not required
plt.boxplot(Loan_data_2.open_acc) #yes not required
plt.boxplot(Loan_data_2.pub_rec) #yes not required
plt.boxplot(Loan_data_2.revol_bal) #yes not required
plt.boxplot(Loan_data_2.total_acc) # yes not required
plt.boxplot(Loan_data_2.out_prncp) #yes not required 
plt.boxplot(Loan_data_2.total_pymnt) #yes not required
plt.boxplot(Loan_data_2.total_rec_prncp) #yes not required
plt.boxplot(Loan_data_2.total_rec_int) #yes not required
plt.boxplot(Loan_data_2.total_rec_late_fee) #yes not required
plt.boxplot(Loan_data_2.recoveries) # yes not required
plt.boxplot(Loan_data_2.last_pymnt_amnt) #yes not required
plt.boxplot(Loan_data_2.collections_12_mths_ex_med) #yes not required
plt.boxplot(Loan_data_2.tot_cur_bal) #yes not required

#%%

plt.boxplot(Loan_data_2.dti) #yes need to check and treat delete 1 row
plt.boxplot(Loan_data_2.delinq_2yrs) #yes need to check and treat delete 1 row
plt.boxplot(Loan_data_2.revol_util) #yes need to check and treat delete 1 row
plt.boxplot(Loan_data_2.acc_now_delinq) #yes check and treat delete 1 row
plt.boxplot(Loan_data_2.tot_coll_amt) #yes check and treat delete 1 row
plt.boxplot(Loan_data_2.total_rev_hi_lim) #yes check and treat delete 3 row

#%%
# Outlier treatment
for x in ['dti','delinq_2yrs','revol_util','acc_now_delinq','tot_coll_amt','total_rev_hi_lim']:
    loan_d = Loan_data_2[ Loan_data_2[x] == max(Loan_data_2[x])]
    Loan_data_2 = Loan_data_2.drop(loan_d.index, axis=0)

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

#%% # Firstly skip below code and run from line number 191

colname_1=loan_training.columns[:]

#%%
   """ Run feature selection method one by one with logistic regression"""
#%%
                """1. Univariate Feature Selection (Chi test)"""
#%%
# First run logistic regression without feature selection 

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


test = SelectKBest(score_func=chi2, k=35)
fit1 = test.fit(x_train, y_train)

print(fit1.scores_)
print(list(zip(colname_1,fit1.get_support())))
x_train = fit1.transform(x_train)
x_test = fit1.transform(x_test)
#print(Loan_data_2)

#%%
        """2. Univariate Feature Selection (Variance Threshold)"""
#%%
from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold()
fit1 = vt.fit(x_train, y_train)
print(fit1.variances_)

x_train = fit1.transform(x_train)
x_test = fit1.transform(x_test)

print(x_train)
print(x_train.shape[1])
print(list(zip(colname_1,fit1.get_support())))
  
    
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
                           """ Logistic Regression Classifier"""
#%%
# Model building
from sklearn.linear_model import LogisticRegression

classifier  = LogisticRegression()

classifier.fit(x_train,y_train)

# Predict y variable
y_pred = classifier.predict(x_test)

print(classifier.coef_)
print(classifier.intercept_)

# Confusion Matrix, Accuracy and Classification report

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(y_test,y_pred)
print(cfm)

print("classification report: ")
print(classification_report(y_test,y_pred))

acc=accuracy_score(y_test, y_pred)
print("Accuracy of the model: ", acc)

#%%
       """ Logistic Regression Classifier with Feature selection method RFE"""
#%%
# Model buiding
from sklearn.linear_model import LogisticRegression

classifier  = LogisticRegression()

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
                    """Tuning (Change Threshold)"""
#%%
# Probability of y predicted variable
y_pred_prob = classifier.predict_proba(x_test)
print(y_pred_prob)

#%%
# Check errors for different threshold value between 0 to 1
for a in np.arange(0,1,0.01):           # range 0 to 1 and split with 0.01
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)  # where function works as if else
    cfm=confusion_matrix(y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]             # cfm[0,1] shows the 0th row 1st column value and cfm[1,0] shows the 1st row and 0th column value
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
          cfm[1,0]," , type 1 error:", cfm[0,1])
    
#%%
# for good confution matrix we are changing the the threshold (0.5) with 0.60
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value > 0.60:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)


#%%
# checking the confusion matrix, accuracy with new threshold

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(y_test,y_pred_class)
print(cfm)
acc=accuracy_score(y_test, y_pred_class)
print("Accuracy of the model: ",acc)
print(classification_report(y_test, y_pred_class))
