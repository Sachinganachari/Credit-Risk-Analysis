# Python Final Project

# 1. Importing important libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import classification_report as cr
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import metrics


# 2.Exploratory data analysis

# 2.1. Overview of dataset and target variable

#Loading dataset 
filename='M:/Sachin My files/Imarticus/Grp projects/XYZ corp/XYZCorp_LendingData.txt'
corp=pd.read_csv(filename,sep='\t',low_memory=False)

#Dataset
corp.shape #Data has 855969 rows and 73 columns
corp.info()
corp.head()
#Summary 
corp.describe()

#Target variable (Y Variable)
corp.default_ind.describe()
sns.countplot('default_ind',data=corp)

#By plot it is clearly visible its an imbalanced dataset 

#Numeric features
numeric_features=corp.select_dtypes(include=[np.number])
numeric_features.dtypes

#Categorical features
categoricals=corp.select_dtypes(exclude=[np.number])
categoricals.dtypes

# 2.2 Detailed data analysis-Null values and unquie values
corp.isnull().sum()
sns.heatmap(corp.isnull(), cbar=False)
    
#Unique values 
corp.nunique() 
#Policy_code has only one level

# 2.3 Detailed data analysis - Analysing numeric discrete and continuous variables

#Issue_d
#For data analysis lets create a function that will split the issue_d variable into month and year 

def getMonth(x):
    return x.split('-')[0]


def getYear(x):
    return x.split('-')[1]


corp['issue_month'] = corp.issue_d.apply(getMonth)
corp['issue_year'] = corp.issue_d.apply(getYear)

#Loan issue year and month
f,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(10,3))
orderby=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sns.countplot(x='issue_month',data=corp,order=orderby,ax=ax1)
sns.countplot(x='issue_year',data=corp,ax=ax2)

#Loan amnt 
corp[corp['default_ind']==0]['loan_amnt'].hist(bins=35,color='blue',label='default 0')
corp[corp['default_ind']==1]['loan_amnt'].hist(bins=35,color='green',label='default 1')
plt.title('Distribution of loan amount by default_ind')
plt.legend()

#funded amnt 
corp[corp['default_ind']==0]['funded_amnt'].hist(bins=35,color='blue',label='default 0')
corp[corp['default_ind']==1]['funded_amnt'].hist(bins=35,color='green',label='default 1')
plt.title('Distribution of funded amount by default_ind')
plt.legend()

#Boxplot of loan amnt and funded amnt 
f,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(10,3))
sns.boxplot(y='default_ind',x='loan_amnt',data=corp,orient='h',ax=ax1)
sns.boxplot(y='default_ind',x='funded_amnt',data=corp,orient='h',ax=ax2)

# Frequency distribution of int_rate and installment
f,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(10,3))
sns.distplot(corp['int_rate'],ax=ax1)
sns.distplot(corp['installment'],ax=ax2)


# 2.4.Detailed data analysis - Analysing categorical variables

# Purpose
sns.countplot(x='purpose',hue='default_ind',data=corp)
plt.xticks(rotation=30)
plt.title('Loan Purpose',fontsize=20)

#Emp_length
sns.countplot(x='emp_length',hue='default_ind',data=corp)

#Term 
f,(ax1,ax2)=plt.subplots(nrows=1,ncols=2)
sns.countplot(x='term',hue='default_ind',data=corp,ax=ax1)
sns.boxplot(x='term',y='loan_amnt',data=corp,ax=ax2)

#Grade 
f,(ax1,ax2)=plt.subplots(nrows=1,ncols=2)
sns.countplot(x='grade',hue='default_ind',data=corp,ax=ax1)
sns.boxplot(x='grade',y='loan_amnt',data=corp,ax=ax2)

#Home_ownership
f,(ax1,ax2)=plt.subplots(nrows=1,ncols=2)
sns.countplot(x='home_ownership',hue='default_ind',data=corp,ax=ax1)
sns.boxplot(x='home_ownership',y='loan_amnt',data=corp,ax=ax2)

#initial_list_status
f,(ax1,ax2)=plt.subplots(nrows=1,ncols=2)
sns.countplot(x='initial_list_status',hue='default_ind',data=corp,ax=ax1)
sns.boxplot(x='initial_list_status',y='loan_amnt',data=corp,ax=ax2)

#Verification stauts
f,(ax1,ax2)=plt.subplots(nrows=1,ncols=2)
sns.countplot(x='verification_status',hue='default_ind',data=corp,ax=ax1)
sns.boxplot(x='verification_status',y='loan_amnt',data=corp,ax=ax2)

#Application_type and pymnt_plan
f,(ax1,ax2)=plt.subplots(nrows=1,ncols=2)
sns.countplot(x='pymnt_plan',hue='default_ind',data=corp,ax=ax1)
sns.countplot(x='application_type',hue='default_ind',data=corp,ax=ax2)



#Summary of EDA
# The year of 2015 and October month were the highest amount of loans were issued 
# Purpose of loan is mostly for debt consolidation
# B and C grades are most dominant ones
# Most income sources were verified before issuing loans
# Policy code has only one level(i.e level 1)
# Application type joint level is less compared to individual
# pymnt_plan has all 'n' values expect five 'y' values

# 2.5 Data Cleaning-Dropping null and irrelevant columns 

#After analysing data lets remove columns which have more then 7lakh null values in each column

null_counts=corp.isnull().sum()
for i in range(len(null_counts)):
    if null_counts[i]>700000:
        del corp['{}'.format(null_counts.index[i])]
# 19 Variables with more then 80% of null values have been removed from the dataset
        
#Lets drop some of irrelevant features from the dataset
corp.drop(['id','member_id','policy_code','application_type','pymnt_plan','issue_year','issue_month'],axis=1,inplace=True)


# 3. Feature Engineering

# 3.1 Imputing missing values

#Generic code to repalce all missing values of numeric variables with mean value and categoricaly varaiables with mode value.

for x in corp.columns[:]:
  if corp[x].dtype=='object':
    corp[x].fillna(corp[x].mode()[0],inplace=True)
  elif corp[x].dtype=='int64' or corp[x].dtype=='float64':
    corp[x].fillna(corp[x].mean(),inplace=True)
    
#Check if any null values are left in dataset
corp.isnull().sum().mean()
    
# 3.2 Data Transformation - Outliers
numeric_features=corp.select_dtypes(include=[np.number])
num_cols=numeric_features.columns
print(num_cols)

#Outliers of each numeric columns in a single page
r=16;c=2;pos=1
fig=plt.figure(figsize=(11,4))
for e in num_cols:
    fig.add_subplot(r,c,pos)
    numeric_features.boxplot(e,vert=False)
    pos+=1

#Dropping columns with more outliers
corp.drop(['revol_bal','dti','recoveries','mths_since_last_major_derog','tot_coll_amt','total_rev_hi_lim','delinq_2yrs','pub_rec','total_rec_late_fee','collection_recovery_fee'],axis=1,inplace=True)
       
# 3.3. Data Transformation-Removing correlated columns
cor=numeric_features.corr()

#Correlation heatmap
plt.subplots(figsize=(20, 9))
sns.heatmap(cor,linewidth=0.3,annot=True,annot_kws={'size':5},cmap='RdYlGn')

#Dropping highly correlated columns
corp.drop(['funded_amnt','funded_amnt_inv','out_prncp_inv','total_pymnt_inv'],axis=1,inplace=True)

# 3.4. Data transformation-Dealing with time variables

#next_pymnt_d
#Has only 3 levels and 29% missing values
corp.next_pymnt_d.nunique()
corp.drop(['next_pymnt_d'],axis=1,inplace=True)

#last_credit_pull_d
#It has 102 levels and missing values
corp.last_credit_pull_d.nunique()
corp.drop(['last_credit_pull_d'],axis=1,inplace=True)

#earliest_cr_line
corp.earliest_cr_line.nunique() # 697 levels
corp.drop(['earliest_cr_line'],axis=1,inplace=True)

#Last payment date
corp['last_pymnt_d']=pd.to_datetime(corp['last_pymnt_d'])
corp['last_pymnt_d']=pd.DatetimeIndex(corp['last_pymnt_d']).year

#issue_d 
corp['issue_d']=pd.to_datetime(corp['issue_d'])
corp['issue_year']=corp['issue_d'].apply(lambda x:x.year)

#Converting time variable to age variable
#What was the loan period from issue date to last_pymnt_date
corp['loan_period']=corp.last_pymnt_d - corp.issue_year
corp.loan_period.value_counts()

#Droping original time variables
corp.drop(['issue_year','last_pymnt_d'],axis=1,inplace=True)
corp.columns

# 3.5. Data transformation-Categorical variables to dummy columns 

#Dropping irrelevant columns
#Emp_title,title and addr_state have more no of levels 
#Zip_code has unique values
#Sub_grade because one column with grade is already present in dataset
corp.drop(['sub_grade','emp_title','title','zip_code','addr_state'],axis=1,inplace=True)

#Converting factor variables to dummy variables
cols=corp.select_dtypes(include=['object']).columns

for f in cols:
    dummy=pd.get_dummies(corp[f],drop_first=True,prefix=f)
    corp=corp.join(dummy)

#Lets drop original categorical variables from dataset
corp.drop(['term','grade','emp_length','home_ownership','verification_status', 'purpose', 'initial_list_status'],axis=1,inplace=True)

# 4.Machine Learning models

#spliting the data into X_train and X_test data
X_train=corp[corp['issue_d']<'2015-06-01']
X_test=corp[corp['issue_d']>='2015-06-01']

#droping issue_d from both dataset as issue_d is time variable dataset
X_train.drop(['issue_d'],axis=1,inplace=True)
X_test.drop(['issue_d'],axis=1,inplace=True)



#Separting Y variable from X_train and X_test dataset and assigning to y_train and y_test
y_train=X_train.default_ind.copy()
y_test=X_test.default_ind.copy()
X_train.drop(['default_ind'],axis=1,inplace=True)
X_test.drop(['default_ind'],axis=1,inplace=True)


# 4.1. Logistic Regression

log_reg=LogisticRegression(C=0.001,random_state=101,solver='liblinear')
log_reg.fit(X_train,y_train)
log_predictions=log_reg.predict(X_test)

#confusion matrix
confusion_matrix(list(y_test),list(log_predictions))

#Accuracy score
acc= accuracy_score(y_test,log_predictions)
print("Accuracy of log_model is: ",acc)

#Classification report
print(cr(y_test,log_predictions))

#K-fload Cross validation
kfold_cv=KFold(n_splits=10)
kfold_log_result=cross_val_score(estimator=log_reg,X=X_train,
y=y_train, cv=kfold_cv)
print(kfold_log_result)


# 4.2.Decision Tree

# 4.2.1. Gini Decision Tree

m_gini=dtc(criterion='gini',max_depth=4,random_state=101,min_samples_leaf=2)
m_gini.fit(X_train,y_train)
p_gini=m_gini.predict(X_test)

#Confusion matrix
confusion_matrix(list(y_test),list(p_gini))

#Accuracy score
acc= accuracy_score(y_test,p_gini)
print("Accuracy of gini_model is: ",acc)

#classification report
print(cr(y_test,p_gini))

#K-fload Cross validation
kfold_cv=KFold(n_splits=10)
kfold_gini_result=cross_val_score(estimator=m_gini,X=X_train,
y=y_train, cv=kfold_cv)
print(kfold_gini_result)

# 4.2.2. Entropy Decision Tree

m_entropy=dtc(criterion='entropy',max_depth=4,random_state=101,min_samples_leaf=2)
m_entropy.fit(X_train,y_train)
p_entropy=m_entropy.predict(X_test)

#Confusion matrix
confusion_matrix(list(y_test),list(p_entropy))

#Accuracy score
acc= accuracy_score(y_test,p_entropy)
print("Accuracy of entropy_model is: ",acc)

#classification report
print(cr(y_test,p_entropy))

#K-fload Cross validation
kfold_cv=KFold(n_splits=10)
kfold_entropy_result=cross_val_score(estimator=m_entropy,X=X_train,
y=y_train, cv=kfold_cv)
print(kfold_entropy_result)

# 4.3. Random Forest
rf_model=rfc(n_estimators=10,random_state=101,criterion='entropy') #Entropy model performed well compared to gini 
rf_model.fit(X_train,y_train)

#predictions
rf_prediction=rf_model.predict(X_test)

#confusion matrix
confusion_matrix(list(y_test),list(rf_prediction))

#Accuracy score
acc= accuracy_score(y_test,rf_prediction)
print("Accuracy of rf_model is: ",acc)

#classification report
print(cr(y_test,rf_prediction))

#K-fload Cross validation
kfold_cv=KFold(n_splits=10)
kfold_rf_result=cross_val_score(estimator=rf_model,X=X_train,
y=y_train, cv=kfold_cv)
print(kfold_rf_result)

# 5. Final Predictions And ROC curve

#After comparing accuracy,recall_scor,f1_score and cross-validation scores of all the models above logistic model is selected for final predictions and plotting ROC Curve for same model

#Final predictions
final_predictions= pd.DataFrame({'actual_values': y_test, 'predicted_values':log_predictions})
final_predictions.head()

#Write file
final_predictions.to_csv('F:/Imarticus/Grp projects/Python Project - Bank Lending/final predictions.csv', index=False)

#ROC Curve
fpr,tpr,threshold=metrics.roc_curve(y_test,log_predictions)
roc_auc=metrics.auc(fpr,tpr)

#plot the curve
plt.title('Receiver operating characteristics')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('true positive rate')
plt.xlabel('false positive rate')
plt.show()

