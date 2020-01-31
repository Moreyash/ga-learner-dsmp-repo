# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df=pd.read_csv(path)
df.head()
df.info
df['INCOME']=df['INCOME'].str.replace('$','')
df['INCOME']=df['INCOME'].str.replace(',','')
print(df['INCOME'])

df['HOME_VAL']=df['HOME_VAL'].str.replace('$','')
df['HOME_VAL']=df['HOME_VAL'].str.replace(',','')
print(df['HOME_VAL'])

df['BLUEBOOK']=df['BLUEBOOK'].str.replace('$','')
df['BLUEBOOK']=df['BLUEBOOK'].str.replace(',','')
print(df['BLUEBOOK'])


df['OLDCLAIM']=df['OLDCLAIM'].str.replace('$','')
df['OLDCLAIM']=df['OLDCLAIM'].str.replace(',','')
print(df['OLDCLAIM'])


df['CLM_AMT']=df['CLM_AMT'].str.replace('$','')
df['CLM_AMT']=df['CLM_AMT'].str.replace(',','')
print(df['CLM_AMT'])

X=df.drop(['CLAIM_FLAG'],1)
y=df['CLAIM_FLAG']
count=y.value_counts()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.3,random_state= 6)
# Code ends here


# --------------
# Code starts here
X_train['INCOME']=X_train['INCOME'].astype(float)/100
X_train['HOME_VAL']=X_train['HOME_VAL'].astype(float)/100
X_train['BLUEBOOK']=X_train['BLUEBOOK'].astype(float)/100
X_train['OLDCLAIM']=X_train['OLDCLAIM'].astype(float)/100
X_train['CLM_AMT']=X_train['CLM_AMT'].astype(float)/100


X_test['INCOME']=X_test['INCOME'].astype(float)/100
X_test['HOME_VAL']=X_test['HOME_VAL'].astype(float)/100
X_test['BLUEBOOK']=X_test['BLUEBOOK'].astype(float)/100
X_test['OLDCLAIM']=X_test['OLDCLAIM'].astype(float)/100
X_test['CLM_AMT']=X_test['CLM_AMT'].astype(float)/100





# Code ends here


# --------------
# Code starts here
X_train.dropna(subset=['YOJ','OCCUPATION'],inplace=True)
X_test.dropna(subset=['YOJ','OCCUPATION'],inplace=True)



y_train=y_train[X_train.index]
y_test=y_test[X_test.index]

X_train['AGE'].fillna((X_train['AGE'].mean()), inplace=True)
X_train['CAR_AGE'].fillna((X_train['CAR_AGE'].mean()), inplace=True)
X_train['INCOME'].fillna((X_train['INCOME'].mean()), inplace=True)
X_train['HOME_VAL'].fillna((X_train['HOME_VAL'].mean()), inplace=True)

X_test['AGE'].fillna((X_test['AGE'].mean()), inplace=True)
X_test['CAR_AGE'].fillna((X_test['CAR_AGE'].mean()), inplace=True)
X_test['INCOME'].fillna((X_test['INCOME'].mean()), inplace=True)
X_test['HOME_VAL'].fillna((X_test['HOME_VAL'].mean()), inplace=True)


# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for i in columns:
    le=LabelEncoder()
    X_train[i]=le.fit_transform(X_train[i].astype(str))
    X_test[i]=le.fit_transform(X_test[i].astype(str))
# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model=LogisticRegression(random_state=6)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)


# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote=SMOTE(random_state= 9)
X_train,y_train= smote.fit_sample(X_train, y_train)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
# Code ends here


# --------------
# Code Starts here
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)
# Code ends here


