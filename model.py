import pandas as pd
import numpy as np

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#model
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
#model performance
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#loading test and test data
df_train = pd.read_csv('/home/rupikat/Downloads/ds_data_big/ds_data/data_train.csv',index_col='id')
df_test = pd.read_csv('/home/rupikat/Downloads/ds_data_big/ds_data/data_test.csv',index_col='id')

#finding the columns having NaN values
nan_column=df_train.isnull().sum()
nan_column=nan_column[nan_column>0]
#print(nan_column.sort_values(ascending=False))
#print('-'*20)
nan_column_test=df_test.isnull().sum()
nan_column_test=nan_column_test[nan_column_test>0]
#print(nan_column_test.sort_values(ascending=False))

#finding the columns of 
bool_cols_train = [col for col in df_train 
             if df_train[[col]].dropna().isin([0, 1]).all().values]
#print("columns in test having binary values:",bool_cols_train)
bool_cols_test = [col for col in df_test
             if df_test[[col]].dropna().isin([0, 1]).all().values]
#print("columns in test having binary values:",bool_cols_test)
fill_na_col_bool_train=list(set(nan_column.index)&set(bool_cols_train))
fill_na_col_bool_test=list(set(nan_column_test.index)&set(bool_cols_test))
fill_na_col_num_train=list(set(nan_column.index)-set(fill_na_col_bool_train))
fill_na_col_num_test=list(set(nan_column_test.index)-set(fill_na_col_bool_test))
#print("Binary columns in train having nan;",fill_na_col_bool_train)
#print("Binary columns in test having nan;",fill_na_col_bool_test)
#print("numeric columns in train having nan;",fill_na_col_num_train)
#print("numeric columns in test having nan;",fill_na_col_num_test)
for col in fill_na_col_num_train:
    mean = df_train[col].mean().astype(np.int32)
    df_train[col].fillna( mean , inplace = True)
for col in fill_na_col_num_test:
    mean = df_test[col].mean().astype(np.int32)
    df_test[col].fillna( mean , inplace = True)
fill_bool=['cat2','cat10','cat5']
for col in fill_bool:
    mode = df_test[col].mode().astype(np.int32)
    df_test[col].fillna(df_test[col].mode()[0] , inplace = True)
    mode = df_train[col].mode().astype(np.int32)
    df_train[col].fillna(df_train[col].mode()[0] , inplace = True)
df_train=df_train.drop(['cat6', 'cat8'], axis=1)
df_test=df_test.drop(['cat6', 'cat8'], axis=1)    
print(df_test.isnull().sum())
print(df_train.isnull().sum())


X_train, X_test, y_train, y_test = train_test_split(df_train.drop('target',axis=1),df_train.target, test_size=0.2,stratify=df_train.target, random_state=4)
logreg=LogisticRegression(
    penalty='l2',C=10,
    n_jobs=-1, verbose=1, 
    solver='sag', multi_class='ovr',
    max_iter=200
)
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)
y_pred_prob=logreg.predict_proba(X_test)

#ROC curve
logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

print("The classification Report")
print(classification_report(y_test, y_pred, digits=6))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
cm = confusion_matrix(y_test, y_pred)
print("The classification matrix is")
print(cm)


#SUBMISSION 
result=logreg.predict_proba(df_test)
submission = pd.DataFrame({"id": df_test.index,
                         "target": result[:,1]})
submission.to_csv("ds_data_submission.csv",index=False)
