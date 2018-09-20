import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df_train = pd.read_csv('data_train.csv',index_col='id')

plt.figure(figsize=(15,7))
plt.subplot(121)
ax = sns.countplot(x="target", data=df_train)
plt.subplot(122)
plt.title("Percentage distribution class 1 and 0")
plt.pie(df_train["target"].value_counts(), labels=["0","1"], colors = ['y', 'g'], startangle=90, autopct='%.1f%%')
plt.show()
