#after filling all the NaN values in df_train,we can plot this

df_corr = df_train.corr(method='spearman').abs()

# order columns and rows by correlation with target
df_corr = df_corr.sort_values('target',axis=0,ascending=False).sort_values('target',axis=1,ascending=False)
print(df_corr.target)
ax=plt.figure(figsize=(20,16)).gca()
sns.heatmap(df_corr,ax=ax,square=True)
