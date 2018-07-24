import pandas as pd

currentdir = "C:\\Users\\Mahe\\Desktop\\padova\\UPDATE" #CHANGE INPUT PATH

df1=pd.read_csv('t1.csv')
df2=pd.read_csv('t2.csv')

df=df1.merge(df2, on='system_call', how='outer', suffixes=('_M', '_B')).fillna(0)

df.to_csv('merged_data_drebin.csv',index=False)


       