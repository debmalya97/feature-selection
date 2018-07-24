import os
import pandas as pd
import numpy as np
import csv
'''

0 means benign and 1 malign

'''


### change the two folder directory here

dir_list = ["D:\\VINOD\\MALINE\\System-call-count-B-Bigram","D:\\VINOD\\System-call-count-M-Bigram"] #CHANGE INPUT PATH


appended_data = []

for l in dir_list: 
    for root, dirs, files in os.walk(l):
        for name in files:
           
            file = open(root+"/"+name,'r') 
            print(name)
            print("\n")
            df = pd.read_csv(file,header=None,encoding='utf8')   #In windows and python3 always pass file object not the path directly in pd.read_csv                
            df = df.rename(columns={0: 'col'})           
            df = pd.DataFrame(df.col.str.split(' ',1).tolist(), columns = ['col1','col2']).T.reset_index(drop=True)          
            df = df.rename(columns=df.iloc[0]).drop(df.index[0]) 
            df = df.loc[:,~df.columns.duplicated()]                        
            appended_data.append(df)
            if l=="D:\\VINOD\\MALINE\\System-call-count-B-Bigram": ###change directory of System-call-count-B-Bigram here
                df['class']=0
            else:
                df['class']=1

print(len(appended_data))

#df = df.loc[:,~df.columns.duplicated()]    

   
appended_data= pd.concat(appended_data, axis=0)
final=appended_data.fillna(0)
  

  
###to make the class column first(optional)  
  
# get a list of columns
cols = list(final)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index('class')))
# use ix to reorder
final = final.ix[:, cols] 
final.to_csv("maline_fvt_original_bigram.csv",index=False)

  


     