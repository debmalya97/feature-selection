import os
import pandas as pd


currentdir = "C:\\Users\\Mahe\\Desktop\\padova\\freq_table\\UNIQ_LIST_benign" #CHANGE INPUT PATH

df = pd.DataFrame(columns=['system_call','frequency','file_frequency']) 
df.set_index('system_call', inplace=True)

    
for root, dirs, files in os.walk(currentdir):
    for name in files:       
        outfile2 = open(root+"/"+name,'r')
        line = outfile2.readline()              
        while line:
            words=line.split(" ")
            
            if words[0] in df.index:
                df.loc[words[0],'frequency']+=int(words[1])
                df.loc[words[0],'file_frequency']+=1
             
            else:
                
                df = df.append(pd.Series([int(words[1]),1], name=words[0], index=df.columns))
                
                
            
            line = outfile2.readline()
            
                          
df.to_csv('t2.csv')
              
                
                
                
            
        