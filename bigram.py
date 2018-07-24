# !/usr/bin/python
#SCRIPT TO PRINT SYSTEM CALL WITH VALUE
import os
from typing import TextIO
import nltk

currentdir = "C:\\Users\\Mahe\\Desktop\\padova\\freq_table\\bigram" #CHANGE INPUT PATH
resultdir = "C:\\Users\\Mahe\\Desktop\\padova\\testing" #CHANGE OUTPUT ACCORDINGLY

for root, dirs, files in os.walk(currentdir):
    for name in files:
        #systemcallcount = {}
        l = []
        #print(os.path.join(root, name))
        outfile1 = open(resultdir + "/" + name, "w+")
        outfile2 = open(root+"/"+name,'r')
        line = outfile2.readline()
        while line:
            l.append(line)
            
         
            
        
            line = outfile2.readline()
        s = "".join(l)
        bigram = list(nltk.bigrams(s.split()))
        print(bigram)
       # %s" % (''.join(ele) for ele in bigram)
        #outfile1.write("%s" % (''.join(ele) for ele in bigram))
        outfile1.write("".join('{}\n'.format(ele) for ele in bigram))
        #print(bigram)
        
        #print("\n")
        

        
        #outfile1.write(result)
        outfile1.write("\n")
            
        outfile1.close()
        outfile2.close()
        
        
        