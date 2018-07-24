'''
Given a extracted system call log file we generate the frequency of the system call logs in each file.

'''


import os

currentdir = "C:\\Users\\Mahe\\Desktop\\padova\\Dataset-20180514T120518Z-001\\Dataset\\Malware\\Others-Shweta" #CHANGE INPUT PATH(\\ for windows only)
resultdir = "C:\\Users\\Mahe\\Desktop\\padova\\Dataset-20180514T120518Z-001\\UNIQ_LIST_malware_others" #CHANGE OUTPUT ACCORDINGLY


if not os.path.exists(resultdir):
    os.makedirs(resultdir)
    
    
for root, dirs, files in os.walk(currentdir):
    for name in files:
        systemcallcount = {}  
        print(os.path.join(root, name))
        outfile1 = open(resultdir + "/" + name, "w+")
        outfile2 = open(root+"/"+name,'r')
        line = outfile2.readline()
        while line:
            if line not in systemcallcount:
                systemcallcount[line] = 1
            else:
                systemcallcount[line] += 1
            line = outfile2.readline()
        outfile2.close()

        for key in systemcallcount.keys():
            value = systemcallcount[key]
            result = str(key.strip('\n'))+" "+str(value)
            print(result)
            outfile1.write(result)
            outfile1.write("\n")
        outfile1.close()
        outfile2.close()