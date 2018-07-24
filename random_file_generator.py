import shutil, random, os


d="E:\\padova\\UPDATE\\random_small_drebin"

filenames = random.sample(os.listdir("E:\\padova\\Dataset-20180514T120518Z-001\\UNIQ_LIST_malware_drebin"),73)
for fname in filenames:
    srcpath = os.path.join("E:\\padova\\Dataset-20180514T120518Z-001\\UNIQ_LIST_malware_drebin", fname)
    shutil.copy(srcpath,d)