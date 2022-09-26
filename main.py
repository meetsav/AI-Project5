import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import subprocess

def configure_git_path():
    return subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode("utf-8").strip()

git_path = configure_git_path()

def giveMax(set):
    temp=set[0]
    for i in range(len(set)):
        if set[i]>temp :
            temp=set[i]

    return temp

def str_to_int(database,row,column):
    str=database[0]
    strlen=database[2]
    str2=database[1]
    str2len=database[3]
    result=database[4]
    dbset=list()
    for i in range(row):
        temp=str[i]
        temp1=str2[i]
        intval=list()
        label1=True
        label2=True
        for j in range(int(strlen[i])):
            if temp[j]=='A':
                intval.append('1')
            if temp[j]=='G':
                intval.append('2')
            if temp[j]=='C':
                intval.append('3')
            if temp[j]=='T':
                intval.append('4')
        for j in range(int(str2len[i])):
            if label2:
                if temp1[j]=='A':
                    intval.append('1')
                if temp1[j]=='G':
                    intval.append('2')
                if temp1[j]=='C':
                    intval.append('3')
                if temp1[j]=='T':
                    intval.append('4')
        cur_len=len(intval)
        for j in range(column-cur_len):
            intval.append('0')
        intval.append(result[i])
        dbset.append(intval)
    db=np.array(dbset)
    return (db)

def getXY(db,max):
    Y=db[max]
    X=db[:1:]
    return X,Y

def merge(dataset1,dataset2):
    dbset=pd.concat([dataset1,dataset2],join='outer',sort=True,ignore_index=True)
    dbset=dbset.sample(frac=1)

    return dbset
def writeTofile(data,flag):
    filename=''
    if flag=='s':
        filename="seq_test.out"
    if flag=='k':
        filename="kmer_test.out"
    if flag=='d':
        filename="deepbind_test.out"
    with open(filename,'w') as file:
        for j in range(len(data)):
            if data[j]==1:
                file.write('+')
            if data[j]==0:
                file.write('-')
    file.close()

def seq_row_input():

    dataset = pd.read_csv(git_path+"/seq_positive_training.csv", header=None)
    dataset1 = pd.read_csv(git_path+"/seq_negative_training.csv", header=None)
    test=pd.read_csv(git_path+"/seq_test.csv",header=None)

    dataset[4]=1
    dataset1[4]=0
    test[4]=1
    data=dataset.dropna()
    data1=dataset1.dropna()
    db=merge(data,data1)
    test.dropna()
    max_test_column=max(test[2])+max(test[3])+1
    max_test_row=len(test)
    max_length_column = max(db[2]) + max(db[3]) + 1
    max_length_row = len(db)

    if max_test_column>max_length_column:
        max_length_column=max_test_column
    else:
        max_test_column=max_length_column

    db = str_to_int(db, max_length_row, max_length_column)
    test=str_to_int(test,max_test_row,max_test_column)
    print(test.shape)
    print(db.shape)
    #test=np.delete(test,max_test_column,1)
    Y = db[:, max_length_column]
    (db,Y,test,'s')

def kmer_input():
    dataset=pd.read_csv(git_path+"/kmer_positive_training.csv",header=None)
    dataset1=pd.read_csv(git_path+"/kmer_negative_training.csv",header=None)
    #print(dataset)
    #print(dataset1)
    kmer=pd.read_csv(git_path+"/kmer_test.csv",header=None)
    kmer=kmer.values
    data=pd.concat([dataset,dataset1],sort=True,ignore_index=True)
    data=data.values
    max=len(data[0])
    kmer_target=data[:,max-1]
    print(len(kmer_target))
    trainAndTest(data,kmer_target,kmer,'k')

def deepbind():

    with open(git_path+"/deepbind_negative_training.csv",'r')  as file:
        data=file.read()
    data=data.split("\n")
    with open(git_path+"/deepbind_positive_training.csv",'r') as file:
        data1=file.read()
    data1=data1.split("\n")

def trainAndTest(X,Y,test,flag):


    classifier=tree.DecisionTreeClassifier()
    classifier=classifier.fit(X,Y)
    predection=classifier.predict(test)
    writeTofile(predection,flag)
kmer_input()
