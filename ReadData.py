import pandas as pd
from csv import reader

def  load_csv(filename):
    dataset=list()
    with open(filename,"r") as file:
        csv_reader=reader(file)
        for row in csv_reader:
            if row=='e,m,p,t,y,!':
                continue
            dataset.append(row)
    return dataset

dataset=load_csv("/home/meet/PycharmProjects/A5/seq_positive_training.csv")


def stringToint(dataset):

    dbset=list()
    for i in range(len(dataset)):
        intdb=list()
        intdb2=list()
        str,str2,str3,str4=dataset[i]
        temp1=True;
        temp2=True;
        for j in range(maximum(int(str3),int(str4))):
            temp=list()
            if j==int(str3):
                temp1=False
            if j==int(str4):
                temp2=False
            if(temp1):
                if str[j] == 'A':
                    intdb.append('1')
                if str[j] == 'C':
                    intdb.append('2')
                if str[j] == 'G':
                    intdb.append('3')
                if str[j] == 'T':
                    intdb.append('4')
            if (temp2):
                if str2[j] == 'A':
                    intdb2.append('1')
                if str2[j] == 'C':
                    intdb2.append('2')
                if str2[j] == 'G':
                    intdb2.append('3')
                if str2[j] == 'T':
                    intdb2.append('4')
        temp.append(intdb)
        temp.append(intdb2)
        dbset.append(temp)

    return dbset

def maximum(a,b):
    if a>b:
        return a
    else:
        return b

def remove_emptyLines(dataset):
    list=['e', 'm', 'p', 't', 'y', '!']
    dbset=dataset
    dbset.remove(list)
    return dbset

print(stringToint(dataset))







#sbn=pd.read_csv("/home/meet/PycharmProjects/A5/seq_positive_training.csv",delimiter=",")
#sbp=pd.read_csv("/home/meet/PycharmProjects/A5/seq_positive_training.csv",delimiter=",")
#Kmer
#kbp=pd.read_csv("/home/meet/PycharmProjects/A5/kmer_positive_training.csv",delimiter=",")
#kbn=pd.read_csv("/home/meet/PycharmProjects/A5/kmer_negative_training.csv",delimiter=",")
#deepbind
#dbp=pd.read_csv("/home/meet/PycharmProjects/A5/deepbind_positive_training.csv",error_bad_lines=False)
#dbn=pd.read_csv("/home/meet/PycharmProjects/A5/deepbind_negative_training.csv",error_bad_lines=False)


#print (dbn)
#print (dbp)

