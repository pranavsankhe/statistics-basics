import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import multivariate_normal
import matplotlib.pyplot as mpl
import pomegranate
from scipy.stats import norm
ip_file = pd.read_excel('DataSet/university data.xlsx')
size=ip_file.shape

def mean(data,col):
    m= data[col].mean()
    print ("Mean of %s is %d" %(col,m))
    return m

def variance(data,col):
    print (np.cov(data[col].dropna(), ddof=0))
    v=data[col].var()
    print( "Variance of %s is %d"%(col,v))
    return v

def std(data,col):
    st=data[col].std()
    print("Standard Deviation of %s is %d"%(col,st))
    return st

mu1=mean(ip_file,"CS Score (USNews)")
mu2=mean(ip_file,"Research Overhead %")
mu3=mean(ip_file,"Admin Base Pay$")
mu4=mean(ip_file,"Tuition(out-state)$")
mu=[mu1,mu2,mu3,mu4]
print (mu)

var1=variance(ip_file,"CS Score (USNews)")
var2=variance(ip_file,"Research Overhead %")
var3=variance(ip_file,"Admin Base Pay$")
var4=variance(ip_file,"Tuition(out-state)$")

sigma1=std(ip_file,"CS Score (USNews)")
sigma2=std(ip_file,"Research Overhead %")
sigma3=std(ip_file,"Admin Base Pay$")
sigma4=std(ip_file,"Tuition(out-state)$")


df=ip_file.iloc[0:49,2:6]
cov_mat=df.cov().round(3)
print (cov_mat)
print  (df.corr().round(3)) #do correlation using numpy

#log likelihood independent variable


X=0
for i in range(0,49):
    X += (multivariate_normal.logpdf(df.iloc[i,:],mu,cov_mat,allow_singular='True'))
print (X)

#model = BayesianNetwork.from_samples(df, algorithm='exact')


mpl.plot()





