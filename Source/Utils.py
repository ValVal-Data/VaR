#Tool functions

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scp
import pandas as pan
import glob as gb
import scipy.optimize as op
from Source.Distribution import *


def load(pth):
    """
    File loading of temporal asset data

    Parameters:
        pth (str): Path to the file.

    Returns:
        data (pan.DataFrame)
    
    Notes:
        - Data should be in tabular format with "  " as separation
    """
    data=pan.read_csv(pth,sep="  ",engine="python")
    data["Date"]=pan.to_datetime(data["Date"],dayfirst=True)
    data=data.set_index("Date")
    data=data.sort_index(ascending=True)
    return(data)
    
#Load all files in Data
def loadAll():
    """
    Load multiple files contain in set files

    Parameters:
        None

    Returns:
        dataClose (pan.DataFrame): Contain closing data with each asset as a column
        dataDiv (pan.DataFrame): Contain dividend data with each asset as a column
        dataCur (pan.DataFrame): Contain currency exchange data with each asset as a column
        info (dic): Dictionnary that contain information from file, such as currency and capitalization
    
    Notes:
        - Data should be in the folder Data
        - They are separated between closing value, currency exchange rate and dividends
        - This separation is specified by a code before the extension separated by '_'
    """
    #Find all files in Data
    filesDiv=gb.glob("Data/*Div.dat")
    filesClose=gb.glob("Data/*Close.dat")
    filesCur=gb.glob("Data/*Cur.dat")

    #Extract info from name
    name=[f.split('_')[1] for f in filesClose]
    currency=[f.split('_')[-3] for f in filesClose]
    distrib=[f.split('_')[-2] for f in filesClose]
    info={na: {"Currency":cu,"Type":ty} for na, cu, ty in zip(name,currency,distrib)}
    nameC=[f.split('-')[0].split('\\')[1] for f in filesCur]
    nameDiv=[f.split('_')[1] for f in filesDiv]

    #Load data
    dataClose=[load(f) for f in filesClose]
    dataDiv=[load(f) for f in filesDiv]
    dataCur=[load(f) for f in filesCur]

    #Merge into a single table
    dataClose=pan.concat(dataClose,axis=1)
    dataClose.columns=name
    dataDiv=pan.concat(dataDiv,axis=1,sort=True)
    dataDiv.columns=nameDiv
    dataCur=pan.concat(dataCur,axis=1)
    dataCur.columns=nameC
    
    #Adjust for dividends
    for fund in nameDiv:
        dataClose[fund+"_div"]=dataDiv[fund]
        dataClose[fund+"_div"]=dataClose[fund+"_div"].fillna(0)
        dataClose[fund]+=dataClose[fund+"_div"]
        dataClose.drop(columns=[fund+"_div"],inplace=True)
        
    #Convert everything to CHF
    for fund in name:
        if info[fund]["Currency"]!="CHF":
            dataClose[fund]*=dataCur[info[fund]["Currency"]]

    #Load portfolio and adjust values
    port=pan.read_csv("Portfolio.txt",sep="\t",engine="python")
    dataClose/=list(dataClose.iloc[0])
    with open("Portfolio.txt","r") as f:
        for line in f:
            l=line.split("\t")
            if l[0] in ["CHF","USD","EUR"]:
                dataClose[l[0]]=int(l[1])*dataCur[l[0]]/float(dataCur[l[0]].iloc[0])
            else:
                dataClose[l[0]]*=int(l[1])
    dataClose["Total"]=dataClose.sum(axis=1)
    return((dataClose,dataDiv, dataCur,info))

def returnsP(data):
    """
    Calculate returns in percentage r=(p1-p0)/p0*100

    Parameters:
        data (pan.DataFrame): Data to convert

    Returns:
        histVar (pan.DataFrame)

    """
    histVar=data.copy(deep=True)
    for fund in data.columns:
        histVar[fund]=(-data[fund]+data[fund].shift(1))/data[fund]*100
    histVar=histVar.dropna()
    return(histVar)

def returnsL(data):
    """
    Calculate returns log returns lr=log(p1/p0)

    Parameters:
        data (pan.DataFrame): Data to convert

    Returns:
        histVar (pan.DataFrame)

    """
    histVar=data.copy(deep=True)
    for fund in data.columns:
        histVar[fund]=np.log(data[fund].shift(1)/data[fund])
    histVar=histVar.dropna()
    return(histVar)

def garch_likelihood(param, r,dist="gaussian"):
    """
    Calculate loglikelihood for GARCH fitting different distributions (Gaussian, Student-t, Skewed-t)

    Parameters:
        param (list): Parameter of the GARCH model
        r (pan.DataFrame): Log returns
        dist (str,default="gaussian"): Choice of the distribution for innovation ("gaussian","t","skewed-t")

    Returns:
        log likelihood (float): Log likelihood total

    Notes:
        - The number of parameter depends on the distributions
            - Gaussian: Alpha, Beta, Omega
            - Student-t: Alpha, Beta, Omega, Nu
            - Skewed-t: Alpha, Beta, Omega, Nu, Lambda
        - There are some constraints applied
            - Nu > 2
            - -1 < Lamda < 1
            - Omega > 0
            - Alpha >= 0
            - Beta >= 0
            - (Alpha + Beta) < 1

    """
    #Parameter definition
    if dist=="gaussian":
        omega,alpha,beta=param
        nu=None
    elif dist=="t":
        omega,alpha,beta,nu=param
        if nu<=2:
            return 1e10
    elif dist=="skewed-t":
        omega,alpha,beta,nu,lam=param
        if nu<=2:
            return 1e10
        if lam>1 or lam<-1:
            return 1e10
    # enforce positivity constraints
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta >= 1):
        return 1e10
    #Calculate conditional variance
    t=len(r)
    sigma2=np.zeros(t)
    sigma2[0]=np.var(r)
    for i in range(1,t):
        sigma2[i]=omega+alpha*(r[i-1]**2)+beta*sigma2[i-1]
    # Standardized residuals
    eps=(r-np.mean(r))/np.sqrt(sigma2)
    #Log likelihood calculation
    if dist=="gaussian":
        c=np.log(scp.norm.pdf(eps))
    elif dist=="t":
        c=np.log(scp.t.pdf(eps,df=nu))
    elif dist=="skewed-t":
        c=np.log(t_skewed_pdf(eps,nu,lam))
    ll=c-0.5*np.log(sigma2)
    return(-np.sum(ll))


def fitGARCH(logHistVar,dist="gaussian"):
    """
    Fit returns with GARCH model

    Parameters:
        logHistVar (pan.DataFrame): Data to convert
        dist (str,default="gaussian"): Choice of the distribution for innovation ("gaussian","t","skewed-t")

    Returns:
        sr (np.array): Residual/volatility
        var (list): List of fitted volatility
        res (res): Log likelihood value for the fit

    """
    if dist == "gaussian":
        param0 = np.array([0.01, 0.05, 0.9])  # initial guess
    elif dist == "t":
        param0 = np.array([0.01, 0.05, 0.9, 8])  # includes nu
    elif dist == "skewed-t":
        param0 = np.array([0.01, 0.05, 0.9, 8, 0])  # includes nu, lambda
    res=[]
    for i in range(len(logHistVar.columns)):
        t=np.asarray(logHistVar.iloc[:,i])
        res.append(op.minimize(garch_likelihood,param0,args=(t,dist),method="bfgs"))
    opt=np.array([i.x for i in res])
    #standardized residuals
    var=[]
    sr=[]
    for i in range(len(logHistVar.columns)):
        var.append(calc_Garch(opt[i],np.asarray(logHistVar.iloc[:,i]),dist))
        sr.append((np.asarray(logHistVar.iloc[:,i])-np.asarray(logHistVar.iloc[:,i]).mean())/var[-1])
    sr=np.array(sr).T
    return((sr,var,res))

def calc_Garch(param,r,dist="gaussian"):
    """
    Calculate volatility for a certain set of parameter

    Parameters:
        logHistVar (pan.DataFrame): Data to convert
        dist (str,default="gaussian"): Choice of the distribution for innovation ("gaussian","t","skewed-t")

    Returns:
        sr (np.array): Residual/volatility
        var (list): List of fitted volatility
        res (res): Log likelihood value for the fit

    """
    if dist=="gaussian":
        omega,alpha,beta=param
    elif dist=="t":
        omega,alpha,beta,nu=param
    elif dist=="skewed-t":
        omega,alpha,beta,nu,lam=param
    t=len(r)
    sigma2=np.zeros(t)
    sigma2[0]=np.var(r)
    for i in range(1,t):
        sigma2[i]=omega+alpha*r[i-1]**2+beta*sigma2[i-1]
    return np.sqrt(sigma2)

def correlMap(logHistVar,dist="gaussian"):
    """
    Calculation of the correlation map, by fitting GARCH

    Parameters:
        logHistVar (pan.DataFrame): Data to convert
        dist (str,default="gaussian"): Choice of the distribution for innovation ("gaussian","t","skewed-t")

    Returns:
        srr.corr() (pan.DataFrame): Correlation matrix
        var (list): List of fitted volatility
        param (list): Optimal GARCH parameters
        srr (list): Standardized residuals 
    """

    sr,var,res=fitGARCH(logHistVar,dist)
    var=np.asarray(var)
    param=[i.x for i in res]
    param=np.asarray(param).T
    srr=pan.DataFrame(sr,columns=logHistVar.columns,index=logHistVar.index)
    #Calculate standardized variance
    for name in srr.columns:
        srr[name]=(srr[name]-srr[name].mean())/srr[name].var()
    return (srr.corr(),var,param,srr)

def increase_corr(mat,alpha=0.3,rho=0.7):
    """
    Increase correlation of a matrix, while preseving positive and definite

    Parameters:
        mat (pan.DataFrame): Data to modify
        alpha(float, default=0.3): How much to change the original data
        rho (float, default=0.7): Target correlation

    Returns:
        mat (pan.DataFrame): Modified matrix
    """
    n=mat.shape[0]
    j=np.full((n,n),rho)
    np.fill_diagonal(j,1.0)
    return (1-alpha)*mat+j*alpha