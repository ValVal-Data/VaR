# Objet and function for Monte carlo simulations

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scp
import pandas as pan
from IPython.display import clear_output
from Source.Distribution import *


class MCSim:
    """
    Monte Carlo engine for CCC-GARCH modeling of VaR

    Parameters:
        initial_values (pan.DataFrame): Initial price for the assets
        volh (list): Historical mean volatility
        drift (float): Drift in the GARCH model
        step (int): Number of step to do for each simulation
        horizon (int): Length of the simulation
        nbsim (int): Number of simulation to run
        q (float): 1-Confidence level

    Attributes:
        initial_values (pan.DataFrame): Initial price for the assets
        volh (list): Historical mean volatility
        drift (float): Drift in the GARCH model
        step (int): Number of step to do for each simulation
        horizon (int): Length of the simulation
        nbsim (int): Number of simulation to run
        q (float): 1-Confidence level
        stress(float): Factor to scale volatility for stress testing
        s_1 (list): Price at t-1 for GARCH calculation
        dist (str,default="gaussian"): Choice of the distribution for innovation ("gaussian","t","skewed-t")
        var (list): Volatility at t=0
        param (list): GARCH parameters
        l (np.array): Cholesky decomposition
        corrst (pan.DataFrame): Correlation map
        cdf_vals (np.array): Calculation of CDF for t-skewed distribution to make code faster
        pdf_vals (np.array): Calculation of PDF for t-skewed distribution to make code faster
        xs (np.array): list of point for calculation of PDF and CDF
    """
    def __init__(self,initial_values:pan.DataFrame,volh:list,drift:float,step:int,horizon:int,nbsim:int,q:float)->None:
        self.s0=initial_values
        self.volh=volh
        self.drift=drift
        self.step=step
        self.horizon=horizon
        self.nbsim=nbsim
        self.q=q
        self.stress=1
    def set_garch(self,corrst:pan.DataFrame,param:list,var:list,l:np.array,dist:str="gaussian")->None:
        """
        Set attributes for CCC-GARCH

        Parameters:
            corrst (pan.DataFrame): Correlation map
            param (list): GARCH parameters
            var (list): Volatility at t=0
            l (np.array): Cholesky decomposition
            dist (str,default="gaussian"): Choice of the distribution for innovation ("gaussian","t","skewed-t")

        Returns:
            None
        """
        self.corrst=corrst
        self.l=np.linalg.cholesky(self.corrst)
        self.param=param
        self.var=var
        self.s_1=self.s0*np.exp(-l)
        self.dist=dist
    def simulate_paths_hist(self)->np.array:
        """
        simulate paths using historical variance

        Returns:
            result (np.array): list of path based on the simulation
        """
        #Simulate value over time using historical volatility
        dt=self.horizon/self.step
        paths=[]
        for i in range(self.nbsim):
            price=self.s0
            path=[price]
            for j in range(self.step):
                z=np.random.normal()
                price=price*np.exp((self.drift-0.5*self.volh**2)*dt+self.volh*np.sqrt(dt)*z)
                path.append(price)
            paths.append(path)
        return np.array(paths)
    def simulate_paths_GARCH(self)->np.array:
        """
        simulate paths using CCC-GARCH

        Returns:
            result (np.array): list of path based on the simulation
        """
        #Simulate value over time using GARCH volatility
        if self.dist=="gaussian":
            omega,alpha,beta=self.param
        elif self.dist=="t":
            omega,alpha,beta,nu=self.param
        elif self.dist == "skewed-t":
            omega,alpha,beta,nu,lam=self.param
            self.make_t_skewed_pdf()
            self.make_t_skewed_cdf()
        paths=[]
        sigmas=[]
        rets=[]
        for i in range(self.nbsim):
            #clear_output(wait=True)
            #print("Sim: "+str(i+1)+"/"+str(self.nbsim))
            price=self.s0
            path=[self.s_1,self.s0]
            sigma=[self.var]
            for j in range(self.step):
                #Calculate correlated shock
                sigma.append(np.sqrt(omega+alpha*np.log(path[-1]/path[-2])**2+beta*sigma[-1]**2))
                if self.dist=="gaussian":
                    z=np.random.randn(len(self.s0))
                    eps= self.l @ z
                elif self.dist=="t":
                    z=scp.t.rvs(df=nu,size=len(self.s0))
                    z=z/((nu/(nu-2))**0.5)#Standardize to variance=1
                    eps= self.l @ z
                elif self.dist=="skewed-t":
                    u=np.random.randn(len(self.s0))
                    y=self.l @ u
                    v=scp.norm.cdf(y)
                    eps=self.t_skewed_ppf(v)
                #Compute price movement
                price=price*np.exp(eps*sigma[-1]*np.sqrt(self.stress)+self.drift)
                path.append(price)
            sigmas.append(sigma)
            paths.append(path)
        return np.array(paths[1:])
    def VaR(self,gar:bool)->list:
        """
        Calculation of Value at Risk and Expected Shortfall

        Parameters:
            gar (bool): True to choose GARCH modeling and False to use historical volatility

        Returns:
            var (list): VaE values
            es (list): ES values
        """
        #Calculate VaR and ES
        if gar:
            data=self.simulate_paths_GARCH()
        else:
            data=self.simulate_paths_hist()
        sf=data[:,-1]
        pl=((sf-self.s0)/self.s0+self.drift)*100
        var_95=list(np.percentile(pl,(1-self.q)*100,axis=0))
        es=[pl[:,i][pl[:,i]>var_95[i]].mean() for i in range(len(var_95))]
        return np.abs(var_95), np.abs(es)
    #Create grid to interpolate cdf and ppf to make code faster
    def make_t_skewed_cdf(self)->None:
        """
        Make t-skewed cdf

        Returns:
            None
        """
        self.cdf_vals=[]
        for i in range(len(self.pdf_vals)):
            temp=inte.cumulative_trapezoid(self.pdf_vals[i],self.xs,initial=0)
            temp/=temp[-1]
            self.cdf_vals.append(temp)
        self.cdf_vals=np.asarray(self.cdf_vals)
    def make_t_skewed_pdf(self)->None:
        """
        Make t-skewed pdf

        Returns:
            None
        """
        omega,alpha,beta,nu,lam=self.param
        self.xs=np.linspace(-5,5,20000)
        self.pdf_vals=[]
        for i in range(len(nu)):
            self.pdf_vals.append(t_skewed_pdf(self.xs,nu[i],lam[i]))
        self.pdf_vals=np.asarray(self.pdf_vals)
    def t_skewed_cdf(self,x:float)->np.array:
        """
        Return t-skewed cdf

        Parameters:
            x (float): Where to calculate cdf

        Returns:
            cdf (np.array)
        """
        r=[]
        for i in range(len(self.cdf_vals)):
            r.append(np.interp(x,self.xs,self.cdf_vals[i]))
        return np.asarray(r)
    def t_skewed_ppf(self,x:float)->np.array:
        """
        Return t-skewed ppf

        Parameters:
            x (float): Where to calculate ppf

        Returns:
            ppf (np.array)
        """
        r=[]
        for i in range(len(self.cdf_vals)):
            r.append(np.interp(x[i],self.cdf_vals[i],self.xs))
        return np.asarray(r)
        