#Function for statistical testing

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scp
import pandas as pan
import scipy.optimize as op

def autocorrelation(x):
    y=x-np.mean(x)
    res=np.correlate(y,y,mode='full')
    acf=res[res.size//2:]
    return acf/acf[0]

def prob_cluster(x,center):
    #Softmax over distances
    d0=np.linalg.norm(x-center[0],axis=1)
    d1=np.linalg.norm(x-center[1],axis=1)
    p1=np.exp(-d1)/(np.exp(-d0)+np.exp(-d1))
    return 1-p1

def checkDistribution(data:pan.DataFrame,col:list,title:str)->None:
    """
    Compare return distribution to gaussian one in individual subpanels

    Parameters:
        data (float): Data to plot
        col (list): List of colors
        title (str): Title of the plot

    Returns:
        None, plot directly
    """
    fig,ax=plt.subplots(2,3,figsize=(10,4))
    fig.suptitle(title)
    for i in range(len(data.columns)):#To distribute the plots on 2 row and 3 columns
        if i>=3:
            j=(1,i-3)
        else:
            j=(0,i)
        ret=data.iloc[:,i]
        mu,sigma=ret.mean(), ret.std()
        x=np.linspace(ret.min(),ret.max(),200)
        pdf=scp.norm.pdf(x,mu,sigma)
        kde=scp.gaussian_kde(ret)
        pdfr=kde(x)
        ax[j].plot(x,pdf,linewidth=2,color='black',linestyle='--')
        ax[j].legend(["Gaussian"],loc="upper left")
        ax[j].set_title(data.columns[i])
        ax[j].fill_between(x,pdfr,color=col[i])
        ax[j].set_xlabel("Weekly return (%)")
        ax[j].set_ylabel("PDF")
    plt.tight_layout()
    plt.show()

def checkDistributionQQ(data:pan.DataFrame,col:list,title:str)->None:
    """
    QQplots with return distribution in individual subpanels

    Parameters:
        data (pan.DataFrame): Data to plot
        col (list): List of colors
        title (str): Title of the plot

    Returns:
        None, plot directly
    """
    fig,ax=plt.subplots(2,3,figsize=(10,4))
    fig.suptitle(title)
    for i in range(len(data.columns)):
        if i>=3:
            j=(1,i-3)
        else:
            j=(0,i)
        
        osm,osr=scp.probplot(data.iloc[:,i],dist="norm")[:2]
        th_q,sam_q=osm
        slope,inter,r=osr
        ax[j].scatter(th_q,sam_q,color=col[i])
        x_line=th_q
        y_line=slope*x_line+inter
        ax[j].plot(x_line,y_line,color="black",linestyle="--")
        
        ax[j].set_title(data.columns[i])
        ax[j].set_xlabel("Normal distribution")
        ax[j].set_ylabel("Sample distribution")
    plt.tight_layout()
    plt.show()

def checkDistributionQQ5(data:pan.DataFrame,col:list,title:str)->None:
    """
    QQplots with return distribution in individual subpanels for 5 plots in line

    Parameters:
        data (float): Data to plot
        col (list): List of colors
        title (str): Title of the plot

    Returns:
        None, plot directly
    """
    fig,ax=plt.subplots(1,5,figsize=(10,3))
    fig.suptitle(title)
    for i in range(len(data.columns)):
        osm,osr=scp.probplot(data.iloc[:,i],dist="norm")[:2]
        th_q,sam_q=osm
        slope,inter,r=osr
        ax[i].scatter(th_q,sam_q,color=col[i])
        x_line=th_q
        y_line=slope*x_line+inter
        ax[i].plot(x_line,y_line,color="black",linestyle="--")
        
        ax[i].set_title(data.columns[i])
        ax[i].set_xlabel("Normal distribution")
        ax[i].set_ylabel("Residual distribution")
    plt.tight_layout()
    plt.show()

def normalityTest(data:pan.DataFrame, alpha:float=0.05)->pan.DataFrame:
    """
    Perform Jarque_Bera and Anderson tests to check if data are normally distributed

    Parameters:
        data (pan.DataFrame): Data to analyze
        alpha (float, default=0.05): 1-Confidence level

    Returns:
        result (pan.DataFrame): Return if the test is rejected based on confidence level
    """
    a=[]
    for name in data.columns:
        t=scp.anderson(data[name],dist="norm",method='interpolate')
        a.append(t.pvalue)
    jb=scp.jarque_bera(data,axis=0).pvalue
    res=[["Reject"if x<alpha else "Fail to reject" for x in row] for row in [a,jb]]
    return(pan.DataFrame(res,columns=data.columns,index=["Anderson-Darling","Jarque-Bera"]).T.round(3))

def ljung_box_test(x:np.array,cl:float=0.05,lag:int=10)->str:
    """
    Perform Ljung-Box test to see if the variance is constant

    Parameters:
        data (np.array): Data to analyse
        cl (float, default=0.05): 1-Confidence level
        lag (int, default=10): Lag up to which the autocorrelation is computed

    Returns:
        result (str): Return if the test is rejected based on confidence level
    """
    x=np.asarray(x)
    n=len(x)
    mean=np.mean(x)
    acf=[]

    #Compute autocorrelation
    for i in range(1,lag+1):
        num=np.sum((x[i:]-mean)*(x[:-i]-mean))
        den=np.sum((x-mean)**2)
        acf.append(num/den)

    #Statistics
    q=n*(n+2)*np.sum([acf[i]**2/(n-(i+1)) for i in range(lag)])
    p_val=1-scp.chi2.cdf(q,df=lag)
    if p_val<cl:
        return("Reject")
    else:
        return("Fail to reject")
def arch_lm_test(x:np.array,cl:float=0.05,lag:int=10)->str:
    """
    Perform Arch-Lm test to see if the variance is constant

    Parameters:
        data (np.array): Data to analyse
        cl (float, default=0.05): 1-Confidence level
        lag (int, default=10): Lag up to which the autocorrelation is computed

    Returns:
        result (str): Return if the test is rejected based on confidence level
    """
    x=np.asarray(x)
    n=len(x)

    #Lagged matrix
    y=np.column_stack([x[lag-i-1:n-i-1] for i in range(lag)])
    y_reg=x[lag:]

    #add constant
    y=np.column_stack([np.ones(len(y)),y])

    #OLS estimate
    beta=np.linalg.lstsq(y,y_reg,rcond=None)[0]
    y_hat=y@beta

    #Compute R2
    ssr=np.sum((y_hat-np.mean(y_reg))**2)
    sst=np.sum((y_reg-np.mean(y_reg))**2)
    r2=ssr/sst

    lm=len(y_reg)*r2
    p_val=1-scp.chi2.cdf(lm,df=lag)
    if p_val<cl:
        return("Reject")
    else:
        return("Fail to reject")

#R2 calc for engle_sheppard_test
def regression_r2(x:np.array,y:np.array)->float:
    """
    Calculation of R^2 for Engle-Sheppart test

    Parameters:
        x (np.array): X values
        y (np.array): Y values

    Returns:
        r2 (float): Return R^2
    """
    x=np.column_stack([np.ones(len(x)),x])
    beta=np.linalg.lstsq(x,y,rcond=None)[0]
    y_hat=x@beta
    ssr=np.sum((y_hat-y.mean())**2)
    sst=np.sum((y-y.mean())**2)

    return (ssr/sst)

def engle_sheppard_test(srr:pan.DataFrame,corrst:pan.DataFrame,lag:int=5,alpha:float=0.05)->str:
    """
    Engle-Sheppart test to test if correlation is constant

    Parameters:
        srr (pan.DataFrame): Standardized residual
        corrst (pan.DataFrame): Correlation map
        lag (int, default=10): Lag up to which the autocorrelation is computed
        alpha (float, default=0.05): 1-Confidence level

    Returns:
        result (str): Return if the test is rejected based on confidence level
    """
    test_stat=0
    num_pairs=0
    for name in range(len(srr.columns)-1):
        for name2 in range(name+1,len(srr.columns)):
            q=srr.iloc[:,name]*srr.iloc[:,name2]-corrst.iloc[name,name2]
            #Lag matrix
            x=np.column_stack([q.shift(l) for l in range(1,lag+1)])
            valid=~np.isnan(x).any(axis=1)
            x=x[valid]
            y=q[valid].values

            r2=regression_r2(x,y)
            test_stat+=r2
            num_pairs+=1
    test_stat*=len(srr.columns)
    df=num_pairs*lag
    p_val=1-scp.chi2.cdf(test_stat,df)
    if p_val<alpha:
        return("Reject")
    else:
        return("Fail to reject")


def kupiec(histVar:pan.DataFrame,t:list,alpha:float=0.025)->str:
    """
    Kupiec test to check if the number of exception to the threhold is in line with confidence level

    Parameters:
        histVar (pan.DataFrame): Historical return data
        t (list): Thresholds
        alpha (float, default=0.05): 1-Confidence level

    Returns:
        result (str): Return if the test is rejected based on confidence level
    """
    ex=[]
    #Calculate number of breach
    for i in range(len(histVar.columns)):
        x=histVar.iloc[:,i].lt(-t[i]).sum()
        n=len(histVar.iloc[:,i])
        ex.append((x,x/n,n-x))
    #Calculate p-value
    lr=[]
    ll=[]
    for elem in ex:
        l=-2*(np.log(((1-alpha)**elem[2])*(alpha**elem[0]))-
              np.log(((1-elem[1])**elem[2])*(elem[1]**elem[0])))
        ll.append(l)
        p_val=1-scp.chi2.cdf(l,df=1)
        if(p_val<alpha):
            lr.append("Reject")
        else:
            lr.append("Fail to Reject")
    ex=np.asarray(ex)
    return lr, ex[:,1], ll

def christoffersens(histVar:pan.DataFrame,t:list,alpha:float=0.025)->list:
    """
    Christoffersen test on a DataFrame to check if the exceptions are independant and is in line with confidence level.

    Parameters:
        histVar (pan.DataFrame): Historical return data
        t (list): Thresholds
        alpha (float, default=0.05): 1-Confidence level

    Returns:
        result (list): Return if the test is rejected based on confidence level

    Notes:
        - For test on single serie, see christoffersen
    """
    ku=kupiec(histVar,t)[2]
    res=[]
    for i in range(len(ku)):
        res.append(christoffersen(np.asarray(histVar.iloc[:,i]),t[i],ku[i],alpha))
    return res

def christoffersen(histVar:pan.DataFrame,t:float,ku:float,alpha:float=0.025)->list:
    """
    Christoffersen test on single data serie to check if the exceptions are independant and is in line with confidence level. 

    Parameters:
        histVar (pan.DataFrame): Historical return data
        t (flaot): Threshold
        ku (float): Value of the Kupiec test for the data
        alpha (float, default=0.05): 1-Confidence level

    Returns:
        result (list): Return if the test is rejected based on confidence level

    Notes:
        - For test on DataFrame, see christoffersens
    """
    p=1-alpha
    i=(histVar < -t).astype(int)
    #Transition count
    i_prev=i[:-1]
    i_curr=i[1:]
    N00=np.sum((i_prev==0)&(i_curr==0))
    N01=np.sum((i_prev==0)&(i_curr==1))
    N10=np.sum((i_prev==1)&(i_curr==0))
    N11=np.sum((i_prev==1)&(i_curr==1))
    pi0=N01/(N00+N01)if(N00+N01)>0 else 0
    pi1=N11/(N10+N11)if(N10+N11)>0 else 0
    pi=(N01+N11)/(N00+N11+N10+N01)
    eps=1e-10
    pi0=max(min(pi0,1-eps),eps)
    pi1=max(min(pi1,1-eps),eps)
    #l_ind=np.log(1-p)*(N00+N10)+np.log(p)*(N01+N11)
    l_ind=np.log(1-pi)*(N00+N10)+np.log(pi)*(N01+N11)
    l_markov=(np.log(1-pi0)*N00+
              np.log(pi0)*N01+
              np.log(1-pi1)*N10+
              np.log(pi1)*N11)
    lr_ind=-2*(l_ind-l_markov)
    lr_cc=lr_ind+ku
    p_val=1-scp.chi2.cdf(lr_cc,df=2)
    #return (p_val,lr_ind,l_ind,l_markov,ku)
    if p_val<alpha:
        return("Reject")
    else:
        return("Fail to reject")