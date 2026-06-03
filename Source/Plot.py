#Contain all function needed to plot

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scp
import pandas as pan
import sklearn as sk
from Source.Mlp import *
from Source.StatTest import *

def plotProb(coded,ncluster):
    center=sk.cluster.KMeans(n_clusters=ncluster,random_state=0).fit(coded).cluster_centers_
    prob=prob_cluster(coded,center)
    plt.plot(prob)
    plt.xlabel("6 month rolling window")
    plt.ylabel("Stress score")
    plt.show()

def plotClusterDim(la,coded):
    fig,ax=plt.subplots(4,4,figsize=(10,6))
    tsne=sk.manifold.TSNE(n_components=2,random_state=0).fit_transform(coded)
    for i in range(len(la)):
        ax[i,0].scatter(coded[:,0],coded[:,1],c=la[i],cmap='viridis')
        ax[i,0].set_xlabel("Dimension 1")
        ax[i,0].set_ylabel("Dimension 2")
        ax[i,1].scatter(coded[:,0],coded[:,2],c=la[i],cmap='viridis')
        ax[i,1].set_xlabel("Dimension 1")
        ax[i,1].set_ylabel("Dimension 3")
        ax[i,2].scatter(coded[:,1],coded[:,2],c=la[i],cmap='viridis')
        ax[i,2].set_xlabel("Dimension 2")
        ax[i,2].set_ylabel("Dimension 3")
        ax[i,3].scatter(tsne[:,0],tsne[:,1],c=la[i],cmap='viridis')
        ax[i,3].set_xlabel("t-SNE 1")
        ax[i,3].set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()

def plotClusterScores(r,coded):
    silhouette,ch,db=clusterData(r,coded,0.01,10,10)
    fig,ax=plt.subplots(1,3,figsize=(12,3))
    ax[0].plot(r,silhouette)
    ax[0].set_xlabel("Number of clusters")
    ax[0].set_ylabel("Silhouette Score")
    ax[0].legend(["K-Means","Gaussian Mixture","Spectral Clustering","DBSCAN","HDBSCAN","Agglomerative"])
    ax[1].plot(r,ch)
    ax[1].set_xlabel("Number of clusters")
    ax[1].set_ylabel("Calinski Harabasz Score")
    ax[2].plot(r,db)
    ax[2].set_xlabel("Number of clusters")
    ax[2].set_ylabel("Davies Bouldin Score")
    plt.tight_layout()
    plt.show()

def plotClusterMean(histVar,la,title,mopti):
    ret=np.array(histVar.rolling(int(mopti["Data"][0])).mean().dropna())[:-1]
    vol=np.array(histVar.rolling(int(mopti["Data"][0])).std().dropna())[:-1]
    r1=[]
    v1=[]
    anova=[]
    for i in range(len(la)):
        r1.append([ret[~la[i].astype(bool)].mean(axis=1),ret[la[i].astype(bool)].mean(axis=1)])
        v1.append([vol[~la[i].astype(bool)].mean(axis=1),vol[la[i].astype(bool)].mean(axis=1)])
        anova.append([scp.f_oneway(r1[-1][0],r1[-1][1]).pvalue,scp.f_oneway(v1[-1][0],v1[-1][1]).pvalue])
    fig,ax=plt.subplots(2,len(la),figsize=(10,4))
    plt.subplots_adjust(wspace=0,hspace=0)
    for i in range(len(la)):
        ax[0,i].set_title(title[i])
        ax[0,i].violinplot([r1[i][0],r1[i][1]],showmeans=True)
        ax[0,i].set_ylim(0.995,1.01)
        ax[0,i].set_xticks([1,2],["Cluster 1","Cluster 2"])
        ax[0,i].text(2,1.009,f"p-value = {anova[i][0]:.1e}",va="top",ha="right")
        ax[1,i].violinplot([v1[i][0],v1[i][1]],showmeans=True)
        ax[1,i].set_ylim(0.008,0.03)
        ax[1,i].set_xticks([1,2],["Cluster 1","Cluster 2"])
        ax[1,i].text(2,0.0285,f"p-value = {anova[i][1]:.1e}",va="top",ha="right")
        if i>0:
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
            ax[1,i].set_yticks([])
        else:
            ax[0,i].set_ylabel("Return")
            ax[1,i].set_ylabel("Volatility")
    plt.show()

def plotEPS(coded):
    nbnn=sk.neighbors.NearestNeighbors(n_neighbors=5).fit(coded)
    dist,id=nbnn.kneighbors(coded)
    k_dist=np.sort(dist[:,4])
    plt.plot(k_dist)
    plt.show()

def plotParamNN(name,opti):
    lon=len(opti["Param"])
    fig,ax=plt.subplots(lon,lon,figsize=(16,16))
    plt.subplots_adjust(wspace=0,hspace=0)
    for i in range(lon):
        for j in range(lon):
            ax[j,i].scatter(opti["Param"][i],opti["Param"][j],c=np.log10(opti["Test"]),cmap='viridis')
            if 0<j<lon-1 and 0<i<lon-1:
                ax[j,i].set_xticks([])
                ax[j,i].set_yticks([])
                continue
            if j>0:
                ax[j,i].set_xticks([])
            if i>0:
                ax[j,i].set_yticks([])
            if j==0:
                ax[j,i].set_xlabel(name[i])
                ax[j,i].xaxis.set_ticks_position("top")
                ax[j,i].xaxis.set_label_position("top")
            if i==0:
                ax[j,i].set_ylabel(name[j])
    plt.show()

def plotLatent(coded):
    msd=np.zeros(len(coded)-1)
    for i in range(len(coded)-1):
        msd[i]=np.mean((coded[i]-coded[i+1])**2)
    fig,ax=plt.subplots(1,4,figsize=(12,3))
    tsne=sk.manifold.TSNE(n_components=2,random_state=0).fit_transform(coded)
    sc1=ax[0].scatter(coded[:,0],coded[:,1],c=coded[:,2],cmap='viridis')
    ax[0].set_xlabel("Dimension 1")
    ax[0].set_ylabel("Dimension 2")
    cb=fig.colorbar(sc1,ax=ax[0])
    cb.set_label("Dimension 3")
    ax[1].plot(msd)
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Mean Squared displacement")
    ax[2].hist(msd,bins=40)
    ax[2].set_ylim(0,50)
    ax[2].set_ylabel("Nb")
    ax[2].set_xlabel("Mean Squared displacement")
    ax[3].scatter(tsne[:,0],tsne[:,1])
    ax[3].set_xlabel("t-SNE 1")
    ax[3].set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()

def plotNNProgess(opti):
    progress=opti["Valid"][:,:,1].copy()
    progress[progress==0]=np.nan
    fig,ax=plt.subplots(1,2,figsize=(16,6))
    for i in range(progress.shape[1]):
        ax[0].plot(progress[:,i])
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    for i in range(progress.shape[1]):
        ax[1].plot(progress[:,i])
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].set_ylim(0,10)
    plt.show()

def plotPriceEvolution(data:pan.DataFrame,title:str)->None:
    """
    Plot price evolution in % change (starting at 0)

    Parameters:
        data (pan.DataFrame): Data frame witch each column being an asset over time
        title (str): Title of the plot

    Returns:
        None, plot the figure directly
    """
    plt.plot(data/list(data.iloc[0])*100-100)#Normalize so they all start at 0
    plt.xlabel("Date")
    plt.ylabel("Evolution (+%)")
    plt.legend(data.columns)
    plt.title(title)
    plt.show()

def plotBar(data1:np.array,data2:np.array,col:list,lim:list,name:str,lab1:list,lab2:list,title:str)->None:
    """
    Two Bar Plots side by side to compare different values

    Parameters:
        data1 (np.array): Data to plot on the left
        data2 (np.array): Data to plot on the right
        col (list): The color scheme
        lim (tuples(2)): Limit for the y axis
        name (str): Label for each column
        lab1 (list): Y axis label for left plot
        lab2 (list): Y axis label for right plot
        title (str): Title of the plot
    
    Returns:
        None, plot the figure directly
    """
    fig,ax=plt.subplots(1,2,figsize=(10,3))
    fig.suptitle(title)
    ax[0].bar(name,data1,color=col)
    ax[0].set_xticks(range(len(name)))
    ax[0].set_xticklabels(name,rotation=45)
    ax[0].set_ylabel(lab1)
    ax[0].set_ylim(lim)
    ax[1].bar(name,data2,color=col)
    ax[1].set_ylabel(lab2)
    ax[1].set_xticks(range(len(name)))
    ax[1].set_xticklabels(name,rotation=45)
    ax[1].set_ylim(lim)
    plt.show()

def plotLineCompare(data1:np.array,data2:np.array,xlab:str,ylab:str,title1:list,title2:list,lim:tuple,titleG:str)->None:
    """
    Two line Plots side by side to compare different values

    Parameters:
        data1 (np.array): Data to plot on the left
        data2 (np.array): Data to plot on the right
        xlab (str): X axis label
        ylab (str): Y axis label
        title1 (list): Title for left plot
        title2 (list): Title for right plot
        lim (tuples(2)): Limit for the y axis
        titleG (str): Title of the figure
    
    Returns:
        None, plot the figure directly
    """
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    fig.suptitle(titleG)
    ax[0].plot(data1)
    ax[0].set_xlabel(xlab)
    ax[0].set_ylabel(ylab)
    ax[0].set_ylim(lim)
    ax[0].set_title(title1)
    ax[0].legend(data1.columns,loc="lower left")
    ax[1].plot(data2)
    ax[1].set_xlabel(xlab)
    ax[1].set_ylabel(ylab)
    ax[1].set_title(title2)
    ax[1].set_ylim(lim)
    ax[1].legend(data2.columns,loc="lower left")
    plt.show()

def checkVisualHeteroskedasticity(data:pan.DataFrame,title:str)->None:
    """
    List line plot with rolling average return to check visually heteroskedasticity

    Parameters:
        data (pan.DataFrame): Data to plot
        title (str): Title of the plot
    
    Returns:
        None, plot the figure directly
    """
    rollingVol=data.rolling(window=48).std()
    rollingVol=rollingVol.dropna()
    fig,ax=plt.subplots(1,2,figsize=(10,3))
    fig.suptitle(title)
    ax[0].plot(data**2)
    ax[0].set_xlabel("Date")
    ax[0].set_ylabel("Squared Return ")
    ax[0].legend(data.columns,loc="upper left")
    ax[1].plot(rollingVol)
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("1-year Rolling volatility")
    ax[1].set_ylim(0,8)
    ax[1].set_xticks(rollingVol.index[::62])
    ax[1].set_xticklabels([d.strftime('%Y') for d in rollingVol.index[::62]])
    ax[1].legend(data.columns,loc="upper left")
    plt.show()


def plotCorrMap(corr:pan.DataFrame,logHistVar:pan.DataFrame)->None:
    """
    Plot correlation map

    Parameters:
        corr (pan.DataFrame): Data to plot
        logHistVar (pan.DataFrame): dataframe from where the map was extracted to get legend
    
    Returns:
        None, plot the figure directly
    """
    corrmap=pan.DataFrame(corr,columns=logHistVar.columns,index=logHistVar.columns)
    fig,ax=plt.subplots(figsize=(6,6))
    cax=ax.imshow(corrmap,cmap='coolwarm',vmin=-1,vmax=1)

    ax.set_xticks(np.arange(len(logHistVar.columns)))
    ax.set_yticks(np.arange(len(logHistVar.columns)))

    ax.set_xticklabels(logHistVar.columns)
    ax.set_yticklabels(logHistVar.columns)

    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    cb=fig.colorbar(cax)
    cb.set_label("Correlation",fontsize=12)
    plt.show()

def plotCompareCorrMap(corr:pan.DataFrame,corr2:pan.DataFrame,logHistVar:pan.DataFrame)->None:
    """
    Plot 2 correlation map side by side to compare

    Parameters:
        corr (pan.DataFrame): Data to plot on the left
        corr2 (pan.DataFrame): Data to plot on the right
        logHistVar (pan.DataFrame): dataframe from where the map was extracted to get legend
    
    Returns:
        None, plot the figure directly
    """
    corrmap=pan.DataFrame(corr,columns=logHistVar.columns,index=logHistVar.columns)
    corrmap2=pan.DataFrame(corr2,columns=logHistVar.columns,index=logHistVar.columns)
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    fig.suptitle("Correlation maps used")
    fig.subplots_adjust(wspace=0.6)
    cax=ax[0].imshow(corrmap,cmap='coolwarm',vmin=-1,vmax=1)

    ax[0].set_xticks(np.arange(len(logHistVar.columns)))
    ax[0].set_yticks(np.arange(len(logHistVar.columns)))
    ax[0].set_title("Model correlation map")
    ax[1].set_title("Stress correlation map")

    ax[0].set_xticklabels(logHistVar.columns)
    ax[0].set_yticklabels(logHistVar.columns)

    cax2=ax[1].imshow(corrmap2,cmap='coolwarm',vmin=-1,vmax=1)

    ax[1].set_xticks(np.arange(len(logHistVar.columns)))
    ax[1].set_yticks(np.arange(len(logHistVar.columns)))

    ax[1].set_xticklabels(logHistVar.columns)
    ax[1].set_yticklabels(logHistVar.columns)   

    plt.setp(ax[0].get_xticklabels(),rotation=45,ha="right")
    plt.setp(ax[1].get_xticklabels(),rotation=45,ha="right")

    cb=fig.colorbar(cax2,orientation='horizontal',ax=ax,fraction=0.05,pad=0.3)
    cb.set_label("Correlation",fontsize=12)
    plt.show()

def plotResidual(srr:pan.DataFrame,colors:list,title:str,bin:int=20)->None:
    """
    Plot residual from GARCH

    Parameters:
        srr (pan.DataFrame): Data to plot
        colors (list): List of the color for the different lines
        title (str): Title of the plot
        bin (int, default=20): Number of bins

    Returns:
        None, plot the figure directly
    """
    fig,ax=plt.subplots(1,5,figsize=(10,3))
    fig.suptitle(title)
    for i in range(len(srr.columns)):  
        ax[i].hist(srr.iloc[:,i],bins=bin,density=True,alpha=1,color=colors[i])
        ax[i].set_title(srr.columns[i])
        ax[i].set_xlim((-max(abs(srr.iloc[:,1])),max(abs(srr.iloc[:,1]))))
        ax[i].set_title(srr.columns[i])
        ax[i].set_xlabel("Residual")
        ax[i].set_ylabel("Density")
    plt.tight_layout()
    plt.show()

def plotDistrib(srr:pan.DataFrame,colors:list,title:str,lim:tuple,bin:int=20)->None:
    """
    Plot residual from GARCH

    Parameters:
        srr (pan.DataFrame): Data to plot
        colors (list): List of the color for the different lines
        title (str): Title of the plot
        li, (tuple): Limit for the x axis
        bin (int, default=20): Number of bins

    Returns:
        None, plot the figure directly
    """
    fig,ax=plt.subplots(1,5,figsize=(10,3))
    fig.suptitle(title)
    for i in range(len(srr.columns)):  
        ax[i].hist(srr.iloc[:,i],bins=bin,density=True,alpha=1,color=colors[i],range=lim)
        ax[i].set_title(srr.columns[i])
        ax[i].set_xlim(lim)
        ax[i].set_title(srr.columns[i])
        ax[i].set_xlabel("Returns (%)")
        ax[i].set_ylabel("Density")
    plt.tight_layout()
    plt.show()

def plotCumulativeExcept(histVarP:pan.DataFrame,t:list,alpha:float=0.025)->None:
    """
    Plot residual from GARCH

    Parameters:
        histVarP (pan.DataFrame): Data to plot
        t (list): List of the thresholds for exceptions
        alpha (float, default=0.025): 1-Confidence level
    
    Returns:
        None, plot the figure directly
    """
    #Calculate cumulative exception rate
    v=[]
    for i in range(len(histVarP.columns)):
        v.append((histVarP.iloc[:,i] < -t[i]).astype(int))
    c=[np.cumsum(x) for x in v]
    c=pan.concat(c,axis=1)
    c.columns=histVarP.columns
    #Calculate expected exception rate
    alpha=0.025
    tt=np.arange(1,len(histVarP)+1)
    ex=tt*alpha
    c["Expected"]=ex
    #Plot
    colors2=plt.rcParams["axes.prop_cycle"].by_key()["color"][:6]
    colors2[-1]='#222222'
    ls=["-"]*5+["--"]
    for i in range(len(c.columns)):
        plt.plot(c.iloc[:,i],color=colors2[i],linestyle=ls[i])
    plt.legend(c.columns)
    plt.xlabel("Date")
    plt.ylabel("Cumulative exception")
    plt.title("Cumulative exception over time")
    plt.show()