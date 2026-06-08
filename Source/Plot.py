#Contain all function needed to plot

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as scp
import pandas as pan
import sklearn as sk
from Source.Mlp import *
from Source.StatTest import *

#Colors
colAccent=["#2AA198"]
colLines=["#0B2239","#234A7D","#4F6FA8"]
colLines2=["#234A7D","#4F6FA8","#0B2239"]
colGray=["#2A2A2A","#7A7A7A","#E5E5E5"]
colBack=["#E6EAF0","#EEF1F5","#F6F7F9"]
scheme=(plt.cycler(color=3*colLines2)+plt.cycler(linestyle=3*["-"]+3*["--"]+3*[":"]))
schemeGray=(plt.cycler(color=3*colGray)+plt.cycler(linestyle=3*["-"]+3*["--"]+3*[":"]))
schemeAccent=plt.cycler(color=colAccent+1000000*colLines2)
cmap=LinearSegmentedColormap.from_list("my_gradient",[colLines[-2],colAccent[-1]])
cmap2=LinearSegmentedColormap.from_list("my_gradient",[colBack[-1],colLines[-2]])
cmap3=LinearSegmentedColormap.from_list("my_gradient",[colLines[-2],colBack[0],colAccent[-1]])
#General
plt.rcParams["font.family"]="Segoe UI"
plt.rcParams["axes.titleweight"]="regular"
plt.rcParams["axes.titlesize"]=12
plt.rcParams["text.color"]=colLines[0]
plt.rcParams["axes.labelcolor"]=colLines[0]
plt.rcParams["axes.edgecolor"]=colLines[0]
plt.rcParams["xtick.color"]=colLines[0]
plt.rcParams["ytick.color"]=colLines[0]
plt.rcParams["axes.prop_cycle"]=scheme
#Label
plt.rcParams["axes.labelsize"]=12
plt.rcParams["axes.labelweight"]="regular"
#Tickes
plt.rcParams["xtick.labelsize"]=10
plt.rcParams["ytick.labelsize"]=10
#Background
plt.rcParams["figure.facecolor"]=colBack[-1]
plt.rcParams["axes.facecolor"]=colBack[-1]

#Functions
def customSuptitle(text,fig, **kwargs):
    return fig.suptitle(text,fontfamily="cambria",fontweight="bold",size=16,**kwargs)

def customTitle(text, **kwargs):
    return plt.title(text,fontfamily="cambria",fontweight="bold",size=16,**kwargs)

def plotProb(coded,ncluster,msd,thr):
    fig,ax=plt.subplots()
    gmm=sk.mixture.GaussianMixture(n_components=ncluster,covariance_type="full")
    gmm.fit(coded)
    prob=gmm.predict_proba(coded)
    ax.plot(prob[:,0])
    ax.set_xlabel("6 month rolling window")
    ax.set_ylabel("Stress score")
    for i in np.where(msd>thr)[0]:
        ax.axvline(x=i,color='black',alpha=0.4,linestyle="--")
    """ax.axhline(y=0.5,color='black')
    ax2=ax.twinx()
    #ax2.plot(msd,color="gray",alpha=0.3)
    ax2.set_ylabel("Mean Squared Displacement",color="gray")
    ax2.fill_between(np.arange(0,len(msd)),msd,0,color="gray",alpha=0.3)"""
    plt.show()

def plotClusterDim(la,coded,title):
    fig,ax=plt.subplots(1,len(la),figsize=(11,3))
    if coded.shape[1]==2:
        tsne=coded
        lab=("Autoencoder 1","Autoencoder 2")
    else:
        tsne=sk.manifold.TSNE(n_components=2,random_state=0).fit_transform(coded)
        lab=("t-SNE 1","t-SNE 2")
    plt.subplots_adjust(wspace=0,hspace=0)
    for i in range(len(la)):
        ax[i].scatter(tsne[:,0],tsne[:,1],c=la[i],cmap=cmap3)
        ax[i].set_xlabel(lab[0])
        if i==0:
            ax[i].set_ylabel(lab[1])
        else:
            ax[i].set_yticks([])
        ax[i].set_title(title[i])
    plt.show()

def plotClusterScores(r,coded,ddbscan,min_db):
    silhouette,ch,db=clusterData(r,coded,ddbscan,min_db,min_db)
    fig,ax=plt.subplots(1,3,figsize=(12,3))
    ax[0].plot(r,silhouette)
    ax[0].set_xlabel("Number of clusters")
    ax[0].set_ylabel("Silhouette Score")
    ax[0].legend(["K-Means","Gaussian Mixture","Spectral Clustering","HDBSCAN","DBSCAN","Agglomerative"])
    ax[1].plot(r,ch)
    ax[1].set_xlabel("Number of clusters")
    ax[1].set_ylabel("Calinski Harabasz Score")
    ax[2].plot(r,db)
    ax[2].set_xlabel("Number of clusters")
    ax[2].set_ylabel("Davies Bouldin Score")
    plt.tight_layout()
    plt.show()

def plotClusterMean(histVar,la,title,win,lim1,lim2,p=0.05):
    ret=np.array(histVar.rolling(win).mean().dropna())[:-1]
    vol=np.array(histVar.rolling(win).std().dropna())[:-1]
    r1=[]
    v1=[]
    anova=[]
    for i in range(len(la)):
        rr=[]
        vv=[]
        for j in np.unique(la[0]):
            mask=la[i]==j
            rr.append(ret[mask].mean(axis=1))
            vv.append(vol[mask].mean(axis=1))
        r1.append(rr)
        v1.append(vv)
    for i in range(len(r1)):
        an=[]
        for j in range(len(r1[i])):
            ann=[]
            for l in range(len(r1[i])):
                ann.append([scp.f_oneway(r1[i][j],r1[i][l]).pvalue,scp.f_oneway(v1[i][j],v1[i][l]).pvalue])
            an.append(ann)
        tmp=np.array(an)
        anova.append((tmp<=p).astype(int))
    fig,ax=plt.subplots(2,len(la),figsize=(10,4))
    plt.subplots_adjust(wspace=0,hspace=0)
    for i in range(len(la)):
        cl=len(np.unique(la[i]))
        ax[0,i].set_title(title[i])
        ax[0,i].violinplot(r1[i],showmeans=True)
        ax[0,i].set_ylim(lim1)
        ax[1,i].violinplot(v1[i],showmeans=True)
        ax[1,i].set_ylim(lim2)
        ax[1,i].set_xticks(range(1,cl+1),[f"Cluster {x+1}" for x in range(cl)])
        plt.setp(ax[1,i].get_xticklabels(),rotation=45)
        if i>0:
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
            ax[1,i].set_yticks([])
        else:
            ax[0,i].set_ylabel("Return")
            ax[1,i].set_ylabel("Volatility")
    plt.show()
    fig,ax=plt.subplots(2,len(la),figsize=(9,4.5))
    plt.subplots_adjust(wspace=0,hspace=0)
    for i in range(len(anova)):
        ax[0,i].set_title(title[i])
        ax[0,i].imshow(anova[i][:,:,0],cmap=cmap3,vmin=-1,vmax=1)
        ax[0,i].set_xticks([])
        ax[1,i].imshow(anova[i][:,:,1],cmap=cmap3,vmin=-1,vmax=1)
        ax[1,i].set_xticks(range(cl),[f"Cluster {x+1}" for x in range(cl)])
        ax[1,i].set_yticks(range(cl),[f"Cluster {x+1}" for x in range(cl)])
        ax[0,i].set_yticks(range(cl),[f"Cluster {x+1}" for x in range(cl)])
        plt.setp(ax[1,i].get_xticklabels(),rotation=45)
        if i>0:
            ax[0,i].set_yticks([])
            ax[1,i].set_yticks([])
        else:
            ax[0,i].set_ylabel("Return")
            ax[1,i].set_ylabel("Volatility")
    plt.show()

def plotEPS(coded,nb):
    nbnn=sk.neighbors.NearestNeighbors(n_neighbors=nb).fit(coded)
    dist,id=nbnn.kneighbors(coded)
    k_dist=np.sort(dist[:,nb-1])
    plt.plot(k_dist)
    plt.xlabel("Neighbor")
    plt.ylabel("Distance")
    plt.show()

def plot3D(coded):
    fig,ax=plt.subplots(1,3,figsize=(10,4))
    ax[0].scatter(coded[:,0],coded[:,1],c=np.arange(0,len(coded)),cmap=cmap)
    ax[0].set_xlabel("Autoencoder 1")
    ax[0].set_ylabel("Autoencoder 2")
    ax[1].scatter(coded[:,0],coded[:,2],c=np.arange(0,len(coded)),cmap=cmap)
    ax[1].set_ylabel("Autoencoder 1")
    ax[1].set_xlabel("Autoencoder 3")
    ax[2].scatter(coded[:,1],coded[:,2],c=np.arange(0,len(coded)),cmap=cmap)
    ax[2].set_ylabel("Autoencoder 2")
    ax[2].set_xlabel("Autoencoder 3")
    plt.tight_layout()
    plt.show()

def plotLatent(coded,nb):
    nbnn=sk.neighbors.NearestNeighbors(n_neighbors=nb).fit(coded)
    dist,id=nbnn.kneighbors(coded)
    k_dist=np.sort(dist[:,nb-1])
    msd=np.zeros(len(coded)-1)
    vel=np.zeros(len(coded)-1)
    for i in range(len(coded)-1):
        msd[i]=np.mean((coded[i]-coded[i+1])**2)
        vel[i]=np.linalg.norm(coded[i]-coded[i+1])
    acc=vel[1:]-vel[:-1]
    fig,ax=plt.subplots(1,5,figsize=(15,4))
    if coded.shape[1]==2:
        tsne=coded
        lab=("Autoencoder 1","Autoencoder 2")
    else:
        tsne=sk.manifold.TSNE(n_components=2,random_state=0).fit_transform(coded)
        lab=("t-SNE 1","t-SNE 2")
    ax[0].plot(msd)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Mean Squared displacement")
    ax[1].plot(vel)
    ax[1].set_ylabel("First-order displacement")
    ax[1].set_xlabel("Time")
    ax[2].plot(acc)
    ax[2].set_ylabel("Second-order displacement")
    ax[2].set_xlabel("Time")
    ax[3].scatter(tsne[:,0],tsne[:,1],c=np.arange(0,len(tsne)),cmap=cmap3)
    ax[3].set_xlabel(lab[0])
    ax[3].set_ylabel(lab[1])
    ax[4].plot(k_dist)
    ax[4].set_xlabel("Neighbor")
    ax[4].set_ylabel("Distance")
    plt.tight_layout()
    plt.show()
    return msd,vel,acc

def plotParamNN(name,opti):
    lon=len(opti["Param"])
    fig,ax=plt.subplots(lon,lon,figsize=(16,16))
    plt.subplots_adjust(wspace=0,hspace=0)
    for i in range(lon):
        for j in range(lon):
            ax[j,i].scatter(opti["Param"][i],opti["Param"][j],c=np.log10(opti["Test"]),cmap=cmap3)
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

def plotNNProgess(opti,perc):
    progress=opti["Valid"][:,:,1].copy()
    progress[progress==0]=np.nan
    top=np.zeros(progress.shape[1])
    for i in range(len(top)):
        tmp=progress[~np.isnan(progress[:,i]),i]
        top[i]=tmp[-1]
    top=np.argsort(top)
    progress=progress[:,top]
    fig,ax=plt.subplots(1,2,figsize=(16,6))
    ax[0].set_prop_cycle(schemeGray)
    ax[1].set_prop_cycle(schemeGray)
    for i in range(1,progress.shape[1]):
        ax[0].plot(progress[:,i])
    ax[0].plot(progress[:,0],color=colAccent[0],linestyle="-",linewidth=4)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    for i in range(1,progress.shape[1]):
        ax[1].plot(progress[:,i])
    ax[1].plot(progress[:,0],color=colAccent[0],linestyle="-",linewidth=4)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].set_ylim(np.nanmin(progress)*(0.99),np.nanmin(progress)*(1+perc))
    plt.show()
    
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
    customSuptitle(title,fig)
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
        ax[j].fill_between(x,pdfr,color=col[i],alpha=0.6)
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
    customSuptitle(title,fig)
    for i in range(len(data.columns)):
        if i>=3:
            j=(1,i-3)
        else:
            j=(0,i)
        
        osm,osr=scp.probplot(data.iloc[:,i],dist="norm")[:2]
        th_q,sam_q=osm
        slope,inter,r=osr
        ax[j].scatter(th_q,sam_q,color=col[i],alpha=0.6)
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
    customSuptitle(title,fig)
    for i in range(len(data.columns)):
        osm,osr=scp.probplot(data.iloc[:,i],dist="norm")[:2]
        th_q,sam_q=osm
        slope,inter,r=osr
        ax[i].scatter(th_q,sam_q,color=col[i],alpha=0.6)
        x_line=th_q
        y_line=slope*x_line+inter
        ax[i].plot(x_line,y_line,color="black",linestyle="--")
        
        ax[i].set_title(data.columns[i])
        ax[i].set_xlabel("Normal distribution")
        ax[i].set_ylabel("Residual distribution")
    plt.tight_layout()
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
    customTitle(title)
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
    customSuptitle(title,fig)
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
    customSuptitle(titleG,fig)
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
    customSuptitle(title,fig)
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
    cax=ax.imshow(corrmap,cmap=cmap,vmin=-1,vmax=1)

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
    customSuptitle("Correlation maps used",fig)
    fig.subplots_adjust(wspace=0.6)
    cax=ax[0].imshow(corrmap,cmap=cmap3,vmin=-1,vmax=1)

    ax[0].set_xticks(np.arange(len(logHistVar.columns)))
    ax[0].set_yticks(np.arange(len(logHistVar.columns)))
    ax[0].set_title("Model correlation map")
    ax[1].set_title("Stress correlation map")

    ax[0].set_xticklabels(logHistVar.columns)
    ax[0].set_yticklabels(logHistVar.columns)

    cax2=ax[1].imshow(corrmap2,cmap=cmap3,vmin=-1,vmax=1)

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
    customSuptitle(title,fig)
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
    customSuptitle(title,fig)
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
    for i in range(len(c.columns)):
        if c.columns[i]=="Expected":
            plt.plot(c.iloc[:,i],color="black",linestyle=":")
        else:
            plt.plot(c.iloc[:,i])
    plt.legend(c.columns)
    plt.xlabel("Date")
    plt.ylabel("Cumulative exception")
    customTitle("Cumulative exception over time")
    plt.show()