import numpy as np
import scipy.stats as scp
import pandas as pan
import torch as tr
import time as ti
import sklearn as sk
from typing import Callable,TypeVar

F=TypeVar("F",bound=Callable[...,object])

class TickData(tr.utils.data.Dataset):
    """
    Tick data for Variational autoencoder. Calculate rolling window: return, volatility, skewedness, kurtosis, cumulative return and z-score

    Parameters:
        pan (pan.DataFrame): Historical return
        window_size (int): Size of the rolling window

    Attributes:
        data (pan.DataFrame): Historical return
        window (int): Size of the rolling window
        tar (np.array): Normalized features

    """
    def __init__(self,df:pan.DataFrame,window_size:int)->None:
        self.data=df
        self.window=window_size
        self.tar=log_square_returns(self.data)
        me=np.zeros((len(self.data)-self.window+1,len(self.data.columns)))
        vol=np.zeros((len(self.data)-self.window+1,len(self.data.columns)))
        ske=np.zeros((len(self.data)-self.window+1,len(self.data.columns)))
        kur=np.zeros((len(self.data)-self.window+1,len(self.data.columns)))
        cumret=np.zeros((len(self.data)-self.window+1,len(self.data.columns)))
        dd=np.zeros((len(self.data)-self.window+1,len(self.data.columns)))
        z=np.zeros((len(self.data)-self.window+1,len(self.data.columns)))
        for id,elem in enumerate(self.data.columns):
            me[:,id]=self.data[elem].rolling(self.window).mean().dropna()
            vol[:,id]=self.data[elem].rolling(self.window).std().dropna()
            ske[:,id]=self.data[elem].rolling(self.window).skew().dropna()
            kur[:,id]=self.data[elem].rolling(self.window).kurt().dropna()
            cumret[:,id]=(1+self.data[elem]).rolling(self.window).apply(lambda x: x.prod()-1).dropna()
            cum_w=(1+self.data[elem]).cumprod()
            rolmax=cum_w.rolling(self.window).max()
            #dd[:,id]=(cum_w/rolmax-1).dropna()
            z[:,id]=((self.data[elem].rolling(self.window).mean()-self.data[elem].mean())/self.data[elem].std()).dropna()
        self.tar=np.array([me,vol,ske,kur,cumret,z]).astype(np.float32)
        #Normalize data
        for i in range(self.tar.shape[0]):
            self.tar[i]-=np.nanmin(self.tar[i])
            self.tar[i]/=np.max(self.tar[i])
    def __len__(self)->int:
        return self.tar.shape[1]-1
    def __getitem__(self,idx:int)->list:
        x=self.tar[:,idx]
        x=x.reshape(-1)
        return tr.tensor(x),tr.tensor(x)

class VariationalAutoencoder(tr.nn.Module):
    """
    Variational autoencoder

    Parameters:
        dimensions (list): input layer, hidden layer and latent space size
        do (float): Dropout value

    Attributes:
        l1 (tr.nn.Linear): Input linear layer of the encoder
        dropout (tr.nn.Dropout): Dropout linear layer of the encoder
        l_mu (tr.nn.Linear): Mu layer of the encoder
        l_logvar (tr.nn.Linear): Logvar layer of the encoder
        l2 (tr.nn.Linear): Hidden linear layer of the decoder
        l3 (tr.nn.Linear): Output linear layer of the decoder

    """
    def __init__(self,dimensions:list,do:float)->None:
        super().__init__()
        dim=[int(x) for x in dimensions]
        self.l1=tr.nn.Linear(dim[0],dim[1])
        self.dropout=tr.nn.Dropout(p=do)
        self.l_mu=tr.nn.Linear(dim[1],dim[2])
        self.l_logvar=tr.nn.Linear(dim[1],dim[2])

        self.l2=tr.nn.Linear(dim[2],dim[1])
        self.l3=tr.nn.Linear(dim[1],dim[0])
    
    def forward(self,x:np.array)->list:
        """
        Forward function of the VAE

        Parameters:
            x (np.array): Data

        Returns:
            result (list): Reconstructed, mu, logvar
        """
        mu,logvar=self.encode(x)
        z=self.reparametrize(mu,logvar)
        recon=self.decode(z)
        return recon,mu,logvar
    
    def encode(self,x:np.array)->list:
        """
        Encode function of the VAE

        Parameters:
            x (np.array): Data

        Returns:
            result (list): Mu, logvar
        """
        h=tr.nn.functional.relu(self.l1(x))
        h=self.dropout(h)
        mu=self.l_mu(h)
        logvar=self.l_logvar(h)
        return mu,logvar
    
    def decode(self,x)->tr.tensor:
        """
        Decode function of the VAE

        Parameters:
            x (np.array): Data

        Returns:
            result (tr.tensor)
        """
        h=self.l2(x)
        return tr.torch.sigmoid(self.l3(h))

    def reparametrize(self,mu:tr.tensor,logvar:tr.tensor):
        """
        Reparametrize function of the VAE

        Parameters:
            mu (tr.tensor): Mean
            logvar (tr.tensor): Log of the variance

        Returns:
            result (tr.tensor)
        """
        std=tr.torch.exp(0.5*logvar)
        eps=tr.torch.randn_like(std)
        return mu+eps*std

def clusterData(r:np.array,coded:np.array,eps:float,samplehdbscan:float,sampledbscan:int)->list:
    """
    Cluster with k-means, Gaussian mixture, Spectral clustering, DBSCAN and HDBSCAN. Compute Silouhette, Calinsky Harabasz and Davies Bouldin scores as function of number of cluster

    Parameters:
        r (np.array): List of the different cluster to test
        coded (np.array): Latent space
        eps (float): EPS for DBSCAN
        samplehdbscan (float): Min cluster size for HDBSCAN
        sampledbscan (int): Min samples for DBSCAN

    Returns:
        silhouette (list): Silouhette score
        ch (list): Calinsky Harabasz score
        db (list): Davies Bouldin score
    """
    silhouette=np.zeros((len(r),6))
    ch=np.zeros((len(r),6))
    db=np.zeros((len(r),6))
    for j,i in enumerate(r):
        km=sk.cluster.KMeans(n_clusters=i,random_state=0).fit_predict(coded)
        gmm=sk.mixture.GaussianMixture(n_components=i).fit_predict(coded)
        spe=sk.cluster.SpectralClustering(n_clusters=i).fit_predict(coded)
        ag=sk.cluster.AgglomerativeClustering(n_clusters=i).fit_predict(coded)
        silhouette[j,0]=sk.metrics.silhouette_score(coded,km)
        silhouette[j,1]=sk.metrics.silhouette_score(coded,gmm)
        silhouette[j,2]=sk.metrics.silhouette_score(coded,spe)
        silhouette[j,5]=sk.metrics.silhouette_score(coded,ag)
        ch[j,0]=sk.metrics.calinski_harabasz_score(coded,km)
        ch[j,1]=sk.metrics.calinski_harabasz_score(coded,gmm)
        ch[j,2]=sk.metrics.calinski_harabasz_score(coded,spe)
        ch[j,5]=sk.metrics.calinski_harabasz_score(coded,ag)
        db[j,0]=sk.metrics.davies_bouldin_score(coded,km)
        db[j,1]=sk.metrics.davies_bouldin_score(coded,gmm)
        db[j,2]=sk.metrics.davies_bouldin_score(coded,spe)
        db[j,5]=sk.metrics.davies_bouldin_score(coded,ag)
    hdbscan=sk.cluster.HDBSCAN(min_cluster_size=samplehdbscan,copy=True).fit_predict(coded)
    dbscan=sk.cluster.DBSCAN(min_samples=sampledbscan,eps=eps).fit_predict(coded)
    silhouette[len(np.unique(hdbscan)),3]=sk.metrics.silhouette_score(coded,hdbscan)
    silhouette[len(np.unique(dbscan)),4]=sk.metrics.silhouette_score(coded,dbscan)
    ch[len(np.unique(hdbscan)),3]=sk.metrics.calinski_harabasz_score(coded,hdbscan)
    ch[len(np.unique(dbscan)),4]=sk.metrics.calinski_harabasz_score(coded,dbscan)
    db[len(np.unique(hdbscan)),3]=sk.metrics.davies_bouldin_score(coded,hdbscan)
    db[len(np.unique(dbscan)),4]=sk.metrics.davies_bouldin_score(coded,dbscan)
    return silhouette, ch, db
    
def vae_loss(recon_x:tr.tensor,x:tr.tensor,mu:tr.tensor,logvar:tr.tensor)->float:
    """
    Loss function for variational autoencoder

    Parameters:
        recon_x (tr.tensor): Reconstructed x
        x (tr.tensor): Real x
        mu (tr.tensor): Mu of the VAE
        logvar (tr.tensor): Logvar of the VAE

    Returns:
        loss (tr.tensor)
    """
    recon_loss=tr.nn.functional.binary_cross_entropy(recon_x,x,reduction='sum')
    kl=-0.5*tr.torch.sum(1+logvar-mu**2-logvar.exp())
    return recon_loss+kl


def log_square_returns(x:np.array,eps:float=1e-8)->np.array:
    """
    Log return with eps to prevent log(0)

    Parameters:
        x (np.array): Returns
        eps=1e-8 (np.array): Offset to prevent log(0)

    Returns:
        loss (np.array)
    """
    return np.log(x**2+eps)

def train_loop(dataLoad:tr.utils.data.DataLoader,model:VariationalAutoencoder,loss_fn:F,optimizer:tr.optim,scale:int=1)->np.array:
    """
    Loop to train Variational Autoencoder

    Parameters:
        dataLoad (tr.utils.data.DataLoader): Data loader to train model on
        model (VariationalAutoencoder): Model to optimize
        loss_fn (F): Loss function
        optimizer (tr.optim): Optimizer
        scale=1 (int): Parameter to scale loss

    Returns:
        loss (np.array)
    """
    model.train()
    train_loss=0
    for x,y in dataLoad:
        pred,mu,logvar=model(x)
        loss=loss_fn(pred,y,mu,logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    return train_loss*scale

def test_loop(dataLoad:tr.utils.data.DataLoader,model:VariationalAutoencoder,loss_fn:F,scale:int=1):
    """
    Loop to test Variational Autoencoder

    Parameters:
        dataLoad (tr.utils.data.DataLoader): Data loader to train model on
        model (VariationalAutoencoder): Model to optimize
        loss_fn (F): Loss function
        scale=1 (int): Parameter to scale loss

    Returns:
        loss (np.array)
    """
    model.eval()
    test_loss=0
    with tr.torch.no_grad():
        for x,y in dataLoad:
            pred,mu,logvar=model(x)
            test_loss+=loss_fn(pred,y,mu,logvar).item()
    return test_loss*scale

def split_data(data,ratio_train:float,ratio_test:float,batch_size:int)->list:
    """
    Split data into train, test and validation set

    Parameters:
        data (TickData): Data to split
        ratio_train (float): Ratio of train
        ratio_test (float): Ratio of test
        batch_size (int): Batch size

    Returns:
        loader (list): Train, test, valid
    """
    n=len(data)
    train_end=int(ratio_train*n)
    test_end=int(ratio_test*n)
    train=tr.utils.data.Subset(data,range(0,train_end))
    test=tr.utils.data.Subset(data,range(train_end,test_end))
    valid=tr.utils.data.Subset(data,range(test_end,n))
    train_loader=tr.utils.data.DataLoader(train,batch_size=batch_size,shuffle=True)
    test_loader=tr.utils.data.DataLoader(test,batch_size=batch_size,shuffle=False)
    valid_loader=tr.utils.data.DataLoader(valid,batch_size=batch_size,shuffle=False)
    return train_loader,test_loader,valid_loader

def random_opti(histVar:pan.DataFrame,number:int,batch_size:list,midlayer:list,h:list,learning_rate:list,dropout:list,loss_fn:F,ratio_train:float,ratio_test:float,epoch:int,early_lim:int,early_ratio:float,early_rate:float,print_rate:int,si:int,fe:int)->dict:
    """
    Loop to do random search for hyperparameter of the Variational Autoencoder

    Parameters:
        histVar (pan.DataFrame): Returns as function of time
        number (int): Number of model to optimize in parallel
        batch_size (int): Batch size
        midlayer(list): List of model architectures
        h (list): List of rolling window sizes
        learning_rate (list): List of learning rates
        dropout (list): List of dropout
        loss_fn (F): Loss function
        ratio_train (float): Ratio of train
        ratio_test (float): Ratio of test
        epoch (int): Maximal number of epochs
        early_lim (int): Number of train loop to do before evaluation
        early_ratio (float): Maximal acceptable ratio of loss compared to the best one
        early_rate (int): Minimal acceptable improvement rate
        print_rate (int): Rate of progress print
        si (int): Number of assets
        fe (int): Number of features

    Returns:
        result (dict): Weights, Model, Test, Valid, Param, r2
    """
    #Generate random values
    bs=np.random.choice(batch_size,size=number)
    mli=np.random.choice(len(midlayer),size=number)
    ml=midlayer[mli]
    hv=np.random.choice(h,size=number)
    do=np.random.uniform(dropout[0],dropout[1],size=number)
    lr=10**np.random.uniform(learning_rate[0],learning_rate[1],size=number)

    #Initialize models
    models=[]
    for i in range(number):
        va={}
        va["Model"]=VariationalAutoencoder(ml[i],do[i]).to("cpu")
        va["Data"]=(hv[i],bs[i],ml[i])
        va["Weights"]=va["Model"].state_dict()
        va["Optimizer"]=tr.optim.Adam(va["Model"].parameters(),lr=lr[i])
        models.append(va)

    #Initialize data
    da={}
    uu=np.unique(np.array([hv,bs]).transpose(),axis=0)
    for i in uu:
        da[tuple(i)]={}
        data=TickData(histVar,i[0])
        train_loader,test_loader,valid_loader=split_data(data,ratio_train,ratio_test,int(i[1]))
        da[tuple(i)]["Train"]=train_loader
        da[tuple(i)]["Test"]=test_loader
        da[tuple(i)]["Valid"]=valid_loader

    #Train
    #id=np.array([x for x in range(0,number,1)])
    id=np.ones(number,dtype=bool)
    removedId=np.ones(number,dtype=bool)
    err=np.zeros((epoch,number,2))
    st=ti.time()
    best=np.inf
    for i in range(epoch):
        if i%print_rate==0:
            print(f"Epoch: {i+1}/{epoch}")
            print(f"Model remaining: {id.sum()}/{number}")
        for j,mod_id in enumerate(id):
            if mod_id==1:
                err[i,j,0]=train_loop(da[models[j]["Data"][:2]]["Train"],models[j]["Model"],loss_fn,models[j]["Optimizer"])
                err[i,j,1]=test_loop(da[models[j]["Data"][:2]]["Valid"],models[j]["Model"],loss_fn)
                if i>0:
                    if err[i,j,1]>err[i-1,j,1]:#Overfitting validation
                        id[j]=0
                    elif err[i,j,0]>err[i-1,j,0]:#Train loss increase
                        id[j]=0
        #Kill bad performer
        if i%early_lim==0 and i>0:#Kill based on criterion
            mask=err[i,:,1]>0
            if best>np.nanmin(err[i,mask,1]):
                best=np.nanmin(err[i,mask,1])
            if i%print_rate==0:
                print(f"Best: {best:.2e}")
            for j,mod_id in enumerate(id):  
                if mod_id:
                    if np.isnan(err[i,j,1]):#Nan value for loss function
                        id[j]=False
                        removedId[j]=False
                        err[i,j]=np.zeros(2)
                        continue
                    rate=(err[i,j,1]-err[i-early_lim,j,1])/err[i-early_lim,j,1]
                    if err[i,j,1]>early_ratio*best:#Loss worst than x percent of the current best
                        id[j]=False
                        removedId[j]=False
                        err[i,j]=np.zeros(2)
                    elif rate>0:
                        id[j]=False
                        err[i,j]=np.zeros(2)
                    elif -rate<early_rate:#Rate of improvement close to zero
                        id[j]=False
                        err[i,j]=np.zeros(2)
                    else:
                        models[j]["Weights"]=models[j]["Model"].state_dict()
        if id.sum()==0:#Kill if there is nothing to optimize anymore
            print("\nAll models converged or got killed")
            break
        if i>0 and i%print_rate==0:
            improve=[-(err[i,x,1]-err[i-1,x,1]) for x in np.where(id)[0]]
            print(f"Mean improvement: {np.nanmean(improve)/np.nanmean(err[i-1,:,1])*100:.1f}%")
        if i%print_rate==0:
            en=ti.time()
            tt=en-st
            st=ti.time()
            if tt>60:
                tt/=60
                print(f"Time: {tt:.1f}m\n")
            else:
                print(f"Time: {tt:.1f}s\n")
    models=np.array(models)[removedId]   
    err=err[:,removedId]

    #Test
    final=np.zeros(len(models))
    for x,mod in enumerate(models):
        final[x]=test_loop(da[mod["Data"][:2]]["Test"],mod["Model"],loss_fn)

    #Calculate R2
    print("\nCalculating R2")
    r2=np.zeros(len(models))
    for nb,i in enumerate(models):
        mod=i["Model"]
        data=TickData(histVar,int(i["Data"][0]))
        data_loader=tr.utils.data.DataLoader(data)
        mod.eval()
        pr=np.zeros((len(data),fe*si))
        tar=np.zeros((len(data),fe*si))
        with tr.torch.no_grad():
            for id,(x,y) in enumerate(data_loader):
                pr[id]=mod(x)[0]
                tar[id]=y
        ssr=np.sum((tar-pr)**2)
        sst=np.sum((tar-tar.mean())**2)
        r2[nb]=1-ssr/sst

    param=[bs[removedId],ml[removedId,-1],ml[removedId,1],hv[removedId],do[removedId],lr[removedId],r2,final]
    return {"Weights":[x["Weights"] for x in models],"Model":[x["Data"] for x in models],"Test":final,"Valid":err,"Param":param,"r2":r2}

def gen_mode(h:int,ms:float,ls:int)->np.array:
    """
    Generate possible architecture for Variational Autoencoder

    Parameters:
        h (int): Size of the input layer
        ms (float): Ratio of division for the hidden layer
        ls (int): Latent space size

    Returns:
        model (np.array): List of possible architecture
    """
    po=np.array([(x,x//y,z) for x in h for y in ms for z in ls])
    msk=np.all(np.diff(po,axis=1)<0,axis=1)
    po=po[msk]
    return po