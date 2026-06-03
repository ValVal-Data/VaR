import numpy as np
import scipy.stats as scp
import pandas as pan
import torch as tr
import time as ti
import sklearn as sk

def clusterData(r,coded,eps,samplehdbscan,sampledbscan):
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

class Autoencoder(tr.nn.Module):
    def __init__(self,dimensions:list,do:float)->None:
        super().__init__()
        dim=[int(x) for x in dimensions]
        self.encoder=tr.nn.Sequential(
            tr.nn.Linear(dim[0],dim[1]),
            tr.nn.Tanh(),
            tr.nn.Dropout(p=do),
            tr.nn.Linear(dim[1],dim[2]),
            tr.nn.Tanh()
        )
        self.decoder=tr.nn.Sequential(
            tr.nn.Linear(dim[2],dim[1]),
            tr.nn.Tanh(),
            tr.nn.Linear(dim[1],dim[1]),
            tr.nn.Tanh(),
            tr.nn.Dropout(p=do),
            tr.nn.Linear(dim[1],dim[0])
        )

    def forward(self,x):
        y=self.encode(x)
        return self.decode(y)
    
    def encode(self,x):
        return self.encoder(x)
    
    def decode(self,x):
        return self.decoder(x)



class TickData(tr.utils.data.Dataset):
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
            self.reg=np.max(abs(self.tar[i]))
            self.tar[i]/=self.reg
    def __len__(self)->int:
        return self.tar.shape[1]-1
    def __getitem__(self,idx:int)->list:
        x=self.tar[:,idx]
        x=x.reshape(-1)
        return tr.tensor(x),tr.tensor(x)

def log_square_returns(x,eps=1e-8):
    return np.log(x**2+eps)

def train_loop(dataLoad:tr.utils.data.DataLoader,model,loss_fn,optimizer,scale=1):
    model.train()
    train_loss=0
    for x,y in dataLoad:
        pred=model(x)
        loss=loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss+=loss.item()
    return train_loss*scale

def test_loop(dataLoad:tr.utils.data.DataLoader,model,loss_fn,scale=100):
    model.eval()
    test_loss=0
    with tr.torch.no_grad():
        for x,y in dataLoad:
            pred=model(x)
            test_loss+=loss_fn(pred,y).item()
    return test_loss*scale

def split_data(data,ratio_train,ratio_test,batch_size):
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

def random_opti(histVar,number,batch_size,midlayer,h,learning_rate,dropout,loss_fn,ratio_train,ratio_test,epoch,early_lim,early_ratio,early_rate,print_rate,si,fe):
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
        va["Model"]=Autoencoder(ml[i],do[i]).to("cpu")
        va["Data"]=(hv[i],bs[i])
        va["Optimizer"]=tr.optim.Adam(va["Model"].parameters(),lr=lr[i])
        models.append(va)

    #Initialize data
    da={}
    uu=np.unique(np.array([hv,bs]).transpose(),axis=0)
    for i in uu:
        da[tuple(i)]={}
        data=TickData(logHistVar,i[0])
        train_loader,test_loader,valid_loader=split_data(data,ratio_train,ratio_test,int(i[1]))
        da[tuple(i)]["Train"]=train_loader
        da[tuple(i)]["Test"]=test_loader
        da[tuple(i)]["Valid"]=valid_loader

    #Train
    #id=np.array([x for x in range(0,number,1)])
    id=np.ones(number,dtype=bool)
    err=np.zeros((epoch,number,2))
    st=ti.time()
    best=np.inf
    for i in range(epoch):
        if i%print_rate==0:
            print(f"Epoch: {i+1}/{epoch}")
            print(f"Model remaining: {id.sum()}/{number}")
        for j,mod_id in enumerate(id):
            if mod_id==1:
                err[i,j,0]=train_loop(da[models[j]["Data"]]["Train"],models[j]["Model"],loss_fn,models[j]["Optimizer"])
                err[i,j,1]=test_loop(da[models[j]["Data"]]["Valid"],models[j]["Model"],loss_fn)
                if i>0:
                    if err[i,j,1]>err[i-1,j,1]:#Overfitting
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
                        continue
                    rate=(err[i,j,1]-err[i-early_lim,j,1])/err[i-early_lim,j,1]
                    if err[i,j,1]>early_ratio*best:#Loss worst than x percent of the current best
                        id[j]=False
                    elif rate>0:
                        id[j]=False
                    elif -rate<early_rate:#Rate of improvement close to zero
                        id[j]=False
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
        
    #Test
    final=np.zeros(number)
    for x,mod in enumerate(models):
        final[x]=test_loop(da[mod["Data"]]["Test"],mod["Model"],loss_fn)

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

    param=[bs,ml[:,-1],ml[:,1],hv,do,lr,r2,final]
    return {"Model":models,"Test":final,"Valid":err,"Param":param,"r2":r2}

def gen_mode(h,ms,ls):
    po=np.array([(x,x//y,z) for x in h for y in ms for z in ls])
    msk=np.all(np.diff(po,axis=1)<0,axis=1)
    po=po[msk]
    return po