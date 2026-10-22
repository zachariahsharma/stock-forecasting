"""Fresh portfolio demonstration runs, 2026-10-22; not historical checkpoints."""
from pathlib import Path
import json, time, hashlib, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoformerConfig, AutoformerForPrediction

OUT=Path(__file__).resolve().parent/'results'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(42);np.random.seed(42);random.seed(42)
device='cuda'
source=Path.home()/'old/content/data/BigTestDataset/CLOSEDATAFINAL.feather'
values=pd.read_feather(source,columns=['AAPL'])['AAPL'].dropna().to_numpy(dtype=np.float32)[-3000:]
n=len(values);a=int(n*.70);b=int(n*.85);window=20
lo,hi=float(values[:a].min()),float(values[:a].max());scale=hi-lo
norm=(values-lo)/scale
np.savez(OUT/'price-data.npz',prices=values,train_end=a,val_end=b,lookback=window)
def make_ds(normalized=True,leaked=False,history=20):
    v=norm if normalized else values
    xx=np.stack([v[t-history:t].copy() for t in range(history,n)])
    yy=v[history:].copy();ids=np.arange(history,n)
    if leaked:xx[:,-1]=yy
    xx=torch.tensor(xx[:,:,None],device=device);yy=torch.tensor(yy[:,None],device=device)
    return [(xx[mask],yy[mask]) for mask in [ids<a,(ids>=a)&(ids<b),ids>=b]]

class MLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Flatten(),nn.Linear(20,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):return self.net(x)
class GRU(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(1,32,2,batch_first=True);self.head=nn.Linear(32,1)
    def forward(self,x):return self.head(self.gru(x)[0][:,-1])
class CNNBiGRU(nn.Module):
    def __init__(self):
        super().__init__();self.cnn=nn.Sequential(nn.Conv1d(1,16,3,padding=1),nn.BatchNorm1d(16),nn.ReLU(),nn.Conv1d(16,32,3,padding=2,dilation=2),nn.BatchNorm1d(32),nn.ReLU(),nn.Conv1d(32,32,3,padding=4,dilation=4),nn.BatchNorm1d(32),nn.ReLU());self.gru=nn.GRU(32,32,batch_first=True,bidirectional=True);self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,1))
    def forward(self,x):return self.head(self.gru(self.cnn(x.transpose(1,2)).transpose(1,2))[0][:,-1])

meta={'created':'2026-10-22','kind':'fresh demonstration runs, not historical results','gpu':torch.cuda.get_device_name(0),'seed':42,'source':'~/old/content/data/BigTestDataset/CLOSEDATAFINAL.feather','ticker':'AAPL','observations':n,'price_series_sha256':hashlib.sha256(values.tobytes()).hexdigest(),'split':{'train':[0,a],'validation':[a,b],'test':[b,n]},'normalization':{'fit':'training prices only','min':lo,'max':hi},'lookback':window,'notes':['Models and hyperparameters are small reconstructions of project stages, not exact archived checkpoints.','Each target is strictly after its input window except the intentionally leaked V3 demonstration.','Validation/test windows may use earlier observed history; all targets stay in their chronological split.','Validation used for checkpoint selection; test used only after selection.','No financial performance or profitability evaluation.'],'models':{}}

def fit(name,cls,normalized=True,leaked=False,epochs=65):
    torch.manual_seed(42);m=cls().to(device);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    ds=make_ds(normalized,leaked);train,val,test=ds
    best=float('inf');state=None;hist=[];start=time.time()
    for epoch in range(epochs):
        m.train();perm=torch.randperm(len(train[0]),device=device)
        for ids in perm.split(128):
            opt.zero_grad(set_to_none=True);loss=((m(train[0][ids])-train[1][ids])**2).mean();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
        m.eval()
        with torch.no_grad():
            tr=float(((m(train[0])-train[1])**2).mean());va=float(((m(val[0])-val[1])**2).mean())
        hist.append({'epoch':epoch+1,'train_mse':tr,'val_mse':va,'train_mse_usd2':tr*(scale**2 if normalized else 1),'val_mse_usd2':va*(scale**2 if normalized else 1)})
        if va<best:best=va;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if (epoch+1)%20==0:print(name,'epoch',epoch+1,'val',round(va,5),flush=True)
    m.load_state_dict(state);m.eval()
    with torch.no_grad():pred=m(test[0]).cpu().numpy().ravel()
    if normalized:pred=pred*scale+lo
    actual=values[b:];mae=float(np.abs(pred-actual).mean());baseline=float(np.abs(values[b-1:-1]-actual).mean())
    result={'history':hist,'test_prediction':pred.tolist(),'test_actual':actual.tolist(),'test_mae_usd':mae,'persistence_mae_usd':baseline,'checkpoint_epoch':int(np.argmin([h['val_mse'] for h in hist])+1),'seconds':round(time.time()-start,2),'normalized':normalized,'intentionally_leaked':leaked}
    (OUT/(name+'.json')).write_text(json.dumps(result,indent=2));meta['models'][name]={k:v for k,v in result.items() if k not in ['history','test_prediction','test_actual']};(OUT/'metadata.json').write_text(json.dumps(meta,indent=2));print(name,'done',result['seconds'],'seconds',flush=True)

for args in [('v1_mlp_raw',MLP,False,False),('v2_mlp_scaled',MLP,True,False),('v3_gru_leaked',GRU,True,True),('v4_gru_corrected',GRU,True,False),('v5_cnn_bigru',CNNBiGRU,True,False)]:fit(*args)

# Genuine Hugging Face Autoformer, reduced dimensions for a short demonstration.
torch.manual_seed(42)
c=AutoformerConfig(prediction_length=1,context_length=20,input_size=1,lags_sequence=[1,2,3],num_time_features=0,num_static_categorical_features=0,encoder_layers=1,decoder_layers=1,d_model=16,encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,label_length=10,scaling='std',num_parallel_samples=100,dropout=.1)
m=AutoformerForPrediction(c).to(device);opt=torch.optim.AdamW(m.parameters(),lr=.0005)
train,val,test=make_ds(True,False,23);hist=[];best=float('inf');state=None;start=time.time()
def af(x,y=None):
    kw=dict(past_values=x.squeeze(-1),past_time_features=x.new_empty(len(x),23,0),past_observed_mask=torch.ones_like(x.squeeze(-1)),future_time_features=x.new_empty(len(x),1,0))
    if y is not None:kw['future_values']=y
    return kw
for epoch in range(35):
    m.train();perm=torch.randperm(len(train[0]),device=device)
    for ids in perm.split(128):
        opt.zero_grad(set_to_none=True);loss=m(**af(train[0][ids],train[1][ids])).loss;loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    m.eval()
    with torch.no_grad():tr=float(m(**af(*train)).loss);va=float(m(**af(*val)).loss)
    hist.append({'epoch':epoch+1,'train_nll':tr,'val_nll':va})
    if va<best:best=va;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    if (epoch+1)%5==0:print('v6_autoformer epoch',epoch+1,'val NLL',round(va,4),flush=True)
m.load_state_dict(state);m.eval();pred=[]
with torch.no_grad():
    for x in test[0].split(64):pred.extend(m.generate(**af(x)).sequences.median(dim=1).values[:,0].cpu().numpy().tolist())
pred=np.asarray(pred)*scale+lo
result={'history':hist,'test_prediction':pred.tolist(),'test_actual':values[b:].tolist(),'test_mae_usd':float(np.abs(pred-values[b:]).mean()),'persistence_mae_usd':float(np.abs(values[b-1:-1]-values[b:]).mean()),'checkpoint_epoch':int(np.argmin([h['val_nll'] for h in hist])+1),'seconds':round(time.time()-start,2),'config':c.to_dict()}
(OUT/'v6_autoformer.json').write_text(json.dumps(result,indent=2));meta['models']['v6_autoformer']={k:v for k,v in result.items() if k not in ['history','test_prediction','test_actual','config']};(OUT/'metadata.json').write_text(json.dumps(meta,indent=2));print('ALL RUNS COMPLETE',flush=True)
