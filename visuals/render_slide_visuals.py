"""Render scientific plots and explanatory diagrams from recorded fresh runs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.ticker import MaxNLocator, LogLocator, NullLocator, FuncFormatter

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results'
OUT=ROOT/'slide-visuals'
for sub in ['wide','square']:
    for stem in ['03-v3-target-leakage','05-v5-cnn-bigru','05b-cnn-bigru-loss','06-v6-autoformer','06b-autoformer-loss']:
        for suffix in ['.png','.svg']:(OUT/sub/(stem+suffix)).unlink(missing_ok=True)
BLUE='#147bc1';ORANGE='#dc7736';INK='#171b21';GRAY='#76808a';PALE='#dcebf7';RED='#d35749';GREEN='#278674'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':19,'axes.labelsize':19,'xtick.labelsize':17,'ytick.labelsize':17,'axes.edgecolor':'#8d949b','axes.linewidth':1,'text.color':INK,'axes.labelcolor':INK,'xtick.color':GRAY,'ytick.color':GRAY,'svg.fonttype':'none'})
records={p.stem:json.loads(p.read_text()) for p in DATA.glob('v*.json')}
prices=np.load(DATA/'price-data.npz')['prices']
meta=json.loads((DATA/'metadata.json').read_text())

def start(title,square=False,badge='NEW RUN'):
    fig=plt.figure(figsize=(6,6) if square else (9,4.8),dpi=200,facecolor='none')
    fig.text(.045,.94,title,fontsize=25 if square else 29,weight='bold',va='top')
    fig.text(.95,.86 if square else .925,badge,ha='right',va='top',fontsize=11,weight='bold',color=GRAY)
    return fig
def save(fig,slug,square=False):
    d=OUT/('square' if square else 'wide');d.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']:fig.savefig(d/(slug+'.'+ext),transparent=True)
    plt.close(fig)
def chart(fig,square=False):
    ax=fig.add_axes([.27,.17,.68,.60] if square else [.18,.17,.77,.61])
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16,color=GRAY);ax.set_axisbelow(True)
    ax.tick_params(length=0,pad=9);return ax
def log_ticks(ax,ys):
    ys=np.asarray(ys);ys=ys[np.isfinite(ys)&(ys>0)]
    def fmt(v,pos):
        return (f'{v:.1e}'.replace('e-0','e-').replace('e+0','e')) if v<.001 or v>=1000 else f'{v:.2g}'
    ax.set_yscale('log');ax.set_yticks(np.geomspace(ys.min(),ys.max(),3));ax.yaxis.set_major_formatter(FuncFormatter(fmt));ax.yaxis.set_minor_locator(NullLocator())
def loss(slug,name,title,square=False,nll=False):
    fig=start(title,square);ax=chart(fig,square);h=records[name]['history'];x=[r['epoch'] for r in h]
    a='train_nll' if nll else 'train_mse';b='val_nll' if nll else 'val_mse'
    ax.plot(x,[r[a] for r in h],lw=3.5,color=BLUE,label='Train')
    ax.plot(x,[r[b] for r in h],lw=3.5,color=ORANGE,label='Validation')
    if not nll:log_ticks(ax,[r[a] for r in h]+[r[b] for r in h])
    else:ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_locator(MaxNLocator(3,integer=True));ax.set_xlabel('Epoch',labelpad=10);ax.set_ylabel('NLL' if nll else 'MSE',labelpad=10)
    ax.legend(loc='upper right',frameon=False,fontsize=14 if square else 18,ncol=2,bbox_to_anchor=(1,1.1 if square else 1.18),handlelength=1.3,columnspacing=.9)
    save(fig,slug,square)

def box(ax,xy,w,h,text,color=BLUE,fontsize=22,fill=True):
    r=FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.012,rounding_size=.025',facecolor=color if fill else 'none',edgecolor=color,lw=2)
    ax.add_patch(r);ax.text(xy[0]+w/2,xy[1]+h/2,text,color='white' if fill else color,fontsize=fontsize,ha='center',va='center',weight='bold');return r
def arrow(ax,a,b,color=GRAY):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=22,lw=2,color=color))
def diagram(title,square=False):
    fig=start(title,square,'SCHEMATIC');ax=fig.add_axes([.04,.07,.92,.75]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');return fig,ax

def mlp(slug,square=False):
    fig,ax=diagram('MLP',square)
    xs=[.1,.38,.66,.91];nums=[4,5,5,1];nodes=[]
    for x,n in zip(xs,nums):nodes.append([(x,float(y)) for y in np.linspace(.30,.80,n)])
    for left,right in zip(nodes,nodes[1:]):
        for a in left:
            for b in right:ax.plot([a[0],b[0]],[a[1],b[1]],color=BLUE,alpha=.21,lw=1.7)
    for j,layer in enumerate(nodes):
        for xy in layer:ax.add_patch(Circle(xy,.027,facecolor=ORANGE if j==3 else BLUE,edgecolor='none'))
    for x,t in zip(xs,['Prices','Hidden','Hidden','Price']):ax.text(x,.15,t,ha='center',fontsize=17 if square else 22,weight='bold')
    save(fig,slug,square)

def normalize(slug,square=False):
    fig=start('Min-Max scaling',square,'DATA DEMO')
    end=meta['split']['train'][1];v=prices[end-60:end];scaled=(v-v.min())/(v.max()-v.min())
    # One observed training window, shown before and after scaling.
    top=.68 if square else .77
    if square:
        positions=[[.19,.50,.75,.23],[.19,.15,.75,.23]]
    else:positions=[[.1,.25,.35,.48],[.61,.25,.35,.48]]
    for i,(pos,data,label,color) in enumerate(zip(positions,[v,scaled],['Raw prices ($)','Scaled 0–1'],[GRAY,BLUE])):
        ax=fig.add_axes(pos);ax.plot(data,lw=4,color=color);ax.set_title(label,fontsize=20 if square else 24,weight='bold',pad=15);ax.spines[['top','right','bottom']].set_visible(False);ax.set_xticks([]);ax.yaxis.set_major_locator(MaxNLocator(2));ax.tick_params(length=0,pad=7)
    fig.text(.5,.045 if square else .10,'Same shape, different range',ha='center',fontsize=19 if square else 23,color=GRAY)
    save(fig,slug,square)

def leakage(slug,square=False):
    fig,ax=diagram('Target leakage',square)
    for y,title,leaked in [(.65,'Answer inside input',True),(.23,'Answer kept separate',False)]:
        ax.text(.02,y+.20,title,fontsize=19 if square else 24,weight='bold',color=RED if leaked else GREEN)
        box(ax,(.03,y),.56,.13,'Past prices',BLUE,17 if square else 23)
        if leaked:
            box(ax,(.59,y),.20,.13,'Answer',RED,16 if square else 22);ax.plot([.01,.81],[y-.035,y-.035],color=RED,lw=3);ax.text(.40,y-.10,'Model input',ha='center',fontsize=16 if square else 19,color=GRAY)
        else:
            arrow(ax,(.61,y+.065),(.73,y+.065),GREEN);box(ax,(.76,y),.20,.13,'Target',GREEN,16 if square else 22);ax.plot([.01,.61],[y-.035,y-.035],color=BLUE,lw=3);ax.text(.31,y-.10,'Model input',ha='center',fontsize=16 if square else 19,color=GRAY)
    save(fig,slug,square)

def leakage_loss(slug,square=False):
    fig=start('Leakage experiment',square);ax=chart(fig,square)
    for name,label,color in [('v3_gru_leaked','Leaked inputs',RED),('v4_gru_corrected','Future target',GREEN)]:
        h=records[name]['history'];ax.plot([r['epoch'] for r in h],[r['val_mse'] for r in h],label=label,color=color,lw=3.5)
    log_ticks(ax,[r['val_mse'] for name in ['v3_gru_leaked','v4_gru_corrected'] for r in records[name]['history']]);ax.xaxis.set_major_locator(MaxNLocator(3,integer=True));ax.set_xlabel('Epoch');ax.set_ylabel('Validation MSE')
    ax.legend(frameon=False,fontsize=12 if square else 18,loc='upper right',bbox_to_anchor=(1,1.1 if square else 1.18),ncol=2,handlelength=1.1,columnspacing=.7);save(fig,slug,square)

def forecast(slug,square=False):
    fig=start('GRU forecast',square);ax=chart(fig,square);r=records['v4_gru_corrected'];n=90
    ax.plot(r['test_actual'][-n:],color=INK,lw=3,label='Actual')
    ax.plot(r['test_prediction'][-n:],color=BLUE,lw=3.5,label='GRU')
    ax.set_xlabel('Held-out steps',labelpad=10);ax.set_ylabel('Price ($)',labelpad=10);ax.yaxis.set_major_locator(MaxNLocator(3));ax.xaxis.set_major_locator(MaxNLocator(3,integer=True));ax.legend(frameon=False,ncol=2,fontsize=16 if square else 19,loc='upper left',bbox_to_anchor=(0,1.17),handlelength=1.2,columnspacing=.8)
    save(fig,slug,square)

def backtest(slug,square=False):
    r=json.loads((DATA/'archived-backtest.json').read_text())
    fig=start('Backtest result',square,'ORIGINAL LOG');ax=chart(fig,square)
    x=[v['step'] for v in r['series']];y=[v['portfolio_value'] for v in r['series']]
    ax.fill_between(x,y,0,color=BLUE,alpha=.12);ax.plot(x,y,color=BLUE,lw=4)
    ax.scatter([x[0],x[-1]],[y[0],y[-1]],color=BLUE,s=45,zorder=5)
    ax.annotate(f'${y[0]:,.0f}',(x[0],y[0]),xytext=(7,12),textcoords='offset points',fontsize=18,weight='bold')
    ax.annotate(f'${y[-1]:,.0f}',(x[-1],y[-1]),xytext=(-4,17),ha='right',textcoords='offset points',fontsize=23 if square else 28,weight='bold',color=BLUE)
    ax.set_ylim(0,90000);ax.set_yticks([0,40000,80000]);ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:'0' if v==0 else f'{v/1000:.0f}k'));ax.xaxis.set_major_locator(MaxNLocator(3,integer=True));ax.set_ylabel('Portfolio ($)');ax.set_xlabel('Backtest step',labelpad=10)
    fig.text(.95,.82 if square else .845,'Unvalidated simulation',ha='right',fontsize=12,color=RED)
    save(fig,slug,square)

def pipeline(slug,kind='cnn',square=False):
    title='CNN + BiGRU' if kind=='cnn' else 'GRU sequence model'
    fig,ax=diagram(title,square)
    if kind=='cnn':
        if square:
            for y,t,col in [(.78,'20 prices',GRAY),(.52,'3× Conv1D',BLUE),(.26,'⇄ BiGRU',GREEN),(.02,'Next price',ORANGE)]:box(ax,(.15,y),.7,.14,t,col,22)
            for y in [.74,.48,.22]:arrow(ax,(.5,y),(.5,y-.055))
        else:
            for x,w,t,col in [(.01,.19,'Prices',GRAY),(.28,.22,'3× CNN',BLUE),(.58,.21,'⇄ GRU',GREEN),(.87,.12,'Price',ORANGE)]:box(ax,(x,.45),w,.22,t,col,21)
            for a,b in [(.21,.26),(.51,.56),(.80,.85)]:arrow(ax,(a,.56),(b,.56))
            ax.text(.39,.28,'Local features',ha='center',fontsize=22,color=GRAY);ax.text(.68,.28,'Sequence',ha='center',fontsize=22,color=GRAY)
    else:
        for i in range(4):
            x=.06+i*.235;box(ax,(x,.45),.15,.20,'GRU',BLUE,17 if square else 23)
            ax.text(x+.075,.27,f'Day {i+1}',fontsize=15 if square else 20,ha='center',color=GRAY)
            if i<3:arrow(ax,(x+.16,.55),(x+.22,.55))
        ax.text(.5,.82,'Remember earlier prices',ha='center',fontsize=20 if square else 27,weight='bold')
    save(fig,slug,square)

def autoformer(slug,square=False):
    fig,ax=diagram('Autoformer',square)
    x=np.linspace(0,1,100);trend=.20*x+.05*np.sin(x*3);season=.055*np.sin(x*26)
    if square:
        ax.plot(.05+x*.70,.79+trend*.3,color=ORANGE,lw=4);ax.text(.05,.70,'Trend',color=ORANGE,fontsize=22,weight='bold')
        arrow(ax,(.78,.81),(.94,.81),ORANGE);arrow(ax,(.94,.78),(.94,.23),ORANGE)
        ax.plot(.05+x*.70,.54+season,color=BLUE,lw=3);ax.text(.05,.41,'Remainder',color=BLUE,fontsize=22,weight='bold')
        arrow(ax,(.42,.36),(.42,.29));box(ax,(.03,.14),.73,.14,'Autocorrelation',GREEN,19);arrow(ax,(.79,.21),(.87,.21));ax.text(.94,.20,'+',fontsize=28,ha='center',va='center',weight='bold');ax.text(.53,.01,'Combine → forecast',fontsize=19,ha='center',color=GRAY)
    else:
        ax.plot(.02+x*.29,.70+trend*.45,color=ORANGE,lw=4);ax.text(.02,.58,'Trend',color=ORANGE,fontsize=23,weight='bold')
        ax.plot(.02+x*.29,.29+season*1.2,color=BLUE,lw=3);ax.text(.02,.13,'Remainder',color=BLUE,fontsize=23,weight='bold')
        arrow(ax,(.34,.77),(.91,.77),ORANGE);arrow(ax,(.92,.74),(.92,.53),ORANGE)
        arrow(ax,(.34,.30),(.43,.35));box(ax,(.45,.23),.31,.25,'Auto-\ncorrelation',GREEN,23);arrow(ax,(.78,.36),(.87,.46));ax.text(.92,.47,'+',fontsize=30,weight='bold',ha='center',va='center');arrow(ax,(.92,.42),(.92,.30));ax.plot(.87+x*.12,.16+trend*.4+season*.4,color=BLUE,lw=4);ax.text(.93,.03,'Forecast',ha='center',fontsize=20,weight='bold')
    save(fig,slug,square)

main=[('01-v1-mlp-loss',lambda s:loss('01-v1-mlp-loss','v1_mlp_raw','MLP loss',s)),('02-v2-normalization',lambda s:normalize('02-v2-normalization',s)),('03-v3-backtest-results',lambda s:backtest('03-v3-backtest-results',s)),('04-v4-gru-forecast',lambda s:forecast('04-v4-gru-forecast',s)),('05-v5-cnn-bigru-loss',lambda s:loss('05-v5-cnn-bigru-loss','v5_cnn_bigru','CNN-BiGRU loss',s)),('06-v6-autoformer-loss',lambda s:loss('06-v6-autoformer-loss','v6_autoformer','Autoformer loss',s,True))]
for _,fn in main:fn(False);fn(True)
mlp('01b-mlp-architecture',True)
loss('02b-normalized-mlp-loss','v2_mlp_scaled','Normalized MLP loss',True)
leakage_loss('03b-leakage-comparison',True)
loss('04b-corrected-gru-loss','v4_gru_corrected','Corrected GRU loss',True)
pipeline('05b-cnn-bigru-architecture','cnn',True)
autoformer('06b-autoformer-architecture',True)

archive=DATA/'archived-2024-loss.json'
if archive.exists():
    original=json.loads(archive.read_text())
    for square in [False,True]:
        fig=start('Archived loss · 2024',square,'ORIGINAL LOG');ax=chart(fig,square)
        for tag,label,color in [('Loss/train','Train',BLUE),('Loss/Test','Logged test',ORANGE)]:
            rows=original['series'][tag];ax.plot([r['step'] for r in rows],[r['value'] for r in rows],lw=3,color=color,label=label)
        log_ticks(ax,[r['value'] for rows in original['series'].values() for r in rows]);ax.xaxis.set_major_locator(MaxNLocator(3,integer=True));ax.set_xlabel('Logged step');ax.set_ylabel('Logged loss');ax.legend(frameon=False,fontsize=13 if square else 18,loc='upper right',bbox_to_anchor=(1,1.1 if square else 1.18),ncol=2,handlelength=1.2,columnspacing=.8);save(fig,'bonus-original-2024-loss',square)

# A slide-size overview: each wide image is shown at the intended aspect ratio.
from PIL import Image,ImageDraw,ImageFont
canvas=Image.new('RGB',(1900,1800),'#ffffff');draw=ImageDraw.Draw(canvas)
for i,(slug,_) in enumerate(main):
    im=Image.open(OUT/'wide'/(slug+'.png')).convert('RGBA');im.thumbnail((900,480));x=35+(i%2)*950;y=30+(i//2)*600
    tile=Image.new('RGBA',im.size,'#e5e5e5');tile.alpha_composite(im);canvas.paste(tile.convert('RGB'),(x,y));draw.text((x,y+500),slug,fill=INK)
canvas.save(OUT/'overview.png')
print('Rendered six wide visuals, twelve square alternatives, SVGs, and overview.',flush=True)
