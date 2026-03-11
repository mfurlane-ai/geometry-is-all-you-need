"""
Cubic Lattice Neural Network — 9×9×9 Publication Run
=====================================================
Tests the core geometric predictions from "Geometry Is All You Need":
  1. Semantic clustering: related concepts cluster in physical proximity.
  2. Geometric sparsity: <<N³ nodes activate per inference.

Architecture: 9×9×9 = 729 nodes, 6-connected (face neighbours only).
Key optimisation: neighbour weights stored as sparse edge list,
backprop is O(N_EDGES) not O(N²).

Authors: Matthew Furlane & Claude (Anthropic)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from collections import defaultdict
import time, json, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── config ───────────────────────────────────────────────────────
N            = 9
N_NODES      = N**3          # 729
EMBED_DIM    = 32
N_CATEGORIES = 5
N_PER_CAT    = 10
N_CONCEPTS   = N_CATEGORIES * N_PER_CAT   # 50
EPOCHS       = 250
LR           = 0.015
SPARSITY_THRESHOLD  = 0.15
PROPAGATION_RADIUS  = 2

print("=" * 62)
print("CUBIC LATTICE SIMULATION  —  9×9×9 Publication Run")
print("=" * 62)
print(f"Lattice: {N}x{N}x{N} = {N_NODES} nodes | "
      f"Concepts: {N_CONCEPTS} | Epochs: {EPOCHS} | LR: {LR}\n")

# ── geometry ─────────────────────────────────────────────────────
def node_idx(x,y,z): return x*N*N + y*N + z
def node_coords(idx):
    x=idx//(N*N); y=(idx%(N*N))//N; z=idx%N; return x,y,z

neighbours = [[] for _ in range(N_NODES)]
for x in range(N):
    for y in range(N):
        for z in range(N):
            i = node_idx(x,y,z)
            for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                nx_,ny_,nz_ = x+dx,y+dy,z+dz
                if 0<=nx_<N and 0<=ny_<N and 0<=nz_<N:
                    neighbours[i].append(node_idx(nx_,ny_,nz_))

# Sparse edge arrays
edges_i, edges_j = [], []
for i in range(N_NODES):
    for j in neighbours[i]:
        edges_i.append(i); edges_j.append(j)
edges_i = np.array(edges_i, dtype=np.int32)
edges_j = np.array(edges_j, dtype=np.int32)
N_EDGES  = len(edges_i)

# Manhattan distance matrix
coords = np.array([node_coords(i) for i in range(N_NODES)], dtype=np.float32)
D = np.sum(np.abs(coords[:,None,:] - coords[None,:,:]), axis=2)

lat_conn  = N_EDGES // 2
flat_conn = N_NODES * N_NODES
print(f"Lattice: {lat_conn:,} edges ({lat_conn/flat_conn*100:.2f}% of flat {flat_conn:,})")
deg = defaultdict(int)
for i in range(N_NODES): deg[len(neighbours[i])] += 1
print(f"Degrees: {dict(sorted(deg.items()))}  (3=corner 4=edge 5=face 6=interior)\n")

# ── dataset ───────────────────────────────────────────────────────
CATEGORIES = {
    0:("Animals",  ["dog","cat","bird","fish","bear","wolf","deer","frog","snake","lion"]),
    1:("Vehicles", ["car","bus","train","plane","boat","bike","truck","ship","rocket","tram"]),
    2:("Food",     ["apple","bread","rice","soup","cake","milk","egg","meat","corn","salt"]),
    3:("Weather",  ["rain","snow","wind","fog","hail","storm","cloud","frost","sleet","heat"]),
    4:("Emotions", ["joy","fear","love","hate","calm","rage","hope","grief","pride","envy"]),
}
CAT_NAMES  = [CATEGORIES[i][0] for i in range(N_CATEGORIES)]
CAT_COLORS = ['#e63946','#2a9d8f','#e9c46a','#457b9d','#6a0572']

bases = []
for c in range(N_CATEGORIES):
    b = np.zeros(EMBED_DIM); seg=EMBED_DIM//N_CATEGORIES
    b[c*seg:(c+1)*seg]=1.0; bases.append(b/np.linalg.norm(b))

embs, labels = [], []
for ci,(cname,words) in CATEGORIES.items():
    for w in words:
        e = bases[ci] + np.random.randn(EMBED_DIM)*0.25
        embs.append(e/np.linalg.norm(e)); labels.append((ci,w))
embs = np.array(embs, dtype=np.float32)

intra,inter=[],[]
for i in range(N_CONCEPTS):
    for j in range(i+1,N_CONCEPTS):
        s=np.dot(embs[i],embs[j])
        (intra if labels[i][0]==labels[j][0] else inter).append(s)
print(f"Dataset: {N_CONCEPTS} concepts, dim={EMBED_DIM}")
print(f"Intra sim: {np.mean(intra):.3f}  Inter sim: {np.mean(inter):.3f}\n")

# ── sparse lattice network ────────────────────────────────────────
class SparseLatticeNet:
    def __init__(self, spatial_reg=0.0):
        self.reg   = spatial_reg
        self.W_in  = np.random.randn(N_NODES, EMBED_DIM).astype(np.float32) * 0.08
        self.b     = np.zeros(N_NODES, dtype=np.float32)
        self.W_nb  = np.random.randn(N_EDGES).astype(np.float32) * 0.04
        self.W_out = np.random.randn(N_CATEGORIES, N_NODES).astype(np.float32) * 0.08
        self.b_out = np.zeros(N_CATEGORIES, dtype=np.float32)
        self.loss_h=[]; self.acc_h=[]

    def fwd(self, x):
        pre1 = self.W_in @ x + self.b
        h1   = np.maximum(0, pre1)
        msg  = np.zeros(N_NODES, dtype=np.float32)
        np.add.at(msg, edges_i, self.W_nb * h1[edges_j])
        pre2 = self.W_in @ x + msg + self.b
        h2   = np.maximum(0, pre2)
        logits = self.W_out @ h2 + self.b_out
        e = np.exp(logits - logits.max()); probs = e/e.sum()
        return probs, h2, pre2, h1, pre1

    def step(self, x, cat):
        probs,h2,pre2,h1,pre1 = self.fwd(x)
        dl = probs.copy(); dl[cat] -= 1.0
        dW_out = np.outer(dl, h2)
        dh2    = self.W_out.T @ dl
        if self.reg > 0:
            dh2 += self.reg * (D @ h2)
        dp2    = dh2 * (pre2>0)
        dW_nb  = dp2[edges_i] * h1[edges_j]
        dh1    = np.zeros(N_NODES, dtype=np.float32)
        np.add.at(dh1, edges_j, self.W_nb * dp2[edges_i])
        dW_in2 = np.outer(dp2, x)
        dp1    = dh1 * (pre1>0)
        dW_in1 = np.outer(dp1, x)
        clip=1.0
        np.clip(dW_out,-clip,clip,out=dW_out)
        np.clip(dW_nb, -clip,clip,out=dW_nb)
        np.clip(dW_in2,-clip,clip,out=dW_in2)
        np.clip(dW_in1,-clip,clip,out=dW_in1)
        self.W_out -= LR*dW_out;  self.b_out -= LR*dl
        self.W_in  -= LR*(dW_in2+dW_in1)
        self.b     -= LR*(dp2+dp1)
        self.W_nb  -= LR*dW_nb
        return -np.log(probs[cat]+1e-10), probs

    def train(self, tag=""):
        t0=time.time()
        for ep in range(EPOCHS):
            tl=0.0; ok=0
            for i in np.random.permutation(N_CONCEPTS):
                l,p = self.step(embs[i], labels[i][0])
                tl+=l; ok+=(np.argmax(p)==labels[i][0])
            self.loss_h.append(tl/N_CONCEPTS)
            self.acc_h.append(ok/N_CONCEPTS)
            if ep%50==0 or ep==EPOCHS-1:
                print(f"  ep {ep:3d}  loss={tl/N_CONCEPTS:.4f}  "
                      f"acc={ok/N_CONCEPTS:.3f}  ({time.time()-t0:.0f}s)")
        print(f"  Done {time.time()-t0:.1f}s\n")

    def acts(self):
        A=np.zeros((N_CONCEPTS,N_NODES),dtype=np.float32)
        for i in range(N_CONCEPTS):
            _,h2,_,_,_=self.fwd(embs[i]); A[i]=h2
        return A

# ── flat baseline ─────────────────────────────────────────────────
class FlatNet:
    def __init__(self):
        self.W1=np.random.randn(N_NODES,EMBED_DIM).astype(np.float32)*0.08
        self.b1=np.zeros(N_NODES,dtype=np.float32)
        self.W2=np.random.randn(N_CATEGORIES,N_NODES).astype(np.float32)*0.08
        self.b2=np.zeros(N_CATEGORIES,dtype=np.float32)
        self.loss_h=[]; self.acc_h=[]
    def step(self,x,cat):
        h=np.maximum(0,self.W1@x+self.b1)
        lg=self.W2@h+self.b2; e=np.exp(lg-lg.max()); p=e/e.sum()
        dl=p.copy(); dl[cat]-=1.0
        dW2=np.outer(dl,h); dh=self.W2.T@dl
        dp=dh*(self.W1@x+self.b1>0); dW1=np.outer(dp,x)
        np.clip(dW2,-1,1,out=dW2); np.clip(dW1,-1,1,out=dW1)
        self.W2-=LR*dW2; self.b2-=LR*dl
        self.W1-=LR*dW1; self.b1-=LR*dp
        return -np.log(p[cat]+1e-10),p
    def train(self):
        t0=time.time()
        for ep in range(EPOCHS):
            tl=0.0; ok=0
            for i in np.random.permutation(N_CONCEPTS):
                l,p=self.step(embs[i],labels[i][0])
                tl+=l; ok+=(np.argmax(p)==labels[i][0])
            self.loss_h.append(tl/N_CONCEPTS); self.acc_h.append(ok/N_CONCEPTS)
            if ep%50==0 or ep==EPOCHS-1:
                print(f"  ep {ep:3d}  loss={tl/N_CONCEPTS:.4f}  "
                      f"acc={ok/N_CONCEPTS:.3f}  ({time.time()-t0:.0f}s)")
        print(f"  Done {time.time()-t0:.1f}s\n")

# ── train ─────────────────────────────────────────────────────────
print("── Model A: Lattice, no spatial reg ────────────────────────")
mA = SparseLatticeNet(0.0);   mA.train()
print("── Model B: Lattice, spatial reg λ=0.001 ───────────────────")
mB = SparseLatticeNet(0.001); mB.train()
print("── Model C: Flat fully-connected baseline ───────────────────")
mC = FlatNet();               mC.train()

# ── measure ───────────────────────────────────────────────────────
print("── Measuring results ────────────────────────────────────────")
theo = (4/3)*np.pi*PROPAGATION_RADIUS**3 / N_NODES

def measure(model, name):
    A   = model.acts()
    ctr = np.argmax(A, axis=1)
    id_, ie_ = [],[]
    for i in range(N_CONCEPTS):
        xi,yi,zi=node_coords(ctr[i])
        for j in range(i+1,N_CONCEPTS):
            xj,yj,zj=node_coords(ctr[j])
            d=abs(xi-xj)+abs(yi-yj)+abs(zi-zj)
            (id_ if labels[i][0]==labels[j][0] else ie_).append(d)
    im,ie=np.mean(id_),np.mean(ie_)
    ratio = ie/im if im>0 else 999.0
    sp=[]; 
    for i in range(N_CONCEPTS):
        thr=SPARSITY_THRESHOLD*A[i].max()
        sp.append((A[i]>thr).sum()/N_NODES)
    msp=np.mean(sp)
    print(f"\n  {name}")
    print(f"  Intra dist: {im:.3f}  Inter dist: {ie:.3f}  Ratio: {ratio:.3f}  "
          f"{'✓ CLUSTERING' if ratio>1.05 else '✗ weak'}")
    print(f"  Sparsity: {msp*100:.2f}% active  (theoretical bound: {theo*100:.2f}%)")
    return dict(intra=im,inter=ie,ratio=ratio,sparsity=msp,
                centroids=ctr,acts=A,intra_d=id_,inter_d=ie_)

rA = measure(mA, "Lattice no reg")
rB = measure(mB, "Lattice spatial reg")

fsp=[]
for i in range(N_CONCEPTS):
    h=np.maximum(0,mC.W1@embs[i]+mC.b1)
    fsp.append((h>SPARSITY_THRESHOLD*h.max()).sum()/N_NODES)
fsp_mean=np.mean(fsp)
print(f"\n  Flat baseline  Sparsity: {fsp_mean*100:.2f}%")

# ── figures ───────────────────────────────────────────────────────
print("\n── Generating figures ───────────────────────────────────────")

# Fig 1: Training curves
fig,axes=plt.subplots(1,2,figsize=(12,4.5),facecolor='white')
for ax,(attr,yl,tl) in zip(axes,[('loss_h','Loss','Training Loss'),('acc_h','Accuracy','Accuracy')]):
    ax.plot(getattr(mA,attr),color='#2a9d8f',lw=2,label='Lattice (no reg)')
    ax.plot(getattr(mB,attr),color='#457b9d',lw=2,ls='--',label='Lattice (spatial reg)')
    ax.plot(getattr(mC,attr),color='#e63946',lw=2,ls=':',label='Flat baseline')
    ax.set_xlabel('Epoch',fontsize=11); ax.set_ylabel(yl,fontsize=11)
    ax.set_title(tl,fontsize=12,fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3); ax.set_facecolor('#fafafa')
plt.suptitle('Figure S1 — Training Convergence  |  9×9×9 Lattice vs. Flat',fontsize=11,fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig9_training_curves.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  fig9_training_curves.png")

# Fig 2: 3D clustering scatter
fig=plt.figure(figsize=(14,6),facecolor='white')
for pi,(res,ttl) in enumerate([(rA,'No Spatial Reg'),(rB,'Spatial Reg λ=0.001')]):
    ax=fig.add_subplot(1,2,pi+1,projection='3d',facecolor='white')
    for i,ni in enumerate(res['centroids']):
        x,y,z=node_coords(ni)
        ax.scatter(x,y,z,c=CAT_COLORS[labels[i][0]],s=45,alpha=0.8,
                   zorder=5,edgecolors='black',linewidths=0.3)
    drawn=set()
    for i in range(0,N_NODES,4):
        xi,yi,zi=node_coords(i)
        for j in neighbours[i]:
            k=(min(i,j),max(i,j))
            if k not in drawn:
                xj,yj,zj=node_coords(j)
                ax.plot([xi,xj],[yi,yj],[zi,zj],color='#dddddd',lw=0.3,alpha=0.35)
                drawn.add(k)
    ax.set_xlabel('X',fontsize=8); ax.set_ylabel('Y',fontsize=8); ax.set_zlabel('Z',fontsize=8)
    ax.set_title(f'{ttl}\nClustering ratio: {res["ratio"]:.2f}',fontsize=10,fontweight='bold')
    ax.view_init(elev=22,azim=38); ax.tick_params(labelsize=7)
handles=[mpatches.Patch(color=CAT_COLORS[i],label=CAT_NAMES[i]) for i in range(N_CATEGORIES)]
fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=9,title='Semantic Category')
plt.suptitle('Figure S2 — Concept Centroid Positions in 9×9×9 Lattice\n'
             'Each dot = peak-activation node for one concept. Same colour = same category.',
             fontsize=11,fontweight='bold')
plt.tight_layout(rect=[0,0.08,1,1])
plt.savefig('/home/claude/fig9_clustering_3d.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  fig9_clustering_3d.png")

# Fig 3: Distance histograms
fig,axes=plt.subplots(1,2,figsize=(12,4.5),facecolor='white')
for ax,(res,ttl) in zip(axes,[(rA,'No Spatial Reg'),(rB,'Spatial Reg')]):
    bins=np.arange(0,N*3+2)-0.5
    ax.hist(res['intra_d'],bins=bins,alpha=0.65,color='#2a9d8f',density=True,
            label=f'Same category (mean={res["intra"]:.2f})')
    ax.hist(res['inter_d'],bins=bins,alpha=0.65,color='#e63946',density=True,
            label=f'Diff. category (mean={res["inter"]:.2f})')
    ax.axvline(res['intra'],color='#2a9d8f',lw=2,ls='--')
    ax.axvline(res['inter'],color='#e63946',lw=2,ls='--')
    ax.set_xlabel('Manhattan Distance Between Concept Centroids',fontsize=10)
    ax.set_ylabel('Density',fontsize=10)
    ax.set_title(f'{ttl}  |  Ratio: {res["ratio"]:.2f}',fontsize=10,fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3); ax.set_facecolor('#fafafa')
plt.suptitle('Figure S3 — Manhattan Distance Distributions  |  9×9×9 Lattice',
             fontsize=11,fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig9_distance_histograms.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  fig9_distance_histograms.png")

# Fig 4: Sparsity
fig,ax=plt.subplots(figsize=(9,5),facecolor='white')
lbls=['Flat\n(fully connected)','Lattice\n(no reg)','Lattice\n(spatial reg)']
vals=[fsp_mean,rA['sparsity'],rB['sparsity']]
cols=['#e63946','#2a9d8f','#457b9d']
bars=ax.bar(lbls,[v*100 for v in vals],color=cols,width=0.5,edgecolor='black',linewidth=0.8)
ax.axhline(theo*100,color='black',lw=1.8,ls='--',
           label=f'Theoretical bound (r=2): {theo*100:.1f}%')
ax.axhline(2.0,color='#888888',lw=1.2,ls=':',
           label='Brain estimate: ~1–2% (Olshausen & Field, 2004)')
for bar,val in zip(bars,vals):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
            f'{val*100:.1f}%',ha='center',va='bottom',fontsize=12,fontweight='bold')
ax.set_ylabel('Mean % Nodes Active Per Inference',fontsize=11)
ax.set_title('Figure S4 — Activation Sparsity  |  9×9×9 Lattice vs. Flat\n729 nodes total',
             fontsize=11,fontweight='bold')
ax.legend(fontsize=9); ax.set_ylim(0,max(vals)*160)
ax.grid(True,alpha=0.3,axis='y'); ax.set_facecolor('#fafafa')
plt.tight_layout()
plt.savefig('/home/claude/fig9_sparsity.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  fig9_sparsity.png")

# Fig 5: Activation heatmap
fig,axes=plt.subplots(1,N_CATEGORIES,figsize=(14,3.5),facecolor='white')
zs=N//2
for ci in range(N_CATEGORIES):
    ax=axes[ci]; idx0=ci*N_PER_CAT; a=rB['acts'][idx0]
    hm=np.zeros((N,N))
    for xi in range(N):
        for yi in range(N): hm[xi,yi]=a[node_idx(xi,yi,zs)]
    im=ax.imshow(hm,cmap='hot',interpolation='nearest',vmin=0)
    ax.set_title(f'{CAT_NAMES[ci]}\n"{labels[idx0][1]}"',fontsize=9,fontweight='bold')
    ax.set_xlabel('Y',fontsize=8); ax.set_ylabel('X',fontsize=8)
    ax.tick_params(labelsize=7); plt.colorbar(im,ax=ax,fraction=0.046)
plt.suptitle('Figure S5 — Activation Heatmap (Z=4 middle slice)  |  9×9×9 spatial reg',
             fontsize=10,fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig9_activation_heatmap.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  fig9_activation_heatmap.png")

# Fig 6: Category territory map
fig,ax=plt.subplots(figsize=(7,7),facecolor='white')
zs=N//2
terr=np.full((N,N),-1)
for xi in range(N):
    for yi in range(N):
        ni=node_idx(xi,yi,zs); cs=np.zeros(N_CATEGORIES)
        for ci in range(N_CONCEPTS): cs[labels[ci][0]]+=rB['acts'][ci][ni]
        terr[xi,yi]=np.argmax(cs)
for xi in range(N):
    for yi in range(N):
        cat=terr[xi,yi]
        ax.add_patch(plt.Rectangle((yi-0.5,xi-0.5),1,1,color=CAT_COLORS[cat],alpha=0.75))
        ax.text(yi,xi,CAT_NAMES[cat][:3],ha='center',va='center',
                fontsize=7.5,fontweight='bold',color='white')
ax.set_xlim(-0.5,N-0.5); ax.set_ylim(-0.5,N-0.5)
ax.set_xlabel('Y axis',fontsize=11); ax.set_ylabel('X axis',fontsize=11)
ax.set_xticks(range(N)); ax.set_yticks(range(N))
handles=[mpatches.Patch(color=CAT_COLORS[i],label=CAT_NAMES[i]) for i in range(N_CATEGORIES)]
ax.legend(handles=handles,loc='upper right',fontsize=9)
ax.set_title('Figure S6 — Category Territory Map  |  Z=4 Middle Slice\n'
             'Dominant semantic category at each node position after training',
             fontsize=10,fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig9_territory_map.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  fig9_territory_map.png")

# ── summary ───────────────────────────────────────────────────────
print("\n"+"="*62)
print("9×9×9 RESULTS SUMMARY")
print("="*62)
print(f"\nAccuracy: A={mA.acc_h[-1]*100:.1f}%  B={mB.acc_h[-1]*100:.1f}%  Flat={mC.acc_h[-1]*100:.1f}%")
print(f"\nClustering ratio (inter/intra Manhattan distance):")
print(f"  No reg     : {rA['ratio']:.3f}  {'✓ CLUSTERING' if rA['ratio']>1.05 else '✗'}")
print(f"  Spatial reg: {rB['ratio']:.3f}  {'✓ CLUSTERING' if rB['ratio']>1.05 else '✗'}")
print(f"  Null hyp   : 1.000")
print(f"\nActivation sparsity:")
print(f"  Flat       : {fsp_mean*100:.2f}%")
print(f"  Lattice A  : {rA['sparsity']*100:.2f}%")
print(f"  Lattice B  : {rB['sparsity']*100:.2f}%")
print(f"  Theo bound : {theo*100:.2f}%  (r=2, paper §4.2.6)")
print(f"  Brain est  : ~1–2%")
print(f"\nConnections: Lattice={lat_conn:,}  Flat={flat_conn:,}  ({lat_conn/flat_conn*100:.3f}%)")

results={
    'lattice':'9x9x9','n_nodes':N_NODES,'epochs':EPOCHS,
    'accuracy':{'no_reg':round(mA.acc_h[-1],4),
                'spatial_reg':round(mB.acc_h[-1],4),
                'flat':round(mC.acc_h[-1],4)},
    'clustering_ratio':{'no_reg':round(rA['ratio'],4),
                        'spatial_reg':round(rB['ratio'],4),'null':1.0},
    'sparsity_pct':{'flat':round(fsp_mean*100,2),
                    'no_reg':round(rA['sparsity']*100,2),
                    'spatial_reg':round(rB['sparsity']*100,2),
                    'theoretical_bound':round(theo*100,2)},
    'connections':{'lattice':lat_conn,'flat':flat_conn,
                   'lattice_pct_of_flat':round(lat_conn/flat_conn*100,4)}
}
with open('/home/claude/simulation_results_9x9x9.json','w') as f:
    json.dump(results,f,indent=2)
print(f"\n✓ Results saved.  ✓ 6 figures saved.")
print("="*62)
