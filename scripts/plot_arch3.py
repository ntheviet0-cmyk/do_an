#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate 3 architecture diagrams with 3 different styles: ARIMA, LSTM, TFT."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUT_DIR = r"C:\Đồ_Án\reports\figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'

def box(ax, xy, w, h, text, face, edge='#444', fs=8, weight='normal', alpha=1.0, style='round,pad=0.02'):
    f = patches.FancyBboxPatch(xy, w, h, boxstyle=style, facecolor=face, edgecolor=edge, linewidth=1.2, alpha=alpha)
    ax.add_patch(f)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha='center', va='center', fontsize=fs, weight=weight, wrap=True)

def arrow(ax, s, e, color='#333', w=1.5):
    ax.annotate('', xy=e, xytext=s, arrowprops=dict(arrowstyle='->', color=color, lw=w, shrinkA=2, shrinkB=3))

# ================= STYLE 1: ARIMA - Pastel Academic (giong Image 1) =================
def draw_arima():
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis('off')
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    ax.text(5, 11.4, 'STYLE 1 — ARIMA Box-Jenkins (Pastel Academic, giong Image 1)', ha='center', fontsize=11, weight='bold')
    ax.text(0.5, 10.9, 'MULTI-VIEW', fontsize=8, weight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='#D6E4F0', ec='#7AA0C4'))

    # Top datasets
    box(ax, (3.5,10.0), 3, 0.6, 'Time_series_dataset\n(sales history)', '#FFFFFF', fs=8)
    # utility matrix
    box(ax, (1.5,9.0), 2, 0.7, 'Utility matrix\nY (t x 1)', '#EAF6EA', fs=8)
    box(ax, (4.5,9.0), 2, 0.7, 'Exog matrix\nX (t x k)', '#EAF6EA', fs=8)
    arrow(ax, (5,10.0), (2.5,9.7)); arrow(ax, (5,10.0), (5.5,9.7))

    box(ax, (1.5,8.0), 2, 0.7, 'ADF Test\nStationarity', '#FFF7E0', fs=8, weight='bold')
    box(ax, (4.5,8.0), 2, 0.7, 'Transformer:\nLog + Scale', '#E8F4FF', fs=8)
    arrow(ax, (2.5,9.0), (2.5,8.7)); arrow(ax, (5.5,9.0), (5.5,8.7))

    box(ax, (1.5,7.0), 2, 0.7, 'Differencing (d)\nΔ^d Y', '#D9F2D0', fs=8)
    box(ax, (4.5,7.0), 2, 0.7, 'ACF / PACF\nIdentify p, q', '#D9F2D0', fs=8)
    arrow(ax, (2.5,8.0), (2.5,7.7)); arrow(ax, (5.5,8.0), (5.5,7.7))

    # Feature conversion purple
    box(ax, (3.0,6.0), 4, 0.7, 'Feature conversion (p, d, q)', '#C9B8E0', fs=9, weight='bold')
    arrow(ax, (2.5,7.0), (4.0,6.7)); arrow(ax, (5.5,7.0), (6.0,6.7))

    # feature vectors beige
    for i, x in enumerate([1.2, 3.5, 5.8]):
        box(ax, (x,5.0), 2, 0.6, f'feature\nvector {i+1}', '#FDE5C0', fs=8)
        arrow(ax, (x+1,6.0), (x+1,5.6))
    ax.text(8.2,5.3,'...',fontsize=14)

    # Big cyan block ARIMA
    big = patches.FancyBboxPatch((0.8,2.5), 8.4, 2.2, boxstyle='round,pad=0.02', facecolor='#A8E0E3', edgecolor='#2A7F84', linewidth=1.5)
    ax.add_patch(big)
    ax.text(5,4.3,'ARIMA(p,d,q) Estimator\nAR + I + MA', ha='center', fontsize=12, weight='bold')
    for i, x in enumerate([1.2,3.5,5.8]):
        arrow(ax,(x+1,5.0),(x+1,4.1), color='#555')
    box(ax,(1.2,3.0),2,0.6,'AR(p)\nOutput','#FDE5C0', fs=8)
    box(ax,(4.0,3.0),2,0.6,'I(d)\nOutput','#FDE5C0', fs=8)
    box(ax,(6.8,3.0),2,0.6,'MA(q)\nOutput','#FDE5C0', fs=8)

    arrow(ax,(5,3.0),(5,2.2), color='red', w=2.5)
    box(ax,(3.8,1.3),2.4,0.7,'Diagnostic\nLjung-Box + AIC','#FFFFFF', fs=8)
    box(ax,(3.8,0.3),2.4,0.7,'Forecast\n1 to h steps','#C9B8E0', fs=8, weight='bold')
    arrow(ax,(5,1.3),(5,1.0))

    fig.tight_layout()
    p=os.path.join(OUT_DIR,'arch_ARIMA_style1_pastel.png')
    fig.savefig(p,dpi=250,bbox_inches='tight'); plt.close(fig)

# ================= STYLE 2: LSTM - Modern Flat Horizontal =================
def draw_lstm():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis('off')
    fig.patch.set_facecolor('#F7F8FA'); ax.set_facecolor('#F7F8FA')
    ax.text(6,6.5,'STYLE 2 — LSTM Seq2One (Modern Flat Minimal)',ha='center',fontsize=11,weight='bold')

    # input window
    box(ax,(0.3,2.5),1.5,2.0,'Input Window\n[t-29..t]\n30 steps','#FFFFFF',edge='#2B6CB0',fs=8,weight='bold')
    # sliding
    box(ax,(2.1,2.5),1.3,2.0,'Scaling\n+ Sliding\nWindow','#EBF4FF',edge='#2B6CB0',fs=8)
    arrow(ax,(1.8,3.5),(2.1,3.5),color='#2B6CB0',w=2)

    # Unrolled cells
    colors=['#FFFFFF','#FFF5EB','#FFF5EB','#FFF5EB']
    labels=['LSTM','x t-2','x t-1','x t']
    for i in range(4):
        x=3.7+i*1.35
        b=patches.FancyBboxPatch((x,2.5),1.2,2.0,boxstyle='round,pad=0.02',facecolor='#FF8C42' if i>0 else '#2B6CB0',edgecolor='#333',linewidth=1.2)
        ax.add_patch(b)
        ax.text(x+0.6,3.9 if i==0 else 4.1,labels[i],ha='center',color='white',fontsize=8,weight='bold')
        if i>0:
            ax.text(x+0.6,3.4,'f,i,o,g\nc,h',ha='center',color='white',fontsize=7)
        else:
            ax.text(x+0.6,3.4,'2-Layer\n64 units',ha='center',color='white',fontsize=7)
        if i>0:
            arrow(ax,(x-0.15,3.5),(x,3.5))
        # h arrow bottom
        if i==3:
            arrow(ax,(x+1.2,3.5),(x+1.7,3.5),color='#FF8C42',w=2)
    arrow(ax,(3.4,3.5),(3.7,3.5),color='#2B6CB0',w=2)
    ax.text(6.2,2.1,'Unrolled over time →',ha='center',fontsize=8, style='italic')

    box(ax,(9.3,2.5),1.1,2.0,'Dropout\n0.2 + Dense\n32','#EBF8FF',edge='#2C5282',fs=8)
    box(ax,(10.6,2.5),1.1,2.0,'Output\nŷ t+1..t+h','#2B6CB0',edge='#1A365D',fs=8,weight='bold')
    # need white text for last
    ax.text(11.15,3.5,'Output\nŷ t+1..t+h',ha='center',va='center',color='white',fontsize=8,weight='bold')
    arrow(ax,(10.4,3.5),(10.6,3.5))

    # gate detail on top
    ax.text(6,5.7,'LSTM Cell detail:',fontsize=9,weight='bold',ha='center')
    for j, (t,c) in enumerate([('Forget\nf_t','#FEB2B2'),('Input\ni_t','#9AE6B4'),('Output\no_t','#90CDF4'),('Cell\ng,c','#FEFCBF')]):
        box(ax,(3.2+j*1.5,4.7),1.3,0.7,t,c,fs=7)
    ax.text(9.8,5.0,'σ / tanh',fontsize=8,bbox=dict(boxstyle='round',fc='white',ec='gray'))

    fig.tight_layout()
    p=os.path.join(OUT_DIR,'arch_LSTM_style2_flat.png')
    fig.savefig(p,dpi=250,bbox_inches='tight'); plt.close(fig)

# ================= STYLE 3: TFT - Dark Neon Multi-view =================
def draw_tft():
    fig, ax = plt.subplots(figsize=(10,11))
    ax.set_xlim(0,10); ax.set_ylim(0,12); ax.axis('off')
    fig.patch.set_facecolor('#0F172A'); ax.set_facecolor('#0F172A')
    def dbox(xy,w,h,text,face,edge='#22D3EE',fs=8,tc='white',wb='bold'):
        f=patches.FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.02',facecolor=face,edgecolor=edge,linewidth=1.3)
        ax.add_patch(f)
        ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=fs,weight=wb,color=tc)
    def darr(s,e,c='#22D3EE'):
        ax.annotate('',xy=e,xytext=s,arrowprops=dict(arrowstyle='->',color=c,lw=1.6))

    ax.text(5,11.4,'STYLE 3 — TFT Multi-Horizon (Dark Neon Multi-View)',ha='center',fontsize=11,weight='bold',color='white')
    ax.text(0.6,10.7,'MULTI-VIEW',fontsize=8,weight='bold',color='#0F172A',bbox=dict(boxstyle='round,pad=0.3',fc='#22D3EE',ec='#22D3EE'))

    dbox((3.2,10.0),3.6,0.6,'Sales_dataset (M5)','#1E293B')
    dbox((0.5,8.9),2.7,0.8,'Static Covariates\nstore, item','#312E81')
    dbox((3.6,8.9),2.8,0.8,'Past Observed\nsales, price','#0E7490')
    dbox((6.8,8.9),2.7,0.8,'Known Future\ndow, event','#831843')
    for x in [1.85,5.0,8.15]:
        darr((5,10.0),(x,9.7))

    dbox((0.5,7.9),2.7,0.6,'Static Encoder','#312E81',fs=7)
    dbox((3.6,7.9),2.8,0.6,'Variable Selection\n+ LSTM Encoder','#0E7490',fs=7)
    dbox((6.8,7.9),2.7,0.6,'Variable Selection\n+ LSTM Decoder','#831843',fs=7)
    for x in [1.85,5.0,8.15]:
        darr((x,8.9),(x,8.5))

    dbox((2.5,6.9),5,0.7,'Gated Residual Network + Multi-Head Attention (4 heads)','#7C3AED',edge='#A78BFA',fs=8)
    for x in [1.85,5.0,8.15]:
        darr((x,7.9),(4.0 if x<3 else 6.0 if x>7 else 5.0,7.6))

    for i,x in enumerate([1.0,3.6,6.2]):
        dbox((x,5.8),2.0,0.7,f'feature\nvector {i+1}','#F59E0B',edge='#FBBF24',tc='#111',fs=8)
        darr((x+1,6.9),(x+1,6.5),c='#A78BFA')
    ax.text(8.6,6.1,'...',color='white',fontsize=14)

    big=patches.FancyBboxPatch((0.6,3.3),8.8,2.2,boxstyle='round,pad=0.02',facecolor='#0E7490',edgecolor='#22D3EE',linewidth=1.6)
    ax.add_patch(big)
    ax.text(5,5.1,'Temporal Fusion Decoder',ha='center',fontsize=12,weight='bold',color='white')
    for x in [2.0,4.6,7.2]:
        darr((x,5.8),(x,4.8),c='white')
    dbox((1.0,3.8),2.0,0.6,'Quantile 0.5','#F59E0B',edge='white',tc='#111',fs=8)
    dbox((4.0,3.8),2.0,0.6,'Quantile 0.9','#F59E0B',edge='white',tc='#111',fs=8)
    dbox((7.0,3.8),2.0,0.6,'Quantile 0.1','#F59E0B',edge='white',tc='#111',fs=8)

    ax.annotate('',xy=(5,2.4),xytext=(5,3.3),arrowprops=dict(arrowstyle='->',color='#EF4444',lw=3))
    dbox((3.5,1.5),3,0.7,'LayerNorm + Dense','#E2E8F0',edge='white',tc='#111',fs=8)
    dbox((3.5,0.5),3,0.7,'Forecast 7-28 days','#22D3EE',edge='#22D3EE',tc='#111',fs=8)
    darr((5,1.5),(5,1.2),c='white')

    fig.tight_layout()
    p=os.path.join(OUT_DIR,'arch_TFT_style3_dark.png')
    fig.savefig(p,dpi=250,bbox_inches='tight',facecolor=fig.get_facecolor()); plt.close(fig)

draw_arima(); draw_lstm(); draw_tft()
