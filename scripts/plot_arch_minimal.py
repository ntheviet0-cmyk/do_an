#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSTM & TFT redrawn in ARIMA pastel style, minimal colors."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
OUT_DIR = r"C:\Đồ_Án\reports\figures"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams['font.family'] = 'DejaVu Sans'

WHITE='#FFFFFF'; BEIGE='#FDE5C0'; PURPLE='#C9B8E0'; CYAN='#A8E0E3'; GRAYLABEL='#EEEEEE'
EDGE='#444444'; CYANEDGE='#2A7F84'

def box(ax, xy, w, h, text, face, fs=8, weight='normal'):
    f = patches.FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.02', facecolor=face, edgecolor=EDGE, linewidth=1.2)
    ax.add_patch(f)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha='center', va='center', fontsize=fs, weight=weight)

def arrow(ax, s, e, color='#333333', w=1.5):
    ax.annotate('', xy=e, xytext=s, arrowprops=dict(arrowstyle='->', color=color, lw=w, shrinkA=2, shrinkB=3))

def base_fig():
    fig, ax = plt.subplots(figsize=(10,11))
    ax.set_xlim(0,10); ax.set_ylim(0,12); ax.axis('off')
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    return fig, ax

def draw_lstm_minimal():
    fig, ax = base_fig()
    ax.text(5,11.4,'LSTM',ha='center',fontsize=14,weight='bold')
    ax.text(0.5,10.9,'MULTI-VIEW',fontsize=8,weight='bold',bbox=dict(boxstyle='round,pad=0.3',fc=GRAYLABEL,ec=EDGE))
    box(ax,(3.5,10.0),3,0.6,'Time_series_dataset\n(sales history)',WHITE,fs=8)
    box(ax,(1.5,9.0),2,0.7,'Utility matrix\nY (t x 1)',WHITE,fs=8)
    box(ax,(4.5,9.0),2,0.7,'Window matrix\nX (t x f)',WHITE,fs=8)
    arrow(ax,(5,10.0),(2.5,9.7)); arrow(ax,(5,10.0),(5.5,9.7))
    box(ax,(1.5,8.0),2,0.7,'Scaling\nNormalize',WHITE,fs=8,weight='bold')
    box(ax,(4.5,8.0),2,0.7,'Sliding Window\n30 steps',WHITE,fs=8)
    arrow(ax,(2.5,9.0),(2.5,8.7)); arrow(ax,(5.5,9.0),(5.5,8.7))
    box(ax,(1.5,7.0),2,0.7,'LSTM Cell\nf, i, o, g',WHITE,fs=8)
    box(ax,(4.5,7.0),2,0.7,'Gates\ntanh + sigmoid',WHITE,fs=8)
    arrow(ax,(2.5,8.0),(2.5,7.7)); arrow(ax,(5.5,8.0),(5.5,7.7))
    box(ax,(3.0,6.0),4,0.7,'Feature conversion (window, horizon)',PURPLE,fs=9,weight='bold')
    arrow(ax,(2.5,7.0),(4.0,6.7)); arrow(ax,(5.5,7.0),(6.0,6.7))
    for i,x in enumerate([1.2,3.5,5.8]):
        box(ax,(x,5.0),2,0.6,f'feature\nvector {i+1}',BEIGE,fs=8)
        arrow(ax,(x+1,6.0),(x+1,5.6))
    ax.text(8.2,5.3,'...',fontsize=14)
    big=patches.FancyBboxPatch((0.8,2.5),8.4,2.2,boxstyle='round,pad=0.02',facecolor=CYAN,edgecolor=CYANEDGE,linewidth=1.5)
    ax.add_patch(big)
    ax.text(5,4.3,'2-Layer LSTM Encoder',ha='center',fontsize=12,weight='bold')
    for x in [1.2,3.5,5.8]:
        arrow(ax,(x+1,5.0),(x+1,4.1),color='#555555')
    box(ax,(1.2,3.0),2,0.6,'Output 0\n(h, c)',BEIGE,fs=8)
    box(ax,(4.0,3.0),2,0.6,'Output 1\n(h, c)',BEIGE,fs=8)
    box(ax,(6.8,3.0),2,0.6,'Output N\n(h, c)',BEIGE,fs=8)
    arrow(ax,(5,3.0),(5,2.2),color='red',w=2.5)
    box(ax,(3.8,1.3),2.4,0.7,'LayerNormalization\nDense',WHITE,fs=8)
    box(ax,(3.8,0.3),2.4,0.7,'Output\nDense forecast',PURPLE,fs=8,weight='bold')
    arrow(ax,(5,1.3),(5,1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR,'arch_LSTM_style1_pastel_minimal.png'),dpi=250,bbox_inches='tight'); plt.close(fig)

def draw_tft_minimal():
    fig, ax = base_fig()
    ax.text(5,11.4,'TFT',ha='center',fontsize=14,weight='bold')
    ax.text(0.5,10.9,'MULTI-VIEW',fontsize=8,weight='bold',bbox=dict(boxstyle='round,pad=0.3',fc=GRAYLABEL,ec=EDGE))
    box(ax,(3.5,10.0),3,0.6,'Sales_dataset (M5)',WHITE,fs=8)
    box(ax,(0.6,9.0),2.6,0.7,'Static\nstore, item',WHITE,fs=8)
    box(ax,(3.7,9.0),2.6,0.7,'Past observed\nsales, price',WHITE,fs=8)
    box(ax,(6.8,9.0),2.6,0.7,'Known future\ndow, event',WHITE,fs=8)
    arrow(ax,(5,10.0),(1.9,9.7)); arrow(ax,(5,10.0),(5.0,9.7)); arrow(ax,(5,10.0),(8.1,9.7))
    box(ax,(0.6,8.0),2.6,0.7,'Static Encoder',WHITE,fs=8,weight='bold')
    box(ax,(3.7,8.0),2.6,0.7,'Variable Selection\n+ LSTM Encoder',WHITE,fs=8)
    box(ax,(6.8,8.0),2.6,0.7,'Variable Selection\n+ LSTM Decoder',WHITE,fs=8)
    arrow(ax,(1.9,9.0),(1.9,8.7)); arrow(ax,(5.0,9.0),(5.0,8.7)); arrow(ax,(8.1,9.0),(8.1,8.7))
    box(ax,(0.6,7.0),2.6,0.7,'GRN\nGated Residual',WHITE,fs=8)
    box(ax,(3.7,7.0),2.6,0.7,'Multi-Head\nAttention x4',WHITE,fs=8)
    box(ax,(6.8,7.0),2.6,0.7,'Add & Norm',WHITE,fs=8)
    arrow(ax,(1.9,8.0),(1.9,7.7)); arrow(ax,(5.0,8.0),(5.0,7.7)); arrow(ax,(8.1,8.0),(8.1,7.7))
    box(ax,(3.0,6.0),4,0.7,'Feature conversion',PURPLE,fs=9,weight='bold')
    arrow(ax,(1.9,7.0),(4.0,6.7)); arrow(ax,(5.0,7.0),(5.0,6.7)); arrow(ax,(8.1,7.0),(6.0,6.7))
    for i,x in enumerate([1.2,3.5,5.8]):
        box(ax,(x,5.0),2,0.6,f'feature\nvector {i+1}',BEIGE,fs=8)
        arrow(ax,(x+1,6.0),(x+1,5.6))
    ax.text(8.2,5.3,'...',fontsize=14)
    big=patches.FancyBboxPatch((0.8,2.5),8.4,2.2,boxstyle='round,pad=0.02',facecolor=CYAN,edgecolor=CYANEDGE,linewidth=1.5)
    ax.add_patch(big)
    ax.text(5,4.3,'Temporal Fusion Decoder',ha='center',fontsize=12,weight='bold')
    for x in [1.2,3.5,5.8]:
        arrow(ax,(x+1,5.0),(x+1,4.1),color='#555555')
    box(ax,(1.2,3.0),2,0.6,'Output 0\nQuantile',BEIGE,fs=8)
    box(ax,(4.0,3.0),2,0.6,'Output 1\nQuantile',BEIGE,fs=8)
    box(ax,(6.8,3.0),2,0.6,'Output N\nQuantile',BEIGE,fs=8)
    arrow(ax,(5,3.0),(5,2.2),color='red',w=2.5)
    box(ax,(3.8,1.3),2.4,0.7,'LayerNormalization\nDense',WHITE,fs=8)
    box(ax,(3.8,0.3),2.4,0.7,'Output\nDense forecast',PURPLE,fs=8,weight='bold')
    arrow(ax,(5,1.3),(5,1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR,'arch_TFT_style1_pastel_minimal.png'),dpi=250,bbox_inches='tight'); plt.close(fig)

def draw_arima_minimal():
    fig, ax = base_fig()
    ax.text(5,11.4,'ARIMA',ha='center',fontsize=14,weight='bold')
    ax.text(0.5,10.9,'MULTI-VIEW',fontsize=8,weight='bold',bbox=dict(boxstyle='round,pad=0.3',fc=GRAYLABEL,ec=EDGE))
    box(ax,(3.5,10.0),3,0.6,'Time_series_dataset\n(sales history)',WHITE,fs=8)
    box(ax,(1.5,9.0),2,0.7,'Utility matrix\nY (t x 1)',WHITE,fs=8)
    box(ax,(4.5,9.0),2,0.7,'Exog matrix\nX (t x k)',WHITE,fs=8)
    arrow(ax,(5,10.0),(2.5,9.7)); arrow(ax,(5,10.0),(5.5,9.7))
    box(ax,(1.5,8.0),2,0.7,'ADF Test\nStationarity',WHITE,fs=8,weight='bold')
    box(ax,(4.5,8.0),2,0.7,'Log + Scale',WHITE,fs=8)
    arrow(ax,(2.5,9.0),(2.5,8.7)); arrow(ax,(5.5,9.0),(5.5,8.7))
    box(ax,(1.5,7.0),2,0.7,'Differencing (d)',WHITE,fs=8)
    box(ax,(4.5,7.0),2,0.7,'ACF / PACF\nIdentify p, q',WHITE,fs=8)
    arrow(ax,(2.5,8.0),(2.5,7.7)); arrow(ax,(5.5,8.0),(5.5,7.7))
    box(ax,(3.0,6.0),4,0.7,'Feature conversion (p, d, q)',PURPLE,fs=9,weight='bold')
    arrow(ax,(2.5,7.0),(4.0,6.7)); arrow(ax,(5.5,7.0),(6.0,6.7))
    for i,x in enumerate([1.2,3.5,5.8]):
        box(ax,(x,5.0),2,0.6,f'feature\nvector {i+1}',BEIGE,fs=8)
        arrow(ax,(x+1,6.0),(x+1,5.6))
    ax.text(8.2,5.3,'...',fontsize=14)
    big=patches.FancyBboxPatch((0.8,2.5),8.4,2.2,boxstyle='round,pad=0.02',facecolor=CYAN,edgecolor=CYANEDGE,linewidth=1.5)
    ax.add_patch(big)
    ax.text(5,4.3,'ARIMA(p,d,q) Estimator',ha='center',fontsize=12,weight='bold')
    for x in [1.2,3.5,5.8]:
        arrow(ax,(x+1,5.0),(x+1,4.1),color='#555555')
    box(ax,(1.2,3.0),2,0.6,'AR(p)\nOutput',BEIGE,fs=8)
    box(ax,(4.0,3.0),2,0.6,'I(d)\nOutput',BEIGE,fs=8)
    box(ax,(6.8,3.0),2,0.6,'MA(q)\nOutput',BEIGE,fs=8)
    arrow(ax,(5,3.0),(5,2.2),color='red',w=2.5)
    box(ax,(3.8,1.3),2.4,0.7,'Diagnostic\nLjung-Box + AIC',WHITE,fs=8)
    box(ax,(3.8,0.3),2.4,0.7,'Forecast\n1 to h steps',PURPLE,fs=8,weight='bold')
    arrow(ax,(5,1.3),(5,1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR,'arch_ARIMA_style1_pastel_minimal.png'),dpi=250,bbox_inches='tight'); plt.close(fig)

draw_lstm_minimal(); draw_tft_minimal(); draw_arima_minimal()
