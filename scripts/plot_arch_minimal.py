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
    big=patches.FancyBboxPatch((0.8,2.4),8.4,2.3,boxstyle='round,pad=0.02',facecolor=CYAN,edgecolor=CYANEDGE,linewidth=1.5)
    ax.add_patch(big)
    ax.text(5,4.45,'2-Layer LSTM Encoder',ha='center',fontsize=12,weight='bold',
            bbox=dict(boxstyle='round,pad=0.25',fc=CYAN,ec='none'))
    arrow(ax,(2.2,5.0),(2.2,4.2),color='#555555')
    arrow(ax,(4.5,5.0),(5.0,4.2),color='#555555')
    arrow(ax,(6.8,5.0),(7.8,4.2),color='#555555')
    box(ax,(1.2,3.7),2,0.5,'LSTM L1-L2\n64 units, dropout 0.2',WHITE,fs=7)
    box(ax,(4.0,3.7),2,0.5,'Last hidden\nhn[-1] gather',WHITE,fs=7)
    box(ax,(6.8,3.7),2,0.5,'FC head\n32-ReLU-Dropout',WHITE,fs=7)
    box(ax,(1.2,2.9),2,0.6,'yhat t+1\nstep 1',BEIGE,fs=8)
    box(ax,(4.0,2.9),2,0.6,'yhat horizon\nH=28 steps',BEIGE,fs=8,weight='bold')
    box(ax,(6.8,2.9),2,0.6,'yhat t+H\nstep 28',BEIGE,fs=8)
    for x in [1.2,4.0,6.8]:
        arrow(ax,(x+1,3.7),(x+1,3.5),color='#555555',w=1.2)
    arrow(ax,(5,2.9),(5,2.1),color='red',w=2.5)
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
    box(ax,(0.6,7.0),2.6,0.7,'Static Context\ncs, ce, cc, ch',WHITE,fs=8)
    box(ax,(3.7,7.0),2.6,0.7,'Local LSTM\nEncoding',WHITE,fs=8)
    box(ax,(6.8,7.0),2.6,0.7,'Future Known\nEncoding',WHITE,fs=8)
    arrow(ax,(1.9,8.0),(1.9,7.7)); arrow(ax,(5.0,8.0),(5.0,7.7)); arrow(ax,(8.1,8.0),(8.1,7.7))
    box(ax,(3.0,6.0),4,0.7,'Feature conversion',PURPLE,fs=9,weight='bold')
    arrow(ax,(1.9,7.0),(4.0,6.7)); arrow(ax,(5.0,7.0),(5.0,6.7)); arrow(ax,(8.1,7.0),(6.0,6.7))
    for i,x in enumerate([1.2,3.5,5.8]):
        box(ax,(x,5.0),2,0.6,f'feature\nvector {i+1}',BEIGE,fs=8)
        arrow(ax,(x+1,6.0),(x+1,5.6))
    ax.text(8.2,5.3,'...',fontsize=14)
    big=patches.FancyBboxPatch((0.8,2.4),8.4,2.3,boxstyle='round,pad=0.02',facecolor=CYAN,edgecolor=CYANEDGE,linewidth=1.5)
    ax.add_patch(big)
    ax.text(5,4.45,'Temporal Fusion Decoder',ha='center',fontsize=12,weight='bold',
            bbox=dict(boxstyle='round,pad=0.25',fc=CYAN,ec='none'))
    # Mui ten tu feature vectors -> decoder (hoi cheo cho thang hang)
    arrow(ax,(2.2,5.0),(2.2,4.2),color='#555555')
    arrow(ax,(4.5,5.0),(5.0,4.2),color='#555555')
    arrow(ax,(6.8,5.0),(7.8,4.2),color='#555555')
    # Hang 1 trong decoder: xu ly thoi gian
    box(ax,(1.2,3.7),2,0.5,'Static Enrichment\nGRN + context',WHITE,fs=7)
    box(ax,(4.0,3.7),2,0.5,'Temporal Attention x4\ncausal mask, shared V',WHITE,fs=7)
    box(ax,(6.8,3.7),2,0.5,'Feed-Forward\nGRN + Add & Norm',WHITE,fs=7)
    # Hang 2 trong decoder: quantile heads
    box(ax,(1.2,2.9),2,0.6,'P10\nlower bound',BEIGE,fs=8)
    box(ax,(4.0,2.9),2,0.6,'P50 median\npoint forecast',BEIGE,fs=8,weight='bold')
    box(ax,(6.8,2.9),2,0.6,'P90\nupper bound',BEIGE,fs=8)
    for x in [1.2,4.0,6.8]:
        arrow(ax,(x+1,3.7),(x+1,3.5),color='#555555',w=1.2)
    arrow(ax,(5,2.9),(5,2.1),color='red',w=2.5)
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
    big=patches.FancyBboxPatch((0.8,2.4),8.4,2.3,boxstyle='round,pad=0.02',facecolor=CYAN,edgecolor=CYANEDGE,linewidth=1.5)
    ax.add_patch(big)
    ax.text(5,4.45,'ARIMA(7,1,1) Estimator',ha='center',fontsize=12,weight='bold',
            bbox=dict(boxstyle='round,pad=0.25',fc=CYAN,ec='none'))
    arrow(ax,(2.2,5.0),(2.2,4.2),color='#555555')
    arrow(ax,(4.5,5.0),(5.0,4.2),color='#555555')
    arrow(ax,(6.8,5.0),(7.8,4.2),color='#555555')
    box(ax,(1.2,3.7),2,0.5,'AR(p=7)\nlag weekly',WHITE,fs=7)
    box(ax,(4.0,3.7),2,0.5,'I(d=1)\nDiff stationary',WHITE,fs=7)
    box(ax,(6.8,3.7),2,0.5,'MA(q=1)\nerror smooth',WHITE,fs=7)
    box(ax,(1.2,2.9),2,0.6,'yhat t+1\nstep 1',BEIGE,fs=8)
    box(ax,(4.0,2.9),2,0.6,'yhat recursive\nH=28 steps',BEIGE,fs=8,weight='bold')
    box(ax,(6.8,2.9),2,0.6,'yhat t+H\nstep 28',BEIGE,fs=8)
    for x in [1.2,4.0,6.8]:
        arrow(ax,(x+1,3.7),(x+1,3.5),color='#555555',w=1.2)
    arrow(ax,(5,2.9),(5,2.1),color='red',w=2.5)
    box(ax,(3.8,1.3),2.4,0.7,'Diagnostic\nLjung-Box + AIC',WHITE,fs=8)
    box(ax,(3.8,0.3),2.4,0.7,'Forecast\n1 to h steps',PURPLE,fs=8,weight='bold')
    arrow(ax,(5,1.3),(5,1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR,'arch_ARIMA_style1_pastel_minimal.png'),dpi=250,bbox_inches='tight'); plt.close(fig)

def draw_prophet_minimal():
    fig, ax = base_fig()
    ax.text(5,11.4,'Prophet',ha='center',fontsize=14,weight='bold')
    ax.text(0.5,10.9,'MULTI-VIEW',fontsize=8,weight='bold',bbox=dict(boxstyle='round,pad=0.3',fc=GRAYLABEL,ec=EDGE))
    box(ax,(3.5,10.0),3,0.6,'Time_series_dataset\n(sales history)',WHITE,fs=8)
    box(ax,(1.5,9.0),2,0.7,'Utility matrix\nY (t x 1)',WHITE,fs=8)
    box(ax,(4.5,9.0),2,0.7,'Exog matrix\nX (t x k)',WHITE,fs=8)
    arrow(ax,(5,10.0),(2.5,9.7)); arrow(ax,(5,10.0),(5.5,9.7))
    box(ax,(1.5,8.0),2,0.7,'Trend\nChangepoints',WHITE,fs=8,weight='bold')
    box(ax,(4.5,8.0),2,0.7,'Regressors\nprice, snap, event',WHITE,fs=8)
    arrow(ax,(2.5,9.0),(2.5,8.7)); arrow(ax,(5.5,9.0),(5.5,8.7))
    box(ax,(1.5,7.0),2,0.7,'Yearly + Weekly\nSeasonality',WHITE,fs=8)
    box(ax,(4.5,7.0),2,0.7,'Holidays\n+ SNAP',WHITE,fs=8)
    arrow(ax,(2.5,8.0),(2.5,7.7)); arrow(ax,(5.5,8.0),(5.5,7.7))
    box(ax,(3.0,6.0),4,0.7,'Feature conversion (ds, y, regressors)',PURPLE,fs=9,weight='bold')
    arrow(ax,(2.5,7.0),(4.0,6.7)); arrow(ax,(5.5,7.0),(6.0,6.7))
    for i,x in enumerate([1.2,3.5,5.8]):
        box(ax,(x,5.0),2,0.6,f'feature\nvector {i+1}',BEIGE,fs=8)
        arrow(ax,(x+1,6.0),(x+1,5.6))
    ax.text(8.2,5.3,'...',fontsize=14)
    big=patches.FancyBboxPatch((0.8,2.4),8.4,2.3,boxstyle='round,pad=0.02',facecolor=CYAN,edgecolor=CYANEDGE,linewidth=1.5)
    ax.add_patch(big)
    ax.text(5,4.45,'Prophet Additive Model',ha='center',fontsize=12,weight='bold',
            bbox=dict(boxstyle='round,pad=0.25',fc=CYAN,ec='none'))
    arrow(ax,(2.2,5.0),(2.2,4.2),color='#555555')
    arrow(ax,(4.5,5.0),(5.0,4.2),color='#555555')
    arrow(ax,(6.8,5.0),(7.8,4.2),color='#555555')
    box(ax,(1.2,3.7),2,0.5,'Trend g(t)\nchangepoints',WHITE,fs=7)
    box(ax,(4.0,3.7),2,0.5,'Season s(t)\nyearly + weekly',WHITE,fs=7)
    box(ax,(6.8,3.7),2,0.5,'Holiday h(t)\n+ regressors',WHITE,fs=7)
    box(ax,(1.2,2.9),2,0.6,'yhat_lower\n95% low',BEIGE,fs=8)
    box(ax,(4.0,2.9),2,0.6,'yhat sum\ntrend+season+reg',BEIGE,fs=8,weight='bold')
    box(ax,(6.8,2.9),2,0.6,'yhat_upper\n95% high',BEIGE,fs=8)
    for x in [1.2,4.0,6.8]:
        arrow(ax,(x+1,3.7),(x+1,3.5),color='#555555',w=1.2)
    arrow(ax,(5,2.9),(5,2.1),color='red',w=2.5)
    box(ax,(3.8,1.3),2.4,0.7,'Diagnostic\nInterval 95%',WHITE,fs=8)
    box(ax,(3.8,0.3),2.4,0.7,'Forecast yhat\n1 to h steps',PURPLE,fs=8,weight='bold')
    arrow(ax,(5,1.3),(5,1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR,'arch_Prophet_style1_pastel_minimal.png'),dpi=250,bbox_inches='tight'); plt.close(fig)

def draw_tft_store_cal_minimal():
    """Draw the final evaluated TFT-Lite pipeline and store calibration stage."""
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10); ax.set_ylim(0, 15); ax.axis('off')
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')

    ax.text(5, 14.45, 'TFT-Lite Global + Store Calibration', ha='center', fontsize=14, weight='bold')
    ax.text(0.5, 13.95, 'FINAL MODEL', fontsize=8, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc=GRAYLABEL, ec=EDGE))

    box(ax, (3.35, 13.05), 3.3, 0.58, 'M5 dataset: 30,490 SKU-store series', WHITE, fs=8)
    box(ax, (0.55, 11.95), 2.7, 0.78, 'Static\nitem, department, category, store, state', WHITE, fs=7)
    box(ax, (3.65, 11.95), 2.7, 0.78, 'Past 56 days\nsales, price, lifecycle, hierarchy context', WHITE, fs=7)
    box(ax, (6.75, 11.95), 2.7, 0.78, 'Known future: 28 days\nweekday, month, event, SNAP', WHITE, fs=7)
    arrow(ax, (5, 13.05), (1.9, 12.73)); arrow(ax, (5, 13.05), (5, 12.73)); arrow(ax, (5, 13.05), (8.1, 12.73))

    box(ax, (0.55, 10.85), 2.7, 0.68, 'Static Embedding\ncontext vector', WHITE, fs=8, weight='bold')
    box(ax, (3.65, 10.85), 2.7, 0.68, 'Past VSN + LSTM Encoder', WHITE, fs=8, weight='bold')
    box(ax, (6.75, 10.85), 2.7, 0.68, 'Future VSN + LSTM Decoder', WHITE, fs=8, weight='bold')
    arrow(ax, (1.9, 11.95), (1.9, 11.53)); arrow(ax, (5, 11.95), (5, 11.53)); arrow(ax, (8.1, 11.95), (8.1, 11.53))

    box(ax, (3.0, 9.78), 4.0, 0.7, 'Static context + encoded temporal features', PURPLE, fs=9, weight='bold')
    arrow(ax, (1.9, 10.85), (4.0, 10.48)); arrow(ax, (5, 10.85), (5, 10.48)); arrow(ax, (8.1, 10.85), (6.0, 10.48))

    big = patches.FancyBboxPatch((0.75, 6.45), 8.5, 2.75, boxstyle='round,pad=0.02', facecolor=CYAN, edgecolor=CYANEDGE, linewidth=1.5)
    ax.add_patch(big)
    ax.text(5, 8.88, 'TFT-Lite Temporal Fusion Block', ha='center', fontsize=12, weight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc=CYAN, ec='none'))
    arrow(ax, (5, 9.78), (5, 8.35), color='#555555')
    box(ax, (1.2, 7.65), 2.1, 0.58, 'Static Enrichment\nGRN + context', WHITE, fs=7)
    box(ax, (3.95, 7.65), 2.1, 0.58, 'Multi-head Attention\n4 heads, causal mask', WHITE, fs=7)
    box(ax, (6.7, 7.65), 2.1, 0.58, 'Feed-forward GRN\nAdd + LayerNorm', WHITE, fs=7)
    arrow(ax, (5, 8.35), (2.25, 8.23), color='#555555', w=1.2)
    arrow(ax, (5, 8.35), (5, 8.23), color='#555555', w=1.2)
    arrow(ax, (5, 8.35), (7.75, 8.23), color='#555555', w=1.2)
    box(ax, (3.6, 6.75), 2.8, 0.55, 'Forecast states for H = 28 days', BEIGE, fs=8, weight='bold')
    arrow(ax, (5, 7.65), (5, 7.30), color='#555555', w=1.2)

    box(ax, (1.2, 5.55), 2.5, 0.68, 'Zero-occurrence head\nP(demand > 0)', WHITE, fs=8)
    box(ax, (6.3, 5.55), 2.5, 0.68, 'Severity head\ndemand if active', WHITE, fs=8)
    arrow(ax, (4.35, 6.75), (2.45, 6.23)); arrow(ax, (5.65, 6.75), (7.55, 6.23))
    box(ax, (3.0, 4.43), 4.0, 0.72, 'Bottom-up SKU forecast = sigmoid(zero) x softplus(severity)', PURPLE, fs=8, weight='bold')
    arrow(ax, (2.45, 5.55), (4.2, 5.15)); arrow(ax, (7.55, 5.55), (5.8, 5.15))

    box(ax, (3.0, 3.25), 4.0, 0.7, 'Validation: fit one calibration factor per store\nfactor = actual store total / predicted store total', BEIGE, fs=8)
    arrow(ax, (5, 4.43), (5, 3.95), color='red', w=2.5)
    box(ax, (3.0, 2.12), 4.0, 0.7, 'Clip factor to [0.8, 1.2] and apply it to all store SKUs', CYAN, fs=8, weight='bold')
    arrow(ax, (5, 3.25), (5, 2.82))
    box(ax, (3.0, 0.95), 4.0, 0.72, 'Final forecast\nTFT-Lite Global + Store Calibration', PURPLE, fs=9, weight='bold')
    arrow(ax, (5, 2.12), (5, 1.67), color='red', w=2.5)

    ax.text(5, 0.35, 'Training: Tweedie loss + 0.2 hierarchical loss + 0.1 zero BCE | Evaluation: WRMSSE-5 (L1, L2, L3, L9, L12)',
            ha='center', fontsize=6.8, style='italic', color='#444444')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'arch_TFT_STORE_CAL_style1_pastel_minimal.png'), dpi=250, bbox_inches='tight')
    plt.close(fig)

def draw_tft_store_cal_detailed_minimal():
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.set_xlim(0, 10); ax.set_ylim(0, 16); ax.axis('off')
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    ax.text(5, 15.25, 'TFT-Lite Global + Store Calibration', ha='center', fontsize=14, weight='bold')
    ax.text(5, 14.65, 'FINAL FULL PIPELINE - FEATURE ENGINEERING INCLUDED', ha='center', fontsize=8, weight='bold')

    box(ax, (0.45, 12.45), 2.1, 0.8, 'M5 dataset\n30,490 SKU-store', WHITE, fs=7)
    box(ax, (2.85, 12.45), 2.1, 0.8, 'Static context\nitem/dept/store/state', WHITE, fs=7)
    box(ax, (5.25, 12.45), 2.1, 0.8, 'Past features\nlags/rolls/price/SNAP', WHITE, fs=7)
    box(ax, (7.65, 12.45), 2.1, 0.8, 'Future features\ncalendar/event/SNAP', WHITE, fs=7)

    arrow(ax, (1.5, 12.45), (1.7, 12.1))
    arrow(ax, (4.5, 12.45), (5.3, 11.65))
    arrow(ax, (4.5, 12.45), (8.6, 11.65))

    box(ax, (0.6, 10.75), 2.2, 0.9, 'Past\nFeatures', WHITE, fs=8, weight='bold')
    box(ax, (2.9, 10.75), 2.2, 0.9, 'Future\nFeatures', WHITE, fs=8, weight='bold')
    box(ax, (5.2, 10.75), 2.2, 0.9, 'Local State', WHITE, fs=8, weight='bold')
    box(ax, (7.5, 10.75), 2.2, 0.9, 'Hierarchy\nContext', WHITE, fs=8, weight='bold')

    arrow(ax, (1.7, 10.75), (3.5, 10.1))
    arrow(ax, (5.0, 10.75), (4.0, 10.1))
    arrow(ax, (6.3, 10.75), (6.0, 10.1))
    arrow(ax, (8.6, 10.75), (7.0, 10.1))

    box(ax, (3.0, 9.3), 4.0, 0.8, 'Encoded features -> VSN + LSTM\ntemporal core', PURPLE, fs=8, weight='bold')
    arrow(ax, (5, 9.3), (5, 8.4))
    box(ax, (3.0, 7.6), 4.0, 0.8, 'Temporal fusion -> Forecast states\nH = 28', CYAN, fs=8, weight='bold')
    arrow(ax, (5, 7.6), (5, 6.7))
    box(ax, (3.0, 5.9), 4.0, 0.8, 'Forecast states -> Zero/severity\nheads -> Expected demand', PURPLE, fs=8, weight='bold')
    arrow(ax, (5, 5.9), (5, 5.0))
    box(ax, (3.0, 4.2), 4.0, 0.8, 'Store calibration:\nfactor = actual / predicted\nstore total', BEIGE, fs=8)
    arrow(ax, (5, 4.2), (5, 3.3))
    box(ax, (3.0, 2.5), 4.0, 0.8, 'Clip factor to [0.8, 1.2]\nApply -> Calibrated forecast', CYAN, fs=8, weight='bold')
    arrow(ax, (5, 2.5), (5, 1.6))
    box(ax, (3.0, 0.8), 4.0, 0.8, 'Final forecast:\nTFT-Lite Global + Store Calibration', PURPLE, fs=9, weight='bold')
    ax.text(5, 0.35, 'Training: Tweedie + hierarchical + zero BCE | Eval: WRMSSE-5 over L1/L2/L3/L9/L12', ha='center', fontsize=6.8, fontstyle='italic', color='#444444')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'arch_TFT_STORE_CAL_detailed_feature_engineering.png'), dpi=250, bbox_inches='tight')
    plt.close(fig)

draw_lstm_minimal(); draw_tft_minimal(); draw_arima_minimal(); draw_prophet_minimal(); draw_tft_store_cal_minimal(); draw_tft_store_cal_detailed_minimal()
