import pandas as pd
import numpy as np
import torch
from copy import copy

import matplotlib.pyplot as plt
import seaborn as sns

from .constants import mmoll_mgdl

sns.set()
sns.set_context('paper')
sns.set_style('white')

def clarke_error_grid(output, target, unit:str='mmoll'):
    """
    Return masks for points belonging to certain areas in the Clarke Error Grid 

    Zone A: Clinically Accurate
        Zone A represents glucose values that deviate from the reference by no more than 20% 
        or are in the hypoglycemic range (<70 mg/dl) when the reference is also <70 mg/dl.

    Zone B: Clinically Acceptable
        Upper and lower zone B represents values that deviate from the reference by >20%.

    Zone C: Overcorrecting
        Zone C values would result in overcorrecting acceptable blood glucose levels; 
        such treatment might cause the actual blood glucose to fall below 70 mg/dl or rise above 180 mg/dl.

    Zone D: Failure to Detect
        Zone D represents "dangerous failure to detect and treat" errors. Actual glucose values are 
        outside of the target range, but patient-generated values are within the target range.

    Zone E: Erroneous treatment
        Zone E is an "erroneous treatment" zone. Patient-generated values within this zone are opposite 
        to the reference values, and corresponding treatment decisions would therefore be opposite to that called for.


    Reference:
    [1] Clarke, W.L., Cox, D., Gonder-Frederick, L. A., Carter, W., & Pohl, S. L. (1987). 
        "Evaluating Clinical Accuracy of Systems for Self-Monitoring of Blood Glucose".
        Diabetes Care, 10(5), 622–628. doi: 10.2337/diacare.10.5.622
    [2] Clarke, WL. (2005). "The Original Clarke Error Grid Analysis (EGA)."
        Diabetes Technology and Therapeutics 7(5), pp. 776-779.
    """
    assert unit == 'mmoll' or unit == 'mgdl', 'Please give either mmoll or mgdl as unit here.'
    
    if unit == 'mmoll':
        target = copy(target) * mmoll_mgdl
        output = copy(output) * mmoll_mgdl

    clarke = {}
    clarke['A'] = (abs(target - output) <= 0.2*target) | ((target < 70) & (output < 70))

    clarke['E'] = ((target >= 180) & (output <= 70)) | ((output >= 180) & (target <= 70)) \
        & ~clarke['A']

    clarke['D'] = ((output >= 70) & (output <= 180) & ((target > 240) | (target < 70)))\
        & ~clarke['A'] & ~clarke['E']

    clarke['C'] = ((target >= 70) & (output >= target + 110)) | ((target <= 180) & (output <= (7/5)*target-182))\
        & ~clarke['A'] & ~clarke['E'] & ~clarke['D']

    clarke['B'] = (abs(target - output) > 0.2*target)\
        & ~clarke['A'] & ~clarke['E'] & ~clarke['D'] & ~clarke['C']

    return dict(sorted(clarke.items()))

def format_clarke_error_grid(ax, lims=[0,450], unit='mmoll'):
    lines = np.array([[lims, lims],
        [[lims[0], 70/1.2],     [70, 70]],
        [[70, 70],              [lims[0], 70*0.8]],
        [[70, lims[1]],         [70*0.8, lims[1]*0.8]],
        [[70/1.2, lims[1]/1.2], [70, lims[1]]],
        [[lims[0], 70],         [180, 180]],
        [[180, 180],            [lims[0], 70]],
        [[180, lims[1]],        [70, 70]],
        [[70, 70],              [70*1.2, lims[1]]],
        [[240, 240],            [70, 180]],
        [[240, lims[1]],        [180, 180]],
        [[182*(5/7), 180],      [lims[0], 70]],
        [[70, lims[1]],         [180, lims[1]+110]]])

    s = 1/mmoll_mgdl if unit == 'mmoll' else 1

    lines *= s

    for x, y in lines:
        ax.plot(x, y, c='k')

    ax.set_xlim(np.array(lims)*s)
    ax.set_ylim(np.array(lims)*s)

    texts = np.array([(30, 15, 'A'),
        (414, 260, 'B'),
        (324, 414, 'B'),
        (160, 414, 'C'),
        (160, 15, 'C'),
        (30, 140, 'D'),
        (414, 120, 'D'),
        (30, 260, 'E'),
        (414, 15, 'E'),
        ])
    texts = pd.DataFrame(texts, columns=['x', 'y', 'text'])
    texts[['x', 'y']] = texts[['x', 'y']].astype(float) * s

    for _, (x, y, text) in texts.iterrows():
        ax.text(x, y, text, fontsize=15)

def plot_clarke_error_grid(output, target, unit='mmoll', save_to=None):
    clarke = clarke_error_grid(output, target, unit)
    clarke = pd.DataFrame({c: mask.flatten().numpy() for c, mask in clarke.items()})

    colors = sns.color_palette('RdYlGn_r', len(clarke.keys()))
    colors[2] = (246/256, 245/256, 128/256)#(1, 1, 128/256)
    colors = {k:c for k, c in zip(clarke.keys(), colors)}

    df = pd.DataFrame(torch.stack([output.flatten(), target.flatten()]).T.numpy(), columns=['output', 'target'])
    df['clarke'] = (clarke == True).idxmax(axis=1)

    fig, ax = plt.subplots(figsize=(8,7))
    # note: we cannot use scatterplot in one line because it does not use hue correctly when areas of the CEG are missing
    for k in clarke.keys():
        sns.scatterplot(data=df[df['clarke'] == k], x='target', y='output', color=colors[k], alpha=.5, s=5)
    format_clarke_error_grid(ax, unit=unit)

    plt.xlabel('True Glucose Concentration (mmol/L)')
    plt.ylabel('Predicted Glucose Concentration (mmol/L)')

    #plt.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.06), fancybox=False)
    plt.title(f"Clarke Error Grid\n"+'    '.join(["{:s}: {:.2f}%".format(k, clarke[k].sum()/len(df)*100) for k in clarke.keys()]), )

    if save_to:
        plt.savefig(save_to+'.pdf', bbox_inches='tight')
        plt.savefig(save_to+'.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()