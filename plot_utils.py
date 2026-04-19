# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:33:27 2025

@author: risc915d
"""

import matplotlib.pyplot as plt
import matplotlib as mpl


def set_default_plot_properties(fontname='Times New Roman', fontsize=14, grid='on', color='white', 
                                interpreter='tex', paper='A4'):
    """
    Sets the default plot properties for Matplotlib plots.

    Args:
    fontname (str): Font type
    fontsize (int): Font size
    grid (str): Grid 'on' or 'off'
    color (str): Figure's background color
    interpreter (str): Text interpreter ('tex', 'latex', or 'none')
    paper (tuple): Figure size in inches (width, height)

    Example: 
    set_default_plot_properties('Times New Roman', 14, 'on', 'white', 'latex', (10, 8))
    """
    #import seaborn as sns
    #sns.set_style('ticks')
    #sns.set_context('paper')
    #sns.set_palette('colorblind')
    plt.style.use('seaborn-v0_8')  # Start with a nice base style
    
    # Define paper sizes in inches (width, height)
    paper_sizes = {
        'A3': (11.69, 16.54),
        'A3.5': (10.0, 14.14),
        'A4': (8.27, 11.69),
        'A5': (5.83, 8.27),
        'A6': (4.13, 5.83),
        'A6.5': (3.54, 5.0),
        'A7': (2.91, 4.13),
        # Quadratic sizes
        'Q3': (11.69, 11.69),
        'Q3.5': (10.0, 10.0),
        'Q4': (8.27, 8.27),
        'Q5': (5.83, 5.83),
        'Q6': (4.13, 4.13),
        'Q6.5': (3.54, 3.54),
        'Q7': (2.91, 2.91)
    }

    # Set figure size based on paper format
    if isinstance(paper, str):
        if paper.endswith('p'):
            figsize = paper_sizes.get(paper[:-1], (8.27, 11.69))  # Default to A4 if not found
            figsize = figsize[::-1]  # Swap width and height for portrait
        else:
            figsize = paper_sizes.get(paper, (8.27, 11.69))  # Default to A4 if not found
    else:
        figsize = paper  # Assume it's already a tuple of (width, height)
    
    mpl.rcParams.update({
        # Text
        'font.family': fontname,
        'font.size': fontsize,
        'text.usetex': interpreter == 'latex',

        # Axes
        'axes.grid': grid == 'on',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1,
        'axes.titlesize': fontsize + 2,
        'axes.labelsize': fontsize,

        # Grid
        'grid.color': 'gray',
        'grid.linestyle': ':',
        'grid.linewidth': 1,

        # Ticks
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.labelsize": fontsize - 2,
        "ytick.labelsize": fontsize - 2,
        #"xtick.direction": 'in',
        #"ytick.direction": 'in',

        # Figure
        'figure.figsize': figsize,
        'figure.facecolor': color,

        # Legend
        'legend.fontsize': fontsize - 2,
        'legend.frameon': True,
        'legend.framealpha': 0.8,

        # Lines
        'lines.linewidth': 2,
        'lines.markersize': 8,

        # Saving figures
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
    })

    # Enable LaTeX rendering if interpreter is 'latex'
    if interpreter == 'latex':
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # Set color cycle for multiple lines
    #plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    #    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    #    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    #])
