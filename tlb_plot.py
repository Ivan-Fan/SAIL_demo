import os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
import plotly.graph_objects as go


def plot_fixed_w_3dbar(df, w, font_size=8):
    
    # set up figure
    fig = plt.figure(figsize=(12,10), dpi=800)
    
    # set up params for plotting
    alpha = 0.5
    x_scale = 0.4
    # cmap = cm.YlGnBu
    cmap = cm.coolwarm
    norm = Normalize(vmin=0, vmax=2)  # Normalize based on the number of methods
    # row_colors = cmap(norm(np.array([2,0.8,0])))
    row_colors = ['#fab8a9', '#f7dcbc', '#bcdef7']

    # add labels for method and parameters
    x_labels = ['SPARTAN','SFA', 'SAX']
    z_labels = np.array([4, 5, 6, 7, 8, 9, 10])

    spartan_tlbs_daa_a = df[(df['method'] == 'SPARTAN') & (df['w'] == w)]['tlb'].values.reshape(-1,1)
    sfa_tlbs_a = df[(df['method'] == 'SFA') & (df['w'] == w)]['tlb'].values.reshape(-1,1)
    sax_tlbs_a = df[(df['method'] == 'SAX') & (df['w'] == w)]['tlb'].values.reshape(-1,1)

    data_top = np.concatenate([spartan_tlbs_daa_a, sfa_tlbs_a, sax_tlbs_a], axis=1)

    # Creating the plots
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for i in range(data_top.shape[1]):
        xpos, ypos = np.meshgrid(x_scale*i, z_labels)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)

        dx = 0.3
        dy = 0.8  # Width and depth of the bars
        dz = data_top[:,i].flatten()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, alpha=alpha, color=row_colors[i])

        # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='white', edgecolor='black', alpha=1.0, zsort='average')

    ax.set_xticks(np.arange(len(x_labels))*x_scale+dx/2)
    ax.set_xticklabels(x_labels, rotation=10, ha='right', fontsize=font_size)
    # ax.xaxis.label.set_position((0.5, 0))  
    ax.set_yticks(z_labels)
    ax.set_yticklabels(z_labels, fontsize=font_size)
    ax.set_ylabel('Alphabet Size', fontsize=font_size)
    # ax.set_zlabel('TLB', fontsize=font_size+1)
    ax.set_zlim(0,1.19)
    ax.set_zticklabels(np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),fontsize=font_size)
    ax.view_init(elev=30, azim=-60)

    # Improve spacing
    plt.tight_layout()

    return fig


def plot_fixed_a_3dbar(df, a, font_size=8):
    
    # set up figure
    fig = plt.figure(figsize=(12,10), dpi=800)
    
    # set up params for plotting
    alpha = 0.5
    x_scale = 0.4
    cmap = cm.YlGnBu
    norm = Normalize(vmin=0, vmax=2)  # Normalize based on the number of methods
    # row_colors = cmap(norm(np.array([2,0.8,0])))
    row_colors = ['#fab8a9', '#f7dcbc', '#bcdef7']

    # add labels for method and parameters
    x_labels = ['SPARTAN','SFA', 'SAX']
    z_labels = np.array([4, 5, 6, 7, 8, 9, 10])

    spartan_tlbs_daa_a = df[(df['method'] == 'SPARTAN') & (df['a'] == a)]['tlb'].values.reshape(-1,1)
    sfa_tlbs_a = df[(df['method'] == 'SFA') & (df['a'] == a)]['tlb'].values.reshape(-1,1)
    sax_tlbs_a = df[(df['method'] == 'SAX') & (df['a'] == a)]['tlb'].values.reshape(-1,1)

    data_top = np.concatenate([spartan_tlbs_daa_a, sfa_tlbs_a, sax_tlbs_a], axis=1)

    # Creating the plots
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for i in range(data_top.shape[1]):
        xpos, ypos = np.meshgrid(x_scale*i, z_labels)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)

        dx = 0.3
        dy = 0.8  # Width and depth of the bars
        dz = data_top[:,i].flatten()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, alpha=alpha, color=row_colors[i])

        # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='white', edgecolor='black', alpha=1.0, zsort='average')

    ax.set_xticks(np.arange(len(x_labels))*x_scale+dx/2)
    ax.set_xticklabels(x_labels, rotation=10, ha='right', fontsize=font_size)
    # ax.xaxis.label.set_position((0.5, 0))  
    ax.set_yticks(z_labels)
    ax.set_yticklabels(z_labels, fontsize=font_size)
    ax.set_ylabel('Word length', fontsize=font_size)
    # ax.set_zlabel('TLB', fontsize=font_size+1)
    ax.set_zlim(0,1.19)
    ax.set_zticklabels(np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),fontsize=font_size)
    ax.view_init(elev=30, azim=-60)

    # Improve spacing
    plt.tight_layout()

    return fig
