# Copyright (C) 2023 Miroslav Vořechovský and Jan Mašek
# MIT licence https://en.wikipedia.org/wiki/MIT_License

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import matplotlib.cm as cm
from itertools import cycle
from scipy.stats import norm, rankdata

def plot_2D_view_of_tiling(nv, ns, t, x, ax, vars_to_plot=[0,1]):
    '''
    Plots a 2D view (scatterplot) of sample with two selected variables into the supplied axis. 
    Uses the supplied information about the number of tiles and number of point per tile to 
    color the displayed points.

    Parameters:
        nv (int): Dimension of the supplied (tiled) design.
        ns (int): Number of points in each tile.
        t (int): The number of strata (tiles) along each dimension.
        x : the design (np.array) to be visualized. Shape: (ns*t**nv, nv)
        ax : axis to be used for plotting
        vars_to_plot : two variables (coordinates) to be plotted. Must be selected from range [0,nv-1]

    '''
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    
    # grid 
    grid_points = np.linspace(  0,1, t+1, endpoint=True) #left  probability bound
    ax.xaxis.set_ticks(grid_points)
    ax.yaxis.set_ticks(grid_points)
    ax.grid(True)
    
    u=vars_to_plot[0]
    v=vars_to_plot[1]
    
    # plot all points as empty circles
    ax.scatter(x[:,u],x[:,v],s=50, facecolors='none', edgecolors='k', alpha = 0.4)
    
    # First reshape the 2d array x into a desired shape
    # rows of x are tiles, columns are dimensions
    block_list = [t for i in range(nv)]  + [ns] # prepare a list of shapes in the form (t,t,...,t,ns)
    ur = np.reshape(x[:,u],block_list)#, order='F'
    vr = np.reshape(x[:,v],block_list)#, order='F'

    #T=t**nv unique colors for groups of ns point in each tile
    idx = np.random.permutation(np.linspace(0, 1,t**nv) )
    colors = iter(cycle(cm.rainbow(idx))) 

    for t1 in range (t):
        for t2 in range (t):
            ax.scatter(ur[t1,t2,:],vr[t1,t2,:],s=300/(ns*t), edgecolors=next(colors),  facecolors='none')
    
    #projections onto axes:
    ax.scatter(  x[:,u],0*x[:,v],s=50, marker = "|", color='k')
    ax.scatter(0*x[:,u],  x[:,v],s=50, marker = "_", color='k')
    
def plot_3D_view_of_tiling(nv, ns, t, x, ax, vars_to_plot=[0,1,2]):
    '''
    Plots a 2D view (scatterplot) of sample with two selected variables into the supplied axis. 
    Uses the supplied information about the number of tiles and number of point per tile to 
    color the displayed points.

    Parameters:
        nv (int): Dimension of the supplied (tiled) design.
        ns (int): Number of points in each tile.
        t (int): The number of strata (tiles) along each dimension.
        x : the design (np.array) to be visualized. Shape: (ns*t**nv, nv)
        ax : axis to be used for plotting
        vars_to_plot : three variables (coordinates) to be plotted. Must be selected from range [0,nv-1]

    '''
    u = vars_to_plot [0]
    v = vars_to_plot [1]
    w = vars_to_plot [2]
    
    ax.set_proj_type('ortho') 
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$x_3$')
    
    # grid
    grid_points = np.linspace(  0,1-1/t, t, endpoint=True) #left  probability bound
    ax.xaxis.set_ticks(grid_points)
    ax.yaxis.set_ticks(grid_points)
    ax.zaxis.set_ticks(grid_points)
    ax.grid(True)

    # plot all points as empty circles
    ax.scatter(x[:,u], x[:,v], x[:,w],s=50, facecolors='none', edgecolors='k', alpha = 0.2)

    block_list = [t for i in range(nv)]  + [ns] # prepare a list of shapes in the form (t,t,...,t,ns)
    ur = np.reshape(x[:,u],block_list)#, order='F'
    vr = np.reshape(x[:,v],block_list)#, order='F'
    wr = np.reshape(x[:,w],block_list)#, order='F'
    
    #T=t**nv unique colors for groups of ns point in each tile  
    idx = np.random.permutation(np.linspace(0, 1,t**nv) )
    
    colors = iter(cycle(cm.rainbow(idx))) 
    # plot all points colored according to the blocks as empty circles
    for t1 in range (t):
        for t2 in range (t):
            for t3 in range (t):
                ax.scatter(ur[t1,t2,t3,:],vr[t1,t2,t3,:],wr[t1,t2,t3,:],s=300/(ns*t), edgecolors=next(colors),  facecolors='none')  
    
    #projections onto axes:
    ax.scatter(  x[:,u],0*x[:,v],0*x[:,w],s=50, marker = "|", color='k')
    ax.scatter(0*x[:,u],  x[:,v],0*x[:,w],s=50, marker = "_", color='k')
    ax.scatter(0*x[:,u],0*x[:,v],  x[:,w],s=50, marker = "_", color='k')
    
def eval_function(x,function_name='prodexp'):
    '''
    Calculates the estimation of the integral of selected function 
    via the arithmetical mean of values in the supplied experiment design x.

    Parameters:
        x : the design (np.array) in which axis 1 is the dimension of the function (number of input random variables nv)
        function_name : string specifying the function

    Returns:
        np.array: of results in the shape provided through the design x, apart from axis 1,
        which is seen as the number of input variables nv.
    '''
    if function_name == 'prodexp':
        return np.mean( np.prod( np.exp(-norm.ppf(x)**2), axis=1 ) )
        
        


