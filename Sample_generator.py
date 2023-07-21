#!/usr/bin/env python
# coding: utf-8

# In[1]:


from SampleTiler import strat_sample_by_tiling
from Tools import plot_2D_view_of_tiling, plot_3D_view_of_tiling
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import FileLink, HTML
from ipywidgets import Output


# Performs a sample construction using the Sample Tiling metod. 
# For details see publication: 
# 
# [1] Jan Mašek & Miroslav Vořechovský: Optimal sampling in a hypercube: stratified sample tiling. Advances in Engineering Software, 2023
#  
# 
#  
# ### Parameters
# | Parameter      | Description                                                                                    | Type          |
# | :-------------- |:----------------------------------------------------------------------------------------------- |:------------- |
# | nv             | Dimension of the design domain                                                                  | Integer       |
# | ns             | Number of points in each tile                                                                   | Integer       |
# | t              | The number of strata (tiles) along each dimension                                               | Integer       |
# | my_tile        | Provided point sample that will be used for tiling. If my_tile is None, random sampling will be conducted according to tile_type. | np array (ns,nv)  |
# | tile_type      | Sampling method used for generation of tiles. Options: 'LH' (Latin Hypercube Sample), 'SRS' (Simple Random Sampling) | String        |
# | one_tile       | If True, tiling from an identical tile sample. If False, each tile is generated randomly.     | Boolean       |
# | var_perms      | Triggers random variable permutations                                                          | Boolean       |
# | rand_revers    | Triggers random variable reversions                                                            | Boolean       |
# | t_shifts       | Triggers minor coordinate shifts to achieve regular 1D projections, see Eq. (5) in [1].                            | Boolean       |
# | b_shifts       | Triggers pillar shifting, see Sec. 3.1 in [1].                                                                     | Boolean       |
# | large_shifts   | Triggers large coordinate shifts for high dimension cluster removal, see Eq. (6)  in [1].                            | Boolean       |
# 
# ### Returns
# | Parameter      | Description                                                                         | Type         |
# | :------------- |:------------------------------------------------------------------------------------|:-------------|
# | Point sample   | Tiled point set. 2D numpy array of Ns coordinates in nv dimensions. (Ns = ns * t\*\*nv)                | np array (Ns,nv)  |
# 
# Feel free to explore the properties of tiled point sets for various input parameters!

# In[10]:


# The number of strata (tiles) along each dimension
t = 5

# The number of points within each tile
ns = 6

# The dimension of the design domain
nv = 3

# The total number of points within the sample
Ns = ns*t**nv

# Sample tiling
x = strat_sample_by_tiling(nv, ns, t, tile_type = 'LH', median = True, my_tile = None, \
                           one_tile = True, var_perms = False, rand_revers = False, \
                           t_shifts = True, b_shifts = False, large_shifts = True)

# Display of a 2D subspace, plot variables in vars_to_plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
plot_2D_view_of_tiling(nv, ns, t, x, ax, vars_to_plot = [0,1])

# Save the 2D scatter plot to a pdf file
plt.savefig("scatter_2D.pdf",dpi = 300,bbox_inches = "tight",pad_inches = 0.01)
display(FileLink("scatter_2D.pdf"))

# Save the sample to a text file
np.savetxt('sample.txt', x, delimiter='\t')
display(FileLink('sample.txt'))

# Save the sample as binary
np.save('sample.npy' ,x)
display(FileLink('sample.npy'))

plt.show()

"""
#Smazat
loaded_array = np.load('sample.npy')
print(loaded_array)
plot_2D_view_of_tiling(nv, ns, t, loaded_array, ax, vars_to_plot = [0,1])
plt.show()
"""


# In[ ]:


# Display of a 3D subspace, plot variables in vars_to_plot
if nv>2:
    from mpl_toolkits.mplot3d import Axes3D
    get_ipython().run_line_magic('matplotlib', 'notebook')
    fig = plt.figure(figsize=(7, 7), layout='constrained')
    ax3 = fig.add_subplot(projection='3d')
    plot_3D_view_of_tiling(nv, ns, t, x, ax3, vars_to_plot=[0,1,2])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




