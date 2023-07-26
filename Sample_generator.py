# Copyright (C) 2023 Miroslav Vořechovský and Jan Mašek
# MIT licence https://en.wikipedia.org/wiki/MIT_License

import numpy as np
from scipy.stats import norm, rankdata, qmc
from scipy import stats
from IPython.display import FileLink, FileLinks

################## root sampling methods ##################### 
def get_LHS_median_sample(Ns, dim):
    return (get_LHS_ranks(Ns,dim) + 0.5)/Ns
    
def get_scrambled_Halton_sample(Ns, dim):
    sampler = qmc.Halton(d=dim, scramble=True)
    return sampler.random(n=Ns)

################## tiling ##################### 
def strat_sample_by_tiling(nv, ns, t, tile_type = 'MC', median = True, my_tile = None, one_tile = True,\
                               var_perms = False, rand_revers = False, 
                               t_shifts = False, b_shifts = False, large_shifts = False):
    '''
    Tile a point sample to generate a tiled point set.

    Parameters:
        nv (int): Dimension of the design domain.
        ns (int): Number of points in each tile.
        t (int): The number of strata (tiles) along each dimension.
        my_tile (np.array): Provided point sample that will be used for tiling.
                            If my_tile is None, random sampling will be conducted according to tile_type.
                            Shape: (ns, nv)
        tile_type (str): Sampling method used for generation of tiles.
                         Options: 'LH' (Latin Hypercube Sample with random pairing), 'SRS' (Simple Random Sampling).
        median (bool): Triggers point positions in strata median for tile_type='LH'.  
        one_tile (bool): If True, tiling from an identical tile sample.
                         If False, each tile is generated randomly.
        var_perms (bool): Triggers random variable permutations.
        rand_revers (bool): Triggers random variable reversions.
        t_shifts (bool): Triggers minor coordinate shifts to achieve regular 1D LH-like projections, see Eq. (5).
        b_shifts (bool): Pillar shifting, see Sec. 3.1.
        large_shifts (bool): Triggers large coordinate shifts for high dimension cluster removal, see Eq. (6).

    Returns:
        np.array: Tiled point set. 2D numpy array of Ns coordinates in nv dimensions. (Ns = ns * t ** nv)
    '''
    
    # Total number of strata (tiles)
    T = t**nv
    
    # prepare all T = t**nv tiles with ns point each.
    # these tiles have edge length 1 and they are located in the unit hypercube starting at the origin
    # shape: (T,ns,nv) 
    tiles = get_T_tiles_in_unit_hypercube(nv, ns, t, tile_type, median, my_tile, one_tile)
    
    ###################################################################################
    # perform the desired scrambling operations
    # 
    if rand_revers:
        rev = np.random.choice([0,1], t**nv * nv, p=[0.5, 0.5]).reshape((t**nv, nv))
        # tiles and directions with rev==0 are reversed (all nsim points in a tile)
        #                      with rev==1 are kept untouched
        tiles = (1-rev[:, None ,:])*(1-tiles) \
                 + rev[:, None ,:] *(  tiles)
        
    if var_perms:
        for r in range(T):
            var = np.random.permutation(nv)
            tiles[r] = np.take(tiles[r], var, axis=1) #reorder variables according to given permutation 'var'
        
    # origins = 'bottom left' corners of all tiles in format (shape t**nv, nv)  
    o = strat_sample_corners(nv, t)
    so = o.copy() #copy of the origins
    
    if t_shifts:
        ds = 1./(ns*t**nv) #the smallest distance between point projections along each variable
        if tile_type == 'MC':
            ds = 1./(ns*t)
        # half of ds and half of original LH distance between two points along a line
        shift_correction = 0.5*ds - 1/(2*t*ns) #corrects for true LHS for t even. Relieves the need of periodic shift
        
        s = get_shifts_of_tiles_individual(t,nv,o)
        #apply shifts to the origins
        so = o + s*ds + shift_correction
        if large_shifts: #additional shifts to break small clouds caused by tiny shifts
            so += s*ds*ns*t 
    
    if b_shifts:
        ds = 1./(ns*t**nv) #the smallest distance between point projections along each variable
        # half of ds and half of original LH distance between two points along a line
        shift_correction = 0.5*ds - 1/(2*t*ns) #corrects for true LHS for t even. Relieves the need of periodic shift
        s = get_shifts_of_tiles_block(t,nv,o)
        so = o + s*ds + shift_correction
        
    # now compose the tiles to a design by adding them to the left corners (potentially shifted)
    # and finally scale the tile down to edge length 1/t
    x_3d = so[:, None , : ] + tiles/t #copy the single tile to all the strata
        
    #periodic correction of the shift in each tile (compare to the original left bound xo+1/t)
    if  t_shifts:# and tile_type == 'MC':
        remainders_after_periodic_shifts = (x_3d - o[:,None,:])%(1/t) #remainders to be added to tile left origins
        x_3d = o[:,None,:] + remainders_after_periodic_shifts #add remainders to tile left origins
        #x_3d = np.where(x_3d > o[:,None,:] + 1./t, x_3d - ((x_3d - o[:,None,:])//(1/t))/t, x_3d)
        #x_3d = np.where(x_3d > o[:,None,:] + 1./t, x_3d - 1./t, x_3d)
        #x_3d = np.where(x_3d < o[:,None,:]       , x_3d + 1./t, x_3d)
        
    if b_shifts:
        x_3d = np.where(x_3d > 1, x_3d - 1., x_3d)
        
    return np.reshape(x_3d,(ns * t**nv,nv))



################# auxiliary methods ###########################

def generate_permutations(t,nv, randomized = True):
    rand_s = np.zeros([nv,t])

    if randomized:
        for v in range(nv):
            rand_s[v,:] = np.random.permutation(np.arange(t))
    else:
        for v in range(nv):
            rand_s[v,:] = np.arange(t)
    return rand_s
    
def get_LHS_ranks(Ns,dim):
    ranks = np.zeros((Ns, dim))
    arr = np.arange(Ns)
    for col in range (dim): #each column separately
        perm = np.random.permutation(arr)
        ranks[:,col] = perm

    return ranks

def get_shifts_of_tiles_block(t,nv,o):
    s = np.zeros((t**nv, nv)) #integer shift for every tile (rows) and along every direction (cols)
    bounds = np.zeros( nv )
    var_list = list(np.arange(0, nv, 1, dtype=int))
    for v in range(nv): #index along individual directions to make a shift
        shift_perm = np.arange(t**(nv-1))
        shift_perm = np.random.permutation(shift_perm)
        v_mask = (o[:,v]==0)
        v_idx = np.where(v_mask)[0] #indices of rows in 's' matrix. fullfiles by t**(nv-1)
        #cur_list = var_list[:v] + var_list[v+1:] #indices of directions except v
        for index in range(len(v_idx)):
            gmask = np.full((t**nv), True) #set all tiles to True
            bounds = o[v_idx[index],:]
            for dim in range(nv):
                if dim != v:
                    gmask = np.logical_and(gmask , np.isclose(o[:,dim], bounds[dim], rtol=0.1/t)   )#   o[:,v]==r/t

            x_idx = np.where(gmask)[0] #indices of rows in 's' matrix
            s[x_idx,v] = shift_perm[index] 
    return s

def get_shifts_of_tiles_individual(t,nv,o):
    s = np.zeros((t**nv, nv))
    for v in range(nv): #index along individual directions to make a shift
        for r in range(t): #index all (slices of) tiles along a given direction
            shift_perm = np.arange(t**(nv-1))
            shift_perm = np.random.permutation(shift_perm)
            mask = np.isclose(o[:,v], r/t, rtol=0.1/t)#   o[:,v]==r/t
            x_idx = np.where(mask)[0] #indices of rows in 's' matrix
            for index in range(len(x_idx)):
                s[x_idx[index],v] = shift_perm[index]
    return s

def get_one_tile_in_unit_hypercube(nv, ns, t, tile_type = 'MC', median = True):
    u = 0.5 #for LHS-median
    if tile_type == 'LH':
        LHS_ranks = get_LHS_ranks(ns,nv)
        if not median: #LHS-random selection from 1D strata
            u = np.random.rand(ns, nv)
        tile = (LHS_ranks+u)/ns
    elif tile_type == 'Halton':
        sampler = qmc.Halton(d=nv, scramble=True)
        tile = sampler.random(n=ns) #a single Halton tile
    else: #if tile_type == 'MC':
        tile = np.random.rand(ns,nv) #a single MC tile 
    return tile

def get_T_tiles_in_unit_hypercube(nv, ns, t, tile_type = 'MC', median = True, my_tile = None, one_tile = True):
    T = t**nv
    
    if one_tile: # repeat a single tile
        if (my_tile is None):  #check whether the tile is provided
            tile = get_one_tile_in_unit_hypercube(nv, ns, t, tile_type, median)

        else:
            if(my_tile.shape[0] != ns or my_tile.shape[1] != nv):
                #check whether the size has a correct size
                print(f'Wrong tile size. Either {my_tile.shape[0]} in not ns={ns}, or nv={my_tile.shape[1]} is not nv={nv}')
                return None
            tile = my_tile
   
        #copy the tile into T individual tiles - obtain a 3D array (shape: T,ns,nv)
        tiles = np.tile(tile,(T, 1,1))
    
    else: # generate independent tiles
        tiles=np.zeros((T,ns,nv))#space for T individual tiles, each ns*nv (shape: T,ns,nv)
        for tile in range(T):
            #write a single tile in format ns points (rows) and nv cols
            tiles[tile,:,:] = get_one_tile_in_unit_hypercube(nv, ns, t, tile_type, median)

    return tiles

def strat_sample_corners(nv, t):
    pa = np.linspace( 0 , 1-1/t , t , endpoint=True) #left tile probability bounds
    pa_list = [pa] * (nv)
    p = np.meshgrid(*pa_list)#,indexing='ij')
    x = np.array(p).reshape((nv, t**nv)).T#returns the standard "design format" (shape t**nv, nv)
    return x#np.flip(x, axis=1)
