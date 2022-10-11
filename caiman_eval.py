import sys, os
from copy import copy
from pathlib import Path
from tqdm import tqdm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tifffile

import twoplib

from caiman.source_extraction.cnmf import cnmf as cnmf

def get_footprints(cnmf):
    """ 
    getting the spatial footprints from a cnmf obj
    reshape cnmf.estimates.A into image format 

    returns array of shape n_comps, xpx, ypx
    """
    A = cnmf.estimates.A
    n_comps = A.shape[1]
    xpx, ypx = cnmf.dims
    F = np.zeros((n_comps, ypx, xpx))
    for i in range(A.shape[1]):
        F[i,:,:] = A[:,i].reshape(ypx,xpx).toarray()

    F = F.swapaxes(1,2) # weird
    return F


def calc_coords_from_footprints(F):
    """ defines a central coordinate for each spatial  footprint.
    currently: center of gravity """

    coords = []
    for i,f in enumerate(F):
        y = np.argmin((np.cumsum(np.sum(f,axis=0)) / np.cumsum(np.sum(f,axis=0))[-1] - 0.5)**2)
        x = np.argmin((np.cumsum(np.sum(f,axis=1)) / np.cumsum(np.sum(f,axis=1))[-1] - 0.5)**2)
        coords.append((x,y))
    return coords

def filter_by_border_dist(F, coords=None, min_dist=8):
    """ footprint is bad if it's coordinate is too close to
    the border

    min_dist is in pixels, makes sense to set it to 1x soma diameter

    TODO operate on coords or footprints? 
    pro coords: it's an operation on coords
    pro F: it filters essentially F. If there are more metrics, maybe they all should be on F?
    """
    
    if coords is None: # avoiding the discussion
        coords = calc_coords_from_footprints(F)
    
    n_comps, xpx, ypx = F.shape
    good_inds = np.ones(n_comps,dtype='bool')

    for i in range(n_comps):
        x,y = coords[i]
        if x < min_dist or x > xpx-min_dist or y < min_dist or y > ypx-min_dist:
            good_inds[i] = False
    return good_inds
    

# metrics to evaluate footprints
# TODO unify  - either pass optional coords arg for these metrics
# or remove it for the filter_by ... 

def compactness(F, size, thresh):
    """ calculates the ratio of signal inside / outside 
    of a bounding box at the size of the soma """
    coords = calc_coords_from_footprints(F)
    slices = twoplib.make_slices(coords, size)

    ratios = []
    for i, sl in enumerate(slices):
        Fi = np.sum(F[sl]) # inner
        f = copy(F[i])
        f[sl] = 0
        Fo = np.sum(f) # outer
        ratios.append(Fi/Fo)
        
    good_inds = np.array(ratios) > thresh
    print("good/bad: %i/%i" % (np.sum(good_inds),good_inds.shape[0]))
    return good_inds, ratios

def fwhm(F, size, thresh):
    """ full width half maximum, in pixels """
    coords = calc_coords_from_footprints(F)
    slices = twoplib.make_slices(coords, size)

    sds = []
    Fs = [F[i][sl] for i,sl in enumerate(slices)]
    for i in range(len(Fs)):
        f = Fs[i]
        try:
            profile = np.sum(f, axis=0) + np.sum(f, axis=1)
            profile = profile / profile.max() # norm
            fwhm = np.sum(profile > 0.5)
            sds.append(fwhm)
        except:
            sds.append(np.nan)
    good_inds = np.array(sds) < thresh
    return good_inds, sds


def plot_footprints(Image, F, save=None):
    """ plotting helper, footprints over image
    save is optional path to which image is saved """

    fig, axes = plt.subplots()

    # background image
    vmin, vmax = np.percentile(Image.flatten(), (5,99.5))
    axes.matshow(Image, cmap='gray', vmin=vmin, vmax=vmax)

    # all the same color
    # rois = F.max(axis=0)
    # roi_mask = np.zeros(rois.shape)
    # roi_mask[:] = np.nan
    # roi_mask[np.where(rois > 0.025)] = 1.0
    # axes.matshow(roi_mask,vmin=0,vmax=1,cmap='cool', alpha=0.9)

    # each with a different color
    roi_mask = np.zeros(Image.shape)
    roi_mask[:] = np.nan
    ix = np.arange(F.shape[0])
    np.random.shuffle(ix)
    for i in ix:
        roi_mask[np.where(F[ix[i],:,:] > 0.05)] = i
    axes.matshow(roi_mask, vmin=0, vmax=F.shape[0], cmap='turbo', alpha=0.9)

    axes.set_xticks([])
    axes.set_yticks([])

    fig.tight_layout()
    if save:
        os.makedirs(Path(save.parent), exist_ok=True)
        fig.savefig(save, dpi=600)

    return axes

def plot_footprints_eval(Fs, good_inds, save=None):
    """ plot result of footprint evaluation, footprints
    on a grid

    save is optional path to which image is saved """

    nrows = np.ceil(np.sqrt(len(Fs))).astype('int32')
    Z = twoplib.tile_slices(Fs, nrows=nrows)
    size = Fs[0].shape[1] # safe for both xy and txy

    # first plot all in grey
    fig, axes = plt.subplots()
    axes.matshow(Z, cmap='Greys_r')

    # then remove bad from Fs
    Fsc = copy(Fs)
    q = np.ones(shape=(size, size))
    q[:] = np.nan
    for i in range(good_inds.shape[0]):
        if good_inds[i] == False:
            Fsc[i] = copy(q)
        else:
            pass

    # tile and replot in color
    Z = twoplib.tile_slices(Fsc, nrows=nrows)
    cmap = mpl.cm.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0)
    axes.matshow(Z, cmap=cmap)
    fig.tight_layout()
    
    if save:
        os.makedirs(Path(save.parent), exist_ok=True)
        fig.savefig(save, dpi=600)

    return axes

