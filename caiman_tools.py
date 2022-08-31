import sys, os
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cv2
try:
    cv2.setNumThreads(8)
except:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params

import logging
logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

os.chdir("/home/georg/code/twoplib") # local hack - to be removed
import twoplib

"""
##     ## ######## ##       ########  ######## ########
##     ## ##       ##       ##     ## ##       ##     ##
##     ## ##       ##       ##     ## ##       ##     ##
######### ######   ##       ########  ######   ########
##     ## ##       ##       ##        ##       ##   ##
##     ## ##       ##       ##        ##       ##    ##
##     ## ######## ######## ##        ######## ##     ##
"""

def memmap_to_array(fname):
    """ loads memmap from disk using caimans tools
    then reshapes to a (t,x,y) stack """
    Yr, dims, T = cm.load_memmap(fname)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    return images


def save_caiman_params(opts, path):
    """ stores both a json and a pickle """

    # based on https://stackoverflow.com/a/52604722
    def default(obj):
        if type(obj) == slice:
            return None
        if type(obj) == Path:
            return str(obj)
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        raise TypeError('Unknown type:', type(obj))

    path = Path(path) # just in case

    # json
    lines = json.dumps(opts.to_dict(), indent=2, default=default)
    with open(path.with_suffix('.json'),'w') as fH:
        fH.writelines(lines)

    # pickle
    with open(path.with_suffix('.pckl'),'wb') as fH:
        pickle.dump(opts,fH)

    print("caiman parameters written to %s" % path)

def load_caiman_params(path):
    """ loads just the pickle """
    path = Path(path).with_suffix('.pckl') # just in case
    with open(path ,'rb') as fH:
        opts = pickle.load(fH)
    return opts

"""
########     ###    ########     ###    ##     ##  ######
##     ##   ## ##   ##     ##   ## ##   ###   ### ##    ##
##     ##  ##   ##  ##     ##  ##   ##  #### #### ##
########  ##     ## ########  ##     ## ## ### ##  ######
##        ######### ##   ##   ######### ##     ##       ##
##        ##     ## ##    ##  ##     ## ##     ## ##    ##
##        ##     ## ##     ## ##     ## ##     ##  ######
"""

def setup_caiman_params(opts, fnames, meta, meta_si, mROI_ix=None):
    """
    MAJOR TODO: make this load params from disk
    or: make this function be the place that stores hardcoded stuff?
    have an own mesoscope specific function?
    """
    
    # image temporal and spatial properties / resolution
    if mROI_ix is None: # meso support
        degx, degy = meta['RoiGroups']['imagingRoiGroup']['rois']['scanfields']['sizeXY']
        xpx,ypx = meta['RoiGroups']['imagingRoiGroup']['rois']['scanfields']['pixelResolutionXY']
    else:
        degx, degy = meta['RoiGroups']['imagingRoiGroup']['rois'][mROI_ix]['scanfields']['sizeXY']
        xpx,ypx = meta['RoiGroups']['imagingRoiGroup']['rois'][mROI_ix]['scanfields']['pixelResolutionXY']

    um_per_deg = 15 # FIXME hardcoded? very much a microscope specific value
    dxy = (degx * um_per_deg / xpx, degy * um_per_deg / ypx) # pixel size

    frame_rate = twoplib.read_float_from_meta(meta_si, "SI.hRoiManager.scanFrameRate")

    data_dict = {
        'fnames': fnames, # the raw files
        'dxy': dxy,
        'fr': frame_rate,
        'decay_time': 0.4 # FIXME HARDCODED!
    }
    opts.change_params(params_dict=data_dict)


    # MOCO
    max_shift_um = (24., 24.)       # maximum shift in um
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
    pw_rigid = True       # flag to select rigid vs pw_rigid motion correction
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)] # maximum allowed rigid shift in pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)]) # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24) # overlap between pathes (size of patch in pixels: strides+overlaps)
    max_deviation_rigid = 3 # maximum deviation allowed for patch with respect to rigid shifts

    mc_dict = {
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': 'copy',
        'use_cuda': True
        # 'indices': (slice(None,None),slice(None,None))
    }
    opts.change_params(params_dict=mc_dict)


    # CNMF - parameters for source extraction and deconvolution
    p = 1                    # order of the autoregressive system
    gnb = 2                  # number of global background components
    merge_thr = 0.85         # merging threshold, max correlation allowed
    rf = 25                  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 6          # amount of overlap between the patches in pixels
    K = 5                    # number of components per patch
    # gSig = [3, 3]            # expected half size of neurons in pixels

    # infer for soma size 10um
    gSig = list([int(5/v) for v in dxy]) # TODO deal with this later

    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    method_init = 'greedy_roi'
    ssub = 2                     # spatial subsampling during initialization
    tsub = 2                     # temporal subsampling during intialization

    # parameters for component evaluation
    cnmf_dict = {
                # 'fnames': None,
                # 'fr':frame_rate,
                'p': p,
                'nb': gnb,
                'rf': rf,
                'K': K,
                'gSig': gSig,
                'stride': stride_cnmf,
                'method_init': method_init,
                'rolling_sum': True,
                'merge_thr': merge_thr,
                #'n_processes': n_processes,
                'only_init': True,
                'ssub': ssub,
                'tsub': tsub
    }

    opts.change_params(params_dict=cnmf_dict)


    # evaluation
    min_SNR = 3  # signal to noise ratio for accepting a component
    rval_thr = 0.85  # space correlation threshold for accepting a component
    cnn_thr = 0.99  # threshold for CNN based classifier
    cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

    eval_dict = {
        'min_SNR': min_SNR,
        'rval_thr': rval_thr,
        'use_cnn': True,
        'min_cnn_thr': cnn_thr,
        'cnn_lowest': cnn_lowest
    }
    opts.change_params(params_dict=eval_dict)

    return opts


"""
##     ##  #######   ######   #######
###   ### ##     ## ##    ## ##     ##
#### #### ##     ## ##       ##     ##
## ### ## ##     ## ##       ##     ##
##     ## ##     ## ##       ##     ##
##     ## ##     ## ##    ## ##     ##
##     ##  #######   ######   #######
"""

def run_caiman_moco(opts):
    """ 
    runs the NoRMCorre algorithm
    https://github.com/flatironinstitute/NoRMCorre
    returns the updated parameters and the motion correction object
    """

    # start local cluster
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    opts.change_params(params_dict=dict(n_processes=n_processes))

    # run moco 
    fnames = opts.get('data', 'fnames')
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True) # this saves in order F by default
    # see https://github.com/flatironinstitute/CaImAn/issues/1000 for my question regarding this

    # update parameters
    opts.change_params(params_dict=dict(mmap_F=mc.mmap_file))
   
    # clean up 
    cm.stop_server(dview=dview)

    return opts, mc


def merge_memmap(opts, remove=True):
    """ this first changes the memmap from F to C internally, then makes
    a concatenated memmmap of the entire recording
    optionally delete all individual memmaps """

    # fname related
    mmap_F_fnames = opts.data['mmap_F']
    fname_mmap_merged = cm.save_memmap(opts.data['mmap_F'], base_name='memmap_', order='C', border_to_0=0)

    mmap_C_fnames = [fname for fname in os.listdir() if fname.startswith('memmap_') and fname.endswith('.mmap')]
    mmap_C_fnames = [fname for fname in mmap_C_fnames if not fname.startswith('memmap__')]
    mmap_C_fnames = list(np.sort(mmap_C_fnames))

    opts.change_params(dict(mmap_C=mmap_C_fnames))

    if remove:
        for fname in mmap_F_fnames:
            print("removing %s" % fname)
            os.remove(fname)
        opts.change_params(dict(mmap_F=None))

        for fname in mmap_C_fnames:
            print("removing %s" % fname)
            os.remove(fname)
        opts.change_params(dict(mmap_C=None))

    return fname_mmap_merged, opts


"""
 ######  ##    ## ##     ## ########
##    ## ###   ## ###   ### ##
##       ####  ## #### #### ##
##       ## ## ## ## ### ## ######
##       ##  #### ##     ## ##
##    ## ##   ### ##     ## ##
 ######  ##    ## ##     ## ##
"""

def run_caiman_cnmf(opts, images):
    """
    runs caimans CNMF / deconvolution algorithm
    images is a (t,x,y) stack (memmaped)
    """

    # starting a new cluster
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    # first CNMF run
    opts.change_params({'p': 0}) # turning off deconvolution

    # RUN
    cnmf1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnmf1.fit(images)
    print("components after initial seed %i" % cnmf1.estimates.A.shape[1])

    # and save result
    outpath = cnmf1.mmap_file[:-5]+'cnmf_init.hdf5'
    cnmf1.save(outpath)
    print("saved to %s" % outpath)

    # rerun CNMF with seeded 
    opts.change_params({'p': 1}) # turning deconvolution back on
    cnmf2 = cnmf1.refit(images, dview=dview) # images are here needed?

    """ TODO understand what is going on here, why new object, why does it get ref to images """

    print("second cnmf run done")
    print("components before selection %i" % cnmf2.estimates.A.shape[1])

    outpath = cnmf2.mmap_file[:-5] + 'cnmf2_all.hdf5'
    cnmf2.save(outpath)
    print("saved to %s" % outpath)

    # EVALUATION
    cnmf2.estimates.evaluate_components(images, opts, dview=dview)

    # update object with selected components
    cnmf2.estimates.select_components(use_object=True)
    print("components after selection %i" % cnmf2.estimates.A.shape[1])

    # save
    cnmf2.mmap_file[:-5] + 'cnmf2_sel.hdf5'
    cnmf2.save(outpath)

    cm.stop_server(dview=dview)

    return cnmf1, cnmf2

"""
########  #### ########  ######## ##       #### ##    ## ########
##     ##  ##  ##     ## ##       ##        ##  ###   ## ##
##     ##  ##  ##     ## ##       ##        ##  ####  ## ##
########   ##  ########  ######   ##        ##  ## ## ## ######
##         ##  ##        ##       ##        ##  ##  #### ##
##         ##  ##        ##       ##        ##  ##   ### ##
##        #### ##        ######## ######## #### ##    ## ########
"""

def caiman_pipeline(folder):
    """ run the entire caiman pipeline on the folder containing the imaging files """
    folder = Path(folder)
    os.chdir(folder)

    fnames = [folder / fname for fname in os.listdir(folder) if fname.endswith('.tif')]
    fnames = list(np.sort(fnames))

    # FOR DEBUG - limit dataset size
    # fnames = fnames[:2] 

    # parameters
    opts = params.CNMFParams()
    meta, meta_si = twoplib.get_meta(fnames[0])
    opts = setup_caiman_params(opts, fnames, meta, meta_si, mROI_ix=None)
    opts_path = folder / 'caiman_params.pckl'
    save_caiman_params(opts, opts_path)

    # run moco
    opts, mc = run_caiman_moco(opts)

    # merge memmap
    fname_mmap_merged, opts = merge_memmap(opts)
    images = memmap_to_array(fname_mmap_merged)

    # cnmf / eval
    cnmf1, cnmf2 = run_caiman_cnmf(opts, images)
    return cnmf1, cnmf2

