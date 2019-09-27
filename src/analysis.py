import numpy as np
import tifffile
import subprocess
import os

from tabulate  import tabulate
from types     import SimpleNamespace
from pathlib   import Path
from itertools import zip_longest

from matplotlib      import pyplot as plt
from scipy           import ndimage
from scipy.ndimage   import label, zoom
from skimage         import io
from skimage.measure import compare_ssim

# import spimagine
from segtools.numpy_utils import normalize3, perm2, collapse2, splt
from segtools.StackVis import StackVis

# from csbdeep.utils.utils import normalize_minmse
import ipdb

def cat(*args,axis=0): return np.concatenate(args, axis)
def stak(*args,axis=0): return np.stack(args, axis)

def imsave(x, name, **kwargs): return tifffile.imsave(str(name), x, **kwargs)
def imread(name,**kwargs): return tifffile.imread(str(name), **kwargs)
# def imsavefiji(x, **kwargs): return tifffile.imsave('/Users/broaddus/Desktop/stack.tiff', x, imagej=True, **kwargs)
def imsavefiji(x, **kwargs): return tifffile.imsave('stack.tiff', x, imagej=True, **kwargs)

# figure_dir      = Path('/Users/broaddus/Dropbox/phd/notes/denoise_paper/res').resolve()
# experiments_dir = Path('/Volumes/project-broaddus/denoise_experiments').resolve()
figure_dir      = Path('/projects/project-broaddus/denoise_experiments/figures/').resolve(); figure_dir.mkdir(exist_ok=True,parents=True)
experiments_dir = Path('/projects/project-broaddus/denoise_experiments/').resolve()
rawdata_dir     = Path('/projects/project-broaddus/rawdata').resolve()

## load data


def eval_01():
  data = np.load(experiments_dir/'flower/e01/e01_fig2_flower.npz')['rgb']
  names = [
    "raw",
    "n2v",
    "xox",
    "plus",
    "bigplus",
    "8xo8x",
    "xxoxx",
    "xxxoxxx",
    "xxxxoxxxx",
    "xxxxxoxxxxx",
    "xxxxxxoxxxxxx",
    "xxxxxxxoxxxxxxx",
    ]
  perm = [0,1,3,4,2,6,7,8,9,10,11,5,]
  data = data[perm]
  names = list(np.array(names)[perm])
  return SimpleNamespace(data=data,names=names)

def eval_02():
  data = np.load(experiments_dir/'flower/e02/e02_fig2_flower.npz')['rgb']
  names = [
    "n2v",
    "n2v^2",
    "n2v plus",
    "n2v bigplus",
    "n2v xxxxoxxxx",
    "n2v xox",
    "n2v xxoxx",
    ]

  return SimpleNamespace(data=data,names=names)

def eval_flower_gt():
  flower_all = imread(rawdata_dir/'artifacts/flower.tif')
  flower_all = normalize3(flower_all,2,99.6)

  ## The old way: break it up into patches
  # flower_all_patches = flower_all[0].reshape((4,256,4,256)).transpose((0,2,1,3)).reshape((16,256,256))
  # flower_all_patches = flower_all_patches[[0,3,5,12]]
  flower_gt = flower_all.mean(0)
  flower_gt_patches = flower_gt.reshape((4,256,4,256)).transpose((0,2,1,3)).reshape((16,256,256))
  flower_gt_patches = flower_gt_patches[[0,3,5,12]]

  # flower_gt = flower_all.mean(0)
  return flower_gt_patches

def fulldata():
  e01 = eval_01()
  e02 = eval_02()
  gt  = eval_flower_gt()
  dat = SimpleNamespace(e01=e01,e02=e02,gt=gt)
  return dat

def fulldata2():
  ## load the flower dataset and build the GT
  flower_all = imread(rawdata_dir/'artifacts/flower.tif')
  flower_all = normalize3(flower_all,2,99.6)
  flower_gt  = flower_all.mean(0)
  flower_gt_patches = flower_gt.reshape((4,256,4,256)).transpose((0,2,1,3)).reshape((16,256,256))
  flower_gt_patches = flower_gt_patches[[0,3,5,12]]

  ## load the predictions from single-phase models (600th epoch)
  img6 = np.load(experiments_dir / 'flower/e01/flower3_6/epochs_npy/arr_600.npy')    # n2v
  img7 = np.load(experiments_dir / 'flower/e01/flower3_7/epochs_npy/arr_600.npy')    # xox
  img8 = np.load(experiments_dir / 'flower/e01/flower3_8/epochs_npy/arr_600.npy')    # plus
  img9 = np.load(experiments_dir / 'flower/e01/flower3_9/epochs_npy/arr_600.npy')    # bigplus
  img10 = np.load(experiments_dir / 'flower/e01/flower3_10/epochs_npy/arr_600.npy')  # 8xo8x
  img11 = np.load(experiments_dir / 'flower/e01/flower3_11/epochs_npy/arr_600.npy')  # xxoxx
  img12 = np.load(experiments_dir / 'flower/e01/flower3_12/epochs_npy/arr_600.npy')  # xxxoxxx
  img13 = np.load(experiments_dir / 'flower/e01/flower3_13/epochs_npy/arr_600.npy')  # xxxxoxxxx
  img14 = np.load(experiments_dir / 'flower/e01/flower3_14/epochs_npy/arr_600.npy')  # xxxxxoxxxxx
  img15 = np.load(experiments_dir / 'flower/e01/flower3_15/epochs_npy/arr_600.npy')  # xxxxxxoxxxxxx
  img16 = np.load(experiments_dir / 'flower/e01/flower3_16/epochs_npy/arr_600.npy')  # xxxxxxxoxxxxxxx

  names = [
    "raw",
    "n2v",
    "xox",
    "plus",
    "bigplus",
    "8xo8x",
    "xxoxx",
    "xxxoxxx",
    "xxxxoxxxx",
    "xxxxxoxxxxx",
    "xxxxxxoxxxxxx",
    "xxxxxxxoxxxxxxx",
    ]


  data = stak(img6, img7, img8, img9, img10, img11, img12, img13, img14, img15, img16,)

  data[:,[2,4]] = normalize3(np.log(normalize3(data[:,[2,4]],0,99)+1e-7)) ## k-space channels
  data[:,[0,3]] = normalize3(data[:,[0,3]]) ## real space channels
  data[:,1]     = normalize3(data[:,1]) ## mask channel ?
  data = data[:,:,:,0] ## remove channels dim 
  data = cat(stak(np.zeros(data[0,0].shape),data[0,0],data[0,2])[None],data[:,[1,3,4]]) ## move raw to front. reshape to ?,4,256,256

  ## put the trials in a sensible order
  perm = [0,1,3,4,2,6,7,8,9,10,11,5,]
  data = data[perm]
  names = list(np.array(names)[perm])
  e01 = SimpleNamespace(data=data,names=names)

  img1 = np.load(experiments_dir/'flower/e02/flower1_1/epochs_npy/arr_400.npy') # n2v^2
  img2 = np.load(experiments_dir/'flower/e02/flower1_2/epochs_npy/arr_400.npy') # n2v plus
  img3 = np.load(experiments_dir/'flower/e02/flower1_3/epochs_npy/arr_400.npy') # n2v bigplus
  img4 = np.load(experiments_dir/'flower/e02/flower1_4/epochs_npy/arr_400.npy') # n2v xxxxoxxxx
  img5 = np.load(experiments_dir/'flower/e02/flower1_5/epochs_npy/arr_400.npy') # n2v xox
  # img6 = np.load(experiments_dir/'flower/e02/flower1_6/epochs_npy/arr_400.npy') # n2v xxoxx

  ## (N2V, OURS 2class, OURS 3class) , (raw, mask, raw fft, pred, pred fft) , n_samples , channels, y , x
  data = stak(img1, img2, img3, img4, img5, ) #img6)

  ## normalize fft and real space separately
  data[:,[2,4]] = normalize3(np.log(normalize3(data[:,[2,4]],0,99)+1e-7))
  data[:,[0,3]] = normalize3(data[:,[0,3]])
  data[:,1]     = normalize3(data[:,1])

  ## remove channels and pad xy with white
  data = data[:,:,:,0]
  # data = np.pad(data,[(0,0),(0,0),(0,0),(0,1),(0,1)],mode='constant',constant_values=1)

  ## reshape to (raw, N2V, ours 2 class, ours 3class) , (real, fft, mask), samples, y, x
  data = cat(stak(np.zeros(data[0,0].shape),data[0,0],data[0,2])[None],data[:,[1,3,4]])

  names = [
    "n2v",
    "n2v^2",
    "n2v plus",
    "n2v bigplus",
    "n2v xxxxoxxxx",
    "n2v xox",
    "n2v xxoxx",
    ]

  e02 = SimpleNamespace(data=data,names=names)
  
  dat = SimpleNamespace(gt=flower_gt_patches,e01=e01,e02=e02)
  return dat

def fulldata_fullpatch():
  ## load the flower dataset and build the GT
  flower_all = imread(rawdata_dir/'artifacts/flower.tif')
  flower_all = normalize3(flower_all,2,99.6)
  flower_gt  = flower_all.mean(0)
  # flower_gt_patches = flower_gt.reshape((4,256,4,256)).transpose((0,2,1,3)).reshape((16,256,256))
  # flower_gt_patches = flower_gt_patches[[0,3,5,12]]

  ## load the predictions from single-phase models (600th epoch)
  img6 = imread(experiments_dir / 'flower/e01/flower3_6/pred_flower.tif')    # 1   n2v
  img7 = imread(experiments_dir / 'flower/e01/flower3_7/pred_flower.tif')    # 2   xox
  img8 = imread(experiments_dir / 'flower/e01/flower3_8/pred_flower.tif')    # 3   plus
  img9 = imread(experiments_dir / 'flower/e01/flower3_9/pred_flower.tif')    # 4   bigplus
  img10 = imread(experiments_dir / 'flower/e01/flower3_10/pred_flower.tif')  # 5   8xo8x
  img11 = imread(experiments_dir / 'flower/e01/flower3_11/pred_flower.tif')  # 6   xxoxx
  img12 = imread(experiments_dir / 'flower/e01/flower3_12/pred_flower.tif')  # 7   xxxoxxx
  img13 = imread(experiments_dir / 'flower/e01/flower3_13/pred_flower.tif')  # 8   xxxxoxxxx
  img14 = imread(experiments_dir / 'flower/e01/flower3_14/pred_flower.tif')  # 9   xxxxxoxxxxx
  img15 = imread(experiments_dir / 'flower/e01/flower3_15/pred_flower.tif')  # 10  xxxxxxoxxxxxx
  img16 = imread(experiments_dir / 'flower/e01/flower3_16/pred_flower.tif')  # 11  xxxxxxxoxxxxxxx

  names_e01 = [
    "n2v",
    "xox",
    "plus",
    "bigplus",
    "8xo8x",
    "xxoxx",
    "xxxoxxx",
    "xxxxoxxxx",
    "xxxxxoxxxxx",
    "xxxxxxoxxxxxx",
    "xxxxxxxoxxxxxxx",
    ]


  data = stak(img6, img7, img8, img9, img10, img11, img12, img13, img14, img15, img16,)

  # data[:,[2,4]] = normalize3(np.log(normalize3(data[:,[2,4]],0,99)+1e-7)) ## k-space channels
  # data[:,[0,3]] = normalize3(data[:,[0,3]]) ## real space channels
  # data[:,1]     = normalize3(data[:,1]) ## mask channel ?

  ## remove channels dim 
  data = data[:,:,0]

  ## move raw to front. reshape to ?,4,256,256
  # data = cat(stak(np.zeros(data[0,0].shape),data[0,0],data[0,2])[None],data[:,[1,3,4]]) 

  ## put the trials in a sensible order

  # perm = [0, 2, 3, 1, 5, 6, 7, 8, 9, 10, 4,]
  # data = data[perm]
  # names = list(np.array(names)[perm])

  e01 = SimpleNamespace(data=data,names=names_e01)

  if False:
    img1 = np.load(experiments_dir/'flower/e02/flower1_1/epochs_npy/arr_400.npy') # n2v^2
    img2 = np.load(experiments_dir/'flower/e02/flower1_2/epochs_npy/arr_400.npy') # n2v plus
    img3 = np.load(experiments_dir/'flower/e02/flower1_3/epochs_npy/arr_400.npy') # n2v bigplus
    img4 = np.load(experiments_dir/'flower/e02/flower1_4/epochs_npy/arr_400.npy') # n2v xxxxoxxxx
    img5 = np.load(experiments_dir/'flower/e02/flower1_5/epochs_npy/arr_400.npy') # n2v xox
    # img6 = np.load(experiments_dir/'flower/e02/flower1_6/epochs_npy/arr_400.npy') # n2v xxoxx

    ## (N2V, OURS 2class, OURS 3class) , (raw, mask, raw fft, pred, pred fft) , n_samples , channels, y , x
    data = stak(img1, img2, img3, img4, img5, ) #img6)

    ## normalize fft and real space separately
    data[:,[2,4]] = normalize3(np.log(normalize3(data[:,[2,4]],0,99)+1e-7))
    data[:,[0,3]] = normalize3(data[:,[0,3]])
    data[:,1]     = normalize3(data[:,1])

    ## remove channels and pad xy with white
    data = data[:,:,:,0]
    # data = np.pad(data,[(0,0),(0,0),(0,0),(0,1),(0,1)],mode='constant',constant_values=1)

    ## reshape to (raw, N2V, ours 2 class, ours 3class) , (real, fft, mask), samples, y, x
    data = cat(stak(np.zeros(data[0,0].shape),data[0,0],data[0,2])[None],data[:,[1,3,4]])

    names = [
      "n2v",
      "n2v^2",
      "n2v plus",
      "n2v bigplus",
      "n2v xxxxoxxxx",
      "n2v xox",
      "n2v xxoxx",
      ]

  # e02 = SimpleNamespace(data=data,names=names)

  dat = SimpleNamespace(gt=flower_gt,e01=e01,all=flower_all) #e02=e02)
  return dat

@DeprecationWarning
def e01_fig2_flower():
  # img1 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_1/epochs_npy/arr_600.npy')
  # img2 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_2/epochs_npy/arr_600.npy')
  # img3 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_3/epochs_npy/arr_600.npy')
  # img4 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_4/epochs_npy/arr_600.npy')
  # img5 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_5/epochs_npy/arr_600.npy') 
  img6 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_6/epochs_npy/arr_600.npy')    # n2v
  img7 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_7/epochs_npy/arr_600.npy')    # xox
  img8 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_8/epochs_npy/arr_600.npy')    # plus
  img9 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_9/epochs_npy/arr_600.npy')    # bigplus
  img10 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_10/epochs_npy/arr_600.npy')  # 8xo8x
  img11 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_11/epochs_npy/arr_600.npy')  # xxoxx
  img12 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_12/epochs_npy/arr_600.npy')  # xxxoxxx
  img13 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_13/epochs_npy/arr_600.npy')  # xxxxoxxxx
  img14 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_14/epochs_npy/arr_600.npy')  # xxxxxoxxxxx
  img15 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_15/epochs_npy/arr_600.npy')  # xxxxxxoxxxxxx
  img16 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_16/epochs_npy/arr_600.npy')  # xxxxxxxoxxxxxxx

  names = [
    # "raw",
    "n2v",
    "xox",
    "plus",
    "bigplus",
    "8xo8x",
    "xxoxx",
    "xxxoxxx",
    "xxxxoxxxx",
    "xxxxxoxxxxx",
    "xxxxxxoxxxxxx",
    "xxxxxxxoxxxxxxx",
    ]

  ## (N2V, OURS 2class, OURS 3class) , (raw, mask, raw fft, pred, pred fft) , n_samples , channels, y , x
  # rgb = stak(img1, img2, img3, img4, img5, img6, img7, img8, img9)
  rgb = stak(img6, img7, img8, img9, img10, img11, img12, img13, img14, img15, img16,)

  # rgb[:,[2,4]] = normalize3(rgb[:,[2,4]], pmin=0, pmax=99.0)
  # rgb[:,[2,4]] = normalize3(np.log(rgb[:,[2,4]]+1e-7))
  rgb[:,[2,4]] = normalize3(np.log(normalize3(rgb[:,[2,4]],0,99)+1e-7)) ## k-space channels
  rgb[:,[0,3]] = normalize3(rgb[:,[0,3]]) ## real space channels
  rgb[:,1]     = normalize3(rgb[:,1]) ## mask channel ?
  rgb = rgb[:,:,:,0] ## remove channels dim 
  rgb = cat(stak(np.zeros(rgb[0,0].shape),rgb[0,0],rgb[0,2])[None],rgb[:,[1,3,4]]) ## reshape to 15,4,256,256

  # rgb = np.pad(rgb,[(0,0),(0,0),(0,0),(0,1),(0,1)],mode='constant',constant_values=1) ## pad xy with white

  # plt.figure()
  # d = np.fft.fftshift(np.fft.fftfreq(256))
  # for i,m in enumerate("N2V,OURS 2class,OURS 3class".split(',')):
  #   plt.plot(d,rgb[i,-1].mean((0,1)),label=f'{m} : avg s,y')
  #   plt.plot(d,rgb[i,-1].mean((0,2)),label=f'{m} : avg s,x')
  # plt.legend()

  ## reshape to (raw, N2V, ours 2 class, ours 3class) , (real, fft, mask), samples, y, x

  # rgb = rgb.reshape((15, 4, 256, 256))[]
  # rgb = cat(stak(np.zeros(rgb[0,0].shape),rgb[0,0],rgb[0,2])[None],rgb[:,[1,3,4]])

  ## models, types, samples, y, x
  # rgb = collapse2(rgb,'mtsyx','mt,sy,x')
  # rgb = rgb[[0,1,2,3,4,6,8,9,11,13,14]]
  # rgb = rgb[[0,1,5,8,3,6,9,2,4,7,10,]]
  # rgb = collapse2(rgb,'myx','y,mx')

  # io.imsave(savedir.parent/'shutterclosed_normalized.png',rgb[:64])
  np.savez_compressed(savedir.parent / 'e01_fig2_flower.npz', rgb=rgb)

  return rgb

@DeprecationWarning
def e02_fig2_flower():
  img1 = np.load(experiments_dir/'flower/e02/flower1_1/epochs_npy/arr_400.npy') # n2v^2
  img2 = np.load(experiments_dir/'flower/e02/flower1_2/epochs_npy/arr_400.npy') # n2v plus
  img3 = np.load(experiments_dir/'flower/e02/flower1_3/epochs_npy/arr_400.npy') # n2v bigplus
  img4 = np.load(experiments_dir/'flower/e02/flower1_4/epochs_npy/arr_400.npy') # n2v xxxxoxxxx
  img5 = np.load(experiments_dir/'flower/e02/flower1_5/epochs_npy/arr_400.npy') # n2v xox
  img6 = np.load(experiments_dir/'flower/e02/flower1_6/epochs_npy/arr_400.npy') # n2v xxoxx

  names = [
    "n2v",
    "n2v^2",
    "n2v plus",
    "n2v bigplus",
    "n2v xxxxoxxxx",
    "n2v xox",
    "n2v xxoxx",
    ]


  ## (N2V, OURS 2class, OURS 3class) , (raw, mask, raw fft, pred, pred fft) , n_samples , channels, y , x
  rgb = stak(img1, img2, img3, img4, img5, img6)

  ## normalize fft and real space separately
  rgb[:,[2,4]] = normalize3(np.log(normalize3(rgb[:,[2,4]],0,99)+1e-7))
  rgb[:,[0,3]] = normalize3(rgb[:,[0,3]])
  rgb[:,1]     = normalize3(rgb[:,1])

  ## remove channels and pad xy with white
  rgb = rgb[:,:,:,0]
  # rgb = np.pad(rgb,[(0,0),(0,0),(0,0),(0,1),(0,1)],mode='constant',constant_values=1)

  ## reshape to (raw, N2V, ours 2 class, ours 3class) , (real, fft, mask), samples, y, x
  rgb = cat(stak(np.zeros(rgb[0,0].shape),rgb[0,0],rgb[0,2])[None],rgb[:,[1,3,4]])

  np.savez_compressed(savedir.parent / 'e02_fig2_flower.npz', rgb=rgb)
  return rgb

## perform analysis

def noise_distributions(dat=None):
  """
  let's look at the noise distribution for several images.
  noise dist means we're comparing an image wrt ground truth. it is a distribution of *residuals*.
  we can build this dist for raw data, one-phase model predictions, and two phase predictions.
  it will be useful to see how this distribution shifts depending on model type and one/two phase predictions.
  does the normalization matter?
  we can also look at the differences between predictions and raw input, which should be equivalent?

  """
  if dat is None: dat = fulldata()

  plt.figure()
  plt.hlines(0,0,100)

  delta = dat.e01.data[:,1] - dat.gt
  x = np.linspace(0,100,101)
  for i,name in enumerate(dat.e01.names):
    plt.plot(x,np.percentile(delta[i].mean(0), x), label=name)

  delta = dat.e02.data[:,1] - dat.gt
  x = np.linspace(0,100,101)
  for i,name in enumerate(dat.e02.names[:-1]):
    plt.plot(x,np.percentile(delta[i].mean(0), x), '--', label=name)
  
  plt.legend()

def print_metrics(dat):

  ## dataset e01

  ys = np.array([normalize_minmse(x, dat.gt) for x in dat.e01.data[:,1]])
  mse = ((dat.gt-ys)**2).mean((1,2,3))
  psnr = 10*np.log10(1/mse)
  ssim = np.array([[compare_ssim(dat.gt[j],ys[i,j]) for j in range(ys.shape[1])] for i in range(ys.shape[0])])

  table = [dat.e01.names,list(mse),list(psnr),list(ssim.mean(1))]
  headers = ['name','mse','psnr','ssim']
  print(tabulate(zip(*table),headers=headers,floatfmt='f',numalign='decimal'))

  ## dataset e02

  ys = np.array([normalize_minmse(x, dat.gt) for x in dat.e02.data[:,1]])
  mse = ((dat.gt-ys)**2).mean((1,2,3))
  psnr = 10*np.log10(1/mse)
  ssim = np.array([[compare_ssim(dat.gt[j],ys[i,j]) for j in range(ys.shape[1])] for i in range(ys.shape[0])])

  table = [dat.e02.names,list(mse),list(psnr),list(ssim.mean(1))]
  headers = ['name','mse','psnr','ssim']
  print(tabulate(zip(*table),headers=headers,floatfmt='f',numalign='decimal'))

def print_metrics_fullpatch(dat):
  """
  Running on full dataset takes XXX seconds.
  """

  ## dataset e01
  # dat.e01.data = dat.e01.data[:,::30]

  ## first do all the denoising models
  # ys = np.array([normalize_minmse(x, dat.gt) for x in dat.e01.data[:,1]])
  ys   = dat.e01.data #[:,0] ys.shape == (11,100,1024,1024) == model,sample_i,y,x
  mse  = ((dat.gt-ys)**2).mean((1,2,3))
  psnr = 10*np.log10(1/mse)
  ssim = np.array([[compare_ssim(dat.gt,ys[i,j]) for j in range(ys.shape[1])] for i in range(ys.shape[0])])
  ssim = ssim.mean(1)

  table = [dat.e01.names, list(mse), list(psnr), list(ssim)]
  table = list(zip_longest(*table))

  ## then we'll do the raw data
  ys   = dat.all # (100,1024,1024) == sample_i,y,x
  mse  = ((dat.gt-ys)**2).mean((0,1,2))
  psnr = 10*np.log10(1/mse)
  ssim = np.array([compare_ssim(dat.gt,ys[i]) for i in range(100)])
  ssim = ssim.mean(0)
  table = [["raw",mse,psnr,ssim]] + table
  
  print(tabulate(table,headers=['name','mse','psnr','ssim'],floatfmt='f',numalign='decimal'))

  ## dataset e02
  if False:

    ys = np.array([normalize_minmse(x, dat.gt) for x in dat.e02.data[:,1]])
    mse = ((dat.gt-ys)**2).mean((1,2,3))
    psnr = 10*np.log10(1/mse)
    ssim = np.array([[compare_ssim(dat.gt[j],ys[i,j]) for j in range(ys.shape[1])] for i in range(ys.shape[0])])

    table = [dat.e02.names,list(mse),list(psnr),list(ssim.mean(1))]
    headers = ['name','mse','psnr','ssim']
    print(tabulate(zip(*table),headers=headers,floatfmt='f',numalign='decimal'))


## Utils or work in progress

def hist(arr):
  x = np.linspace(0,100,101)
  plt.figure()
  plt.plot(x,np.percentile(arr,x),'.-')

def signals_in_kspace():
  xs, ys = [],[]
  for i in range(100):
    x = np.random.rand(256)-0.5
    y = cat(np.fft.fft(x)[1:129].real, np.fft.fft(x)[1:129].imag)
    xs.append(x)
    ys.append(y)
  xs = np.array(xs)
  ys = np.array(ys)
  return xs, ys

def plot_image_k_space(img):
  fftx = np.fft.fft(img,axis=2)
  fftxmean = fftx.mean((0,1))
  fftxmean[0] = 0
  # fftxmean = np.fft.fftshift(fftxmean)
  fftxmean = np.abs(fftxmean)
  plt.plot(fftxmean)
  ffty = np.fft.fft(img,axis=1)
  fftymean = ffty.mean((0,2))
  fftymean[0] = 0
  # fftymean = np.fft.fftshift(fftymean)
  fftymean = np.abs(fftymean)
  plt.plot(fftymean)

def plot_avg_1dimage(img):

  plt.plot(img.mean((0,1)))
  plt.plot(img.mean((0,2)))

  if True:
    plt.figure()
    x = np.fft.fft(img,axis=2)
    x = x.mean((0,1))
    x[0] = 0
    # x = np.fft.ifft(x)
    x = np.fft.fftshift(x)
    d = np.fft.fftfreq(x.shape[0])
    d = np.fft.fftshift(d)
    x = np.abs(x)
    plt.plot(d*x.shape[0],x)

    x = np.fft.fft(img,axis=1)
    x = x.mean((0,2))
    x[0] = 0
    # x = np.fft.ifft(x)
    x = np.fft.fftshift(x)
    d = np.fft.fftfreq(x.shape[0])
    d = np.fft.fftshift(d)
    x = np.abs(x)
    plt.plot(d*x.shape[0],x)
    plt.semilogy()

def fft2d():
  "slice 32 is almost all noise. the x/y differences are very clear here!"
  img  = imread('/Users/broaddus/Desktop/Projects/devseg_data/raw/celegans_isbi/Fluo-N3DH-CE/01/t190.tif')
  x  = np.fft.fft2(img[32])
  x[0,0] = 0
  x  = np.fft.fftshift(x)
  iss = StackVis(np.log(np.abs(x+1e-7)))
  d0 = np.fft.fftshift(np.fft.fftfreq(x.shape[0]))
  d1 = np.fft.fftshift(np.fft.fftfreq(x.shape[1]))
  plt.plot(d0, np.log(np.abs(x+1e-7)).mean(1))
  plt.plot(d1, np.log(np.abs(x+1e-7)).mean(0))

def fft2d2(img):
  x  = np.fft.fft2(img)
  x[0,0] = 0
  x  = np.fft.fftshift(x)
  iss = StackVis(np.log(np.abs(x+1e-7)))
  d0 = np.fft.fftshift(np.fft.fftfreq(x.shape[0]))
  d1 = np.fft.fftshift(np.fft.fftfreq(x.shape[1]))
  plt.plot(d0, np.log(np.abs(x+1e-7)).mean(1),label='mean1')
  plt.plot(d1, np.log(np.abs(x+1e-7)).mean(0),label='mean0')
  plt.legend()


def plot_x(img):
  x = img.mean((0,1))
  freal = plt.figure()
  freal.gca().plot(x)

  fkspa = plt.figure()
  x = np.fft.fft(x)
  x = np.abs(x)
  x[0] = 0
  x = np.fft.fftshift(x)
  d = np.fft.fftfreq(x.shape[0])
  d = np.fft.fftshift(d)
  cv = np.ones(9)/9
  x = np.convolve(x,cv,mode='same')
  fkspa.gca().plot(d,x,label='fft')
  plt.semilogy()
  
  for order in [0,1,3]:
    x = img.mean((0,1))
    x = x[::5]
    x = zoom(x,5,order=order)
    freal.gca().plot(x)

    x = np.fft.fft(x)
    x = np.abs(x)
    x[0] = 0
    x = np.fft.fftshift(x)
    d = np.fft.fftfreq(x.shape[0])
    d = np.fft.fftshift(d)
    cv = np.ones(9)/9
    x = np.convolve(x,cv,mode='same')
    fkspa.gca().plot(d,x,label=f'order {order}')

  x = img.mean((0,1))
  k = np.r_[0:3,3:-1:-1]
  k = k/k.sum()
  x = np.convolve(x,k,mode='same')
  freal.gca().plot(x)

  x = np.fft.fft(x)
  x = np.abs(x)
  x[0] = 0
  x = np.fft.fftshift(x)
  d = np.fft.fftfreq(x.shape[0])
  d = np.fft.fftshift(d)
  cv = np.ones(9)/9
  x = np.convolve(x,cv,mode='same')
  fkspa.gca().plot(d,x,label='conv')

  x = img.mean((0,2))
  freal.gca().plot(x)
  x = np.fft.fft(x)
  x = np.abs(x)
  x[0] = 0
  x = np.fft.fftshift(x)
  d = np.fft.fftfreq(x.shape[0])
  d = np.fft.fftshift(d)
  cv = np.ones(9)/9
  x = np.convolve(x,cv,mode='same')
  fkspa.gca().plot(d,x,label='y dim')

  plt.semilogy()
  plt.legend()

def quickload():
  img  = imread('/Users/broaddus/Desktop/Projects/devseg_data/raw/celegans_isbi/Fluo-N3DH-CE/01/t190.tif')
  pimg = imread('/Users/broaddus/Desktop/Projects/devseg_data/cl_datagen/noise2void2 all timepoints 10x patches w rot augmentation/pimgs/pimg100.tif/')
  return img, pimg[0]


rsync_down = """

e01_fig2_flower.npz
e02_fig2_flower.npz
fig2_flower.npz
fig2_cele.npz

n2v2_flower.py
torch_models.py
n2v2_cele.py

# data.npz
# cl_torch.py
# cl_torch_selfsup.py
# cl_datagen2.py

# recfield_xy.png
# recfield_xz.png
# data_xy.png
# data_xz.png
loss.png
# lossdist.pdf
# lossdist_unsorted.pdf
# histogram_img_pimg.pdf
# slicelist.pkl

# multi_losses.png

epochs/
# epochs_npy/
movie/
# counts/
# pts/
pimgs/
"""

rsync_up = """
"""

def sync(savedir, subdir):
  subprocess.run([f"mkdir -p {savedir}"],shell=True)
  name = "rsync_down_fly.txt"
  with open(name,'w') as f:
    f.write(rsync_down)

  d1 = "efal:/projects/project-broaddus/denoise/" + subdir
  d2 = "/Users/broaddus/Desktop/Projects/denoise/" + subdir
  files = f"--files-from {name}"
  cmd = f"rsync -avr {files} {d1} {d2}"
  res = subprocess.run([cmd], shell=True)
  print(res)
  os.remove(name)

def syncall():
  subdir  = f"flower/e01/" #flower3_10/"
  sync(savedir / subdir, subdir)

  # for i in range(6):
  #   subdir  = f"flower/e02/flower1_{i}/"
  #   sync(savedir / subdir, subdir)





