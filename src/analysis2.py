import numpy as np
import tifffile
import subprocess
import os

from tabulate  import tabulate
from types     import SimpleNamespace
from pathlib   import Path
from itertools import zip_longest
from glob      import glob

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
import csv

def cat(*args,axis=0): return np.concatenate(args, axis)
def stak(*args,axis=0): return np.stack(args, axis)

def imsave(x, name, **kwargs): return tifffile.imsave(str(name), x, **kwargs)
def imread(name,**kwargs): return tifffile.imread(str(name), **kwargs)
# def imsavefiji(x, **kwargs): return tifffile.imsave('/Users/broaddus/Desktop/stack.tiff', x, imagej=True, **kwargs)
# def imsavefiji(x, **kwargs): return tifffile.imsave('stack.tiff', x, imagej=True, **kwargs)

# figure_dir      = Path('/Users/broaddus/Dropbox/phd/notes/denoise_paper/res').resolve()
# experiments_dir = Path('/Volumes/project-broaddus/denoise_experiments').resolve()
figure_dir      = Path('/projects/project-broaddus/denoise_experiments/figures/').resolve(); figure_dir.mkdir(exist_ok=True,parents=True)
experiments_dir = Path('/projects/project-broaddus/denoise_experiments/').resolve()
rawdata_dir     = Path('/projects/project-broaddus/rawdata').resolve()


def load_flower():

  ## load the flower dataset and build the GT
  flower_all = imread(rawdata_dir/'artifacts/flower.tif')
  flower_all = normalize3(flower_all,2,99.6)
  flower_gt  = flower_all.mean(0)
  # flower_gt_patches = flower_gt.reshape((4,256,4,256)).transpose((0,2,1,3)).reshape((16,256,256))
  # flower_gt_patches = flower_gt_patches[[0,3,5,12]]

  ## load the predictions from single-phase models (600th epoch)
  img6  = imread(experiments_dir / 'flower/e01/flower3_6/pred_flower.tif')   # 0  n2v
  img7  = imread(experiments_dir / 'flower/e01/flower3_7/pred_flower.tif')   # 1  xox
  img8  = imread(experiments_dir / 'flower/e01/flower3_8/pred_flower.tif')   # 2  plus
  img9  = imread(experiments_dir / 'flower/e01/flower3_9/pred_flower.tif')   # 3  bigplus
  img10 = imread(experiments_dir / 'flower/e01/flower3_10/pred_flower.tif')  # 4  8xo8x
  img11 = imread(experiments_dir / 'flower/e01/flower3_11/pred_flower.tif')  # 5  xxoxx
  img12 = imread(experiments_dir / 'flower/e01/flower3_12/pred_flower.tif')  # 6  xxxoxxx
  img13 = imread(experiments_dir / 'flower/e01/flower3_13/pred_flower.tif')  # 7  xxxxoxxxx
  img14 = imread(experiments_dir / 'flower/e01/flower3_14/pred_flower.tif')  # 8  xxxxxoxxxxx
  img15 = imread(experiments_dir / 'flower/e01/flower3_15/pred_flower.tif')  # 9  xxxxxxoxxxxxx
  img16 = imread(experiments_dir / 'flower/e01/flower3_16/pred_flower.tif')  # 10  xxxxxxxoxxxxxxx

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

  nlm  = imread("/projects/project-broaddus/denoise_experiments/shutter/e01/nlm/denoised.tif")
  bm3d = np.array([imread(x) for x in sorted(glob("/projects/project-broaddus/denoise_experiments/flower/e01/bm3d/*.tif"))])

  n2gt = imread("/projects/project-broaddus/denoise_experiments/flower/e01/n2gt2/pred.tif")
  n2gt = n2gt[:,0] ## get rid of singleton channel

  e01 = SimpleNamespace(data=data,names=names_e01)
  dat = SimpleNamespace(gt=flower_gt,e01=e01,all=flower_all,bm3d=bm3d,n2gt=n2gt,nlm=nlm) #e02=e02)
  return dat

def load_shutter():
  ## load the flower dataset and build the GT
  raw_all = imread(rawdata_dir/'artifacts/shutterclosed.tif')
  raw_all = normalize3(raw_all,2,99.6)
  raw_gt  = raw_all.mean(0)
  # raw_gt_patches = raw_gt.reshape((4,256,4,256)).transpose((0,2,1,3)).reshape((16,256,256))
  # raw_gt_patches = raw_gt_patches[[0,3,5,12]]

  ## load the predictions from single-phase models (600th epoch)
  img0 = imread(experiments_dir / 'shutter/e01/mask00/pred.tif')  # n2v
  img1 = imread(experiments_dir / 'shutter/e01/mask01/pred.tif')
  img2 = imread(experiments_dir / 'shutter/e01/mask02/pred.tif')
  img3 = imread(experiments_dir / 'shutter/e01/mask03/pred.tif')
  img4 = imread(experiments_dir / 'shutter/e01/mask04/pred.tif')
  img5 = imread(experiments_dir / 'shutter/e01/mask05/pred.tif')
  img6 = imread(experiments_dir / 'shutter/e01/mask06/pred.tif')
  img7 = imread(experiments_dir / 'shutter/e01/mask07/pred.tif')
  img8 = imread(experiments_dir / 'shutter/e01/mask08/pred.tif')

  names_e01 = [
    "n2v",
    "xox",
    # "plus",
    # "bigplus",
    "xxoxx",
    "xxxoxxx",
    "xxxxoxxxx",
    "xxxxxoxxxxx",
    "xxxxxxoxxxxxx",
    "xxxxxxxoxxxxxxx",
    "8xo8x",
    ]


  data = stak(img0, img1, img2, img3, img4, img5, img6, img7, img8,)

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

  nlm  = imread("/projects/project-broaddus/denoise_experiments/shutter/e01/nlm/denoised.tif")
  bm3d = np.array([imread(x) for x in sorted(glob("/projects/project-broaddus/denoise_experiments/shutter/e01/bm3d/*.tif"))])

  n2gt = imread("/projects/project-broaddus/denoise_experiments/shutter/e01/n2gt2/pred.tif")
  n2gt = n2gt[:,0] ## get rid of singleton channel

  e01 = SimpleNamespace(data=data,names=names_e01)
  dat = SimpleNamespace(gt=raw_gt,e01=e01,all=raw_all,bm3d=bm3d,n2gt=n2gt,nlm=nlm) #e02=e02)
  return dat

def load_cele():
  raw  = np.array([imread(f"/lustre/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01/t{i:03d}.tif") for i in [0,10,100,189]])
  raw  = normalize3(raw,2,99.6)  
  n2v  = np.array([imread(f"/projects/project-broaddus/denoise_experiments/cele/e01/cele1/pimgs/pimg01_{i:03d}.tif") for i in [0,10,100,189]])
  n2v  = n2v[:,0]
  n2v2 = np.array([imread(f"/projects/project-broaddus/denoise_experiments/cele/e01/cele4/pimgs/pimg01_{i:03d}.tif") for i in [0,10,100,189]])
  n2v2 = n2v2[:,0]
  nlm  = np.array([imread(f"/projects/project-broaddus/denoise_experiments/cele/e01/nlm/denoised{i:03d}.tif") for i in [0,10,100,189]])
  dat  = SimpleNamespace(raw=raw,n2v2=n2v2,nlm=nlm,n2v=n2v)
  return dat

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

def print_metrics_fullpatch(dat, nth=1, outfile=None):
  """
  Running on full dataset takes XXX seconds.
  """

  ## first do all the denoising models
  # ys = np.array([normalize_minmse(x, dat.gt) for x in dat.e01.data[:,1]])
  
  def single(ys,name):
    ys   = ys[::nth]
    mse  = ((dat.gt-ys)**2).mean((0,1,2))
    psnr = 10*np.log10(1/mse)
    ssim = np.array([compare_ssim(dat.gt,ys[i].astype(np.float64)) for i in range(ys.shape[0]//50)])
    ssim = ssim.mean()
    return [name, mse, psnr, ssim]

  table = []
  
  ## shape == (11,100,1024,1024) == model,sample_i,y,x
  n2v2 = dat.e01.data 
  for i in range(n2v2.shape[0]):
    table.append(single(n2v2[i], dat.e01.names[i]))

  table.append(single(dat.all,"RAW"))
  table.append(single(dat.gt[None],"GT"))
  table.append(single(dat.n2gt,"N2GT"))
  table.append(single(dat.bm3d,"BM3D"))
  table.append(single(dat.nlm,"NLM"))

  # table = [dat.e01.names, list(mse), list(psnr), list(ssim)]
  # table = list(zip_longest(*table))
  
  header=['name','mse','psnr','ssim']
  if outfile:
    with open(outfile, "w", newline="\n") as f:
      writer = csv.writer(f)
      writer.writerows([header] + table)
  else:
    print(tabulate(table,headers=header,floatfmt='f',numalign='decimal'))

  return table

def make_visual_table(dat,outfile=None):
  names = "RAW NLM BM3D N2V N2V2 N2GT GT".split(' ')
  rgb   = stak(dat.all[0], dat.nlm[0], dat.bm3d[0], dat.e01.data[0,0], dat.e01.data[7,0], dat.n2gt[0], dat.gt)
  rgb   = normalize3(rgb)
  y,x   = 256, 256 ## top left pixel location
  rgb   = rgb[:,y:y+256,x:x+256].transpose((1,0,2)).reshape((256,-1))
  print(rgb.shape)
  io.imsave(outfile,rgb)

def make_visual_table_cele(dat, outfile=None):
  names = "RAW NLM N2V N2V2".split(' ')
  rgb = stak(dat.raw[0], dat.nlm[0], dat.n2v[0], dat.n2v2[0])
  rgb = normalize3(rgb)
  z,y,x = 14, 256, 256 ## top left pixel location
  rgb   = rgb[:,z,y:y+256,x:x+256].transpose((1,0,2)).reshape((256,-1))
  io.imsave(outfile,rgb)
