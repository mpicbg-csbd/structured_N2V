import numpy as np
import tifffile
import subprocess
import os
import csv

from tabulate  import tabulate
from types     import SimpleNamespace
from pathlib   import Path
from itertools import zip_longest
from glob      import glob

from matplotlib      import pyplot as plt
plt.switch_backend('Agg')
from scipy           import ndimage
from scipy.ndimage   import label, zoom
from scipy.signal    import correlate2d
from skimage         import io
from skimage.measure import compare_ssim
from numpy.fft import fft2,ifft2,ifftshift

from uncertainties import unumpy

# import spimagine
from segtools.numpy_utils import normalize3, perm2, collapse2, splt
from segtools.StackVis import StackVis
import utils


# from csbdeep.utils.utils import normalize_minmse
# import ipdb
# import json
# from os.path import join as pjoin


def writecsv(list_of_lists,outfile):
  with open(outfile, "w", newline="\n") as f:
    writer = csv.writer(f)
    writer.writerows(list_of_lists)

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



def load_single_and_eval_metrics__flower(loaddir):
  flower_all = imread(rawdata_dir/'artifacts/flower.tif')
  flower_all = normalize3(flower_all,2,99.6)
  flower_gt  = flower_all.mean(0)

  ## deal with heterogeneous file names  
  loaddir = Path(loaddir)
  if (loaddir / 'denoised.tif').exists():
    img = imread(loaddir / 'denoised.tif')
  elif (loaddir / 'pred.tif').exists():
    img = imread(loaddir / 'pred.tif')
  elif (loaddir / 'img000.tif').exists():
    img = np.array([imread(loaddir / f'img{n:03d}.tif') for n in range(100)])

  ## deal with singleton channels
  if img.shape[1]==1: img=img[:,0]

  met = eval_single_metrics(flower_gt, img)

  header=['mse','psnr','ssim']
  writecsv([header,met], loaddir / 'table.csv')

def eval_single_metrics(gt,ys,nth=1):
  ys   = ys[::nth]
  mse  = ((gt-ys)**2).mean((0,1,2))
  psnr = 10*np.log10(1/mse)
  ssim = np.array([compare_ssim(gt,ys[i].astype(np.float64)) for i in range(ys.shape[0]//50)])
  ssim = ssim.mean()
  return [mse,psnr,ssim]

def single(csvfile):
  r = list(csv.reader(open(csvfile), delimiter=','))
  return [float(x) for x in r[1]]

def make_metrics_table(all_result_files,outdir):
  outdir = Path(outdir)  
  res  = utils.recursive_map2(single, all_result_files)
  
  def un(x): return unumpy.uarray(x.mean(0),x.std(0)*1)
  n2v  = un(np.array(res['n2v'])[0])
  n2v2  = un(np.array(res['n2v'])[5])
  n2gt = un(np.array(res['n2gt']))
  nlm  = np.array(res['nlm'])
  bm3d = np.array(res['bm3d'])
  ## MSE & PSNR & SSIM
  s1 = "{0:1ueS} & {1:1uL} & {2:1uL}" ## numbers w uncertainties
  s2 = "{0:1e} & {1:1f} & {2:1f}" ## normal floats
  lines = [
    s1.format(*n2v),
    s1.format(*n2v2),
    s1.format(*n2gt),
    s2.format(*nlm),
    s2.format(*bm3d),
  ]
  table = " \\\\\n".join(lines)
  table = table.replace('\\pm','$\\pm$')

  print(table, file=open(outdir / 'table.tex','w'))
  return table

def merge_all_results(all_result_files,outdir):
  outdir = Path(outdir)
  res  = utils.recursive_map2(single, all_result_files)

  for n,metric in enumerate(['mse','psnr','ssim']):
    n2v  = np.array(res['n2v'])[...,n]
    n2gt = np.array(res['n2gt'])[...,n]
    nlm  = np.array(res['nlm'])[...,n]
    bm3d = np.array(res['bm3d'])[...,n]

    plt.figure()
    i=0
    for d in n2v:
      plt.plot([i]*len(d),  d, '.'); i+=1
    plt.plot([i]*len(n2gt), n2gt, '.'); i+=1
    plt.plot([i]*len([nlm]),  [nlm], '.'); i+=1
    plt.plot([i]*len([bm3d]), [bm3d], '.'); i+=1
    plt.savefig(outdir/f'table_{metric}.pdf')

def correlation_analysis(rawdata,savedir,removeGT=False):
  savedir = Path(savedir); savedir.mkdir(exist_ok=True,parents=True)

  img  = imread(rawdata)
  
  png_name = 'autocorr_img.png'
  pdf_name = 'autocorr_plot.pdf'

  if removeGT: 
    img = img-img.mean(0)
    png_name = 'autocorr_img-gt.png'
    pdf_name = 'autocorr_plot-gt.pdf'

  corr = np.array([autocorrelation(img[i]) for i in range(10)])
  corr = corr.mean(0)
  a,b  = corr.shape
  corr = corr[a//2-30:a//2+30, b//2-30:b//2+30]
  corr = corr.real / corr.real.max()
  io.imsave(savedir / png_name,normalize3(corr))

  d0 = np.arange(corr.shape[0])-corr.shape[0]//2
  d1 = np.arange(corr.shape[1])-corr.shape[1]//2
  plt.figure()
  a,b = corr.shape
  # y = corr.mean(1); y = y/y.max()
  y = corr[:,b//2]
  plt.plot(d0,y,'.-', label='y profile')
  # y = corr.mean(0); y = y/y.max()
  y = corr[a//2,:]
  plt.plot(d1,y,'.-', label='x profile')
  plt.legend()
  plt.savefig(savedir / pdf_name)


def autocorrelation(x):
    """
    2D autocorrelation
    remove mean per-patch (not global GT)
    normalize stddev to 1
    """
    x = (x - np.mean(x))/np.std(x)
    # x = np.pad(x, [(50,50),(50,50)], mode='constant')
    f = fft2(x)
    p = np.abs(f)**2
    pi = ifft2(p)
    pi = np.fft.fftshift(pi)
    return pi

## old data loading

def load_flower():

  ## load the flower dataset and build the GT
  flower_all = imread(rawdata_dir/'artifacts/flower.tif')
  flower_all = normalize3(flower_all,2,99.6)
  flower_gt  = flower_all.mean(0)

  ## load the predictions from single-phase models (600th epoch)
  img0  = np.array([imread(experiments_dir / f'flower/e01/mask00_{n}/pred.tif') for n in range(5)])
  img1  = np.array([imread(experiments_dir / f'flower/e01/mask01_{n}/pred.tif') for n in range(5)])
  img2  = np.array([imread(experiments_dir / f'flower/e01/mask02_{n}/pred.tif') for n in range(5)])
  img3  = np.array([imread(experiments_dir / f'flower/e01/mask03_{n}/pred.tif') for n in range(5)])
  img4  = np.array([imread(experiments_dir / f'flower/e01/mask04_{n}/pred.tif') for n in range(5)])
  img5  = np.array([imread(experiments_dir / f'flower/e01/mask05_{n}/pred.tif') for n in range(5)])
  img6  = np.array([imread(experiments_dir / f'flower/e01/mask06_{n}/pred.tif') for n in range(5)])
  img7  = np.array([imread(experiments_dir / f'flower/e01/mask07_{n}/pred.tif') for n in range(5)])
  img8  = np.array([imread(experiments_dir / f'flower/e01/mask08_{n}/pred.tif') for n in range(5)])

  names = "N2V 1x 2x 3x 4x 5x 6x 7x 8x".split(' ')

  if False:

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

    names = [
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

  data = stak(img0, img1, img2, img3, img4, img5, img6, img7, img8,)
  # data = stak(img6, img7, img8, img9, img10, img11, img12, img13, img14, img15, img16,)

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
  # nlm_vals = [5,10,50,100,200,500]
  nlm  = np.array([imread(f"/projects/project-broaddus/denoise_experiments/flower/e01/nlm/0010/denoised.tif") for n in nlm_vals])
  bm3d = np.array([imread(x) for x in sorted(glob("/projects/project-broaddus/denoise_experiments/flower/e01/bm3d/*.tif"))])

  n2gt = imread("/projects/project-broaddus/denoise_experiments/flower/e01/n2gt2/pred.tif")
  n2gt = n2gt[:,0] ## get rid of singleton channel

  e01 = SimpleNamespace(data=data,names=names)
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

  names = [
    "n2v",
    "1x",
    "2x",
    "3x",
    "4x",
    "5x",
    "6x",
    "7x",
    "8x",
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

  e01 = SimpleNamespace(data=data,names=names)
  dat = SimpleNamespace(gt=raw_gt,e01=e01,all=raw_all,bm3d=bm3d,n2gt=n2gt,nlm=nlm) #e02=e02)
  return dat

def load_cele():
  raw  = np.array([imread(f"/lustre/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01/t{i:03d}.tif") for i in [0,10,100,189]])
  raw  = normalize3(raw,2,99.6)  
  n2v  = np.array([imread(f"/projects/project-broaddus/denoise_experiments/cele/e01/cele1/pimgs/pimg01_{i:03d}.tif") for i in [0,10,100,189]])
  n2v  = n2v[:,0]
  n2v2 = np.array([imread(f"/projects/project-broaddus/denoise_experiments/cele/e01/cele3/pimgs/pimg01_{i:03d}.tif") for i in [0,10,100,189]])
  n2v2 = n2v2[:,0]
  nlm  = np.array([imread(f"/projects/project-broaddus/denoise_experiments/cele/e01/nlm/denoised{i:03d}.tif") for i in [0,10,100,189]])
  dat  = SimpleNamespace(raw=raw,n2v2=n2v2,nlm=nlm,n2v=n2v)
  return dat

## old evaluation

def eval_single(gt,ys,name,nth=1):
  met = eval_single_metrics(gt,ys,nth)
  return [name] + met

def nlmeval(nlm_vals,outfile):
  flower_all = imread(rawdata_dir/'artifacts/flower.tif')
  flower_all = normalize3(flower_all,2,99.6)
  flower_gt  = flower_all.mean(0)  
  nlm  = np.array([imread(f"/projects/project-broaddus/denoise_experiments/flower/e01/nlm/{n:04d}/denoised.tif") for n in nlm_vals])

  table = []
  for i in range(nlm.shape[0]):
    table.append(eval_single(flower_gt,nlm[i],nlm_vals[i]))
  
  header=['name','mse','psnr','ssim']
  with open(outfile, "w", newline="\n") as f:
    writer = csv.writer(f)
    writer.writerows([header] + table)

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
  
  table = []
  
  ## shape == (11,100,1024,1024) == model,sample_i,y,x
  n2v2 = dat.e01.data
  for i in range(n2v2.shape[0]):
    res = np.array([eval_single(dat.gt,n2v2[i,j], dat.e01.names[i]) for j in n2v2[i].shape[0]])

    table.append(eval_single(dat.gt,n2v2[i], dat.e01.names[i]))

  table.append(eval_single(dat.gt,dat.all,"RAW"))
  table.append(eval_single(dat.gt,dat.gt[None],"GT"))
  table.append(eval_single(dat.gt,dat.n2gt,"N2GT"))
  table.append(eval_single(dat.gt,dat.bm3d,"BM3D"))
  table.append(eval_single(dat.gt,dat.nlm[i], "NLM"))
  # for i in range(dat.nlm.shape[0]):

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
