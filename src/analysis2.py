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

import matplotlib
from matplotlib      import pyplot as plt
plt.switch_backend('Agg')
# import matplotlib.gridspec as gridspec
from matplotlib import font_manager


def set_global_times_font():
  # prop = font_manager.FontProperties(fname='/projects/project-broaddus/install/fonts/times/Times New Roman Bold.ttf',size=10)
  font = font_manager.get_font('/projects/project-broaddus/install/fonts/times/Times New Roman Bold.ttf')
  font_manager.fontManager.ttflist.append(font)
  # font_dirs = ['/projects/project-broaddus/install/fonts/times/',]
  # font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
  # font_list = font_manager.createFontList(font_files)
  # font_manager.fontManager.ttflist.append(font_list[-1])
  # return prop, font_manager.fontManager.ttflist, font
  matplotlib.rcParams['font.family'] = 'Times New Roman'
  matplotlib.rcParams['font.size'] = 10
  # matplotlib.rcParams['font.weight'] = 'normal'

# /projects/project-broaddus/install/fonts/times/
## times new roman doesn't exist on cluster!?
# plt.rc('font',**{'family':['serif'], 'size':10, 'serif':['LiberationSerif-Regular']})
# plt.rc('text', usetex=False)

from scipy           import ndimage
from scipy.ndimage   import label, zoom
from scipy.signal    import correlate2d
from skimage         import io
from skimage.measure import compare_ssim
from numpy.fft import fft2,ifft2,ifftshift,fftshift

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

import pickle
## zero parameter functions. produce final results. never change targets.
## these functions reference files.py, and typically they require many results from 
## within files.py, bringing them all together for analysis.

import json
import shutil
import files


## but also make figures!

def load_flower_coleman(n2v='2', n2v2='06_4'):
  "load from local copy of fileserver data"
  d = SimpleNamespace()
  d.flower = imread("/projects/project-broaddus/rawdata/artifacts/flower.tif")
  d.flower = normalize3(d.flower,2,99.6)

  d.n2v    = imread(f"/projects/project-broaddus/denoise_experiments/flower/e01/mask00_{n2v}/pred.tif")
  d.n2v    = d.n2v[:,0].astype(np.float32)

  d.n2v2   = imread(f"/projects/project-broaddus/denoise_experiments/flower/e01/mask{n2v2}/pred.tif")
  d.n2v2   = d.n2v2[:,0].astype(np.float32)

  return d

def save_w_filter(e):
  def f(k,v):
    if 'fft' in k: return v
    else: return v[15]
  e2 = SimpleNamespace(**{k:f(k,v) for k,v in e.__dict__.items()})
  pickle.dump(e2,open("/projects/project-broaddus/denoise_experiments/fig_data/e2.pkl",'wb'))

## Intro

def load_figdata():
  d = SimpleNamespace()
  d.flower       = imread(files.flowerdata)
  d.flower       = normalize3(d.flower,2,99.6)
  d.gt           = d.flower.mean(0)
  d.shutter      = imread(files.shutterdata)
  d.n2v          = imread(files.n2v2_dirs[0][2]+'pred.tif')[:,0]
  d.n2v2         = imread(files.n2v2_dirs[6][4]+'pred.tif')[:,0]

  d.flower  = d.flower[15]
  d.shutter = d.shutter[15]
  d.n2v     = d.n2v[15]
  d.n2v2    = d.n2v2[15]

  d.flower  = d.flower.astype(np.float32)
  d.gt      = d.gt.astype(np.float32)
  d.shutter = d.shutter.astype(np.float32)
  d.n2v     = d.n2v.astype(np.float32)
  d.n2v2    = d.n2v2.astype(np.float32)

  pickle.dump(d, open(files.d_figdata + 'figdata.pkl','wb'))
  return d


@DeprecationWarning
def intro(d):

  x,y,dx,dy = 200,200,64,64
  ss = slice(y,y+dy), slice(x,x+dx)
  flower       = d.flower[ss]
  gt           = d.gt[ss]
  diff         = d.shutter[ss]
  n2v          = d.n2v[ss]
  n2v2         = d.n2v2[ss]

  # flower       = io.imread("fig_data_old/flower.png")
  # gt           = io.imread("fig_data_old/gt.png")
  # diff         = io.imread("fig_data_old/shutter.png")
  # n2v          = io.imread("fig_data_old/n2v.png")
  # n2v2         = io.imread("fig_data_old/n2v2.png")

  fig, ax = plt.subplots(nrows=2, ncols=3,figsize=(3.39,3.39*2/3), dpi=109*2)

  corr_fl = np.load(files.d_figdata + "ca/corr_raw_fl.npy")
  corr_fl = np.load(files.d_figdata + "ca/corr_raw_flGT.npy")
  corr_shut = np.load(files.d_figdata + "ca/corr_raw_shut.npy")
  corr = corr_shut

  # ax[1,2].imshow(autocorr_img,cmap='magma',interpolation='nearest')
  fontdict = {'color':'white'}

  y,x,dx,dy = 256,256,64,64
  ss = slice(y,y+dy), slice(x,x+dx)

  import os
  from matplotlib import font_manager as fm, rcParams
  fpath = "/projects/project-broaddus/install/fonts/times/Times New Roman Bold.ttf" #"fonts/ttf/cmr10.ttf")
  prop  = fm.FontProperties(fname=fpath,size=10)
  name = os.path.split(fpath)[1]

  ax[0,0].imshow(flower,cmap='magma',interpolation='nearest')
  ax[0,0].set_axis_off()
  ax[0,0].text(0.03,0.88,u"RAW",transform=ax[0,0].transAxes,fontproperties=prop,color='white')

  ax[1,0].imshow(gt,cmap='magma',interpolation='nearest')
  ax[1,0].set_axis_off()
  ax[1,0].text(0.03,0.88,"GT",transform=ax[1,0].transAxes,fontproperties=prop,color='white')

  ax[0,2].imshow(diff,cmap='magma',interpolation='nearest')
  ax[0,2].set_axis_off()
  ax[0,2].text(0.03,0.88,"NOISE",transform=ax[0,2].transAxes,fontproperties=prop,color='white')

  # ax[1,2].imshow(corr,cmap='magma',interpolation='nearest')
  # ax[1,2].set_axis_off()
  # ax[1,2].plot(corr[5,:])
  ax[1,2].text(0.0,0.88,"CORR",transform=ax[1,2].transAxes,fontproperties=prop,color='black')

  ax[1,1].imshow(n2v2,cmap='magma',interpolation='nearest')
  ax[1,1].set_axis_off()
  ax[1,1].text(0.03,0.88,"N2V$_s$",transform=ax[1,1].transAxes,fontproperties=prop,color='white')

  ax[0,1].imshow(n2v,cmap='magma',interpolation='nearest')
  ax[0,1].set_axis_off()
  ax[0,1].text(0.03,0.88,"N2V",transform=ax[0,1].transAxes,fontproperties=prop,color='white')

  fig.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=0.03,hspace=0.03)


  d0 = np.arange(corr.shape[0]) - corr.shape[0]//2
  d1 = np.arange(corr.shape[1]) - corr.shape[1]//2

  ax[1,2].set_yticks([])
  ax[1,2].set_xticks([-10,10])
  ax[1,2].spines['left'].set_position('zero')
  ax[1,2].spines['bottom'].set_position('zero')
  ax[1,2].spines['right'].set_visible(False)
  ax[1,2].spines['top'].set_visible(False)

  a,b = corr.shape
  y = corr[:,b//2]
  ax[1,2].plot(d0,y,'.-',lw=1,ms=1.5,label='y')
  y = corr[a//2,:]
  ax[1,2].plot(d1,y,'.-',lw=1,ms=1.5,label='x')
  ax[1,2].grid(False, which='both')

  ax11ins = ax[1,2].inset_axes([0.0,0.45,0.38,0.38])
  central_slice = slice(a//2-4,a//2+5), slice(b//2-4,b//2+5)
  ax11ins.imshow(corr[central_slice],cmap='gray',interpolation='nearest')

  ax11ins.axhline(corr[central_slice].shape[0]/2-0.5,c='C1',lw=.5)
  ax11ins.axvline(corr[central_slice].shape[1]/2-0.5,c='C0',lw=.5)
  ax11ins.set_yticks([])
  ax11ins.set_xticks([0,8])
  ax11ins.set_xticklabels([-4,4])
  ax11ins.tick_params(axis='x',size=2,labelsize=8,pad=2,)
  for s in ['left', 'bottom', 'right', 'top']: ax11ins.spines[s].set_visible(False)

  # ax[1,2].legend(loc='upper right',handlelength=2.0)

  fig.savefig(files.d_figdata + "intro.pdf")
  return fig,ax


## FFT

def add_fft(d):
  def _fft(img):
    img2 = img[15]
    img2 = (img2-img2.mean())/img2.std()
    img2 = np.abs(fftshift(fft2(img2)))
    # img2 = normalize3(img2,0,99)
    return img2
  d.fft_flower = _fft(d.flower)
  d.fft_n2v  = _fft(d.n2v)
  d.fft_n2v2 = _fft(d.n2v2)

def choose_the_mask(d):
  fig,axs = plt.subplots(nrows=2,ncols=3,figsize=(3.39,3.39*2/3),dpi=127*2)
  b,c = 1024,1024
  ss1 = slice(b//2-20,b//2+20), slice(c//2-20,c//2+20)
  ss2 = slice(b//2-256,b//2+256), slice(c//2-256,c//2+256)
  ss2 = slice(None), slice(None)

  e = SimpleNamespace()
  e.flower = d.flower[15][ss1]
  e.n2v = d.n2v[15][ss1].astype(np.float64)
  e.n2v2 = d.n2v2[15][ss1]
  e.fft_flower = d.fft_flower[ss2]
  e.fft_n2v = d.fft_n2v[ss2]
  e.fft_n2v2 = d.fft_n2v2[ss2]
  for k,v in e.__dict__.items(): io.imsave(f'../../denoise_experiments/fig_data/mask_{k}.png',v)

  axs[0,0].imshow(e.flower)
  axs[0,1].imshow(e.n2v)
  axs[0,2].imshow(e.n2v2)
  axs[1,0].imshow(e.fft_flower)
  axs[1,1].imshow(e.fft_n2v)
  axs[1,2].imshow(e.fft_n2v2)
  for a in axs.flat: a.axis('off')
  fig.subplots_adjust(left=.005,bottom=.005,right=.995,top=.995,wspace=0.005,hspace=0.005)
  return fig,axs

def choose_the_mask_just_ffts(d):
  fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(3.39,3.39*1/3),dpi=127*2)

  # import os
  # from matplotlib import font_manager as fm, rcParams
  # fpath = os.path.join("/projects/project-broaddus/install/fonts/times/Times New Roman Bold.ttf") #"fonts/ttf/cmr10.ttf")
  # prop  = fm.FontProperties(fname=fpath,size=10)
  # fname = os.path.split(fpath)[1]

  d.fft_flower, d.fft_n2v, d.fft_n2v2 = normalize3(stak(d.fft_flower, d.fft_n2v, d.fft_n2v2,),2,99)

  axs[0].imshow(d.fft_flower, interpolation='nearest')
  axs[0].text(0.03,0.88,"RAW", transform=axs[0].transAxes, color='white')
  axs[1].imshow(d.fft_n2v, interpolation='nearest')
  axs[1].text(0.03,0.88,"N2V", transform=axs[1].transAxes, color='white')
  axs[2].imshow(d.fft_n2v2, interpolation='nearest')
  axs[2].text(0.03,0.88,"N2V$_s$", transform=axs[2].transAxes, color='white')
  for a in axs.flat: a.axis('off')
  fig.subplots_adjust(left=.005,bottom=.005,right=.995,top=.995,wspace=0.005,hspace=0.005)
  fig.savefig('../../denoise_experiments/fig_data/ffts.pdf')

  return fig,axs

def savemany():
  """
  Ran it! After visual inspection the best result is '06_4' ... 
  And after performing the same visual optimization over N2V=='00_?' we get '00_2' as the best.
  NOTE: both searches only update the top right panel
  """
  # for s in [f'0{n}_{m}' for n in [3,4,5,6] for m in [0,1,2,3,4]]:
  # for s in [f'00_{m}' for m in [0,1,2,3,4]]: 
  s = '06_4'
  e = load_flower_coleman(s)
  add_fft(e)
  fig,axs = choose_the_mask(e)
  fig.savefig(f'../../denoise_experiments/fig_data/choose{s}.pdf')

##

def gather_results_to_figdata_dir():
  Path(files.d_figdata).mkdir(exist_ok=True,parents=True)
  res  = utils.recursive_map2(csv2floatList, files.all_tables)
  json.dump(res,open(files.d_figdata + 'allscores.json','w'))
  
  flower = imread(files.flowerdata)
  gt = normalize3(flower,2,99.6).mean(0)

  imsave(flower[:5], files.d_figdata + 'flower.tif')
  imsave(gt, files.d_figdata + 'gt.tif')
  imsave(imread(files.n2v2_dirs[0][0]+'pred.tif')[:5,0], files.d_figdata + 'n2v2.tif')
  imsave(imread(files.n2gt_dirs[0]+'pred.tif')[:5,0], files.d_figdata + 'n2gt.tif')
  imsave(np.array([imread(files.bm3d_dir+f'img{n:03d}.tif') for n in range(5)]), files.d_figdata + 'bm3d.tif')
  imsave(imread(files.nlm_dir+'denoised.tif')[:5], files.d_figdata + 'nlm.tif')
  
  imsave(imread(files.shutterdata)[:5], files.d_figdata + 'shutter.tif')
  
  imsave(np.array([imread(f) for f in files.celedata]), files.d_figdata + 'cele.tif')
  cele_n2v2_dirs  

def make_predtifs_smaller():
  for p in glob("/projects/project-broaddus/denoise_experiments/flower/e01/mask??_?/pred.tif"):
    img = imread(p)
    imsave(img.astype(np.float16), p, compress=9)

### utils 

def csv2floatList(csvfile):
  r = list(csv.reader(open(csvfile), delimiter=','))
  return [float(x) for x in r[1]]

## parameterized funcs. have no knowledge of filesys.

def load_prediction_and_eval_metrics__generic(rawdata, loaddir):
  raw_all = imread(rawdata)
  raw_all = normalize3(raw_all,2,99.6)
  gt  = raw_all.mean(0)

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

  met = eval_single_metrics(gt, img)

  header=['mse','psnr','ssim']
  writecsv([header,met], loaddir / 'table.csv')

def eval_single_metrics(gt,ys,nth=1):
  ys   = ys[::nth]
  mse  = ((gt-ys)**2).mean((0,1,2))
  psnr = 10*np.log10(1/mse)
  ssim = np.array([compare_ssim(gt,ys[i].astype(np.float64)) for i in range(ys.shape[0]//50)])
  ssim = ssim.mean()
  return [mse,psnr,ssim]

def correlation_analysis(rawdata,savedir,name,removeGT=False):
  savedir = Path(savedir); savedir.mkdir(exist_ok=True,parents=True)

  img  = imread(rawdata)
  
  # png_name = f'autocorr_img_{name}.png'
  # pdf_name = f'autocorr_plot_{name}.pdf'

  if removeGT: img = img-img.mean(0)

  corr = np.array([autocorrelation(img[i]) for i in range(10)])
  corr = corr.mean(0)
  a,b  = corr.shape
  corr = corr[a//2-15:a//2+15, b//2-15:b//2+15]
  corr = corr / corr.max()
  # io.imsave(savedir / png_name,normalize3(corr))
  np.save(savedir/f'corr_raw_{name}.npy',corr)

  ## This is now done entirely during figure plotting, locally.

  # d0 = np.arange(corr.shape[0])-corr.shape[0]//2
  # d1 = np.arange(corr.shape[1])-corr.shape[1]//2
  # plt.figure()
  # a,b = corr.shape
  # # y = corr.mean(1); y = y/y.max()
  # y = corr[:,b//2]
  # plt.plot(d0,y,'.-', label='y profile')
  # # y = corr.mean(0); y = y/y.max()
  # y = corr[a//2,:]
  # plt.plot(d1,y,'.-', label='x profile')
  # plt.legend()
  # plt.savefig(savedir / pdf_name)

### utils

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
    return pi.real

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
