import gputools
import tifffile
from segtools.numpy_utils import normalize3
from pathlib import Path
from subprocess import run
import numpy as np

def cat(*args,axis=0): return np.concatenate(args, axis)
def stak(*args,axis=0): return np.stack(args, axis)

def imsave(x, name, **kwargs): return tifffile.imsave(str(name), x, **kwargs)
def imread(name,**kwargs): return tifffile.imread(str(name), **kwargs)


def nlm_2d(rawdata, savedir, sigma=0.1, **kwargs):
  # dir = "nlm"
  # savedir = denoise_experiments / f'flower/e01/{dir}/'
  savedir = Path(savedir)
  savedir.mkdir(exist_ok=True,parents=True)
  # img  = imread(rawdata_dir / 'artifacts/flower.tif')
  img  = imread(rawdata)
  pmin, pmax = 2, 99.6
  img  = normalize3(img,pmin,pmax,axs=(1,2)).astype(np.float32,copy=False)
  pimg = []
  for x in img:
    ## gputools.denoise.nlm2(data, sigma, size_filter=2, size_search=3)
    ## for noise level of sigma_0, choose sigma = 1.5*sigma_0
    x = gputools.denoise.nlm2(x,sigma,**kwargs)
    pimg.append(x)
  pimg = np.array(pimg)
  imsave(pimg, savedir/f'denoised.tif')

def nlm_3d_cele(savedir, sigma=0.1, **kwargs):
  # dir = "nlm"
  # savedir = denoise_experiments / f'flower/e01/{dir}/'
  celedata = '/lustre/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01/'
  savedir = Path(savedir); savedir.mkdir(exist_ok=True,parents=True)
  # img  = imread(rawdata_dir / 'artifacts/flower.tif')
  for i in [0,10,100,189]:

    img  = imread(celedata + f"t{i:03d}.tif")
    pmin, pmax = 2, 99.6
    img  = normalize3(img,pmin,pmax,axs=(1,2)).astype(np.float32,copy=False)
    ## gputools.denoise.nlm2(data, sigma, size_filter=2, size_search=3)
    ## for noise level of sigma_0, choose sigma = 1.5*sigma_0
    pimg = gputools.denoise.nlm3(img,sigma,**kwargs)
    imsave(pimg, savedir / f'denoised{i:03d}.tif')

def bm3d_2d(rawdata, savedir, sigma=0.1, **kwargs):
  img = imread(rawdata)
  img = normalize3(img,2,99.6)

  bm3d = "/projects/project-broaddus/comparison_methods/bm3d/build/bm3d"
  savedir = Path(savedir); savedir.mkdir(exist_ok=True,parents=True)
  tmpdir = savedir / 'tmp/'; tmpdir.mkdir(exist_ok=True,parents=True)

  for i in range(100):
    tmpname = tmpdir  / f"img{i:03d}.tif"
    outname = savedir / f"img{i:03d}.tif"

    if not tmpname.exists():
      imsave(img[i], tmpname)
    run(f"{bm3d} {tmpname} 0.1 {outname}",shell=True)