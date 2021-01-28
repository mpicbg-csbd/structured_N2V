import gputools
import tifffile
from tifffile import imread,imsave
from segtools.numpy_utils import normalize3
from pathlib import Path
from subprocess import run
import numpy as np
from segtools.ns2dir import load,save
import itertools
from glob import glob

def cat(*args,axis=0): return np.concatenate(args, axis)
def stak(*args,axis=0): return np.stack(args, axis)

def imsave(x, name, **kwargs): return tifffile.imsave(str(name), x, **kwargs)
def imread(name,**kwargs): return tifffile.imread(str(name), **kwargs)

from scipy.optimize import minimize_scalar
flowerdata = '/projects/project-broaddus/rawdata/artifacts/flower.tif'
flowerdir  = '/projects/project-broaddus/denoise_experiments/flower/e01/'
# home = Path('/projects/project-broaddus/')

def optimize_nlm():
  img = imread(flowerdata)
  img = normalize3(img,2,99.6)
  gt = img.mean(0)

  def obj(sigma):
    res = gputools.denoise.nlm2(img[1],sigma)
    return ((gt-res)**2).mean()

  # for s in [0.01,0.1,0.5,0.9,2.0]:
  #   print(obj(s))

  print(minimize_scalar(obj, bracket=(0.1,0.5,0.9)))

def nlm_2d(rawdata, savedir, **kwargs):
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
    sigma = 0.1826499502297115 ## obtained through optimization vs GT
    x = gputools.denoise.nlm2(x,sigma,**kwargs)
    pimg.append(x)
  pimg = np.array(pimg)
  imsave(pimg, savedir/f'denoised.tif')

def nlm_3d_cele(savedir, sigma=0.1, **kwargs):

  celedata = '/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01/'
  savedir  = Path(savedir); savedir.mkdir(exist_ok=True,parents=True)
  # img  = imread(rawdata_dir / 'artifacts/flower.tif')
  for i in [0,10,100,189]:

    img  = imread(celedata + f"t{i:03d}.tif")
    pmin, pmax = 2, 99.6
    img  = normalize3(img,pmin,pmax,axs=(1,2)).astype(np.float32,copy=False)
    ## gputools.denoise.nlm2(data, sigma, size_filter=2, size_search=3)
    ## for noise level of sigma_0, choose sigma = 1.5*sigma_0
    pimg = gputools.denoise.nlm3(img,sigma,**kwargs)
    imsave(pimg, savedir / f'denoised{i:03d}.tif')

def nlm_2d_cele_just189():
  name = '/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01/t189.tif'
  img = imread(name)
  img = normalize3(img,2,99.6)
  img = img[22]

  r_sigma = np.linspace(.3,.6,10) + 0.36
  r_size_filer = [3,4,5] #[1,2,3,4,5,6]
  r_size_serach = [6,7,8,9,10] #[5,10,15,20]
  count = 0
  for p1,p2,p3 in itertools.product(r_sigma,r_size_filer,r_size_serach):
    print(count)
    img2 = gputools.denoise.nlm2(img,p1,size_filter=p2,size_search=p3)
    name2 = f"/projects/project-broaddus/denoise_experiments/cele/e01/nlm2_2d/t{count:03d}.tif"
    save(img2, name2)
    count += 1

def nlm_3d_cele_just189(n):
  name = '/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01/t189.tif'
  img  = imread(name)
  img  = normalize3(img,2,99.6)

  r_sigma = np.linspace(.3,.6,10)
  r_size_filer = [1,2,3] #,4,5,6]
  r_size_serach = [1,5,10,15] #[5,10,15,20]

  count = 0
  for p1,p2,p3 in itertools.product(r_sigma,r_size_filer,r_size_serach):
  # count = n
  # p1,p2,p3 = list(itertools.product(r_sigma,r_size_filer,r_size_serach))[n]

    print(count)
    img2 = gputools.denoise.nlm3(img,p1,size_filter=p2,size_search=p3)
    name2 = f"/projects/project-broaddus/denoise_experiments/cele/e01/nlm2_3d/t{count:03d}.tif"
    save(img2, name2)
    save(img2[22],name2.replace("nlm2_3d/","nlm2_3d_s22/"))
    count += 1

def nlmupdate():
  name = "/projects/project-broaddus/denoise_experiments/cele/e01/nlm2_3d_s22/"
  for i in range(120,150):
    fi = name + f"t{i:03d}.tif"
    print(fi)
    # os.remove(fi)


## run once. throwaway.

def run01():
  p = "/projects/project-broaddus/denoise_experiments/cele/e01/nlm2_3d/"
  for name in sorted(glob(p + '*.tif')):
    print(name)
    img = imread(name)
    save(img[22], name.replace("nlm2_3d/","nlm2_3d_s22/"))

## BM3d/ bm4d stuff

def optimize_bm3d():
  img = imread(flowerdata)
  img = normalize3(img,2,99.6)
  gt = img.mean(0)
  bm3d = "/projects/project-broaddus/comparison_methods/bm3d/build/bm3d"
  tmp = flowerdir + 'bm3d/eg0.tif'
  imsave(img[0], tmp)
  outname = flowerdir + 'bm3d/res0.tif'

  def obj(sigma):
    run(f"{bm3d} {tmp} {sigma} {outname}",shell=True)
    res = imread(outname)
    return ((gt-res)**2).mean()

  print(minimize_scalar(obj, bracket=(0.01, 0.3, 0.5)))

def bm3d_3d_cele_just189():
  name  = '/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01/t189.tif'
  name2 = name.replace("rawdata/celegans_isbi/","denoise_experiments/cele/e01/bm3d2/tmp/")
  img   = imread(name)
  img   = normalize3(img,2,99.6)
  img   = img[22]
  imsave(img,name2)

  bm3d = "/projects/project-broaddus/comparison_methods/bm3d/build/bm3d"
  r_sigma = np.arange(10)*0.6 + 0.3
  count = 0
  for sigma in r_sigma:
    print(count)
    name3 = Path(name2).parent / f"out_{count}.tif"
    run(f"{bm3d} {name2} {sigma} {name3}",shell=True)
    count += 1
    img = load(name3)
    print(img.max(),img.dtype)
    m = np.isnan(img)
    print(f'nan: {m.sum()}')
    img[m]=0
    print(f'nan: {m.sum()}')
    save(img,name3)
    # img2 = gputools.denoise.nlm3(img,p1,size_filter=p2,size_search=p3)
    # save(img2, name.replace(f"rawdata/celegans_isbi/","denoise_experiments/cele/e01/nlm2/t{count}.tif"))

def bm4d():

  # call = '/Applications/MATLAB.app/bin/matlab -nosplash -nodesktop –nojvm -r "denoise_file($FILENAME,$SIGMA),quit"'
  name  = '/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01/t189.tif'
  img   = load(name)
  img   = normalize3(img,2,99.6)
  img   = (img).astype(np.float32)
  ## fail: np.float64,np.float32,uint32,uint16,uint8
  name2 = '/projects/project-broaddus/denoise_experiments/data/celegans_isbi/Fluo-N3DH-CE/01/t189_f32.tif'
  imsave(img,name2)
  sigmas = [0.04, 0.05, 0.07, 0.1 , 0.14, 0.19, 0.26, 0.35, 0.49, 0.67]
  sigmas = [0.60, 0.7, 0.8, 0.9]
  for s in sigmas:
  # call  = f'/sw/apps/matlab/current/bin/matlab -nosplash -nodesktop -nojvm -r "denoise_file({name2},{sigma}),quit"'
    call  = f"""
    cd /projects/project-broaddus/denoise_experiments/data/bm4d
    /sw/apps/matlab/current/bin/matlab -nosplash -nodesktop -nojvm -r "denoise_file(\'{name2}\',{s}),quit"
    """
    run(call,shell=True)

def bm3d_2d(rawdata, savedir, **kwargs):
  img = imread(rawdata)
  img = normalize3(img,2,99.6)

  bm3d = "/projects/project-broaddus/comparison_methods/bm3d/build/bm3d"
  savedir = Path(savedir); savedir.mkdir(exist_ok=True,parents=True)
  tmpdir  = savedir / 'tmp/'; tmpdir.mkdir(exist_ok=True,parents=True)

  sigma = 0.15488 ## optimized vs GT

  for i in range(100):
    tmpname = tmpdir  / f"img{i:03d}.tif"
    outname = savedir / f"img{i:03d}.tif"

    if not tmpname.exists():
      imsave(img[i], tmpname)
    run(f"{bm3d} {tmpname} {sigma} {outname}",shell=True)

