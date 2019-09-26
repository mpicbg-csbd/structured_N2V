import gputools
import tifffile
from segtools.numpy_utils import normalize3
from pathlib import Path

def cat(*args,axis=0): return np.concatenate(args, axis)
def stak(*args,axis=0): return np.stack(args, axis)

def imsave(x, name, **kwargs): return tifffile.imsave(str(name), x, **kwargs)
def imread(name,**kwargs): return tifffile.imread(str(name), **kwargs)

def nlm2d(rawdata, savedir, sigma=0.1, **kwargs):
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
