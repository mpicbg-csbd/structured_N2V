import numpy as np
import tifffile
from tabulate import tabulate
import subprocess
import os
from  types    import SimpleNamespace
from pathlib import Path

from matplotlib             import pyplot as plt
from scipy import ndimage
from scipy.ndimage              import label, zoom
from skimage import io
from skimage.measure import compare_ssim
from skimage.measure import compare_ssim

# import spimagine
from segtools.numpy_utils import normalize3, perm2, collapse2, splt
from segtools.StackVis import StackVis

from csbdeep.utils.utils import normalize_minmse


def cat(*args,axis=0): return np.concatenate(args, axis)
def stak(*args,axis=0): return np.stack(args, axis)

def imsave(x, name, **kwargs): return tifffile.imsave(str(name), x, **kwargs)
def imread(name,**kwargs): return tifffile.imread(str(name), **kwargs)
def imsavefiji(x, **kwargs): return tifffile.imsave('/Users/broaddus/Desktop/stack.tiff', x, imagej=True, **kwargs)

savedir = Path('/Users/broaddus/Desktop/Projects/denoise/').resolve()

def syncall():
  subdir  = f"flower/e01/" #flower3_10/"
  sync(savedir / subdir, subdir)

  # for i in range(6):
  #   subdir  = f"flower/e02/flower1_{i}/"
  #   sync(savedir / subdir, subdir)

## load data

def eval_01():
  denoise_dir = Path('/Volumes/project-broaddus/denoise/')
  data = np.load(denoise_dir/'flower/e01/e01_fig2_flower.npz')['rgb']
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
  # iss = StackVis(data)
  return SimpleNamespace(data=data,names=names)

def eval_02():
  denoise_dir = Path('/Volumes/project-broaddus/denoise/')
  data = np.load(denoise_dir/'flower/e02/e02_fig2_flower.npz')['rgb']
  names = [
    "n2v",
    "n2v^2",
    "n2v plus",
    "n2v bigplus",
    "n2v xxxxoxxxx",
    "n2v xox",
    "n2v xxoxx",
    ]

  # iss = StackVis(data)
  # print(names)
  # return iss
  return SimpleNamespace(data=data,names=names)

def eval_flower_gt():
  flower_all = imread('/Users/broaddus/Downloads/20190520_tl_25um_20msec_01pc_488_130EM_Conv.tif')
  flower_all = normalize3(flower_all,2,99.6)
  flower_all_patches = flower_all[0].reshape((4,256,4,256)).transpose((0,2,1,3)).reshape((16,256,256))
  flower_all_patches = flower_all_patches[[0,3,5,12]]
  flower_gt = flower_all.mean(0)
  flower_gt_patches = flower_gt.reshape((4,256,4,256)).transpose((0,2,1,3)).reshape((16,256,256))
  flower_gt_patches = flower_gt_patches[[0,3,5,12]]
  return flower_gt_patches

def fulldata():
  e01 = eval_01()
  e02 = eval_02()
  gt  = eval_flower_gt()
  dat = SimpleNamespace(e01=e01,e02=e02,gt=gt)
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
  for i,name in enumerate(dat.e01.names):
    x = np.linspace(0,100,101)
    plt.plot(x,np.percentile(delta[i].mean(0), x), label=name)

  delta = dat.e02.data[:,1] - dat.gt
  for i,name in enumerate(dat.e02.names):
    x = np.linspace(0,100,101)
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






