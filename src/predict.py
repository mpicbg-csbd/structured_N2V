import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import sys
# import ipdb
import warnings
import shutil
import pickle
import itertools

from  types    import SimpleNamespace
from  math     import floor,ceil
from  pathlib  import Path

import tifffile
import numpy         as np

from segtools.numpy_utils import normalize3

import torch_models

def cat(*args,axis=0): return np.concatenate(args, axis)
def stak(*args,axis=0): return np.stack(args, axis)

def imsave(x, name, **kwargs): return tifffile.imsave(str(name), x, **kwargs)
def imread(name,**kwargs): return tifffile.imread(str(name), **kwargs)

def apply_net_tiled_2d(net,img):
  """
  Applies func to image with dims Channels,Y,X
  """

  # borders           = [8,20,20] ## border width within each patch that is thrown away after prediction
  # patchshape_padded = [32,240,240] ## the size of the patch that we feed into the net. must be divisible by 8 or net fails.
  # patchshape        = [16,200,200] ## must be divisible by 8 to avoid artifacts.
  # stride            = [16,200,200] ## same as patchshape in this case
  def f(n,m): return (ceil(n/m)*m)-n ## f(n,m) gives padding needed for n to be divisible by m
  def g(n,m): return (floor(n/m)*m)-n ## f(n,m) gives un-padding needed for n to be divisible by m

  b,c = img.shape[1:]
  r,s = f(b,8),f(c,8) ## calculate extra border needed for stride % 8 = 0

  YPAD,XPAD = 24,24

  img_padded = np.pad(img,[(0,0),(YPAD,YPAD+r),(XPAD,XPAD+s)],mode='constant') ## pad for patch borders
  output = np.zeros(img.shape)
  
  # zs = np.r_[:a:16]
  ys = np.r_[:b:200]
  xs = np.r_[:c:200]

  for x,y in itertools.product(xs,ys):
    re,se = min(y+200,b+r), min(x+200,c+s)
    be,ce = min(y+200,b),   min(x+200,c)
    patch = img_padded[:,y:re+2*YPAD,x:se+2*XPAD]
    patch = torch.from_numpy(patch).cuda().float()
    with torch.no_grad():
      patch = net(patch[None])[0,:,YPAD:-YPAD,XPAD:-XPAD].detach().cpu().numpy()
    output[:,y:be,x:ce] = patch[:,:be-y,:ce-x]

  return output

def predict_on_full_2d_stack(rawdata, savedir, weights):
    savedir = Path(savedir)
    net  = torch_models.Unet2_2d(16,[[1],[1]],finallayer=nn.ReLU).cuda()
    net.load_state_dict(torch.load(weights))
    img  = imread(rawdata)
    pmin, pmax = 2, 99.6
    img  = normalize3(img,pmin,pmax,axs=(1,2)).astype(np.float32,copy=False)
    pimg = []
    for x in img:
      x = apply_net_tiled_2d(net,x[None])
      pimg.append(x)
    pimg = np.array(pimg)
    imsave(pimg.astype(np.float16), savedir/f'pred.tif', compress=9)


