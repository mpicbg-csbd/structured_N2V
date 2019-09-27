import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import sys
import ipdb
import itertools
import warnings
import shutil
import pickle

from  pprint   import pprint
from  types    import SimpleNamespace
from  math     import floor,ceil
from  pathlib  import Path

import tifffile
import numpy         as np
import skimage.io    as io
import matplotlib.pyplot as plt
plt.switch_backend("agg")

from scipy.ndimage        import zoom, label
from skimage.feature      import peak_local_max
from skimage.segmentation import find_boundaries
from skimage.measure      import regionprops
from skimage.morphology   import binary_dilation

from segtools.numpy_utils import collapse2, normalize3, plotgrid
from segtools import color
from segtools.defaults.ipython import moviesave

import torch_models


## copy and paste this command into bash to run a job via the job management queueing system.

# bashcmd = """
# srcdir=/lustre/projects/project-broaddus/denoise_code/src/
# flowerdir=/lustre/projects/project-broaddus/denoise_experiments/flower/e01/n2gt/
# rm -rf $flowerdir/*
# mkdir -p $flowerdir/{epochs,epochs_npy,pimgs,pts,movie,counts,models}
# cp $srcdir/n2gt_2d.py $srcdir/torch_models.py $flowerdir
# srun -J n2gt_2d -n 1 -c 1 --mem=128000 -p gpu --gres=gpu:1 --time=12:00:00 -e $flowerdir/stderr -o $flowerdir/stdout \
# time python3 $flowerdir/n2gt_2d.py &
# """

# savedir = Path(experiments_dir/'flower/e01/n2gt').resolve() #/flower3_9/')

## lightweight funcs and utils

def init_dirs(savedir):
  savedir.mkdir(exist_ok=True)
  (savedir/'epochs/').mkdir(exist_ok=True)
  (savedir/'epochs_npy/').mkdir(exist_ok=True)
  (savedir/'pimgs/').mkdir(exist_ok=True)
  (savedir/'pts/').mkdir(exist_ok=True)
  (savedir/'movie/').mkdir(exist_ok=True)
  (savedir/'counts/').mkdir(exist_ok=True)
  (savedir/'models/').mkdir(exist_ok=True)

def wipe_dirs(savedir):
  if savedir.exists():
    shutil.rmtree(savedir)
    savedir.mkdir()

def cat(*args,axis=0): return np.concatenate(args, axis)
def stak(*args,axis=0): return np.stack(args, axis)

def imsave(x, name, **kwargs): return tifffile.imsave(str(name), x, **kwargs)
def imread(name,**kwargs): return tifffile.imread(str(name), **kwargs)

def pklload(name):
  return pickle.load(open(name,'rb'))

def pklsave(obj,name):
  par = Path(name).parent
  par.mkdir(exist_ok=True,parents=True)
  pickle.dump(obj,open(name,'wb'))

def i2rgb(img):
  if img.shape[-1] == 1: img = img[...,[0,0,0]]
  if img.shape[-1] == 2: img = img[...,[0,1,1]]
  if img.shape[-1]  > 3: img = img[...,None][...,[0,0,0]]
  img = img.astype(np.float)
  return img

def receptivefield(net):
  "calculate and show the receptive field or receptive kernel"
  def rfweights(m):
    if type(m) == nn.Conv2d:
      m.weight.data.fill_(1/(5*5)) ## conv kernel 3*5*5
      m.bias.data.fill_(0.0)
  net.apply(rfweights);
  x0 = np.zeros((256,256)); x0[128,128]=1;
  xout = net.cuda()(torch.from_numpy(x0)[None,None].float().cuda()).detach().cpu().numpy()
  io.imsave(savedir/'recfield_xy.png',normalize3(xout[0,128]))
  io.imsave(savedir/'recfield_xz.png',normalize3(xout[0,:,128]))

def init_weights(m):
  "use as arg in net.apply()"
  if type(m) == nn.Conv2d:
    torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
    m.bias.data.fill_(0.05)

def std_weights(m):
  "use as arg in net.apply()"
  if type(m) == nn.Conv3d:
    print("{:.5f} {:.5f}".format(float(m.weight.std()), float(m.bias.mean())))

def random_slice(img_size, patch_size):
  assert len(img_size) == len(patch_size)
  def f(d,s):
    if s == -1: return slice(None)
    start = np.random.randint(0,d-s+1)
    end   = start + s
    return slice(start,end)
  return tuple([f(d,s) for d,s in zip(img_size, patch_size)])

def init_training_artifacts():
  ta = SimpleNamespace()
  ta.losses = []
  ta.lossdists = []
  ta.e = 0
  return ta

## heavier meaty functions

@DeprecationWarning
def datagen(savedir=None):

  # img = imread(f'/lustre/projects/project-broaddus/rawdata/artifacts/flower.tif')[:10]
  img = imread(f'/lustre/projects/project-broaddus/denoise_experiments/flower/e02/pred_flower.tif')[:10]
  # img = imread(f'/lustre/projects/project-broaddus/rawdata/artifacts/shutterclosed.tif')[0]

  print(img.shape)
  # pmin, pmax = np.random.uniform(1,3), np.random.uniform(99.5,99.8)
  pmin, pmax = 2, 99.6
  print(f"pmin = {pmin}; pmax = {pmax}")
  img = normalize3(img,pmin,pmax).astype(np.float32,copy=False)
  data = img.reshape((-1, 4,256,4,256)).transpose((0,1,3,2,4)).reshape((-1,1,256,256))

  # patch_size = (256,256)
  # slicelist = []
  # def random_patch():
  #   ss = random_slice(img.shape, patch_size)

  #   ## select patches with interesting content. FIXME
  #   while img[ss].mean() < 0.0:
  #     ss = random_slice(img.shape, patch_size)
  #   x  = img[ss].copy()
  #   slicelist.append(ss)

  #   ## augment
  #   # noiselevel = 0.2
  #   # x += np.random.uniform(0,noiselevel,(1,)*3)*np.random.uniform(-1,1,x.shape)
  #   # for d in [0,1,2]:
  #   #   if np.random.rand() < 0.5:
  #   #     x  = np.flip(x,d)

  #   return (x,)

  # data = np.array([random_patch() for _ in range(24)])


  # data = np.load('../../devseg_data/cl_datagen/d003/data.npz')
  print("data.shape: ", data.shape)

  #SCZYX
  if savedir:
    rgb = collapse2(data[:,:],'scyx','s,y,x,c')[...,[0,0,0]]
    rgb = normalize3(rgb)
    rgb = plotgrid([rgb],10)
    io.imsave(savedir/'data_xy_flower.png',rgb)
    np.savez_compressed(savedir/'data_flower.npz',data=data,pmin=pmin,pmax=pmax)
    # pklsave(slicelist, savedir/'slicelist2.pkl')

  dg = SimpleNamespace()
  dg.data = data
  dg.pmin = pmin
  dg.pmax = pmax

  return dg

def setup(savedir):

  savedir = Path(savedir)
  wipe_dirs(savedir)
  init_dirs(savedir)

  data = imread('/lustre/projects/project-broaddus/rawdata/artifacts/flower.tif')
  data = normalize3(data,2,99.6) ## normalize across all dims?

  d = SimpleNamespace()
  d.net = torch_models.Unet2_2d(16,[[1],[1]],finallayer=nn.ReLU)
  d.net.load_state_dict(torch.load('/lustre/projects/project-broaddus/denoise_experiments/flower/models/net_randinit.pt'))

  # d.net.apply(init_weights);
  d.savedir = savedir

  d.xs = torch.from_numpy(data).float()
  d.xs = d.xs.reshape(100,4,256,4,256,1).permute(0,1,3,5,2,4) #.reshape((-1,256,256))
  d.ys = d.xs.mean(0)

  io.imsave(savedir / 'xs.png', collapse2(d.xs[0,:,:,0].numpy(),"12yx","1y,2x"))
  io.imsave(savedir / 'ys.png', collapse2(d.ys[:,:,0].numpy(),"12yx","1y,2x"))
  d.cuda = False

  return d

def train(d,ta=None,end_epoch=300,already_on_cuda=False):
  if ta is None: ta = init_training_artifacts()

  ## setup const variables necessary for training
  batch_size = 4
  inds       = np.arange(0,d.xs.shape[0])
  patch_size = d.xs.shape[4:]
  # xs = d.xs.reshape((100,4,256,4,256)).permute((0,1,3,2,4)) #.reshape((-1,256,256))
  # ys = d.xs.mean(0).reshape((4,256,4,256)).permute((0,2,1,3))
  d.ws = torch.ones(d.xs.shape).float()

  ## set up variables for monitoring training
  # d.example_xs = d.xs[inds[::floor(np.sqrt(len(inds)))]].clone()
  d.example_xs = d.xs[[0,3,5,12],0,0].reshape(-1,1,256,256).clone().cpu()
  d.xs_fft     = torch.fft((d.example_xs-d.example_xs.mean())[...,None][...,[0,0]],2).norm(p=2,dim=-1)
  d.xs_fft     = torch.from_numpy(np.fft.fftshift(d.xs_fft,axes=(-1,-2)))
  lossdist   = torch.zeros(d.xs.shape[0]) - 2

  ## move vars to gpu
  # if d.cuda is False:
  d.net  = d.net.cuda()
  d.xs   = d.xs.cuda()
  d.ys   = d.ys.cuda()
  d.xs_fft = d.xs_fft.cuda()
  d.example_xs = d.example_xs.cuda()
  d.ws = d.ws.cuda()

  ## initialize optimizer (must be done after moving data to gpu ?)
  opt = torch.optim.Adam(d.net.parameters(), lr = 2e-4)

  plt.figure()
  for e in range(ta.e,end_epoch+1):
    ta.e = e
    np.random.shuffle(inds)
    ta.lossdists.append(lossdist.numpy().copy())
    lossdist[...] = -1
    print(f"\r epoch {e}", end="")
  
    for b in range(ceil(d.xs.shape[0]/batch_size)):
      idxs = inds[b*batch_size:(b+1)*batch_size]
      x1   = d.xs[idxs]
      w1   = d.ws[idxs]
      # y1   = d.ys[idxs]

      x1  = x1.reshape(-1,1,256,256)
      y1p = d.net(x1)
      # x1  = x1.reshape(-1,4,4,256,256)
      # y1p = y1p.reshape(-1,4,4,256,256)
      y1p = y1p.reshape(4,4,4,1,256,256)

      # ipdb.set_trace()

      dims = (1,2,3,4,5) ## all dims except batch

      # ipdb.set_trace()
      loss_per_patch = ((y1p-d.ys)**2).mean(dims)

      # loss_per_patch = (w1 * torch.abs(y1p-y1t)).sum(dims) / w1.sum(dims) #+ 1e-3*(y1p.mean(dims)).abs()
      # loss_per_patch = (w1 * -(y1t*torch.log(y1p + 1e-7) + (1-y1t)*torch.log((1-y1p) + 1e-7))).sum(dims) / w1.sum(dims) #+ 1e-2*(y1p.mean(dims)).abs()
      lossdist[idxs] = loss_per_patch.detach().cpu()
      loss = loss_per_patch.mean()
      ta.losses.append(float(loss))
      
      opt.zero_grad()
      loss.backward()
      opt.step()

    ## predict on examples and save each epoch
    if e%10==0:
      with torch.no_grad():
        example_yp = d.net(d.example_xs)

        ## compute fft from predictions
        yp_fft = torch.fft((example_yp - example_yp.mean())[...,None][...,[0,0]],2).norm(p=2,dim=-1) #.cpu().detach().numpy()
        ## shift frequency domain s.t. zer freq is at center of array
        yp_fft = torch.from_numpy(np.fft.fftshift(yp_fft.cpu(),axes=(-1,-2))).cuda()

        ## stack (real space, -weights-, real fft, predictions, and prediction fft) along a new dimension
        rgb = torch.stack([d.example_xs, d.xs_fft, example_yp, yp_fft],0).cpu().detach().numpy()
        arr = rgb.copy()
        ## first normalize each type to [0,1] independently
        rgb = normalize3(rgb,axs=(1,2,3,4)) # dims=type,samples,channels,y,x
        ## then normalize fft's and real-space dims separately 
        rgb[[1,3]] = normalize3(rgb[[1,3]],pmin=0,pmax=99.0,axs=(1,2,3,4))

        ## remove channels and permute into a 2D image
        rgb = collapse2(rgb[:,:,0],'tsyx','sy,tx')

        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          if e%10==0:  io.imsave(d.savedir / f'epochs/rgb_{e:03d}.png', rgb)
          if e%100==0: np.save(d.savedir / f'epochs_npy/arr_{e:03d}.npy', arr)

    ## plot loss 
    batches_per_epoch = ceil(d.xs.shape[0]/batch_size)
    epochs = np.arange(len(ta.losses)) / batches_per_epoch
    plt.clf()
    plt.plot(epochs,ta.losses)
    # plt.ylim(np.mean(ta.losses)-3*np.std(ta.losses),np.mean(ta.losses)+3*np.std(ta.losses))
    plt.yscale('log')
    plt.xlabel(f'1 epoch = {batches_per_epoch} batches')
    plt.savefig(d.savedir/f'loss.png',dpi=300)
    
    ## save model weights
    if e%100==0:
      torch.save(d.net.state_dict(), d.savedir/f'models/net{e:03d}.pt')

  pklsave(ta.losses,d.savedir/f'losses.pkl')
  torch.save(d.net.state_dict(), d.savedir/f'models/net{ta.e:03d}.pt')
  return ta

## 

def multitrain(d):

  if False:
    torch.manual_seed(jj)
    net.apply(init_weights);
    torch.manual_seed(42)
    net.load_state_dict(torch.load('/lustre/projects/project-broaddus/devseg_data/cl_datagen/rsrc/net_random_init_unet2.pt'))
    np.random.seed(jj)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  lossesjj = []
  jj=0
  for jj in range(j,6):
    d.savedir = savedir / f'jj{jj:03d}'; init_dirs(d.savedir)
    ta = init_training_artifacts()
    train(d,ta,100)
    lossesjj.append(ta.losses)
    predict_movies(d)

  plt.figure()
  for loss in lossesjj:
    plt.plot(np.convolve(loss,np.ones(50)/50,mode='valid'),lw=1)
  plt.yscale('log')
  plt.savefig(savedir/'multi_losses.png',dpi=300)

## shared plotting

def plot_losses(d,ta):
  
  ## plot simple loss trajectory

  plt.figure()
  plt.plot(ta.losses)
  plt.ylim(0,ta.losses[0])
  plt.savefig(d.savedir/'loss.pdf')

  ## plot loss distribution trajectories

  lds = ta.lossdists[1::3]
  N   = len(lds)
  colors = color.pastel_colors_RGB(N,max_saturation=0.9,brightness=0.8,shuffle=False)
  # colors = np.arange(N)[:,None][:,[0,0,0]] * (15,-15,15) + (15,240,15)
  # colors = colors/255
  plt.figure()
  for i in np.arange(N):
    plt.plot(sorted(lds[i]),'.',color=colors[i]+[0.25])
  # plt.ylim(0,np.max(lds))
  # plt.scatter(np.r_[0:N],np.ones(N)*1,c=colors)
  plt.savefig(savedir / 'lossdist.pdf')
  plt.figure()
  for i in np.arange(N):
    plt.plot(lds[i],'.',color=colors[i]+[0.25])
  # plt.scatter(np.r_[0:N],np.ones(N)*1,c=colors)
  plt.savefig(d.savedir / 'lossdist_unsorted.pdf')

def histograms():
  "cumulative dist of pixel values in img and pimg"
  plt.figure()
  x = np.linspace(0,100,100)
  plt.plot(x,np.percentile(img,x),label='img')
  plt.plot(x,np.percentile(pimg,x),label='pimg')
  plt.legend()
  plt.savefig(savedir/'histogram_img_pimg.pdf')

if __name__=='__main__':
  print("Training...")
  # params = pklload(sys.argv[1]) if len(sys.argv) > 1 else {}
  # print(params)
  # net = torch_models.Unet(32,[[1],[1]]).cuda()
  # net.load_state_dict(torch.load(savedir/'net.pt'))
  # analysis({'net':net})
  # train()

  # d  = setup()
  # ta = train(d,end_epoch=1001)

  # predict_on_full_flower_for_all_e01_models()

  # e01_fig2_flower()
  # d = SimpleNamespace()
  # d.net = torch_models.Unet2(16,[[1],[1]],finallayer=nn.ReLU).cuda()
  # d.net.load_state_dict(torch.load(savedir/'net099.pt'))
  # print(summary(d.net))

  # plot_losses(d,ta)
  # predict_on_full_flower(d)

info = """

Mon Sep 23
copied from n2v2_flower.py

"""