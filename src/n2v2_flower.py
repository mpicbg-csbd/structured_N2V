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

## bash command to run this script on the cluster. replace `00x` with uniqe id.

## copy and paste this command into bash to run a job via the job management queueing system.

bashcmd = """
mkdir -p /lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower_test/
cp n2v2_flower.py /lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower_test/
srun -J flw3_10 -n 1 -c 1 --mem=128000 -p gpu --gres=gpu:1 --time=12:00:00 -e std.err.flower_test -o std.out.flower_test time python3 /lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower_test/n2v2_flower.py &
"""

bashcmd = """
mkdir -p /lustre/projects/project-broaddus/denoise_experiments/flower/e02/flower1_1/
cp n2v2_flower.py /lustre/projects/project-broaddus/denoise_experiments/flower/e02/flower1_1/
srun -J flw1_1 -n 1 -c 1 --mem=128000 -p gpu --gres=gpu:1 --time=12:00:00 -e std.err.flower1_1 -o std.out.flower1_1 time python3 /lustre/projects/project-broaddus/denoise_experiments/flower/e02/flower1_1/n2v2_flower.py &
"""

savedir = Path('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower_test') #/flower3_9/')

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

  shutil.copy2('/lustre/projects/project-broaddus/denoise_code/src/n2v2_flower.py', savedir)
  shutil.copy2('/lustre/projects/project-broaddus/denoise_code/src/torch_models.py', savedir)

def wipe_dirs(savedir):
  if savedir.exists():
    shutil.rmtree(savedir)
    savedir.mkdir()
  # for x in (savedir/'epochs/').glob('*.png'): x.unlink()
  # for x in (savedir/'rgbs/').glob('*.png'): x.unlink()
  # for x in (savedir/'pimgs/').glob('*.png'): x.unlink()
  # for x in (savedir/'pts/').glob('*.png'): x.unlink()
  # for x in (savedir/'movie/').glob('*.png'): x.unlink()
  # for x in (savedir/'counts/').glob('*.png'): x.unlink()
  # for x in savedir.glob('*.png'): x.unlink()
  # for x in savedir.glob('*.pdf'): x.unlink()
  # for x in savedir.glob('*.pkl'): x.unlink()
  # for x in savedir.glob('*.py'): x.unlink()
  # for x in savedir.glob('*.npz'): x.unlink()

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

## heavier meaty functions

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

def setup(params={}):

  wipe_dirs(savedir)
  init_dirs(savedir)

  # dg = datagen(savedir=savedir)
  # data = dg.data
  # data = np.load('/lustre/projects/project-broaddus/devseg_data/cl_datagen/grid/data_shutter.npz')['data']
  data = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/data_flower3.npz')['data']
  # data = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e02/data_flower.npz')['data']

  d = SimpleNamespace()
  d.net = torch_models.Unet2_2d(16,[[1],[1]],finallayer=nn.ReLU).cuda()
  d.net.load_state_dict(torch.load('/lustre/projects/project-broaddus/denoise_experiments/flower/models/net_randinit.pt'))

  d.net.apply(init_weights);
  d.savedir = savedir

  # d.net.load_state_dict(torch.load('/lustre/projects/project-broaddus/devseg_data/cl_datagen/d000/jj000/net250.pt'))
  # torch.save(d.net.state_dict(), '/lustre/projects/project-broaddus/devseg_data/cl_datagen/rsrc/net_random_init_unet2.pt')

  d.x1_all  = torch.from_numpy(data).float().cuda()
  return d

def init_training_artifacts():
  ta = SimpleNamespace()
  ta.losses = []
  ta.lossdists = []
  ta.e = 0
  return ta

def train(d,ta=None,end_epoch=301):
  if ta is None: ta = init_training_artifacts()
  batch_size = 4
  inds = np.arange(0,d.x1_all.shape[0])
  # example_xs = d.x1_all[inds[::floor(np.sqrt(len(inds)))]].clone()
  example_xs = d.x1_all[[0,3,5,12]].clone()
  xs_fft = torch.fft((example_xs-example_xs.mean())[...,None][...,[0,0]],2).norm(p=2,dim=-1)
  xs_fft = torch.from_numpy(np.fft.fftshift(xs_fft.cpu(),axes=(-1,-2))).cuda()
  w1_all = torch.ones(d.x1_all.shape).float().cuda()
  opt = torch.optim.Adam(d.net.parameters(), lr = 2e-5)
  lossdist = torch.zeros(d.x1_all.shape[0]) - 2

  patch_size = d.x1_all.shape[2:]

  plt.figure()
  for e in range(ta.e,end_epoch):
    ta.e = e
    np.random.shuffle(inds)
    ta.lossdists.append(lossdist.numpy().copy())
    lossdist[...] = -1
    print(f"\r epoch {e}", end="")
  
    for b in range(ceil(d.x1_all.shape[0]/batch_size)):
      idxs = inds[b*batch_size:(b+1)*batch_size]
      x1   = d.x1_all[idxs] #.cuda()
      w1   = w1_all[idxs] #.cuda()

      def random_pixel_mask():
        n = int(np.prod(patch_size) * 0.02)
        x_inds = np.random.randint(0,patch_size[1],n)
        y_inds = np.random.randint(0,patch_size[0],n)
        # z_inds = np.random.randint(0,32,64*64*1)
        ma = np.zeros(patch_size)
        ma[y_inds,x_inds] = 2
        return ma
      
      def sparse_3set_mask():
        "build random mask for small number of central pixels"
        n = int(np.prod(patch_size) * 0.02)
        x_inds = np.random.randint(0,patch_size[1],n)
        y_inds = np.random.randint(0,patch_size[0],n)
        ma = np.zeros(patch_size)

        # ma = binary_dilation(ma)

        for i in [1,2,3,4,5,6,7]:
          m = x_inds-i >= 0;            ma[y_inds[m],x_inds[m]-i] = 1
          m = x_inds+i < patch_size[1]; ma[y_inds[m],x_inds[m]+i] = 1
        # for i in [1]:
        #   m = y_inds-i >= 0;            ma[y_inds[m]-i,x_inds[m]] = 1
        #   m = y_inds+i < patch_size[0]; ma[y_inds[m]+i,x_inds[m]] = 1

        ma = ma.astype(np.uint8)
        ma[y_inds,x_inds] = 2

        return ma

      def checkerboard_mask():
        ma = np.indices(patch_size).transpose((1,2,0))
        ma = np.floor(ma/(1,256)).sum(-1) %2==0
        ma = 2*ma
        if e%2==1: ma = 2-ma
        return ma

      ma = sparse_3set_mask()
      # ipdb.set_trace()
      # return ma

      ## apply mask to input
      w1[:,:] = torch.from_numpy(ma.astype(np.float)).cuda()
      x1_damaged = x1.clone()
      x1_damaged[w1>0] = torch.rand(x1.shape).cuda()[w1>0]

      y1p = d.net(x1_damaged)

      dims = (1,2,3) ## all dims except batch
      dx = 0.15*torch.abs(y1p[:,:,:,1:] - y1p[:,:,:,:-1])
      dy = 0.15*torch.abs(y1p[:,:,1:] - y1p[:,:,:-1])

      if False:
        dy = 0.25*torch.abs(y1p[:,:,:,1:] - y1p[:,:,:,:-1])
        dz = 0.05*torch.abs(y1p[:,:,1:] - y1p[:,:,:-1])
        c0,c1,c2 = 0.0, 0.15, 1.0
        potential = 2e2 * ((y1p-c0)**2 * (y1p-c2)**2) ## rough locations for three classes
        resid = torch.abs(y1p-x1)**2
        loss_per_patch = resid.mean(dims) + dx.mean(dims) #+ dy.mean(dims) + dz.mean(dims) #+ potential.mean(dims)
      
      tm = (w1==2).float() ## target mask
      loss_per_patch = (tm * torch.abs(y1p-x1)**2).sum(dims) / tm.sum(dims) # + dx.mean(dims) + dy.mean(dims) #+ dz.mean(dims)
      # ipdb.set_trace()

      # loss_per_patch = (w1 * torch.abs(y1p-y1t)).sum(dims) / w1.sum(dims) #+ 1e-3*(y1p.mean(dims)).abs()
      # loss_per_patch = (w1 * -(y1t*torch.log(y1p + 1e-7) + (1-y1t)*torch.log((1-y1p) + 1e-7))).sum(dims) / w1.sum(dims) #+ 1e-2*(y1p.mean(dims)).abs()
      lossdist[idxs] = loss_per_patch.detach().cpu()
      loss = loss_per_patch.mean()
      ta.losses.append(float(loss))
      
      opt.zero_grad()
      loss.backward()
      opt.step()

    ## predict on examples and save each epoch

    with torch.no_grad():
      example_yp = d.net(example_xs)
      # xs_fft = xs_fft/xs_fft.max()
      yp_fft = torch.fft((example_yp - example_yp.mean())[...,None][...,[0,0]],2).norm(p=2,dim=-1) #.cpu().detach().numpy()
      yp_fft = torch.from_numpy(np.fft.fftshift(yp_fft.cpu(),axes=(-1,-2))).cuda()
      # yp_fft = yp_fft/yp_fft.max()

      rgb = torch.stack([example_xs,w1[[0]*len(example_xs)]/2,xs_fft,example_yp,yp_fft],0).cpu().detach().numpy()
      arr = rgb.copy()
      # type,samples,channels,y,x
      rgb = normalize3(rgb,axs=(1,2,3,4))
      rgb[[2,4]] = normalize3(rgb[[2,4]],pmin=0,pmax=99.0,axs=(1,2,3,4))
      # return rgb
      # remove channels and permute
      rgb = collapse2(rgb[:,:,0],'tsyx','sy,tx')
      # arr = collapse2(arr[:,:,0],'tsyx','sy,tx')

      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if e%10==0:  io.imsave(d.savedir / f'epochs/rgb_{e:03d}.png', rgb)
        if e%100==0: np.save(d.savedir / f'epochs_npy/arr_{e:03d}.npy', arr)

    batches_per_epoch = ceil(d.x1_all.shape[0]/batch_size)
    epochs = np.arange(len(ta.losses)) / batches_per_epoch
    plt.clf()
    plt.plot(epochs,ta.losses)
    # plt.ylim(np.mean(ta.losses)-3*np.std(ta.losses),np.mean(ta.losses)+3*np.std(ta.losses))
    plt.yscale('log')
    plt.xlabel(f'1 epoch = {batches_per_epoch} batches')
    plt.savefig(d.savedir/f'loss.png',dpi=300)
    if e%100==0:
      torch.save(d.net.state_dict(), savedir/f'models/net{e:03d}.pt')

  pklsave(ta.losses,d.savedir/f'losses.pkl')
  torch.save(d.net.state_dict(), d.savedir/f'models/net{ta.e:03d}.pt')
  return ta

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

## prediction and analysis

def apply_net_tiled(net,img):
  """
  Applies func to image with dims Channels,Z,Y,X
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

def e01_fig2_flower():
  # img1 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_1/epochs_npy/arr_600.npy')
  # img2 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_2/epochs_npy/arr_600.npy')
  # img3 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_3/epochs_npy/arr_600.npy')
  # img4 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_4/epochs_npy/arr_600.npy')
  # img5 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_5/epochs_npy/arr_600.npy')
  img6 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_6/epochs_npy/arr_600.npy')
  img7 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_7/epochs_npy/arr_600.npy')
  img8 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_8/epochs_npy/arr_600.npy')
  img9 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_9/epochs_npy/arr_600.npy')
  img10 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_10/epochs_npy/arr_600.npy')
  img11 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_11/epochs_npy/arr_600.npy')
  img12 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_12/epochs_npy/arr_600.npy')
  img13 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_13/epochs_npy/arr_600.npy')
  img14 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_14/epochs_npy/arr_600.npy')
  img15 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_15/epochs_npy/arr_600.npy')
  img16 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_16/epochs_npy/arr_600.npy')

  ## (N2V, OURS 2class, OURS 3class) , (raw, mask, raw fft, pred, pred fft) , n_samples , channels, y , x
  # rgb = stak(img1, img2, img3, img4, img5, img6, img7, img8, img9)
  rgb = stak(img6, img7, img8, img9, img10, img11, img12, img13, img14, img15, img16,)

  # rgb[:,[2,4]] = normalize3(rgb[:,[2,4]], pmin=0, pmax=99.0)
  # rgb[:,[2,4]] = normalize3(np.log(rgb[:,[2,4]]+1e-7))
  rgb[:,[2,4]] = normalize3(np.log(normalize3(rgb[:,[2,4]],0,99)+1e-7))
  rgb[:,[0,3]] = normalize3(rgb[:,[0,3]])
  rgb[:,1]     = normalize3(rgb[:,1])

  ## remove channels and pad xy with white
  rgb = rgb[:,:,:,0]
  # rgb = np.pad(rgb,[(0,0),(0,0),(0,0),(0,1),(0,1)],mode='constant',constant_values=1)

  # plt.figure()
  # d = np.fft.fftshift(np.fft.fftfreq(256))
  # for i,m in enumerate("N2V,OURS 2class,OURS 3class".split(',')):
  #   plt.plot(d,rgb[i,-1].mean((0,1)),label=f'{m} : avg s,y')
  #   plt.plot(d,rgb[i,-1].mean((0,2)),label=f'{m} : avg s,x')
  # plt.legend()

  ## reshape to (raw, N2V, ours 2 class, ours 3class) , (real, fft, mask), samples, y, x

  # rgb = rgb.reshape((15, 4, 256, 256))[]
  rgb = cat(stak(np.zeros(rgb[0,0].shape),rgb[0,0],rgb[0,2])[None],rgb[:,[1,3,4]])

  ## models, types, samples, y, x
  # rgb = collapse2(rgb,'mtsyx','mt,sy,x')
  # rgb = rgb[[0,1,2,3,4,6,8,9,11,13,14]]
  # rgb = rgb[[0,1,5,8,3,6,9,2,4,7,10,]]
  # rgb = collapse2(rgb,'myx','y,mx')

  # io.imsave(savedir.parent/'shutterclosed_normalized.png',rgb[:64])
  np.savez_compressed(savedir.parent / 'e01_fig2_flower.npz', rgb=rgb)

  return rgb

def e02_fig2_flower():
  img6 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e02/flower1_1/epochs_npy/arr_400.npy')
  img7 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e02/flower1_2/epochs_npy/arr_400.npy')
  img8 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e02/flower1_3/epochs_npy/arr_400.npy')
  img9 = np.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e02/flower1_4/epochs_npy/arr_400.npy')

  ## (N2V, OURS 2class, OURS 3class) , (raw, mask, raw fft, pred, pred fft) , n_samples , channels, y , x
  rgb = stak(img6, img7, img8, img9)

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

def predict_on_full_flower():
  "make movies scrolling through z"

  net  = torch_models.Unet2_2d(16,[[1],[1]],finallayer=nn.ReLU).cuda()
  # Rob Jenkin (Alana) 540 692 0113
  net.load_state_dict(torch.load('/lustre/projects/project-broaddus/denoise_experiments/flower/e01/flower3_6/models/net600.pt'))
  img  = imread(f'/lustre/projects/project-broaddus/rawdata/artifacts/flower.tif')
  # pmin, pmax = np.random.uniform(1,3), np.random.uniform(99.5,99.8)
  pmin, pmax = 2, 99.6
  img  = normalize3(img,pmin,pmax,axs=(1,2)).astype(np.float32,copy=False)
  pimg = []
  for x in img:
    # x = torch.from_numpy(x).cuda()
    # x = net(x[None])
    x = apply_net_tiled(net,x[None])
    pimg.append(x)
  pimg = np.array(pimg)
  # return img, net, pimg
  # pimg = apply_net_tiled(net,img[:,None])
  imsave(pimg, savedir/f'pred_flower.tif')


  # rgb = cat(img, pimg[0], axis=1)
  # rgb = rgb.clip(min=0)
  # moviesave(normalize3(rgb), savedir/f'movie/vert{ds}_{i:03d}.mp4', rate=4)
  # imsave(pimg, savedir/f'pimgs/pimg{ds}_{i:03d}.tif')


  ## make histogram of pimg values at points

  # for name in sorted((savedir/'pimgs/').glob('*.tif')):
  #   pimg = imread(savedir/f'pimgs/pimg{i:03d}.tif')

  ## 2d rgb pngs
  # imsave(pimg, savedir/f'pimg/pimg000.tif',compress=8)
  # rgb1 = cat(pimg[0,:64].max(0), pimg[0,64:].max(0))[...,None]
  # rgb2 = cat(img[0,:64].max(0), img[0,64:].max(0))[...,None][...,[0,0,0]]
  # rgb2[...,[0]] += rgb1
  # rgb2 = normalize3(rgb2)
  # io.imsave(savedir/'rgbs/rgb001.png',rgb2)

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
  # d = setup()
  # ta = train(d,end_epoch=1001)
  e01_fig2_flower()
  # d = SimpleNamespace()
  # d.net = torch_models.Unet2(16,[[1],[1]],finallayer=nn.ReLU).cuda()
  # d.net.load_state_dict(torch.load(savedir/'net099.pt'))
  # print(summary(d.net))

  # plot_losses(d,ta)
  # predict_on_full_flower(d)


info = """

How to use this script
======================

## Interactive mode

First, you have to be on the cluster on a gpu node!
Then open up ipython terminal and import as `fl`.
Then you can initialize an untrained model with:

```
import n2v2_flower as fl
d = fl.setup(); ta = fl.init_training_artifacts();
```

Now you can train the model with 

```
fl.train(d,ta)
```

## run the script

First, you have to be on the cluster on falcon (not a gpu node).
Then adjust the values of `bashcmd` and `savedir` at the top of this file to point to your chosen save directory.
Then simply copy the text of `bashcmd` and paste it into the terminal.

Enjoy!


File history and todo
=====================

BUG: Why doesn't the line `from utils import point_matcher` fail when moving/running the script from a new folder?
BUGFIX: in sparse_3set_mask() the mask was the wrong shape.



"""