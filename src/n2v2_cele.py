import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import sys
# import ipdb
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
# from scipy.ndimage.morphology import binary_dilation
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

# bashcmd = """
# mkdir -p /lustre/projects/project-broaddus/devseg_data/cl_datagen/grid/cele4/
# cp n2v2_cele.py /lustre/projects/project-broaddus/devseg_data/cl_datagen/grid/cele4/
# srun -J cele4 -n 1 -c 1 --mem=128000 -p gpu --gres=gpu:1 --time=12:00:00 -e std.err000 -o std.out000 time python3 /lustre/projects/project-broaddus/devseg_data/cl_datagen/grid/cele4/n2v2_cele.py &
# """

# savedir = Path('/lustre/projects/project-broaddus/devseg_data/cl_datagen/grid/cele4/')

## lightweight, universal funcs and utils

def init_dirs(savedir):
  savedir.mkdir(exist_ok=True)
  (savedir/'epochs/').mkdir(exist_ok=True)
  (savedir/'epochs_npy/').mkdir(exist_ok=True)
  # (savedir/'pts/').mkdir(exist_ok=True)
  # (savedir/'counts/').mkdir(exist_ok=True)
  (savedir/'models/').mkdir(exist_ok=True)

  # shutil.copy2('/lustre/projects/project-broaddus/devseg_code/detect/n2v2_cele.py', savedir)
  # shutil.copy2('/lustre/projects/project-broaddus/devseg_code/detect/torch_models.py', savedir)

def wipe_dirs(savedir):
  if savedir.exists():
    shutil.rmtree(savedir)
    savedir.mkdir()

def cat(*args,axis=0):  return np.concatenate(args, axis)
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

def receptivefield3d(net,kern=(3,5,5)):
  "calculate and show the receptive field or receptive kernel"
  def rfweights(m):
    if type(m) == nn.Conv3d:
      m.weight.data.fill_(1/np.prod(kern)) ## conv kernel 3*5*5
      m.bias.data.fill_(0.0)
  net.apply(rfweights);
  x0 = np.zeros((128,128,128)); x0[64,64,64]=1;
  xout = net.cuda()(torch.from_numpy(x0)[None,None].float().cuda()).detach().cpu().numpy()
  io.imsave(savedir/'recfield_xy.png',normalize3(xout[0,0,64]))
  io.imsave(savedir/'recfield_xz.png',normalize3(xout[0,0,:,64]))

def receptivefield2d(net,kern=(5,5)):
  "calculate and show the receptive field or receptive kernel"
  def rfweights(m):
    if type(m) == nn.Conv2d:
      m.weight.data.fill_(1/np.prod(kern)) ## conv kernel 3*5*5
      m.bias.data.fill_(0.0)
  net.apply(rfweights);
  x0 = np.zeros((256,256)); x0[128,128]=1;
  xout = net.cuda()(torch.from_numpy(x0)[None,None].float().cuda()).detach().cpu().numpy()
  io.imsave(savedir/'recfield_xy.png',normalize3(xout[0,0]))

def init_weights(m):
  "use as arg in net.apply()"
  if type(m) in [nn.Conv2d, nn.Conv3d]:
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

def datagen(params={}, savedir=None):
  data = []

  times = np.r_[:190]

  for i in times:
    img = imread(f'/lustre/projects/project-broaddus/devseg_data/raw/celegans_isbi/Fluo-N3DH-CE/01/t{i:03d}.tif')


    pmin, pmax = np.random.uniform(1,3), np.random.uniform(99.5,99.8)
    img = normalize3(img,pmin,pmax).astype(np.float32,copy=False)

    slicelist = []
    def random_patch():
      ss = random_slice(img.shape, (32,64,64))
      ## select patches with interesting content. 0.02 is chosen by manual inspection.
      while img[ss].mean() < 0.03:
        ss = random_slice(img.shape, (32,64,64))
      x  = img[ss].copy()
      slicelist.append(ss)
  
      ## augment
      # noiselevel = 0.2
      # x += np.random.uniform(0,noiselevel,(1,)*3)*np.random.uniform(-1,1,x.shape)
      # for d in [0,1,2]:
      #   if np.random.rand() < 0.5:
      #     x  = np.flip(x,d)

      return (x,)

    data.append([random_patch() for _ in range(10)]) #ts(xys)czyx

  data = np.array(data)

  print("data.shape: ", data.shape)

  if savedir:
    rgb = collapse2(data[:,:,:,16],'tscyx','ty,sx,c')[...,[0,0,0]]
    rgb = normalize3(rgb)
    io.imsave(savedir/'data_xy_cele.png',rgb)
    rgb = collapse2(data[:,:,:,:,32],'tsczx','tz,sx,c')[...,[0,0,0]]
    rgb = normalize3(rgb)
    io.imsave(savedir/'data_xz_cele.png',rgb)
    np.savez_compressed(savedir/'data_cele.npz',data)
    pklsave(slicelist, savedir/'slicelist_cele.pkl')

  return data

def setup(savedir):

  savedir = Path(savedir)
  wipe_dirs(savedir)
  init_dirs(savedir)

  # data = cl_datagen2.datagen_self_sup(s=4, savedir=savedir)
  # data = cl_datagen2.datagen_all_kinds(savedir=savedir)

  data = np.load('/lustre/projects/project-broaddus/denoise_experiments/cele/e01/data_cele.npz')['arr_0']
  # data = datagen(savedir=savedir)
  data = collapse2(data[None],'rtsczyx','c,ts,r,z,y,x')[0]

  d = SimpleNamespace()
  d.net = torch_models.Unet2(16,[[1],[1]],finallayer=nn.ReLU).cuda()
  d.net.load_state_dict(torch.load('/lustre/projects/project-broaddus/denoise_experiments/flower/models/net_randinit3D.pt'))
  # d.net.apply(init_weights);
  d.savedir = savedir
  # torch.save(d.net.state_dict(), '/lustre/projects/project-broaddus/devseg_data/cl_datagen/grid/net_randinit3D.pt')

  d.x1_all  = torch.from_numpy(data).float().cuda()
  return d

def train(d,ta=None,end_epoch=301,xmask=[0,1,2],ymask=[1,2,3,4,5]):
  if ta is None: ta = init_training_artifacts()
  batch_size = 4
  inds = np.arange(0,d.x1_all.shape[0])
  example_xs = d.x1_all[inds[::floor(np.sqrt(len(inds)))]].clone()
  xs_fft = torch.fft((example_xs-example_xs.mean())[...,None][...,[0,0]],3).norm(p=2,dim=-1)
  xs_fft = torch.from_numpy(np.fft.fftshift(xs_fft.cpu(),axes=(-1,-2,-3))).cuda()
  w1_all = torch.ones(d.x1_all.shape).float().cuda()
  opt = torch.optim.Adam(d.net.parameters(), lr = 4e-6)
  lossdist = torch.zeros(d.x1_all.shape[0]) - 2

  patch_size = d.x1_all.shape[2:]
  # ipdb.set_trace()

  if False:
    kern355 = -torch.ones((3,3,3))
    kern355[1,1,1] = 26
    kern355 /= 26
    kern355 = kern355[None,None].cuda() # batch and channels

  plt.figure()
  for e in range(ta.e,end_epoch+1):
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
        n = int(np.prod(patch_size) * 0.9)
        x_inds = np.random.randint(0,patch_size[2],n)
        y_inds = np.random.randint(0,patch_size[1],n)
        z_inds = np.random.randint(0,patch_size[0],n)
        ma = np.zeros(patch_size)
        ma[z_inds,y_inds,x_inds] = 2
        return ma
      
      def sparse_3set_mask():
        "build random mask for small number of central pixels"
        n = int(np.prod(patch_size) * 0.02)
        x_inds = np.random.randint(0,patch_size[2],n)
        y_inds = np.random.randint(0,patch_size[1],n)
        z_inds = np.random.randint(0,patch_size[0],n)
        ma = np.zeros(patch_size)
        
        for i in xmask:
          m = x_inds-i >= 0;            ma[z_inds[m], y_inds[m],x_inds[m]-i] = 1
          m = x_inds+i < patch_size[2]; ma[z_inds[m], y_inds[m],x_inds[m]+i] = 1
        for i in ymask:
          m = y_inds-i >= 0;            ma[z_inds[m], y_inds[m]-i,x_inds[m]] = 1
          m = y_inds+i < patch_size[1]; ma[z_inds[m], y_inds[m]+i,x_inds[m]] = 1

        ma = ma.astype(np.uint8)
        ma[z_inds,y_inds,x_inds] = 2
        return ma

      def checkerboard_mask():
        ma = np.indices(patch_size).transpose((1,2,3,0))
        ma = np.floor(ma/(2,16,2)).sum(-1) %2==0
        ma = 2*ma
        if e%2==1: ma = 2-ma
        return ma

      # ma = random_pixel_mask()
      ma = sparse_3set_mask()
      # ipdb.set_trace()
      # return ma

      ## apply mask to input
      w1[:,:] = torch.from_numpy(ma.astype(np.float)).cuda()
      x1_damaged = x1.clone()
      x1_damaged[w1>0] = torch.rand(x1.shape).cuda()[w1>0]

      y1p = d.net(x1_damaged)

      dims = (1,2,3,4) ## all dims except batch

      if False:
        dx = 0.15*torch.abs(y1p[:,:,:,:,1:] - y1p[:,:,:,:,:-1])
        dy = 0.15*torch.abs(y1p[:,:,:,1:] - y1p[:,:,:,:-1])
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

    if e%10==0:
      with torch.no_grad():
        example_yp = d.net(example_xs)
        # xs_fft = xs_fft/xs_fft.max()
        yp_fft = torch.fft((example_yp - example_yp.mean())[...,None][...,[0,0]],2).norm(p=2,dim=-1) #.cpu().detach().numpy()
        yp_fft = torch.from_numpy(np.fft.fftshift(yp_fft.cpu(),axes=(-1,-2,-3))).cuda()
        # yp_fft = yp_fft/yp_fft.max()

        rgb = torch.stack([example_xs,w1[[0]*len(example_xs)]/2,xs_fft,example_yp,yp_fft],0).cpu().detach().numpy()
        arr = rgb.copy()
        # type,samples,channels,z,y,x
        rgb = normalize3(rgb,axs=(1,2,3,4,5))
        rgb[[2,4]] = normalize3(rgb[[2,4]],pmin=0,pmax=99.0,axs=(1,2,3,4,5))
        # return rgb
        # remove channels,z and permute
        rgb = collapse2(rgb[:,:,0,patch_size[0]//2],'tsyx','sy,tx')
        # arr = collapse2(arr[:,:,0],'tsyx','sy,tx')

        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          io.imsave(d.savedir / f'epochs/rgb_{e:03d}.png', rgb)
          if e%20==0: np.save(d.savedir / f'epochs_npy/arr_{e:03d}.npy', arr)

    batches_per_epoch = ceil(d.x1_all.shape[0]/batch_size)
    epochs = np.arange(len(ta.losses)) / batches_per_epoch
    plt.clf()
    plt.plot(epochs,ta.losses)
    # plt.ylim(np.mean(ta.losses)-3*np.std(ta.losses),np.mean(ta.losses)+3*np.std(ta.losses))
    plt.yscale('log')
    plt.xlabel(f'1 epoch = {batches_per_epoch} batches')
    plt.savefig(d.savedir/f'loss.png',dpi=300)
    if e%50==0:
      torch.save(d.net.state_dict(), d.savedir/f'models/net{e:03d}.pt')

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

  a,b,c = img.shape[1:]
  q,r,s = f(a,8),f(b,8),f(c,8) ## calculate extra border needed for stride % 8 = 0

  ZPAD,YPAD,XPAD = 8,24,24

  img_padded = np.pad(img,[(0,0),(ZPAD,ZPAD+q),(YPAD,YPAD+r),(XPAD,XPAD+s)],mode='constant') ## pad for patch borders
  output = np.zeros(img.shape)

  zs = np.r_[:a:16]
  ys = np.r_[:b:200]
  xs = np.r_[:c:200]

  for x,y,z in itertools.product(xs,ys,zs):
    qe,re,se = min(z+16,a+q),min(y+200,b+r),min(x+200,c+s)
    ae,be,ce = min(z+16,a),min(y+200,b),min(x+200,c)
    patch = img_padded[:,z:qe+2*ZPAD,y:re+2*YPAD,x:se+2*XPAD]
    patch = torch.from_numpy(patch).cuda().float()
    patch = net(patch[None])[0,:,ZPAD:-ZPAD,YPAD:-YPAD,XPAD:-XPAD].detach().cpu().numpy()
    output[:,z:ae,y:be,x:ce] = patch[:,:ae-z,:be-y,:ce-x]

  return output

def analyze_losses(d,ta):
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


@DeprecationWarning
def fig2_shutterclosed_comparison():
  img1 = np.load('/lustre/projects/project-broaddus/devseg_data/cl_datagen/grid/cele1/epochs_npy/arr_080.npy')
  img2 = np.load('/lustre/projects/project-broaddus/devseg_data/cl_datagen/grid/cele2/epochs_npy/arr_080.npy')
  img3 = np.load('/lustre/projects/project-broaddus/devseg_data/cl_datagen/grid/cele3/epochs_npy/arr_080.npy')
  img4 = np.load('/lustre/projects/project-broaddus/devseg_data/cl_datagen/grid/cele4/epochs_npy/arr_080.npy')
  img5 = np.load('/lustre/projects/project-broaddus/devseg_data/cl_datagen/grid/cele5/epochs_npy/arr_080.npy')

  ## (N2V, OURS 2class, OURS 3class) , (raw, mask, raw fft, pred, pred fft) , n_samples , channels, y , x

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
  np.savez_compressed(savedir.parent / 'fig2_cele.npz', rgb=rgb)

  return rgb


def predict_movies(d):
  "make movies scrolling through z"
  d.savedir = Path(d.savedir)
  (d.savedir/'pimgs/').mkdir(exist_ok=True)
  (d.savedir/'movie/').mkdir(exist_ok=True)  

  ds = "01"
  for i in [0,10,100,189]:
  # for i in [189]:
    img = imread(f'/lustre/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{ds}/t{i:03d}.tif')
    # lab = imread(f'/lustre/projects/project-broaddus/devseg_data/raw/celegans_isbi/Fluo-N3DH-CE/{ds}_GT/TRA/man_track{i:03d}.tif')

    # pmin, pmax = np.random.uniform(1,3), np.random.uniform(99.5,99.8)
    pmin,pmax = 2, 99.6
    img = normalize3(img,pmin,pmax).astype(np.float32,copy=False)

    with torch.no_grad():
      pimg = apply_net_tiled(d.net,img[None])

    rgb = cat(img, pimg[0], axis=1)
    rgb = rgb.clip(min=0)
    # moviesave(normalize3(rgb), d.savedir/f'movie/vert{ds}_{i:03d}.mp4', rate=4)
    imsave(pimg.astype(np.float16), d.savedir/f'pimgs/pimg{ds}_{i:03d}.tif', compress=9)

    if False:
      rgb = i2rgb(img)
      rgb[...,[0,2]] = pimg[0,...,None][...,[0,0]]
      rgb[...,1] -= pimg[0]
      rgb = rgb.clip(min=0)
      moviesave(normalize3(pimg[0]), d.savedir/f'movie/pimg{i:03d}.mp4', rate=4) ## set i=30 and i=150 to get res022 and res023.
      moviesave(normalize3(rgb), d.savedir/f'movie/mix{i:03d}.mp4', rate=4)

def predict_from_new(modeldir):
    d = SimpleNamespace()
    d.savedir = modeldir
    d.net = torch_models.Unet2(16,[[1],[1]],finallayer=nn.ReLU).cuda()
    d.net.load_state_dict(torch.load(modeldir+'models/net100.pt'))
    predict_movies(d)

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
  fig2_shutterclosed_comparison()

  # d = setup()
  # ta = train(d,end_epoch=100)
  # analyze_losses(d,ta)
  # predict_movies(d)

  # d = SimpleNamespace()
  # d.net = torch_models.Unet2(16,[[1],[1]],finallayer=nn.ReLU).cuda()
  # d.net.load_state_dict(torch.load(savedir/'net099.pt'))
