import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def conv_many(*cs):
  N = len(cs)-1
  convs = [nn.Conv3d(cs[i],cs[i+1],(3,5,5),padding=(1,2,2)) for i in range(N)]
  relus = [nn.ReLU() for i in range(N)]
  res = [0]*N*2
  res[::2] = convs
  res[1::2] = relus
  return nn.Sequential(*res)

def conv2(c0,c1,c2):
  return nn.Sequential(
    nn.Conv3d(c0,c1,(3,5,5),padding=(1,2,2)),
    # nn.BatchNorm3d(c2),
    nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    nn.Conv3d(c1,c2,(3,5,5),padding=(1,2,2)),
    # nn.BatchNorm3d(c2),
    nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    )

class Unet2(nn.Module):
  """
  The same exact shape we used for models submitted to ISBI. 2,767,777 params.
  """

  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU):
    super(Unet2, self).__init__()

    self.l_ab = conv2(io[0][0] ,1*c, 1*c)
    self.l_cd = conv2(1*c, 2*c, 2*c)
    self.l_ef = conv2(2*c, 4*c, 2*c)
    self.l_gh = conv2(4*c, 2*c, 1*c)
    self.l_ij = conv2(2*c, 1*c, 1*c)
    
    self.l_k  = nn.Sequential(nn.Conv3d(1*c,io[1][0],(1,1,1),padding=0), finallayer())

  def load_old_state(self, state):
    self.load_state_dict(state,strict=False) ## loads most things correctly, but now we have to fix the missing keys
    self.l_ef[0].weight.data[...] = state['l_e.0.weight'].data
    self.l_ef[0].bias.data[...]   = state['l_e.0.bias'].data
    self.l_ef[2].weight.data[...] = state['l_f.0.weight'].data
    self.l_ef[2].bias.data[...]   = state['l_f.0.bias'].data
    self.l_gh[0].weight.data[...] = state['l_g.0.weight'].data
    self.l_gh[0].bias.data[...]   = state['l_g.0.bias'].data
    self.l_gh[2].weight.data[...] = state['l_h.0.weight'].data
    self.l_gh[2].bias.data[...]   = state['l_h.0.bias'].data

  def forward(self, x):

    c1 = self.l_ab(x)
    c2 = nn.MaxPool3d((1,2,2))(c1)
    c2 = self.l_cd(c2)
    c3 = nn.MaxPool3d((1,2,2))(c2)
    c3 = self.l_ef(c3)

    c3 = F.interpolate(c3,scale_factor=(1,2,2))
    c3 = torch.cat([c3,c2],1)
    c3 = self.l_gh(c3)
    c3 = F.interpolate(c3,scale_factor=(1,2,2))
    c3 = torch.cat([c3,c1],1)
    c3 = self.l_ij(c3)
    out1 = self.l_k(c3)

    return out1

class Unet3(nn.Module):
  """
  The same exact shape we used for models submitted to ISBI. 2,767,777 params.
  """

  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU):
    super(Unet3, self).__init__()

    self.l_ab = conv2(io[0][0] ,c, c)
    self.l_cd = conv2(1*c, 2*c, 2*c)
    self.l_ef = conv2(2*c, 4*c, 4*c)
    self.l_gh = conv2(4*c, 8*c, 4*c)
    self.l_ij = conv2(8*c, 4*c, 2*c)
    self.l_kl = conv2(4*c, 2*c, 1*c)
    self.l_mn = conv2(2*c, 1*c, 1*c)
    
    self.l_o  = nn.Sequential(nn.Conv3d(1*c,io[1][0],(1,1,1),padding=0), finallayer())

  def forward(self, x):

    c1 = self.l_ab(x)
    c2 = nn.MaxPool3d((1,2,2))(c1)
    c2 = self.l_cd(c2)
    c3 = nn.MaxPool3d((1,2,2))(c2)
    c3 = self.l_ef(c3)
    c4 = nn.MaxPool3d((1,2,2))(c3)
    c4 = self.l_gh(c4)
    c4 = F.interpolate(c4,scale_factor=(1,2,2))
    c4 = torch.cat([c4,c3],1)
    c4 = self.l_ij(c4)
    c4 = F.interpolate(c4,scale_factor=(1,2,2))
    c4 = torch.cat([c4,c2],1)
    c4 = self.l_kl(c4)
    c4 = F.interpolate(c4,scale_factor=(1,2,2))
    c4 = torch.cat([c4,c1],1)
    c4 = self.l_mn(c4)
    out1 = self.l_o(c4)

    return out1


def conv2_2d(c0,c1,c2):
  return nn.Sequential(
    nn.Conv2d(c0,c1,(5,5),padding=(2,2)),
    # nn.BatchNorm3d(c2),
    nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    nn.Conv2d(c1,c2,(5,5),padding=(2,2)),
    # nn.BatchNorm3d(c2),
    nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    )


class Unet2_2d(nn.Module):
  """
  The same exact shape we used for models submitted to ISBI. 2,767,777 params.
  """

  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU):
    super(Unet2_2d, self).__init__()

    self.l_ab = conv2_2d(io[0][0] ,1*c, 1*c)
    self.l_cd = conv2_2d(1*c, 2*c, 2*c)
    self.l_ef = conv2_2d(2*c, 4*c, 2*c)
    self.l_gh = conv2_2d(4*c, 2*c, 1*c)
    self.l_ij = conv2_2d(2*c, 1*c, 1*c)
    
    self.l_k  = nn.Sequential(nn.Conv2d(1*c,io[1][0],(1,1),padding=0), finallayer())

  def load_old_state(self, state):
    self.load_state_dict(state,strict=False) ## loads most things correctly, but now we have to fix the missing keys
    self.l_ef[0].weight.data[...] = state['l_e.0.weight'].data
    self.l_ef[0].bias.data[...]   = state['l_e.0.bias'].data
    self.l_ef[2].weight.data[...] = state['l_f.0.weight'].data
    self.l_ef[2].bias.data[...]   = state['l_f.0.bias'].data
    self.l_gh[0].weight.data[...] = state['l_g.0.weight'].data
    self.l_gh[0].bias.data[...]   = state['l_g.0.bias'].data
    self.l_gh[2].weight.data[...] = state['l_h.0.weight'].data
    self.l_gh[2].bias.data[...]   = state['l_h.0.bias'].data

  def forward(self, x):

    c1 = self.l_ab(x)
    c2 = nn.MaxPool2d((2,2))(c1)
    c2 = self.l_cd(c2)
    c3 = nn.MaxPool2d((2,2))(c2)
    c3 = self.l_ef(c3)

    c3 = F.interpolate(c3,scale_factor=(2,2))
    c3 = torch.cat([c3,c2],1)
    c3 = self.l_gh(c3)
    c3 = F.interpolate(c3,scale_factor=(2,2))
    c3 = torch.cat([c3,c1],1)
    c3 = self.l_ij(c3)
    out1 = self.l_k(c3)

    return out1


def conv_res(c0,c1,c2):
  return nn.Sequential(
    nn.Conv3d(c0,c1,(3,5,5),padding=(1,2,2)),
    # nn.BatchNorm3d(c2),
    nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    nn.Conv3d(c1,c2,(3,5,5),padding=(1,2,2)),
    # nn.BatchNorm3d(c2),
    # nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    )

class Res1(nn.Module):
  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU):
    super(Unet2, self).__init__()

    self.l_ab = conv2(io[0][0] ,1*c, 1*c)
    self.l_cd = conv2(1*c, 2*c, 2*c)
    self.l_ef = conv2(2*c, 1*c, 1*c)
    # self.l_gh = conv2(4*c, 2*c, 1*c)
    # self.l_ij = conv2(2*c, 1*c, 1*c)
    
    self.l_k  = nn.Sequential(nn.Conv3d(1*c,io[1][0],(1,1,1),padding=0), finallayer())

  def forward(self, x):

    c1 = nn.Relu()(self._lab(x)  + x)
    c2 = nn.Relu()(self._lcd(c1) + c1)
    c3 = nn.Relu()(self._lef(c2) + c2)
    out1 = self.l_k(c3)

    return out1