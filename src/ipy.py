import tifffile
from tifffile import imread,imsave
from segtools.numpy_utils import normalize3
from pathlib import Path
from subprocess import run
import numpy as np
from segtools.ns2dir import load,save
import itertools
from glob import glob
from skimage.measure import regionprops


def pts(mantrack): return np.array([r.centroid for r in regionprops(mantrack)],np.int)

def run02():
  p = "/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/01_GT/TRA/"
  for name in sorted(glob(p + "*.tif")):
    print(name)
    img = load(name)
    imgpts = pts(img)
    print(imgpts.shape)
    save(imgpts,name.replace("rawdata/","denoise_experiments/data/pts/"))

def nlmupdate():
  name = "/projects/project-broaddus/denoise_experiments/cele/e01/nlm2_3d_s22/"
  for i in range(120,150):
    fi = name + f"t{i:03d}.tif"
    print(fi)
    # os.remove(fi)

def copy_best_denoised_2martin():
  shutil.copy("../../denoise_experiments/cele/e01/nlm2_3d_s22/t005.tif", "/fileserver/CSBDeep/coleman/structn2v/t189_z22_nlm.tif")
  # shutil.copy("../../denoise_experiments/cele/e01/nlm2_3d_s22/t005.tif", "/fileserver/CSBDeep/coleman/structn2v/t189_z22_nlm.tif")
  img = load("../../denoise_experiments/data/bm4d/output/t189_f32_bm4d_sigma_0.90.mat")
  save(img.u[22].astype(np.float32),"/fileserver/CSBDeep/coleman/structn2v/t189_z22_bm4d.tif")


