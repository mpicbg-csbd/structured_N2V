# from types import SimpleNamespace
from utils import flatten, recursive_map2

flowerdata = '/lustre/projects/project-broaddus/rawdata/artifacts/flower.tif'
flowerdir  = '/lustre/projects/project-broaddus/denoise_experiments/flower/e01/'

## flower data

## flowerdir  = '/Users/broaddus/Desktop/falconhome/denoise_experiments/flower/e01/'

## n2v2_dirs has axes "model" x "trial"
n2v2_dirs      = [[flowerdir + f'mask{n:02d}_{m}/' for m in range(6)] for n in range(13)]
n2v2_tables    = recursive_map2(lambda d: d + 'table.csv', n2v2_dirs)
n2gt_dirs      = [flowerdir + f"n2gt2/{n}/" for n in range(6)]
n2gt_tables    = recursive_map2(lambda d: d + 'table.csv', n2gt_dirs)
nlm_dir        = flowerdir + "nlm/"
nlm_tables     = nlm_dir + 'table.csv'
bm3d_dir       = flowerdir + "bm3d/"
bm3d_tables    = bm3d_dir + 'table.csv'

all_tables     = {'n2v':n2v2_tables, 'n2gt':n2gt_tables, 'nlm':nlm_tables, 'bm3d':bm3d_tables}

## results from flower analysis

d_figdata = '/lustre/projects/project-broaddus/denoise_experiments/fig_data/'

flower_results = [
                  d_figdata + 'table_mse.pdf',
                  d_figdata + 'table_psnr.pdf',
                  d_figdata + 'table_ssim.pdf',
                  ]
flower_table_tex = d_figdata + 'table.tex'
d_correlation_analysis = d_figdata + 'ca/'

## shutter data

shutterdata  = '/lustre/projects/project-broaddus/rawdata/artifacts/shutterclosed.tif'
shutterdir   = '/lustre/projects/project-broaddus/denoise_experiments/shutter/e01/'
shutter_dir_nlm = shutterdir + 'nlm/'
shutter_dir_bm3d = shutterdir + 'bm3d/'

## c. elegans

times = [0,10,100,189]
celedata = [f'/lustre/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01/t{n:03d}.tif' for n in times]
celedir  = '/lustre/projects/project-broaddus/denoise_experiments/cele/e01/'
cele_n2v2_dirs  = [celedir + f'mask_1_x{n:02d}y{m:02d}/' for n in [0,1] for m in [0,4,8]]
## TODO: this list is incomplete. probably an antipattern.
cele_n2v2_pimgs = [d + 'pimgs/pimg01_000.tif' for d in cele_n2v2_dirs]

cele_nlm_vals = [5,10,50,100,200,500] + [9,11,40,70] + [14,20,30]
cele_nlm_dirs = [celedir + f'nlm/{n:04d}/' for n in cele_nlm_vals]





