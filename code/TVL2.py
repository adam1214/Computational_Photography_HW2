"""
Solve problems by ProxImaL
"""

# Proximal doc: http://www.proximal-lang.org/en/latest/tutorial/index.html
import sys

from proximal.utils.utils import *
from proximal.halide.halide import *
from proximal.lin_ops import *
from proximal.prox_fns import *
from proximal.algorithms import *

#import cvxpy as cvx
import numpy as np

from PIL import Image

import time

import imageio
from load_psnr import PSNR_UCHAR3

############################################################

#%% Load image
path = '../data/blurred_image/edgetaper/blur_edgetaper.png'
# path = '../data/blurred_image/curiosity_medium.png'
img = Image.open(path)  # opens the file using Pillow - it's not an array yet
b = np.asfortranarray(im2nparray(img))

# Kernel
K = Image.open('../data/kernel/kernel_medium.png')  # opens the file using Pillow - it's not an array yet
K = np.asfortranarray(im2nparray(K))
K /= np.sum(K)

K_rgb = np.zeros((K.shape[0], K.shape[1], 3))
K_rgb[:,:,0] = K
K_rgb[:,:,1] = K
K_rgb[:,:,2] = K
K = K_rgb

#%% Now test the solver with some sparse gradient deconvolution
lamb = 0.01
eps_abs_rel = 1e-3
test_solver = 'pc'
max_iters = 1000

tstart = time.time()

#%% rgb channels
x = Variable(b.shape)

# model the problem by proximal
prob = Problem(sum_squares( conv(K, x, dims=2) - b ) + lamb * group_norm1( grad(x, dims = 2), [3] ) + nonneg(x)) # formulate problem

# solve the problem
result = prob.solve(verbose=True,solver=test_solver,x0=b,eps_abs=eps_abs_rel, eps_rel=eps_abs_rel,max_iters=max_iters) # solve problem
x = x.value

# record time
t_int = time.time() - tstart
print( "Elapsed time: %f seconds.\n" %t_int )

# output color image
x = np.clip(x, 0, 1.0)*255.0
x = (x+0.5).astype(np.uint8)
imageio.imwrite('../result/deblur_edgetaper_norm2.png', x)

# compare with reference answer
img_ref_norm2 = Image.open('../ref_ans/curiosity_medium/deblur_edgetaper_norm2.png')
img_ref_norm2 = np.asarray(img_ref_norm2)
your_norm2 = Image.open('../result/deblur_edgetaper_norm2.png')
your_norm2 = np.asarray(your_norm2)

print("psnr = %f" %PSNR_UCHAR3(img_ref_norm2, your_norm2))

