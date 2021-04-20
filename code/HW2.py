import time

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from load_psnr import PSNR_UCHAR3
from deconv_RL_BRL import *

import imageio

# ignore warning
import warnings
warnings.filterwarnings("ignore")

#%% Load image and kernel
'''
Change the input file name/path here
'''
input_filename = 'curiosity_small.png'
kernel_filename = 'kernel_small.png'

input_filepath = '../data/blurred_image/'+input_filename
kernel_filepath = '../data/kernel/'+kernel_filename
img = Image.open(input_filepath)  # opens the file using Pillow - it's not an array yet
img_in = np.asarray(img)
k = Image.open(kernel_filepath)  # opens the file using Pillow - it's not an array yet
k_in = np.asarray(k)

#%% Show image and kernel
plt.figure()
plt.imshow(img_in)
plt.title('Original blurred image')
plt.show()

plt.figure()
plt.imshow(k_in, cmap='gray')
plt.title('blur kernel')
plt.show()


############# RL deconvolution #############
print ("start RL deconvolution...")

# RL parameters
"""
Adjust parameters here
"""
# for RL
max_iter_RL = 25

# deblur in linear domain or not
to_linear = 'False'; #'True' for deblur in linear domain, 'False' for deblur in nonlinear domain

# RL deconvolution
RL_start = time.time()
RL_result = RL(img_in, k_in, max_iter_RL, to_linear)
RL_end = time.time()

# show RL result
plt.figure()
plt.imshow(RL_result)
plt.title('RL-deblurred image')
plt.show()

# store image
imageio.imwrite('../result/RL_'+ 's' +'_iter%d.png' %(max_iter_RL), RL_result)

# compare with reference answer and show processing time
img_ref_RL = Image.open('../ref_ans/curiosity_small/rl_deblur20.png')

img_ref_RL = np.asarray(img_ref_RL)
your_RL = Image.open('../result/RL_'+ 's' +'_iter%d.png' %(max_iter_RL))
your_RL = np.asarray(your_RL)

print("psnr = %f" %PSNR_UCHAR3(img_ref_RL, your_RL))

RL_period = RL_end - RL_start
print("RL process time = %f sec"%RL_period)


############# RL energy #############
I_filename = 'RL_s_iter25.png'
I_filepath = '../result/' + I_filename
I = Image.open(I_filepath)
I_in = np.asarray(I)

print ("start RL energy...")

# calculate in linear domain or not
to_linear = 'False'; #'True' for calculate in linear domain, 'False' for calculate in nonlinear domain

RL_energy_start = time.time()
energy =  RL_energy(img_in, k_in, I_in, to_linear)
RL_energy_end = time.time()

# compare with reference answer and show processing time
'''
change dictionary keys 'RL_a', 'RL_b' here
'''
energy_dict = np.load('../ref_ans/energy_dict.npy',allow_pickle='TRUE').item()
print("Error = %f %%" % ( abs(1-energy/energy_dict['RL_a'])*100) )

RL_energy_period = RL_energy_end - RL_energy_start
print("RL process time = %f sec"%RL_energy_period)


############# BRL deconvolution #############
print ("start BRL deconvolution...")

# RL&BRL parameters
"""
Adjust parameters here
"""
# for BRL
max_iter_BRL = 25
rk = 6
sigma_r = 50.0/255/255
lamb_da = 0.03/255

# deblur in linear domain or not
to_linear = 'False'; #'True' for deblur in linear domain, 'False' for deblur in nonlinear domain

# BRL deconvolution
BRL_start = time.time()
BRL_result = BRL(img_in, k_in, max_iter_BRL, lamb_da, sigma_r, rk, to_linear)
BRL_end = time.time()

# show BRL result
plt.figure()
plt.imshow(BRL_result)
plt.title('BRL-deblurred image')
plt.show()

# store image
imageio.imwrite('../result/BRL_'+ 's' +'_iter%d_rk%d_si%0.2f_lam%0.3f.png' %(max_iter_BRL, rk, sigma_r*255*255, lamb_da*255), BRL_result)

# compare with reference answer
img_ref_BRL = Image.open('../ref_ans/curiosity_small/brl_deblur_lam0.03.png')
img_ref_BRL = np.asarray(img_ref_BRL)
your_BRL = Image.open('../result/BRL_'+ 's' +'_iter%d_rk%d_si%0.2f_lam%0.3f.png' %(max_iter_BRL, rk, sigma_r*255*255, lamb_da*255))
your_BRL = np.asarray(your_BRL)

print("psnr = %f" %PSNR_UCHAR3(img_ref_BRL, your_BRL))

BRL_period = BRL_end - BRL_start
print("BRL process time = %f sec"%BRL_period)


############# BRL energy #############
I_filename = 'BRL_s_iter25_rk6_si50.00_lam0.030.png'
I_filepath = '../result/' + I_filename
I = Image.open(I_filepath)
I_in = np.asarray(I)

print ("start BRL energy...")

# calculate in linear domain or not
to_linear = 'False'; #'True' for calculate in linear domain, 'False' for calculate in nonlinear domain

"""
Adjust parameters here
"""
rk = 6
sigma_r = 50.0/255/255
lamb_da = 0.03/255

BRL_energy_start = time.time()
energy =  BRL_energy(img_in, k_in, I_in, lamb_da, sigma_r, rk, to_linear)
BRL_energy_end = time.time()

# compare with reference answer and show processing time
'''
change dictionary key 'BRL_a', 'BRL_b', 'BRL_c', 'BRL_d' here
'''
energy_dict = np.load('../ref_ans/energy_dict.npy',allow_pickle='TRUE').item()
print("Error = %f %%" % ( abs(1-energy/energy_dict['BRL_a'])*100) )

BRL_energy_period = BRL_energy_end - BRL_energy_start
print("BRL process time = %f sec"%BRL_energy_period)


