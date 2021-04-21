"""
RL and BRL functions
"""
import scipy
import time

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from load_psnr import PSNR_UCHAR3
from deconv_RL_BRL import *
from scipy.signal import convolve2d

import imageio

# ignore warning
import warnings
warnings.filterwarnings("ignore")

def RL(img_in, k_in, max_iter, to_linear):
    """ RL deconvolution
            Args:
                img_in (int np.array, value:[0,255]): Blurred image
                k_in (int np.array, value:[0,255]): blur kernel
                max_iter (int): total iteration count
                to_linear (bool): The flag indicates whether the input image should deblur in linear domain color space or not
            Returns:
                im_out (int np.array, value:[0,255]): RL-deblurred image
            Todo:
                RL deconvolution
    """
    # Use floating-point operations and normalize the intensity I to [0, 1] before deblurring
    B = img_in/255.
    
    # Normalize the blur kernel
    k_in = k_in.astype(np.float32)/np.sum(k_in)
    
    gamma = 2.2
    if to_linear == 'True':
        B = np.power(B, gamma)

    I_cur = B
    K_star = np.flip(k_in)

    for update_cnt in range(0, max_iter, 1):
        A = np.zeros(I_cur.shape)

        A[:,:,0] = convolve2d(I_cur[:,:,0], k_in, boundary='symm', mode='same')
        A[:,:,1] = convolve2d(I_cur[:,:,1], k_in, boundary='symm', mode='same')
        A[:,:,2] = convolve2d(I_cur[:,:,2], k_in, boundary='symm', mode='same')
        C = B/A
        
        D = np.zeros(I_cur.shape)
        D[:,:,0] = convolve2d(C[:,:,0], K_star, boundary='symm', mode='same')
        D[:,:,1] = convolve2d(C[:,:,1], K_star, boundary='symm', mode='same')
        D[:,:,2] = convolve2d(C[:,:,2], K_star, boundary='symm', mode='same')

        I_next = I_cur * D # 元素點乘
        I_cur = I_next
    
    if to_linear == 'True':
        I_cur = np.power(I_cur, 1/gamma)
    
    for ch in range(0, I_cur.shape[2], 1):
        for i in range(0, I_cur.shape[0], 1):
            for j in range(0, I_cur.shape[1], 1):
                if I_cur[i][j][ch] > 1:
                    I_cur[i][j][ch] = 1
                elif I_cur[i][j][ch] < 0:
                    I_cur[i][j][ch] = 0
                
                I_cur[i][j][ch] = round(I_cur[i][j][ch] * 255.)

    return I_cur.astype(np.uint8)

def BRL(img_in, k_in, max_iter, lamb_da, sigma_r, rk, to_linear):
    """ BRL deconvolution
            Args:
                img_in (int np.array, value:[0,255]): Blurred image
                k_in (int np.array, value:[0,255]): blur kernel
                max_iter (int): total iteration count
                lamb_da (float): BRL parameter
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                to_linear (bool): The flag indicates whether the input image should deblur in linear domain color space or not
            Returns:
                im_out (int np.array, value:[0,255]): BRL-deblurred image
            Todo:
                BRL deconvolution
    """

    #return BRL_result


def RL_energy(img_in, k_in, I_in, to_linear):
    """ RL energy
            Args:
                img_in (int np.array, value:[0,255]): Blurred image
                k_in (int np.array, value:[0,255]): blur kernel
                I_in (int np.array, value:[0,255]): Your deblured image
                to_linear (bool): The flag indicates whether the input image should deblur in linear domain color space or not
            Returns:
                energy (float): RL-deblurred energy
            Todo:
                RL energy
    """

    
    #return energy

def BRL_energy(img_in, k_in, I_in, lamb_da, sigma_r, rk, to_linear):
    """ BRL energy
            Args:
                img_in (int np.array, value:[0,255]): Blurred image
                k_in (int np.array, value:[0,255]): blur kernel
                I_in (int np.array, value:[0,255]): Your deblured image
                lamb_da (float): BRL parameter
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                to_linear (bool): The flag indicates whether the input image should deblur in linear domain color space or not
            Returns:
                energy (float): BRL-deblurred energy
            Todo:
                BRL energy
    """

    
    #return energy

if __name__ == "__main__":
    # for debug
    '''
    Change the input file name/path here
    '''
    input_filename = 'curiosity_medium.png'
    kernel_filename = 'kernel_medium.png'

    input_filepath = '../data/blurred_image/'+input_filename
    kernel_filepath = '../data/kernel/'+kernel_filename
    img = Image.open(input_filepath)  # opens the file using Pillow - it's not an array yet
    img_in = np.asarray(img)
    k = Image.open(kernel_filepath)  # opens the file using Pillow - it's not an array yet
    k_in = np.asarray(k)
    
    # Show image and kernel
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
    max_iter_RL = 55

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
    img_ref_RL = Image.open('../ref_ans/curiosity_medium/rl_deblur55.png')

    img_ref_RL = np.asarray(img_ref_RL)
    your_RL = Image.open('../result/RL_'+ 's' +'_iter%d.png' %(max_iter_RL))
    your_RL = np.asarray(your_RL)

    print("psnr = %f" %PSNR_UCHAR3(img_ref_RL, your_RL))

    RL_period = RL_end - RL_start
    print("RL process time = %f sec"%RL_period)
