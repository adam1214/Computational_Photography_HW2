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
    k_in = k_in.astype(np.float32)/np.sum(k_in)
    img_in = img_in.astype(np.float32)/255.

    gamma = 2.2
    if to_linear == 'True':
        print('BRL in Linear')
        img_in = np.power(img_in, gamma)

    B = img_in
    I_cur = img_in
    I_next = np.zeros(I_cur.shape)
    K_star = np.flip(k_in)

    r_omega = 0.5 * rk
    sigma_s = (r_omega/3.)**2
    
    omega_window_size = int(2*r_omega) + 1
    x, y = np.mgrid[0:omega_window_size, 0:omega_window_size] - (omega_window_size-1)/2
    gau_kernel = np.exp(-(x**2+y**2)/(2.*sigma_s))
    gau_kernel = np.stack((gau_kernel, gau_kernel, gau_kernel), axis=2)
    
    pdsize = int(omega_window_size/2)
    for update_cnt in range(0, max_iter, 1):
        print('iter:' + str(update_cnt+1) + '/' + str(max_iter))
        padded = np.zeros((I_cur.shape[0]+pdsize*2, I_cur.shape[1]+pdsize*2))
        padded = np.stack((padded,padded,padded), axis=2)
        for ch in range(0,3,1):
            # Pad the Image, Assume Square filter
            padded[:,:,ch] = np.pad(I_cur[:,:,ch], ((pdsize, pdsize), (pdsize, pdsize)), 'symmetric')
        
        E_B_I_t = np.zeros(I_cur.shape)
        for i in range(pdsize, padded.shape[0] - pdsize, 1):
            for j in range(pdsize, padded.shape[1] - pdsize, 1):
                value_kernel = np.exp(-((padded[i,j,:] - padded[i-pdsize:i+pdsize+1, j-pdsize:j+pdsize+1,:])**2) / (2. * sigma_r)) * ((padded[i,j,:] - padded[i-pdsize:i+pdsize+1, j-pdsize:j+pdsize+1,:]) / sigma_r)
                total_kernel = gau_kernel * value_kernel
                E_B_I_t[i-pdsize,j-pdsize,0] = np.sum(total_kernel[:,:,0]) # scalar
                E_B_I_t[i-pdsize,j-pdsize,1] = np.sum(total_kernel[:,:,1]) # scalar
                E_B_I_t[i-pdsize,j-pdsize,2] = np.sum(total_kernel[:,:,2]) # scalar
        
        convolve2d_term1 = np.zeros(I_cur.shape)
        convolve2d_term1[:,:,0] = convolve2d(I_cur[:,:,0], k_in, boundary='symm', mode='same')
        convolve2d_term1[:,:,1] = convolve2d(I_cur[:,:,1], k_in, boundary='symm', mode='same')
        convolve2d_term1[:,:,2] = convolve2d(I_cur[:,:,2], k_in, boundary='symm', mode='same')
        
        convolve2d_term2 = np.zeros(I_cur.shape)
        C = B/convolve2d_term1
        convolve2d_term2[:,:,0] = convolve2d(C[:,:,0], K_star, boundary='symm', mode='same')
        convolve2d_term2[:,:,1] = convolve2d(C[:,:,1], K_star, boundary='symm', mode='same')
        convolve2d_term2[:,:,2] = convolve2d(C[:,:,2], K_star, boundary='symm', mode='same')


        I_next = (I_cur/(1. + lamb_da * 2. * E_B_I_t)) * convolve2d_term2
        I_cur = I_next

    if to_linear == 'True':
        I_cur = np.power(I_cur, (1/gamma))
    
    # Clipping
    (i, j, c) = np.where(I_cur > 1)
    for index in range(len(c)):
        I_cur[i[index], j[index], c[index]] = 1

    (i, j, c) = np.where(I_cur < 0)
    for index in range(len(c)):
        I_cur[i[index], j[index], c[index]] = 0

    I_cur = np.round(I_cur*255.)
    
    return I_cur.astype(np.uint8)

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
    I_in = I_in/255.
    B = img_in/255.
    k_in = k_in.astype(np.float32)/np.sum(k_in)
    
    energy = np.zeros(I_in.shape)

    A = np.zeros(I_in.shape)
    A[:,:,0] = convolve2d(I_in[:,:,0], k_in, boundary='symm', mode='same')
    A[:,:,1] = convolve2d(I_in[:,:,1], k_in, boundary='symm', mode='same')
    A[:,:,2] = convolve2d(I_in[:,:,2], k_in, boundary='symm', mode='same')
    
    C = np.log(A, dtype=np.float32)
    energy = energy + (A - B*C)

    return np.sum(energy[:,:,0])+np.sum(energy[:,:,1])+np.sum(energy[:,:,2])

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
    I_in = I_in/255.
    B = img_in/255.
    k_in = k_in.astype(np.float32)/np.sum(k_in)
    
    RL_energy = np.zeros(I_in.shape)

    A = np.zeros(I_in.shape)
    A[:,:,0] = convolve2d(I_in[:,:,0], k_in, boundary='symm', mode='same')
    A[:,:,1] = convolve2d(I_in[:,:,1], k_in, boundary='symm', mode='same')
    A[:,:,2] = convolve2d(I_in[:,:,2], k_in, boundary='symm', mode='same')
    
    C = np.log(A, dtype=np.float32)
    RL_energy = RL_energy + (A - B*C)
    RL_energy_scalar = np.sum(RL_energy[:,:,0])+np.sum(RL_energy[:,:,1])+np.sum(RL_energy[:,:,2])
    ############################### RL term above###############################
    
    r_omega = 0.5 * rk
    sigma_s = (r_omega/3.)**2
    
    omega_window_size = int(2*r_omega) + 1
    x, y = np.mgrid[0:omega_window_size, 0:omega_window_size] - (omega_window_size-1)/2
    gau_kernel = np.exp(-(x**2+y**2)/(2.*sigma_s))
    gau_kernel = np.stack((gau_kernel, gau_kernel, gau_kernel), axis=2)

    E_B_I = np.zeros(I_in.shape)
    pdsize = int(omega_window_size/2)
    padded = np.zeros((I_in.shape[0]+pdsize*2, I_in.shape[1]+pdsize*2))
    padded = np.stack((padded,padded,padded), axis=2)
    for ch in range(0,3,1):
        # Pad the Image, Assume Square filter
        padded[:,:,ch] = np.pad(I_in[:,:,ch], ((pdsize, pdsize), (pdsize, pdsize)), 'symmetric')

    for i in range(pdsize, padded.shape[0] - pdsize, 1):
        for j in range(pdsize, padded.shape[1] - pdsize, 1):
            value_kernel = 1 - np.exp(-((padded[i,j,:] - padded[i-pdsize:i+pdsize+1, j-pdsize:j+pdsize+1,:])**2) / (2. * sigma_r))
            total_kernel = gau_kernel * value_kernel
            E_B_I[i-pdsize,j-pdsize,0] = np.sum(total_kernel[:,:,0]) # scalar
            E_B_I[i-pdsize,j-pdsize,1] = np.sum(total_kernel[:,:,1]) # scalar
            E_B_I[i-pdsize,j-pdsize,2] = np.sum(total_kernel[:,:,2]) # scalar
    
    total_energy_scalar = 0
    total_energy_scalar = total_energy_scalar + (RL_energy_scalar + lamb_da*np.sum(E_B_I[:,:,0]))
    total_energy_scalar = total_energy_scalar + (RL_energy_scalar + lamb_da*np.sum(E_B_I[:,:,1]))
    total_energy_scalar = total_energy_scalar + (RL_energy_scalar + lamb_da*np.sum(E_B_I[:,:,2]))

    return total_energy_scalar/3


if __name__ == "__main__":
    # for debug
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
    
    # Show image and kernel
    plt.figure()
    plt.imshow(img_in)
    plt.title('Original blurred image')
    plt.show()

    plt.figure()
    plt.imshow(k_in, cmap='gray')
    plt.title('blur kernel')
    plt.show()
    '''
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
    imageio.imwrite('../my_RL_BRL_result/RL_'+ 's' +'_iter%d.png' %(max_iter_RL), RL_result)

    # compare with reference answer and show processing time
    img_ref_RL = Image.open('../ref_ans/curiosity_medium/rl_deblur55.png')

    img_ref_RL = np.asarray(img_ref_RL)
    your_RL = Image.open('../my_RL_BRL_result/RL_'+ 's' +'_iter%d.png' %(max_iter_RL))
    your_RL = np.asarray(your_RL)

    print("psnr = %f" %PSNR_UCHAR3(img_ref_RL, your_RL))

    RL_period = RL_end - RL_start
    print("RL process time = %f sec"%RL_period)
    
    ############# RL energy #############
    I_filename = 'RL_s_iter55.png'
    I_filepath = '../my_RL_BRL_result/' + I_filename
    I = Image.open(I_filepath)
    I_in = np.asarray(I)

    print ("start RL energy...")

    # calculate in linear domain or not
    to_linear = 'False'; #'True' for calculate in linear domain, 'False' for calculate in nonlinear domain

    RL_energy_start = time.time()
    energy =  RL_energy(img_in, k_in, I_in, to_linear)
    RL_energy_end = time.time()

    # compare with reference answer and show processing time
    
    # change dictionary keys 'RL_a', 'RL_b' here
    
    energy_dict = np.load('../ref_ans/energy_dict.npy',allow_pickle='TRUE').item()
    print("Error = %f %%" % ( abs(1-energy/energy_dict['RL_b'])*100) )

    RL_energy_period = RL_energy_end - RL_energy_start
    print("RL process time = %f sec"%RL_energy_period)
    '''
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
    imageio.imwrite('../my_RL_BRL_result/BRL_'+ 's' +'_iter%d_rk%d_si%0.2f_lam%0.3f.png' %(max_iter_BRL, rk, sigma_r*255*255, lamb_da*255), BRL_result)

    # compare with reference answer
    img_ref_BRL = Image.open('../ref_ans/curiosity_medium/brl_deblur_lam0.006.png')
    img_ref_BRL = np.asarray(img_ref_BRL)
    your_BRL = Image.open('../my_RL_BRL_result/BRL_'+ 's' +'_iter%d_rk%d_si%0.2f_lam%0.3f.png' %(max_iter_BRL, rk, sigma_r*255*255, lamb_da*255))
    your_BRL = np.asarray(your_BRL)

    print("psnr = %f" %PSNR_UCHAR3(img_ref_BRL, your_BRL))

    BRL_period = BRL_end - BRL_start
    print("BRL process time = %f sec"%BRL_period)
    '''
    ############# BRL energy #############
    I_filename = 'BRL_s_iter55_rk12_si25.00_lam0.006.png'
    I_filepath = '../my_RL_BRL_result/' + I_filename
    I = Image.open(I_filepath)
    I_in = np.asarray(I)

    print ("start BRL energy...")

    # calculate in linear domain or not
    to_linear = 'False'; #'True' for calculate in linear domain, 'False' for calculate in nonlinear domain

    """
    Adjust parameters here
    """
    rk = 12
    sigma_r = 25.0/255/255
    lamb_da = 0.006/255

    BRL_energy_start = time.time()
    energy =  BRL_energy(img_in, k_in, I_in, lamb_da, sigma_r, rk, to_linear)
    BRL_energy_end = time.time()

    # compare with reference answer and show processing time
    
    #change dictionary key 'BRL_a', 'BRL_b', 'BRL_c', 'BRL_d' here
    
    energy_dict = np.load('../ref_ans/energy_dict.npy',allow_pickle='TRUE').item()
    print("Error = %f %%" % ( abs(1-energy/energy_dict['BRL_d'])*100) )

    BRL_energy_period = BRL_energy_end - BRL_energy_start
    print("BRL process time = %f sec"%BRL_energy_period)
    '''