import scipy
import time

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from load_psnr import PSNR_UCHAR3
from deconv_RL_BRL import *
from scipy.signal import convolve2d
import cv2
import imageio

# ignore warning
import warnings
warnings.filterwarnings("ignore")

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
    if omega_window_size % 2 == 0:
        omega_window_size += 1

    x, y = np.mgrid[0:omega_window_size, 0:omega_window_size] - (omega_window_size-1)/2
    gau_kernel = -(x**2+y**2)/(2.*sigma_s)
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
                value_kernel_1 = -((padded[i,j,:] - padded[i-pdsize:i+pdsize+1, j-pdsize:j+pdsize+1,:])**2) / (2. * sigma_r)
                value_kernel_2 = (padded[i,j,:] - padded[i-pdsize:i+pdsize+1, j-pdsize:j+pdsize+1,:]) / sigma_r
                total_kernel = np.exp(gau_kernel+value_kernel_1) * value_kernel_2
                '''
                E_B_I_t[i-pdsize,j-pdsize,0] = np.sum(total_kernel[:,:,0]) # scalar
                E_B_I_t[i-pdsize,j-pdsize,1] = np.sum(total_kernel[:,:,1]) # scalar
                E_B_I_t[i-pdsize,j-pdsize,2] = np.sum(total_kernel[:,:,2]) # scalar
                '''
                E_B_I_t[i-pdsize,j-pdsize,:] = np.sum(total_kernel, axis=(0,1))

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

if __name__ == "__main__":
    # resize the blurred image & kernel image first
    image1 = cv2.imread('../my_deblur/blurred_image/straight_blurred.JPG')
    image2 = cv2.imread('../my_deblur/kernel/straight_kernel.JPG', cv2.IMREAD_GRAYSCALE)

    image1_resize = cv2.resize(image1, (684, 1210), interpolation=cv2.INTER_AREA)
    image2_resize = cv2.resize(image2, (13, 13), interpolation=cv2.INTER_AREA)

    cv2.imwrite('../my_deblur/my_image_straight.png', image1_resize)
    cv2.imwrite('../my_deblur/my_kernel_straight.png', image2_resize)
    
    '''
    Change the input file name/path here
    '''
    input_filename = 'my_image_straight.png'
    kernel_filename = 'my_kernel_straight.png'

    input_filepath = '../my_deblur/'+input_filename
    kernel_filepath = '../my_deblur/'+kernel_filename
    img = Image.open(input_filepath)  # opens the file using Pillow - it's not an array yet
    img_in = np.asarray(img)
    k = Image.open(kernel_filepath)  # opens the file using Pillow - it's not an array yet
    pix = np.array(k)

    for i in range(0, pix.shape[0], 1):
        for j in range(0, pix.shape[1], 1):
            pix_val = pix[i][j]
            pix_val = round(((pix_val - 0.0)/(255.0 - 0.0)) * 255.)
            if pix_val <= 77:
                pix_val = 0
            pix[i][j] = pix_val

    Image.fromarray(pix).save('../my_deblur/my_kernel_straight_edit.png')

    k = Image.open('../my_deblur/my_kernel_straight_edit.png')  # opens the file using Pillow - it's not an array yet
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

    ############# BRL deconvolution #############
    print ("start BRL deconvolution...")

    # RL&BRL parameters
    """
    Adjust parameters here
    """
    # for BRL
    max_iter_BRL = 25
    rk = 6
    sigma_r = 5.0/255/255
    lamb_da = 0.001/255

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
    imageio.imwrite('../my_deblur/my_deblur_straight.png', BRL_result)

    BRL_period = BRL_end - BRL_start
    print("BRL process time = %f sec"%BRL_period)