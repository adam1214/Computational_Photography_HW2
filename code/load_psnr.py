"""
provided psnr function
"""

import numpy as np

def PSNR_UCHAR3(input_1, input_2, peak=255):
    [row,col,channel] = input_1.shape
    if input_1.shape != input_2.shape:
        print ("Warning!! Two image have different shape!!")
        return 0
    mse = ((input_1 - input_2)**2).sum() / float(row * col * channel)
    
    return 20*np.log10(peak) - 10*np.log10(mse)
