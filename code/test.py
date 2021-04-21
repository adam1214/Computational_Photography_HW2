from scipy.signal import convolve2d
import numpy as np
image =  np.array([[ 1 ,  2 ,  3 ,  4 ,  5 ,  6 ,  7 ] , 
                   [ 8 ,  9 ,  10 ,  11 ,  12 ,  13 ,  14 ] , 
                   [ 15 ,  16 ,  17 ,  18 ,  19 ,  20 ,  21 ] , 
                   [ 22 ,  23 ,  24 ,  25,  26 ,  27 ,  28 ] , 
                   [ 29 ,  30 ,  31 ,  32 ,  33 ,  34 ,  35 ] , 
                   [ 36 ,  37 ,  38 ,  39 ,  40 ,  41 ,  42 ] , 
                   [ 43 ,  44 ,  45 ,  46 ,  47 ,  48 ,  49 ] ])

filter_kernel =  np.array([ [ 1 ,  2 ,  3 ] , 
                            [ 4 ,  5 ,  6 ] , 
                            [ 7 ,  8 ,  9 ] ])

a = convolve2d(image, np.flip(filter_kernel), boundary='symm', mode='valid')
print(np.flip(filter_kernel))
print(filter_kernel)