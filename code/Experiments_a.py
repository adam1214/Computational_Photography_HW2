import numpy as np
from PIL import Image
import imageio
import matplotlib.pyplot as plt

def DFT(img, LINE_pixel_loc):
    img = np.mean(img, axis=2) / 255 # 每個pixel位置的三個維度相加取平均，然後做noralization使其落在0-1之間 shape:(684,1210)
    img_1d = img[LINE_pixel_loc,:] # 將該2D平面圖之scanline取出(橫線)，變成1D. shape:(1210,)
    img_1d_fft = np.fft.fft(img_1d) # Compute the one-dimensional discrete Fourier Transform. shape:(1210,)
    img_1d_fft_log_mag = np.log10(np.abs(img_1d_fft))
    return img_1d_fft_log_mag

if __name__ == "__main__":
    # Load images
    true_img = np.asarray(Image.open('../data/curiosity.png'))

    rl_s = np.asarray(Image.open('../my_RL_BRL_result/RL_s_iter25.png'))
    rl_m = np.asarray(Image.open('../my_RL_BRL_result/RL_m_iter55.png'))

    brl_s_03 = np.asarray(Image.open('../my_RL_BRL_result/BRL_s_iter25_rk6_si50.00_lam0.030.png'))
    brl_s_06 = np.asarray(Image.open('../my_RL_BRL_result/BRL_s_iter25_rk6_si50.00_lam0.060.png'))
    brl_m_001 = np.asarray(Image.open('../my_RL_BRL_result/BRL_m_iter55_rk12_si25.00_lam0.001.png'))
    brl_m_006 = np.asarray(Image.open('../my_RL_BRL_result/BRL_m_iter55_rk12_si25.00_lam0.006.png'))

    tv_n1 = np.asarray(Image.open('../my_RL_BRL_result/deblur_edgetaper_norm1.png'))
    tv_n2 = np.asarray(Image.open('../my_RL_BRL_result/deblur_edgetaper_norm2.png'))
    tv_poisson = np.asarray(Image.open('../my_RL_BRL_result/deblur_edgetaper_poisson.png'))

    # create scanline
    img_scanline = true_img.copy()
    LINE_pixel_loc = 350
    img_scanline[LINE_pixel_loc, :, 0] = 255
    img_scanline[LINE_pixel_loc, :, 1] = 255
    img_scanline[LINE_pixel_loc, :, 2] = 255
    imageio.imwrite('../Experiment_b/img_scanline.png', img_scanline)

    # estimate 1-D DFT by this scanline (every image)
    true_img_DFT = DFT(true_img, LINE_pixel_loc)

    rl_s_DFT = DFT(rl_s, LINE_pixel_loc)
    rl_m_DFT = DFT(rl_m, LINE_pixel_loc)

    brl_s_03_DFT = DFT(brl_s_03, LINE_pixel_loc)
    brl_s_06_DFT = DFT(brl_s_06, LINE_pixel_loc)
    brl_m_001_DFT = DFT(brl_m_001, LINE_pixel_loc)
    brl_m_006_DFT = DFT(brl_m_006, LINE_pixel_loc)

    tv_n1_DFT = DFT(tv_n1, LINE_pixel_loc)
    tv_n2_DFT = DFT(tv_n2, LINE_pixel_loc)
    tv_poisson_DFT = DFT(tv_poisson, LINE_pixel_loc)

    n_xpoint = int(len(true_img_DFT)/2) # 605
    x = np.linspace(0, np.pi, num=n_xpoint) # 等差數列 差值為pi size=605

    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    plt.plot(x, true_img_DFT[0:n_xpoint], label='true image')

    plt.plot(x, rl_s_DFT[0:n_xpoint], label='RL_small kernel')
    plt.plot(x, rl_m_DFT[0:n_xpoint], label='RL_medium kernel')

    plt.plot(x, brl_s_03_DFT[0:n_xpoint], label='BRL_small kernel_lam=0.03')
    plt.plot(x, brl_s_06_DFT[0:n_xpoint], label='BRL_small kernel_lam=0.06')
    plt.plot(x, brl_m_001_DFT[0:n_xpoint], label='BRL_medium kernel_lam=0.001')
    plt.plot(x, brl_m_006_DFT[0:n_xpoint], label='BRL_medium kernel_lam=0.006')

    plt.plot(x, tv_n1_DFT[0:n_xpoint], label='TV_n1')
    plt.plot(x, tv_n2_DFT[0:n_xpoint], label='TV_n2')
    plt.plot(x, tv_poisson_DFT[0:n_xpoint], label='TV_poisson')

    plt.title('1-D DFT Scanline')
    plt.xlabel('Frequency')
    plt.ylabel('Log Magnitude')

    plt.legend(loc='upper right')

    step = np.pi/5
    plt.xticks([0, step, 2*step, 3*step, 4*step, np.pi], [r'0', r'$\pi/5$', r'$2\pi/5$', r'$3\pi/5$', r'$4\pi/5$', r'$\pi$'])
    
    plt.savefig('../Experiment_b/1_D_DFT_Scanline.png')