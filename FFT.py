import numpy as np


def spatial_to_frequency(img):
    img_tmp = np.round(img * 256)
    f = np.fft.fft2(img_tmp.astype(int))
    f_shift = np.fft.fftshift(f)
    img_frequency = (np.log(np.abs(f_shift+6000))/16 - 0.65) * 5
    return img_frequency
