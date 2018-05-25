import numpy as np


def spatial_to_frequency(img):
    img_tmp = np.round(img * 256)
    print(img_tmp.shape)
    f = np.fft.fft2(img_tmp.astype(int))
    f_shift = np.fft.fftshift(f)
    print(np.abs(f_shift))
    img_frequency = 10 * np.log(np.abs(f_shift)+0.0001) / 256
    return img_frequency
