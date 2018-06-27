from Gabor_Kernel import *
from Sparse_Convolution_Noise import *


class Gabor_Noise:
    def __init__(self, size=512, grid_size=50, point_num=64, K=1, a=0.06, F_0=0.1, omega_0=0.7, anisotropic=True):
        self.size = size
        self.grid_size = grid_size
        self.a = a
        self.K = K
        self.F_0 = F_0
        self.omega_0 = omega_0
        self.point_num = point_num
        self.anisotropic = anisotropic
        self.sparse_noise = []
        gabor_kernel = Gabor_Kernel(img_size=size, K=K, a=a, F_0=F_0, omega_0=omega_0)
        if anisotropic:
            scn = Sparse_Convolution_Noise(width=size, height=size, grid_size=grid_size, point_num=point_num)
            self.sparse_noise = scn.noise
            self.img = cv2.filter2D(scn.img, -1, gabor_kernel.img)
        else:
            self.img = np.zeros([size, size])
            # tbd
            for i in range(self.size):
                for j in range(self.size):
                    #print("process:{0}%".format(round((i*self.size+j) * 100 / (self.size**2))), end="\r")
                    print("process:{0}%".format(round((i * self.size + j) * 100 / (self.size ** 2))))
                    self.img[i][j] = self.noise(i, j)

    def gabor(self, K, a, F_0, omega_0, x, y):
        return K * exp(-pi * (a ** 2) * (x * x + y * y)) * cos(2*pi*F_0*(x*cos(omega_0) + y*sin(omega_0)))

    def noise(self, x, y):
        x /= self.grid_size
        y /= self.grid_size
        sum = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                sum += self.cell(int(x)+i, int(y)+j, x - int(x) - i, y - int(y) - j)
        return sum

    def cell(self, i, j, x, y):
        seed(i*self.size+j)
        sum = 0
        for i in range(self.point_num):
            x_i = random()
            y_i = random()
            w_i = random()*2 - 1
            if i == 0 and j == 0:
                self.sparse_noise.append({'x': int(x_i * self.size), 'y': int(y_i * self.size), 'w': w_i})
            omega_0_i = random()*2*pi
            sum += w_i * self.gabor(self.K, self.a, self.F_0, omega_0_i, (x-x_i)*self.grid_size, (y-y_i)*self.grid_size)
        return sum

    def spacial_display(self, wait=True):
        cv2.namedWindow('Gabor_Noise_spacial', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Gabor_Noise_spacial', self.img/2+0.5)
        if wait:
            cv2.waitKey(0)

    def frequency_display(self, wait=True):
        cv2.namedWindow('Gabor_Noise_frequency', cv2.WINDOW_AUTOSIZE)
        img_frequency = spatial_to_frequency(self.img)
        cv2.imshow('Gabor_Noise_frequency', img_frequency)
        if wait:
            cv2.waitKey(0)

    def frequency_simulate_calculate(self):
        img_frequency = np.zeros([self.size, self.size])
        dx = self.F_0 * sin(self.omega_0)
        dy = self.F_0 * cos(self.omega_0)
        half_size = self.size / 2
        c1 = [0.5 + dx, 0.5 + dy]
        c2 = [0.5 - dx, 0.5 - dy]
        lambda_energy_anisotropic = 0.2
        lambda_energy_isotropic = 0.26
        d2 = (dx ** 2 + dy ** 2)
        if self.anisotropic:
            for i in range(self.size):
                for j in range(self.size):
                    x1 = i / self.size - c1[0]
                    y1 = j / self.size - c1[1]
                    x2 = i / self.size - c2[0]
                    y2 = j / self.size - c2[1]
                    value = self.K / (2 * self.a ** 2) * (exp(-pi / (self.a ** 2) * (x1 * x1 + y1 * y1)) +
                                                          exp(-pi / (self.a ** 2) * (x2 * x2 + y2 * y2)))
                    img_frequency[i][j] = lambda_energy_anisotropic * log(value + 0.00001)
        else:
            for i in range(self.size):
                for j in range(self.size):
                    d1 = ((i - half_size) ** 2 + (j - half_size) ** 2) / (self.size ** 2)
                    value = self.K ** 2 / (4 * sqrt(2) * pi * self.F_0 * (self.a ** 3)) * \
                            exp(-2 * pi / (self.a ** 2) * abs(d1 - d2))
                    img_frequency[i][j] = lambda_energy_isotropic * log(value + 0.00001)
        return img_frequency

    def frequency_simulate_display(self):
        img_frequency = self.frequency_simulate_calculate()
        cv2.namedWindow('Gabor_Kernel_simulate_frequency', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Gabor_Kernel_simulate_frequency', img_frequency)
        cv2.waitKey(0)

    def sparse_convolution_noise_calculate(self):
        scn = np.ones([self.size, self.size]) / 2
        for n in self.sparse_noise:
            scn[n['x'], n['y']] = n['w'] * 0.5 + 0.5
        return scn

    def sparse_convolution_noise_display(self):
        scn = self.sparse_convolution_noise_calculate()
        cv2.namedWindow('Sparse_Convolution_Noise', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Sparse_Convolution_Noise', scn)
        cv2.waitKey(0)


if __name__ == '__main__':
    #gabor = Gabor_Noise(size=128, grid_size=50, point_num=16, anisotropic=False)
    gabor = Gabor_Noise(size=512, point_num=24, K=1, a=0.06, F_0=0.2, omega_0=0.7, anisotropic=True)
    gabor.spacial_display()
    gabor.frequency_display()
    gabor.sparse_convolution_noise_display()
