from Gabor_Noise import *
import time


def get_para(hint, type, default_value):
    print(hint, end='')
    input_value = input()
    if input_value == '':
        input_value = default_value
    else:
        try:
            input_value = type(input_value)
        except TypeError:
            raise(RuntimeError('Wrong input value!'))
    return input_value


def rand(a, b):
    return a + random()*(b-a)


if __name__ == '__main__':
    flag = True
    mode = None
    while flag:
        print('请选择希望查看的Kernal/Noise类型\n(0: Exit, '
              '1: Gaussian Kernel, 2: Harmonic Kernel, 3: Gabor Kernel, 4: Gabor Noise 5: Random Anisotropic Noise '
              '6: Random Isotropic Noise)：')
        choice = input()
        if choice in ['0', '1', '2', '3', '4', '5', '6']:
            mode = int(choice)
        else:
            mode = -1
            print('选择类型有误！请重新输入！')

        # End the program
        if mode == 0:
            exit()

        # Gaussian Kernel
        if mode == 1:
            size = get_para('Image Size=', int, 256)
            k = get_para('K=', float, 1)
            a = get_para('a=', float, 0.06)
            kernel = Gaussian_Kernel(size, k, a)
            kernel.spacial_display()
            kernel.frequency_simulate_display()
            cv2.destroyAllWindows()

        # Harmonic Kernel
        if mode == 2:
            size = get_para('Image Size=', int, 256)
            F_0 = get_para('F_0=', float, 0.15)
            omega_0 = get_para('omega_0=', float, 0.7)
            kernel = Harmonic_Kernel(size, F_0, omega_0)
            kernel.spacial_display()
            kernel.frequency_simulate_display()
            cv2.destroyAllWindows()

        # Gabor Kernel
        if mode == 3:
            size = get_para('Image Size=', int, 256)
            k = get_para('K=', float, 1)
            a = get_para('a=', float, 0.06)
            F_0 = get_para('F_0=', float, 0.15)
            omega_0 = get_para('omega_0=', float, 0.7)
            kernel = Gabor_Kernel(size, k, a, F_0, omega_0)
            kernel.spacial_display()
            kernel.frequency_simulate_display()
            cv2.destroyAllWindows()

        # Gabor Noise
        if mode == 4:
            size = get_para('Image Size=', int, 256)
            kernel_size = get_para('Kernel Size=', int, 40)
            point_num = get_para('Points per Kernel=', int, 16)
            k = get_para('K=', float, 1)
            a = get_para('a=', float, 0.06)
            F_0 = get_para('F_0=', float, 0.15)
            omega_0 = get_para('omega_0=', float, 0.7)
            anisotropic = get_para('If anisotropic? (Y: anisotropic, N: isotropic)  ', str, 'Y')
            if anisotropic == 'Y' or anisotropic == 'y':
                anisotropic = True
            elif anisotropic == 'N' or anisotropic == 'n':
                anisotropic = False
            else:
                raise(RuntimeError('Wrong input value!'))
            kernel = Gabor_Noise(size, kernel_size, point_num, k, a, F_0, omega_0, anisotropic)
            kernel.spacial_display()
            kernel.frequency_simulate_display()
            cv2.destroyAllWindows()

        # Random Gabor Noise (Anisotropic)
        if mode == 5:
            cv2.namedWindow('Gabor_Kernel_spacial', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Gabor_Kernel_frequency', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Sparse_Convolution_Noise', cv2.WINDOW_AUTOSIZE)
            seed(time.time())
            while 1:
                size = 256
                kernel_size = randint(32, 64)
                point_num = randint(8, 16)
                k = rand(0.8, 1.2)
                a = rand(0.03, 0.06)
                F_0 = rand(0.06, 0.14)
                omega_0 = 2 * pi / 32 * randint(0, 32)
                anisotropic = True
                kernel = Gabor_Noise(size, kernel_size, point_num, k, a, F_0, omega_0, anisotropic)
                print('Kernel Size =', kernel_size, ', Points per Kernel =', point_num, ', K =', '%.4f'%k,
                      ', a =', '%.4f'%a, ', F_0 =', '%.4f'%F_0, ', omega_0 =', '%.4f'%omega_0)
                cv2.imshow('Gabor_Kernel_spacial', kernel.img / 2 + 0.5)
                cv2.imshow('Gabor_Kernel_frequency', kernel.frequency_simulate_calculate())
                cv2.imshow('Sparse_Convolution_Noise', kernel.sparse_convolution_noise_calculate())
                # press q to exit
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

        if mode == 6:
            cv2.namedWindow('Gabor_Kernel_spacial', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Gabor_Kernel_frequency', cv2.WINDOW_AUTOSIZE)
            seed(time.time())
            while 1:
                size = 128
                kernel_size = randint(32, 64)
                point_num = randint(4, 16)
                k = rand(0.8, 1.2)
                a = rand(0.04, 0.06)
                F_0 = rand(0.06, 0.14)
                omega_0 = 2 * pi / 32 * randint(0, 32)
                anisotropic = False
                kernel = Gabor_Noise(size, kernel_size, point_num, k, a, F_0, omega_0, anisotropic)
                print('Kernel Size =', kernel_size, ', Points per Kernel =', point_num, ', K =', '%.4f' % k,
                      ', a =', '%.4f' % a, ', F_0 =', '%.4f' % F_0, ', omega_0 =', '%.4f' % omega_0)
                cv2.imshow('Gabor_Kernel_spacial', kernel.img / 2 + 0.5)
                cv2.imshow('Gabor_Kernel_frequency', kernel.frequency_simulate_calculate())
                # press q to exit
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
