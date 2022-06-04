from configparser import NoOptionError
import math
from cv2 import resize
from numpy import arange, exp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import Bases

from numpy.random import default_rng

def MSE(img1, img2, n):

    return np.sum((img1-img2)**2) / n



def show_images(image_number,mse, images):

    n: int = len(images)
    f = plt.figure()
    images_names = ['', 'Image_1', 'Image_2', 'Image_12' ]
    image_name = images_names[image_number]
    titles = ['original image', 'noisy ' + image_name , 'reconstructed image' + ", MSE is: " +"{:g}".format(mse)]
    for i in range(n):
        f.add_subplot(1, n, i + 1).title.set_text(titles[i])
        plt.imshow(images[i])

    plt.show(block=True)


def  getNoiseMatrix(frequency, mean = 1/10,  Standard_deviation  = 1/20, m = 256 , n =256):

        matrix = []

        for row in range(m):

            amplitude = default_rng().normal(loc = mean, scale = Standard_deviation)
            phase = default_rng().uniform(low = 0 , high = 2*math.pi)

            row = range(n)
            row = [amplitude * math.cos(2*math.pi*frequency* j + phase) for j in row ]

            matrix.append(row)

        return np.array(matrix)


def myDFT(img):

    dft_matrix = Bases.Fourier(256, 0,1).matrix
    return (dft_matrix @ img.transpose()).transpose()

def myIDFT(img):

    idft_matrix = np.matrix.conjugate(Bases.Fourier(256, 0,1).matrix)
    return (idft_matrix @ img.transpose()).transpose()


class NoisyImage():

    def __init__(self):

        img = Image.open('horse.jpg').convert('L')
        img = img.resize((256,256))
        img = np.array(img) / 255
        self.original_image = img

        noise_matrix_1 = getNoiseMatrix(frequency = 1/8)
        noise_matrix_2 = getNoiseMatrix(frequency = 1/32)
        noise_matrix_3 = np.add(noise_matrix_1, noise_matrix_2) / 2

        self.noise1 = noise_matrix_1

        # show_images([Image.fromarray(np.zeros_like(noise_matrix_1)), Image.fromarray(255*noise_matrix_1),Image.fromarray(255*noise_matrix_2),Image.fromarray(255*noise_matrix_3) ])

        noisy_image_1 =  img + noise_matrix_1
        noisy_image_2 =  img + noise_matrix_2
        noisy_image_3 =  img + noise_matrix_3

        # show_images([Image.fromarray(255*img), Image.fromarray(255*noisy_image_1),Image.fromarray(255*noisy_image_2),Image.fromarray(255*noisy_image_3) ])

        self.img1 = noisy_image_1
        self.img2 = noisy_image_2
        self.img12 = noisy_image_3

def do( transform = 'auto'):
    noisy_images = NoisyImage()
    image_number = 3
    img1_frequencies = [32,224]
    img2_frequencies = [8,248]
    img3_frequencies = np.append(img1_frequencies,img2_frequencies)

    if transform == 'my':
        
        original_fft = myDFT(noisy_images.original_image)
        noisy_fft_rep = myDFT(noisy_images.img2)

    else:

        original_fft = np.fft.fft2(noisy_images.original_image, norm = "ortho")
        noisy_fft_rep = np.fft.fft2(noisy_images.img12, norm = "ortho")


    original_wavelet = np.sum(abs(original_fft), axis=0)
    noisy_wavelet = np.sum(abs(noisy_fft_rep), axis=0)

    for row in noisy_fft_rep:
        for col in img3_frequencies:
            # row[col] = (row[col+1])
            # row[col] = (row[col+1] +row[col-1])/ 2
            row[col] = 0

    reconstructed_wavelet = np.sum(abs(noisy_fft_rep), axis=0)

    if transform == 'my':
        recon_img = myIDFT(noisy_fft_rep).real
    else :
        recon_img = np.fft.ifft2(noisy_fft_rep, norm = "ortho").real

    #analysis
    plt.plot(range(256) , noisy_wavelet, label = 'noisy', color = 'r')
    plt.plot(range(256) , original_wavelet, label = 'original', color ='b')
    plt.plot(range(256) , reconstructed_wavelet, label = 'reconstructed', color ='g')
    plt.legend(loc='upper left')
    plt.title(' Frequency Domain - Image 1_2 ')

    mse = MSE(noisy_images.original_image, recon_img, 256**2)
    show_images( image_number,mse, [Image.fromarray(255*noisy_images.original_image),Image.fromarray(255*noisy_images.img12),Image.fromarray(255*recon_img)])

do()



