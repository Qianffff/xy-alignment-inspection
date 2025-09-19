import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
from skimage.restoration import denoise_nl_means, estimate_sigma
import cv2
from skimage.measure import block_reduce
from Kernel_and_convolution import *
from Cross_detection import *
from Image_creation import *
from Function_graveyard import *
from Variables_and_constants import *

# Transformations from Poisson noise to Gaussian noise and back
def anscombe_transform(image):
    return 2.0 * np.sqrt(image + 3.0 / 8.0)

def inverse_anscombe_transform(transformed):
    return (transformed / 2.0) ** 2 - 3.0 / 8.0

def denoise_image(image):
    transformed = anscombe_transform(image)
    sigma_est = estimate_sigma(transformed, channel_axis=None)

    denoised_transformed = denoise_nl_means(
        transformed,
        h=1.15 * sigma_est,
        fast_mode=True,
        patch_size=5,
        patch_distance=6,
        channel_axis=None)
    denoised_image = inverse_anscombe_transform(denoised_transformed)
    return denoised_image