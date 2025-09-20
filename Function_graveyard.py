import numpy as np
import matplotlib.pyplot as plt
from Image_creation import *
from Variables_and_constants import *


def plot_kernel(half_pixel_width_gaussian_kernel,sigma,shift_x=0,shift_y=0):
    # Plot the kernel
    plt.figure()
    plt.imshow(gauss_kernel(half_pixel_width_gaussian_kernel*2+1, sigma,shift_x,shift_y))
    plt.title('Beam profile')
    plt.xlabel('px')
    plt.ylabel('px')
    plt.show(block=False)
    plt.pause(0.5)
    return


def calculate_CNR(picture_grid,
                  cross_pixel_left_side,cross_pixel_right_side,
                  cross_pixel_top_side,cross_pixel_bottom_side,
                  cross_pixel_half_width_plus,cross_pixel_half_width_minus):
    # Calculate contrast to noise ratio (CNR)

    cross_sum = 0 # Sum of secondary electron yields of all pixels in the cross
    cross_pixels = 0 # Number of pixels in the cross
    # Sum up the verticle line
    cross_sum += np.sum(picture_grid[cross_pixel_left_side:cross_pixel_right_side,cross_pixel_half_width_minus:cross_pixel_half_width_plus])
    cross_pixels += np.size(picture_grid[cross_pixel_left_side:cross_pixel_right_side,cross_pixel_half_width_minus:cross_pixel_half_width_plus])
    # Sum up the horizontal line
    cross_sum += np.sum(picture_grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,cross_pixel_top_side:cross_pixel_bottom_side]) 
    cross_pixels += np.size(picture_grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,cross_pixel_top_side:cross_pixel_bottom_side])
    # Remove doubly counted region
    cross_sum -= np.sum(picture_grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,cross_pixel_half_width_minus:cross_pixel_half_width_plus])
    cross_pixels -= np.size(picture_grid[cross_pixel_half_width_minus:cross_pixel_half_width_plus,cross_pixel_half_width_minus:cross_pixel_half_width_plus])
    # Calculate average
    cross_average = cross_sum/cross_pixels

    # Calculate background std, number of pixels, and average secondary electron yield
    background_left, background_right, background_top, background_bottom = int(pixels*1/20), int(pixels*8/20), int(pixels*1/20), int(pixels*8/20)
    background_grid = picture_grid[background_left:background_right, background_top:background_bottom]
    background_sum = np.sum(background_grid)
    background_pixels = np.size(background_grid)
    background_std = np.std(background_grid)
    background_average = background_sum/background_pixels

    # Calculate CNR
    CNR = np.abs(cross_average-background_average)/background_std
    return CNR


def find_rotation(img, x, y,cross_length=100e-9,cross_width=14e-9,frame_width=1e-6):
    score = np.zeros(90)
    for i in np.arange(0,90):
        img_no_noise = real_image(cross_length=cross_length,cross_line_width=cross_width,shift_x=x,shift_y=y,rotation=i)[0]
        # img_no_noise = img_no_noise/np.max(img_no_noise)
        score[i] = np.mean((img-img_no_noise)**2)
    best_i = np.argmin(score)
    angles_refined = np.arange(best_i-2.5,best_i+2.5,0.01)
    score_refined = np.zeros(np.shape(angles_refined))
    for j,angle in enumerate(angles_refined):
        img_no_noise = real_image(cross_length=cross_length,cross_line_width=cross_width,shift_x=x,shift_y=y,rotation=angle)[0]
        # img_no_noise = img_no_noise/np.max(img_no_noise)
        score_refined[j] = np.mean((img-img_no_noise)**2)
    best_angle = angles_refined[np.argmin(score_refined)]
    return best_angle

# if __name__ == "__main__":
#     rotation_find_boolean = False
#
#     # Angle of the cross
#     black_white_grid = detect_and_plot_harris_corners(picture_grid_denoised,dot_radius=1,dot_alpha=0.25,k=0.24,percentile=intensity_threshold)
#     if rotation_find_boolean == True:
#         found_rotation = find_rotation(black_white_grid,shift_real_x,shift_real_y,cross_length=cross_length,cross_width=cross_line_width)
#         print(f"Found rotation = {found_rotation}")
#         print(f"Angle error = {found_rotation-rotation:.2f}")