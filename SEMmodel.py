from Kernel_and_convolution import *
from Cross_detection import *
from Image_creation import *
from Denoise_functions import *
from Function_graveyard import *
from Variables_and_constants import *

# Generate wafer image
grid, cross_center = real_image()  

# Use Gaussian distribution to meassure image
picture_grid = measure_image(grid,200e-9,SNR)

# Calculate the position of the cross from the image
cross_center_measured_px = cross_position(picture_grid,intensity_threshold)
cross_center_measured = cross_center_measured_px * pixel_width # Convert from pixels to meters
# Difference between calculated cross center position and actual position (in m)
error = np.linalg.norm([cross_center[0] - cross_center_measured[0], cross_center[1] - cross_center_measured[1]])
# Listing some values of variables used in the simulation
time_to_make_picture = pixels**2*scan_time_per_pixel
# print(f"Time to make image = {time_to_make_picture:.5f} seconds")
# print(f"Scan time per pixel = {scan_time_per_pixel*1e6:.5f} µs")
# print(f"Error = {error*1e9:.3f} nm")

# ===================== Plot =====================
if show_plots == True:
    import matplotlib.pyplot as plt

    # 1. Secondary electron yield grid
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(grid)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=24)  # <-- Make colorbar ticks readable
    plt.tight_layout()
    plt.savefig("Secondary electron yield grid.svg")
    plt.show(block=False)
    plt.pause(0.5)

    # 2. Simulated SEM image
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(picture_grid)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=24)
    plt.tight_layout()
    plt.savefig("Simulated SEM image.svg")
    plt.show(block=False)
    plt.pause(0.5)

    # 3. Black and white image (already handled inside your function)
    black_white_grid = detect_and_plot_harris_corners(
        denoise_image(picture_grid),
        dot_radius=1,
        dot_alpha=0.25,
        k=0.24,
        percentile=intensity_threshold
    )

    # 4. Denoised image with marker
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(denoise_image(picture_grid))
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=24)
    ax.scatter(cross_center_measured_px[1], cross_center_measured_px[0], 
            c='red', marker='+', s=200, label='Center')
    ax.legend()
    plt.tight_layout()
    plt.savefig("Simulated SEM image denoised.svg")
    plt.show(block=False)
    plt.pause(0.5)

    # 5. Denoised image with marker
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(denoise_image(picture_grid))
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=24)
    plt.tight_layout()
    plt.savefig("Simulated SEM image denoised without marker.svg")
    plt.show(block=True)
    plt.pause(0.5)
