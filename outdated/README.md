WELCOME TO THE README OF THIS REPOSITORY
This file serves as the documentation of the code in this repository.

The algorithm works as follows:
1.  \textbf{Input:} pixels\_x, pixels\_y, intensity\_beam, scan\_time\_per\_pixel, microscope parameters (aberrations, source size, etc.)


2.  \textbf{Step 1: Initialize wafer grid} 
3. Create a 2D grid of size $(pixels\_x \times pixels\_y)$ representing the SE escape factor per pixel
4. Optionally add features, e.g., a vertical line of high SE escape factor in the middle: $grid[:, pixels\_x/2] += 1$

5. \textbf{Step 2: Determine optimal beam current and Gaussian width}
6. From microscope parameters, estimate the optimal beam current $I_{\text{beam}}$
7. Using $I_{\text{beam}}$, calculate the beam probe size
8. Convert probe size into Gaussian width by:
$
\sigma = \frac{\text{FWHM}}{2 \sqrt{2 \ln 2}}
$
where FWHM (or FW50) is obtained from the beam spot model
9. Construct the 2D Gaussian kernel of width $\sigma$ and normalize it

10. \textbf{Step 3: Calculate expected secondary electron distribution}
11. Convolve chip grid with Gaussian kernel:
$
\text{expected\_SE} = \text{convolve2d}(grid, kernel)
$
12. Scale by beam current and scan time:
$
\text{expected\_SE} \gets \text{expected\_SE} \cdot I_{\text{beam}} \cdot t_{\text{scan}}
$

13. \textbf{Step 4: Apply Poisson statistics to model shot noise}
14. For each pixel, draw a random integer from Poisson($\lambda = \text{expected\_SE}(x,y)$)
15. Store the result in picture\_grid

16. \textbf{Step 5: Output}
17. Display or save picture\_grid representing the detected secondary electrons per pixel