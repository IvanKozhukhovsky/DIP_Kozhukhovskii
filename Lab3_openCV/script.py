import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise


# Original Image
image_name = 'dog_640_400.jpg'
I = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

############################## Noise Generation ###############################
# Image with saul and pepper noise
I_s_and_p = random_noise(I, mode='s&p', amount=0.2, \
                          salt_vs_pepper=0.4).astype(np.float32) 

# Image with additive gaussian noise
I_gaussian = random_noise(I, mode='gaussian', mean=0.1, var=0.05).astype(np.float32)  

# Image with multiplicative gaussian noise
I_speckle = random_noise(I, mode='speckle', mean=0.1, var=0.05).astype(np.float32) 

# Image with quantization (poission) noise
I_poisson = random_noise(I, mode='poisson').astype(np.float32) 

# Save results
f, ax = plt.subplots(5, 1, figsize=(10,12))

ax[0].imshow(I, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off') 

ax[1].imshow(I_s_and_p, cmap='gray')
ax[1].set_title('Saul and Pepper Noise')
ax[1].axis('off') 

ax[2].imshow(I_gaussian, cmap='gray')
ax[2].set_title('Additional Gaussian Noise')
ax[2].axis('off') 

ax[3].imshow(I_speckle, cmap='gray')
ax[3].set_title('Multiplicative Gaussian Noise')
ax[3].axis('off') 

ax[4].imshow(I_poisson, cmap='gray')
ax[4].set_title('Poisson Noise')
ax[4].axis('off') 

f.suptitle('Noise Generation', fontsize=20, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('Noise_Generation.png')
plt.show()
###############################################################################

############################### Gaussian Filter ###############################
# Base parameteres
sigma_X = 1.5
kernel_size = (5, 5)

# Filters for all images
I_gaus_filter_original = cv2.GaussianBlur(I, kernel_size, sigma_X)
I_gaus_filter_s_and_p = cv2.GaussianBlur(I_s_and_p, kernel_size, sigma_X)
I_gaus_filter_gaussian = cv2.GaussianBlur(I_gaussian, kernel_size, sigma_X)
I_gaus_filter_speckle = cv2.GaussianBlur(I_speckle, kernel_size, sigma_X)
I_gaus_filter_poisson = cv2.GaussianBlur(I_poisson, kernel_size, sigma_X)

# Save results
f, ax = plt.subplots(5, 2, figsize=(10,12))

ax[0, 0].imshow(I, cmap='gray')
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off') 

ax[0, 1].imshow(I_gaus_filter_original, cmap='gray')
ax[0, 1].set_title('Original Image Gaus-Filtered')
ax[0, 1].axis('off') 

ax[1, 0].imshow(I_s_and_p, cmap='gray')
ax[1, 0].set_title('Saul and Pepper Noise')
ax[1, 0].axis('off') 

ax[1, 1].imshow(I_gaus_filter_s_and_p, cmap='gray')
ax[1, 1].set_title('Saul and Pepper Noise Gaus-Filtered')
ax[1, 1].axis('off') 

ax[2, 0].imshow(I_gaussian, cmap='gray')
ax[2, 0].set_title('Additional Gaussian Noise')
ax[2, 0].axis('off') 

ax[2, 1].imshow(I_gaus_filter_gaussian, cmap='gray')
ax[2, 1].set_title('Additional Gaussian Noise Gaus-Filtered')
ax[2, 1].axis('off') 

ax[3, 0].imshow(I_speckle, cmap='gray')
ax[3, 0].set_title('Multiplicative Gaussian Noise')
ax[3, 0].axis('off') 

ax[3, 1].imshow(I_gaus_filter_speckle, cmap='gray')
ax[3, 1].set_title('Multiplicative Gaussian Noise Gaus-Filtered')
ax[3, 1].axis('off') 

ax[4, 0].imshow(I_poisson, cmap='gray')
ax[4, 0].set_title('Poisson Noise')
ax[4, 0].axis('off') 

ax[4, 1].imshow(I_gaus_filter_poisson, cmap='gray')
ax[4, 1].set_title('Poisson Noise Gaus-Filtered')
ax[4, 1].axis('off') 

f.suptitle('Gaussian Filter', fontsize=20, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('Gaussian_Filter.png')
plt.show()
###############################################################################

########################### Counterharmonic Filter ############################
def Contrgarmonic_filter(I, Q, kernel_size):
    eps = 1e-11
    I_Q = (I + eps) ** Q
    kernel = np.ones(kernel_size, dtype=np.float32)
    numerator = cv2.filter2D(I_Q * I, -1, kernel)
    denominator = cv2.filter2D(I_Q, -1, kernel) + eps
    
    return np.clip((numerator / denominator).astype(np.float32), 0, 1)


I_salt = random_noise(I, mode='salt')
I_pepper = random_noise(I, mode='pepper')

I_salt_filter = Contrgarmonic_filter(I_salt, Q=-0.5, kernel_size=(5, 5))
I_pepper_filter = Contrgarmonic_filter(I_pepper, Q=0.5, kernel_size=(5, 5))

f, ax = plt.subplots(2,2, figsize=(10,12))

ax[0,0].imshow(I_salt, cmap='gray')
ax[0,0].set_title('Salt Noise')
ax[0,0].axis('off') 

ax[1,0].imshow(I_salt_filter, cmap='gray')
ax[1,0].set_title('Salt Noise Counterharmonic-Filtered')
ax[1,0].axis('off') 

ax[0,1].imshow(I_pepper, cmap='gray')
ax[0,1].set_title('Pepper Noise')
ax[0,1].axis('off') 

ax[1,1].imshow(I_pepper_filter, cmap='gray')
ax[1,1].set_title('Pepper Noise Counterharmonic-Filtered')
ax[1,1].axis('off') 

f.suptitle('Counterharmonic Filter', \
            ha='center', fontsize=20, fontweight='bold')

plt.tight_layout()
plt.savefig('Counterharmonic_Filter.png')
plt.show()
###############################################################################

################################ Median Filter ################################
kernel_size = 5

# Filters for all images
I_median_filter_original = cv2.medianBlur(I, kernel_size)
I_median_filter_s_and_p = cv2.medianBlur(I_s_and_p, kernel_size)
I_median_filter_gaussian = cv2.medianBlur(I_gaussian, kernel_size)
I_median_filter_speckle = cv2.medianBlur(I_speckle, kernel_size)
I_median_filter_poisson = cv2.medianBlur(I_poisson, kernel_size)

# Save results
f, ax = plt.subplots(5, 2, figsize=(10,12))

ax[0, 0].imshow(I, cmap='gray')
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off') 

ax[0, 1].imshow(I_median_filter_original, cmap='gray')
ax[0, 1].set_title('Original Image Median-Filtered')
ax[0, 1].axis('off') 

ax[1, 0].imshow(I_s_and_p, cmap='gray')
ax[1, 0].set_title('Saul and Pepper Noise')
ax[1, 0].axis('off') 

ax[1, 1].imshow(I_median_filter_s_and_p, cmap='gray')
ax[1, 1].set_title('Saul and Pepper Noise Median-Filtered')
ax[1, 1].axis('off') 

ax[2, 0].imshow(I_gaussian, cmap='gray')
ax[2, 0].set_title('Additional Gaussian Noise')
ax[2, 0].axis('off') 

ax[2, 1].imshow(I_median_filter_gaussian, cmap='gray')
ax[2, 1].set_title('Additional Gaussian Noise Median-Filtered')
ax[2, 1].axis('off') 

ax[3, 0].imshow(I_speckle, cmap='gray')
ax[3, 0].set_title('Multiplicative Gaussian Noise')
ax[3, 0].axis('off') 

ax[3, 1].imshow(I_median_filter_speckle, cmap='gray')
ax[3, 1].set_title('Multiplicative Gaussian Noise Median-Filtered')
ax[3, 1].axis('off') 

ax[4, 0].imshow(I_poisson, cmap='gray')
ax[4, 0].set_title('Poisson Noise')
ax[4, 0].axis('off') 

ax[4, 1].imshow(I_median_filter_poisson, cmap='gray')
ax[4, 1].set_title('Poisson Noise Gaus-Filtered')
ax[4, 1].axis('off') 

f.suptitle('Median Filter', fontsize=20, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('Median_Filter.png')
plt.show()
###############################################################################

################## Weighted Median Filter And Ranked Filter ###################
# weighted median filter - expansion of the median filter
# rank filter - generalisation of the median filter
def weight_rank_filter(I, kernel_size, kernel, rank):  
    I_out = np.copy(I)
    I_copy = cv2.copyMakeBorder(I, int((kernel_size[0] - 1) / 2), int(kernel_size[0] / 2),
                                    int((kernel_size[1] - 1) / 2), int(kernel_size[1] / 2), cv2.BORDER_REPLICATE)
    rows, cols = I_copy.shape[0:2]
    kernel_fl = kernel.flatten()
    index_center_kernel = (kernel_size[0] // 2, kernel_size[1] // 2)
    for i in range(index_center_kernel[0], rows - index_center_kernel[0]):
        for j in range(index_center_kernel[1], cols - index_center_kernel[1]):
            window = I_copy[i-index_center_kernel[0]:i+index_center_kernel[0]+1,
                            j-index_center_kernel[1]:j+index_center_kernel[1]+1].flatten()
            pixels = np.sort(np.repeat(window, kernel_fl))
            I_out[i-index_center_kernel[0], j-index_center_kernel[1]] = pixels[rank]
    return I_out

kernel_size = (5,5) #размер маски для медианы
kernel = np.array([
        [1,1,1,1,1],
        [1,2,2,2,1],
        [2,3,4,3,2],
        [1,2,2,2,1],
        [1,1,1,1,1]], 
                    dtype=np.int64)  #матрица весов для медианного фильтра 5x5
rank = 10  # ранг рангового фильтра

I_weighted_and_ranked_median_filter_original = weight_rank_filter(I, kernel_size, kernel, rank)
I_weighted_and_ranked_median_filter_s_and_p = weight_rank_filter(I_s_and_p, kernel_size, kernel, rank)
I_weighted_and_ranked_median_filter_gaussian = weight_rank_filter(I_gaussian, kernel_size, kernel, rank)
I_weighted_and_ranked_median_filter_speckle = weight_rank_filter(I_speckle, kernel_size, kernel, rank)
I_weighted_and_ranked_median_filter_poisson = weight_rank_filter(I_poisson, kernel_size, kernel, rank)

# Save results
f, ax = plt.subplots(5, 2, figsize=(10,12))

ax[0, 0].imshow(I, cmap='gray')
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off') 

ax[0, 1].imshow(I_weighted_and_ranked_median_filter_original, cmap='gray')
ax[0, 1].set_title('Original Image Weighted-Median-Ranked-Filtered')
ax[0, 1].axis('off') 

ax[1, 0].imshow(I_s_and_p, cmap='gray')
ax[1, 0].set_title('Saul and Pepper Noise')
ax[1, 0].axis('off') 

ax[1, 1].imshow(I_weighted_and_ranked_median_filter_s_and_p, cmap='gray')
ax[1, 1].set_title('Saul and Pepper Noise Weighted-Median-Ranked-Filtered')
ax[1, 1].axis('off') 

ax[2, 0].imshow(I_gaussian, cmap='gray')
ax[2, 0].set_title('Additional Gaussian Noise')
ax[2, 0].axis('off') 

ax[2, 1].imshow(I_weighted_and_ranked_median_filter_gaussian, cmap='gray')
ax[2, 1].set_title('Additional Gaussian Noise Weighted-Median-Ranked-Filtered')
ax[2, 1].axis('off') 

ax[3, 0].imshow(I_speckle, cmap='gray')
ax[3, 0].set_title('Multiplicative Gaussian Noise')
ax[3, 0].axis('off') 

ax[3, 1].imshow(I_weighted_and_ranked_median_filter_speckle, cmap='gray')
ax[3, 1].set_title('Multiplicative Gaussian Noise Weighted-Median-Ranked-Filtered')
ax[3, 1].axis('off') 

ax[4, 0].imshow(I_poisson, cmap='gray')
ax[4, 0].set_title('Poisson Noise')
ax[4, 0].axis('off') 

ax[4, 1].imshow(I_weighted_and_ranked_median_filter_poisson, cmap='gray')
ax[4, 1].set_title('Poisson Noise Weighted-Median-Ranked-Filtered')
ax[4, 1].axis('off') 

f.suptitle('Weighted And Ranked Median Filter', fontsize=20, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('Weighted_and_Ranked_Median_Filter.png')
plt.show()
###############################################################################

########################### Adaptive Median Filter ############################
def adaptive_median_filter(I, s_max):  
    I_out = np.copy(I)
    I_copy = cv2.copyMakeBorder(I, 1, 1, 1, 1, cv2.BORDER_REPLICATE) #border padding = 1
    rows, cols = I_copy.shape[0:2]
    for i in range(1, rows - 1):     
        for j in range(1, cols - 1):   #проход по изобр. с паддингом, поэтому обход не по всему изображению
            # шаг 1
            flag = False  #индикатор того, дошел ли s до s_max
            for s in range(3, s_max+1, 2): #цикл по размеру фильтра
                index_center_kernel = (s // 2, s // 2) #кордината центра фильтра
                # выходим из цикла подбора размера фильтра, 
                # если фильтр не может увеличиваться из-за достижения края картинки
                if (i - index_center_kernel[0]) < 0 or (j-index_center_kernel[1]) < 0:    
                    break
                window = I_copy[i-index_center_kernel[0]:i+index_center_kernel[0]+1,
                                j-index_center_kernel[1]:j+index_center_kernel[1]+1]
                z_min = window.min()
                z_max = window.max()
                z_med = np.median(window)
                A_1 = z_med - z_min
                A_2 = z_med - z_max 
                #выходим из цикла подбора размера фильтра, если достигнуто необх. условие 
                if A_1 > 0 and A_2 < 0:  
                    break
                # если все плохо и мы не вышли из цикла, s=s_max - не меняем текущий пиксель
                if s == s_max:
                    I_out[i-index_center_kernel[0], j-index_center_kernel[1]] = I_copy[i, j]
                    flag = True
            # шаг 2
            if not flag:       #если flag=False, то мы еще не дали пикселю значение
                B_1 = I_copy[i, j] - z_min
                B_2 = I_copy[i, j] - z_max
                if B_1 > 0 and B_2 < 0:  
                    I_out[i-index_center_kernel[0], j-index_center_kernel[1]] = I_copy[i, j]
                else:
                    I_out[i-index_center_kernel[0], j-index_center_kernel[1]] = z_med
    return I_out

s_max = 30

I_adaptive_median_filter_original = adaptive_median_filter(I, s_max)
I_adaptive_median_filter_s_and_p = adaptive_median_filter(I_s_and_p, s_max)
I_adaptive_median_filter_gaussian = adaptive_median_filter(I_gaussian, s_max)
I_adaptive_median_filter_speckle = adaptive_median_filter(I_speckle, s_max)
I_adaptive_median_filter_poisson = adaptive_median_filter(I_poisson, s_max)

# Save results
f, ax = plt.subplots(5, 2, figsize=(10,12))

ax[0, 0].imshow(I, cmap='gray')
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off') 

ax[0, 1].imshow(I_adaptive_median_filter_original, cmap='gray')
ax[0, 1].set_title('Original Image Adaptive-Median-Filtered')
ax[0, 1].axis('off') 

ax[1, 0].imshow(I_s_and_p, cmap='gray')
ax[1, 0].set_title('Saul and Pepper Noise')
ax[1, 0].axis('off') 

ax[1, 1].imshow(I_adaptive_median_filter_s_and_p, cmap='gray')
ax[1, 1].set_title('Saul and Pepper Noise Adaptive-Median-Filtered')
ax[1, 1].axis('off') 

ax[2, 0].imshow(I_gaussian, cmap='gray')
ax[2, 0].set_title('Additional Gaussian Noise')
ax[2, 0].axis('off') 

ax[2, 1].imshow(I_adaptive_median_filter_gaussian, cmap='gray')
ax[2, 1].set_title('Additional Gaussian Adaptive-Median-Filtered')
ax[2, 1].axis('off') 

ax[3, 0].imshow(I_speckle, cmap='gray')
ax[3, 0].set_title('Multiplicative Gaussian Noise')
ax[3, 0].axis('off') 

ax[3, 1].imshow(I_adaptive_median_filter_speckle, cmap='gray')
ax[3, 1].set_title('Multiplicative Gaussian Noise Adaptive-Median-Filtered')
ax[3, 1].axis('off') 

ax[4, 0].imshow(I_poisson, cmap='gray')
ax[4, 0].set_title('Poisson Noise')
ax[4, 0].axis('off') 

ax[4, 1].imshow(I_adaptive_median_filter_poisson, cmap='gray')
ax[4, 1].set_title('Poisson Noise Adaptive-Median-Filtered')
ax[4, 1].axis('off') 

f.suptitle('Adaptive Median Filter', fontsize=20, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('Adaptive_Median_Filter.png')
plt.show()
###############################################################################

################################ Wiener Filter ################################
def wiener_filter(I, kernel_size, var_noise):  
    eps = 1e-11
    I_out = np.copy(I)
    I_copy = cv2.copyMakeBorder(I, int((kernel_size[0] - 1) / 2), int(kernel_size[0] / 2),
                                    int((kernel_size[1] - 1) / 2), int(kernel_size[1] / 2), cv2.BORDER_REPLICATE)
    I_copy_power = I_copy**2
    
    rows, cols = I_copy.shape[0:2]
    index_center_kernel = (kernel_size[0] // 2, kernel_size[1] // 2)
    for i in range(index_center_kernel[0], rows - index_center_kernel[0]):
        for j in range(index_center_kernel[1], cols - index_center_kernel[1]):
            window = I_copy[i-index_center_kernel[0]:i+index_center_kernel[0]+1,
                            j-index_center_kernel[1]:j+index_center_kernel[1]+1]
            window_power = I_copy_power[i-index_center_kernel[0]:i+index_center_kernel[0]+1,
                            j-index_center_kernel[1]:j+index_center_kernel[1]+1]
            m = np.sum(window) / (kernel_size[0] * kernel_size[1])
            var = np.sum(window_power - m ** 2) / (kernel_size[0] * kernel_size[1])
            I_out[i-index_center_kernel[0], j-index_center_kernel[1]] =\
                m + ((var - var_noise) / (var+eps)) * (I_copy[i,j] - m)
                
    return np.clip(I_out, 0, 1).astype(np.float32)

kernel_size = (5, 5)
var_noise = 0.05

I_wiener_filter_original = wiener_filter(I, kernel_size, var_noise)
I_wiener_filter_s_and_p = wiener_filter(I_s_and_p, kernel_size, var_noise)
I_wiener_filter_gaussian = wiener_filter(I_gaussian, kernel_size, var_noise)
I_wiener_filter_speckle = wiener_filter(I_speckle, kernel_size, var_noise)
I_wiener_filter_poisson = wiener_filter(I_poisson, kernel_size, var_noise)

# Save results
f, ax = plt.subplots(5, 2, figsize=(10,12))

ax[0, 0].imshow(I, cmap='gray')
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off') 

ax[0, 1].imshow(I_wiener_filter_original, cmap='gray')
ax[0, 1].set_title('Original Image Wiener-Filtered')
ax[0, 1].axis('off') 

ax[1, 0].imshow(I_s_and_p, cmap='gray')
ax[1, 0].set_title('Saul and Pepper Noise')
ax[1, 0].axis('off') 

ax[1, 1].imshow(I_wiener_filter_s_and_p, cmap='gray')
ax[1, 1].set_title('Saul and Pepper Noise Wiener-Filtered')
ax[1, 1].axis('off') 

ax[2, 0].imshow(I_gaussian, cmap='gray')
ax[2, 0].set_title('Additional Gaussian Noise')
ax[2, 0].axis('off') 

ax[2, 1].imshow(I_wiener_filter_gaussian, cmap='gray')
ax[2, 1].set_title('Additional Gaussian Noise Wiener-Filtered')
ax[2, 1].axis('off') 

ax[3, 0].imshow(I_speckle, cmap='gray')
ax[3, 0].set_title('Multiplicative Gaussian Noise')
ax[3, 0].axis('off') 

ax[3, 1].imshow(I_wiener_filter_speckle, cmap='gray')
ax[3, 1].set_title('Multiplicative Gaussian Noise Wiener-Filtered')
ax[3, 1].axis('off') 

ax[4, 0].imshow(I_poisson, cmap='gray')
ax[4, 0].set_title('Poisson Noise')
ax[4, 0].axis('off') 

ax[4, 1].imshow(I_wiener_filter_poisson, cmap='gray')
ax[4, 1].set_title('Poisson Noise Wiener-Filtered')
ax[4, 1].axis('off') 

f.suptitle('Wiener Filter', fontsize=20, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('Wiener_Filter.png')
plt.show()
###############################################################################

############################## High-pass Filter ###############################
Gx = np.array([[1, 0], [0, -1]])
Gy = np.array([[0, 1], [-1, 0]])
I_x = cv2.filter2D(I, -1, Gx)
I_y = cv2.filter2D(I, -1, Gy)
I_roberts = cv2.magnitude(I_x, I_y)

Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
Gx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
I_x = cv2.filter2D(I, -1, Gx)
I_y = cv2.filter2D(I, -1, Gy)
I_previtt = cv2.magnitude(I_x, I_y)

sobelx = cv2.Sobel(I, -1, 1, 0)
sobely = cv2.Sobel(I, -1, 0, 1)
I_sobel = cv2.magnitude(sobelx, sobely)

I_laplacian = np.absolute(cv2.Laplacian(I, -1))

I_int = np.round(255 * I, 0).astype(np.uint8)
I_canny = cv2.Canny(I_int, 90, 200)

# Save results
f, ax = plt.subplots(5,1, figsize=(10,12))

ax[0].imshow(I_roberts, cmap='gray')
ax[0].set_title('Roberts Filter')
ax[0].axis('off') 

ax[1].imshow(I_sobel, cmap='gray')
ax[1].set_title('Sobel Filter')
ax[1].axis('off') 

ax[2].imshow(I_laplacian, cmap='gray')
ax[2].set_title('Laplacian Filter')
ax[2].axis('off') 

ax[3].imshow(I_previtt, cmap='gray')
ax[3].set_title('Previtt Filter')
ax[3].axis('off') 

ax[4].imshow(I_canny, cmap='gray')
ax[4].set_title('Canny Filter')
ax[4].axis('off') 

f.suptitle('High-pass Filters', fontsize=20, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('High-pass_Filter.png')
plt.show()
###############################################################################