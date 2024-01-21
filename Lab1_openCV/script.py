import numpy as np
import cv2
import matplotlib.pyplot as plt


# I is an RGB - image
# Number of histogram bins
histSize = 256

# Histogram range
# The upper boundary is exclusive
histRange = (0, 256)

# Get image
image_name = 'chess_512_512.jpg' # dog_640_400 lion_512_512 chess_512_512
I = cv2.imread(image_name, cv2.IMREAD_COLOR) 

# Split an image into color layers
# OpenCV stores RGB image as BGR
I_BGR = cv2.split(I)

# Define colors to plot the histograms 
colors = ('b','g','r')

# Define hist collection
CH = []

# Compute and plot the image histograms 
for i, color in enumerate(colors): 
    hist = cv2.calcHist(I_BGR, [i], None, [ histSize ], histRange) 
    CH_color = np.cumsum(hist) / (I.shape[0] * I.shape[1])
    CH_color = CH_color[:, np.newaxis].astype(np.float32)
    CH.append(CH_color)
    plt.plot(hist, color = color) 
plt.title('Starting Image Histogram') 
plt.show()

# Converting a list of cumulative histograms to an array
CH = np.concatenate(CH, axis=1)
###############################################################################

######################### Stretching the dynamic range ########################   
for alpha in range(1, 11):  
    # Convert to floating point
    if I.dtype == np.uint8:
        I_new = I.astype(np.float32 ) / 255
    else :
        I_new = I
    
    # We need to process layers separately
    I_BGR = cv2.split( I_new )
    I_new_BGR = []
    
    for layer in I_BGR:
        I_min = layer.min()
        I_max = layer.max()
        I_new = np.clip(((( layer - I_min ) / ( I_max - I_min )) ** (alpha / 10)),\
                        0, 1)
        I_new_BGR.append( I_new )
        
    # Merge back
    I_new = cv2.merge( I_new_BGR )
    
    # Convert back to uint if needed
    if (I.dtype == np.uint8):
        I_new = ( 255 * I_new ).clip( 0, 255 ).astype( np.uint8 )
        
    # Save results
    cv2.imwrite('Stretching_the_dynamic_range_with_alpha=' + str(alpha/10) +\
                '.jpg', I_new)
    
    # Compute and plot the image histograms 
    I_BGR = cv2.split( I_new )
    for i, color in enumerate(colors): 
        hist = cv2.calcHist(I_BGR , [i] , None , [ histSize ], histRange ) 
        plt.plot(hist, color = color) 
    text_title = 'Image Histogram After Stretching, alpha=' + str(alpha/10)
    plt.title(text_title) 
    plt.show()
###############################################################################

############################ Uniform transformation ###########################
if I.dtype == np.uint8 :
    I = I.astype(np.float32) / 255
    I_BGR = cv2.split(I)

# Convert to floating point
if I.dtype == np.uint8:
    I_new = I.astype(np.float32 ) / 255
else :
    I_new = I
    

for k, layer in enumerate(I_BGR):
    I_min = layer.min()
    I_max = layer.max()

    I_new[:, :, k] = np.clip((I_max - I_min) * CH[(np.round(layer*255, 0).astype(np.uint8)), k] + I_min, 0, 1)
    # I_new[:, :, k] = np.interp(layer, np.arange(256), (I_max - I_min) * \
    #                             CH[:, k] + I_min)

cv2.imwrite('Uniform_Transformation.jpg', (I_new * 255).astype(np.uint8))
###############################################################################

######################### Exponential transformation ##########################
if I.dtype == np.uint8 :
    I = I.astype(np.float32) / 255
    I_BGR = cv2.split(I)
    
# Convert to floating point
if I.dtype == np.uint8:
    I_new = I.astype(np.float32 ) / 255
else :
    I_new = I
    
for alpha in range(1, 11): 
    eps = 1e-7
    
    for k, layer in enumerate(I_BGR):
        I_min = layer.min()
        I_new[:, :, k] = np.clip(I_min - (1 / alpha) * np.log(1 - CH[(np.round(layer*255, 0).astype(np.uint8)), k] + eps), 0, 1)

    cv2.imwrite('Exponential_Transformation_with_alpha=' + str(alpha) +\
                '.jpg', (I_new * 255).astype(np.uint8))  
###############################################################################

######################### Rayleigh law transformation #########################
if I.dtype == np.uint8 :
    I = I.astype(np.float32) / 255
    I_BGR = cv2.split(I)
    
# Convert to floating point
if I.dtype == np.uint8:
    I_new = np.empty_like(I, dtype=np.float32) / 255
else :
    I_new = np.empty_like(I, dtype=np.float32)
    
for alpha in range(1, 11): 
    eps = 1e-7
    
    for k, layer in enumerate(I_BGR):
        I_min = layer.min()
        I_new[:, :, k] = np.clip(I_min + (2 * (alpha/10)**2 * np.log(1 / (1 - CH[(np.round(layer*255, 0).astype(np.uint8)),k]+eps)))**0.5, 0, 1)
    
    cv2.imwrite('Rayleigh_Law_Transformation_with_alpha=' + str(alpha/10) +\
                '.jpg', (I_new * 255).astype(np.uint8))  
###############################################################################

##################### Conversion by the law of degree 2/3 #####################
if I.dtype == np.uint8 :
    I = I.astype(np.float32) / 255
    I_BGR = cv2.split(I)
    
# Convert to floating point
if I.dtype == np.uint8:
    I_new = np.empty_like(I, dtype=np.float32) / 255
else :
    I_new = np.empty_like(I, dtype=np.float32)
    
for k, layer in enumerate(I_BGR):
    I_new[:, :, k] = np.clip(CH[(np.round(layer*255, 0).astype(np.uint8)),k] ** (2/3), 0, 1)

cv2.imwrite('Conversion_By_The_Law_Of_Degree_2div3.jpg', (I_new * 255).astype(np.uint8))  
###############################################################################

# ########################## Hyperbolic transformation ########################
if I.dtype == np.uint8 :
    I = I.astype(np.float32) / 255
    I_BGR = cv2.split(I)
    
# Convert to floating point
if I.dtype == np.uint8:
    I_new = np.empty_like(I, dtype=np.float32) / 255
else :
    I_new = np.empty_like(I, dtype=np.float32)
    
for alpha in range(1, 11): 
    for k, layer in enumerate(I_BGR):
        I_min = layer.min()
        I_new[:, :, k] = np.clip((alpha/100)**CH[(np.round(layer*255, 0).astype(np.uint8)),k], 0, 1)
   
    cv2.imwrite('Hyperbolic_Transformation_with_alpha=' + str(alpha/100) +\
                '.jpg', (I_new * 255).astype(np.uint8))   
###############################################################################

############################## Apply CLAHE method #############################
I = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    
for i in range(1, 11):
    for j in range (1, 11):
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=i, tileGridSize=(j, j))
        I_clahed = clahe.apply(I)
        
        # Save results
        cv2.imwrite('I_clahed' + '_clipLimit=' + str(i) + '_tileGridSize=(' \
                    + str(j) + ',' + str(j) + ')' + '.jpg', I_clahed)
###############################################################################
    
############################# Get the Lookup Table ############################
for alpha in range(1, 21):  
    lut = np.arange(256, dtype = np.uint8)
    
    for layer in I_BGR:
        I_min = layer.min()
        I_max = layer.max()
        lut = (lut - I_min) / (I_max - I_min)
        lut = np.where(lut > 0, lut, 0)
        lut = np.clip(255 * np.power(lut, alpha / 10), 0, 255)
        I_new = cv2.LUT(I, lut)
    
    # Save results
    cv2.imwrite('Lookup_table_with_alpha=' + str(alpha/10) + '.jpg', I_new)
###############################################################################

################################ Image Profile ################################
I = cv2.imread('barcode.jpg', cv2.IMREAD_COLOR)
profile = I[ round(I.shape[0] / 2), :]

# Save results
plt.plot(profile) 
text_title = 'Image Profile'
plt.title(text_title) 
plt.savefig('Image_Profile.png')
plt.show()    
###############################################################################

############################### Image projection ##############################
I = cv2.imread('passport.jpg', cv2.IMREAD_COLOR)

if I.ndim == 2:
    ProjI_y = np.sum(I, 1) / 255 / I.shape[1] 
    ProjI_x = np.sum(I, 0) / 255 / I.shape[0] 
else:
    ProjI_y = np.sum(I, (1 , 2)) / 255 / I.shape[1] 
    ProjI_x = np.sum(I, (0 , 2)) / 255 / I.shape[0] 
    
# Save results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
ax1.plot(ProjI_y)
ax1.set_title('Projection along Y-axis')
ax1.set_xlabel('Pixel Position')
ax1.set_ylabel('Normalized Intensity')
ax2.plot(ProjI_x)
ax2.set_title('Projection along X-axis')
ax2.set_xlabel('Pixel Position')
ax2.set_ylabel('Normalized Intensity')
ax3.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
ax3.set_title('Image')
plt.tight_layout()
plt.savefig('projections_and_image.png')
plt.show()
###############################################################################

