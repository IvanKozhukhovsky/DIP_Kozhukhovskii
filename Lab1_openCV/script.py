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
I = cv2.imread('dog.jpg', cv2.IMREAD_COLOR)
# cv2.imshow('dog', I)
# cv2.waitKey(0)

# Split an image into color layers
# OpenCV stores RGB image as BGR
I_BGR = cv2.split(I)

# Define colors to plot the histograms 
colors = ('b','g','r')

# Compute and plot the image histograms 
for i, color in enumerate(colors): 
    hist = cv2.calcHist(I_BGR , [i] , None , [ histSize ], histRange ) 
    plt.plot(hist, color = color) 
plt.title('Starting Image Histogram') 
plt.show()

######################### Stretching the dynamic range ########################   
# for alpha in range(1, 11):  
#     # Convert to floating point
#     if I.dtype == np.uint8:
#         I_new = I.astype(np.float32 ) / 255
#     else :
#         I_new = I
    
#     # We need to process layers separately
#     I_BGR = cv2.split( I_new )
#     I_new_BGR = []
    
#     for layer in I_BGR:
#         I_min = layer.min()
#         I_max = layer.max()
#         I_new = np.clip(((( layer - I_min ) / ( I_max - I_min )) ** (alpha / 10)),\
#                        0, 1)
#         I_new_BGR.append( I_new )
        
#     # Merge back
#     I_new = cv2.merge( I_new_BGR )
    
#     # Convert back to uint if needed
#     if (I.dtype == np.uint8):
#         I_new = ( 255 * I_new ).clip( 0, 255 ).astype( np.uint8 )
        
#     # Save results
#     cv2.imwrite('Stretching_the_dynamic_range_with_alpha=' + str(alpha/10) +\
#                 '.jpg', I_new)
    
#     # Compute and plot the image histograms 
#     I_BGR = cv2.split( I_new )
#     for i, color in enumerate(colors): 
#         hist = cv2.calcHist(I_BGR , [i] , None , [ histSize ], histRange ) 
#         plt.plot(hist, color = color) 
#     text_title = 'Image Histogram After Stretching, alpha=' + str(alpha/10)
#     plt.title(text_title) 
#     plt.show()
###############################################################################

############################ Uniform transformation ###########################
bgr = []

# Convert to floating point
if I.dtype == np.uint8:
    I_new = I.astype(np.float32 ) / 255
else :
    I_new = I
    

for k, layer in enumerate(I_BGR):
    I_min = layer.min()
    I_max = layer.max()
    
    # Compute and plot the image histograms 
    for i, color in enumerate(colors): 
        hist = cv2.calcHist(I_BGR , [i] , None , [ histSize ], histRange ) 
        bgr.append(hist)
        plt.plot(hist, color = color) 
    plt.title('Starting Image Histogram') 
    plt.show()
        
    H = cv2.merge(bgr)

    #Равномерное преобразование
    numRows = I.shape[0]
    numCols = I.shape[1]
    CH = np.cumsum(H) / (numRows * numCols)

    I_new[:, :, k] = (I_max - I_min) * CH[I[:, :, k]] + I_min

cv2.imwrite('Uniform_Transformation.jpg', I_new)   
###############################################################################

######################### Exponential transformation ##########################

###############################################################################

############################## Apply CLAHE method #############################
# for i in range(1, 11):
#     for j in range (1, 11):
#         # Create CLAHE object
#         clahe = cv2.createCLAHE(clipLimit=i, tileGridSize=(j, j))
        
#         # CLAHE applying
#         I_clahed_b = clahe.apply(I[:, :, 0])
#         I_clahed_g = clahe.apply(I[:, :, 1])
#         I_clahed_r = clahe.apply(I[:, :, 2])
        
#         I_clahed = cv2.merge([I_clahed_b, I_clahed_g, I_clahed_r])
        
#         # Save results
#         cv2.imwrite('I_clahed' + '_clipLimit=' + str(i) + '_tileGridSize=(' \
#                     + str(j) + ',' + str(j) + ')' + '.jpg', I_clahed)
###############################################################################
    
############################# Get the Lookup Table ############################
# for alpha in range(1, 11):  
#     lut = np.arange(256, dtype = np.uint8)
    
#     for layer in I_BGR:
#         I_min = layer.min()
#         I_max = layer.max()
#         lut = (lut - I_min) / (I_max - I_min)
#         lut = np.where(lut > 0, lut, 0)
#         lut = np.clip(255 * np.power(lut, alpha / 10), 0, 255)
#         I_new = cv2.LUT(I, lut)
    
#     # Save results
#     cv2.imwrite('Lookup_table_with_alpha=' + str(alpha/10) + '.jpg', I_new)
###############################################################################

################################ Image Profile ################################
# I = cv2.imread( 'barcode.jpg', cv2.IMREAD_COLOR )
# profile = I[ round(I.shape[0] / 2), :]

# # Save results
# cv2.imwrite('Image_Profile.jpg', profile)
# plt.plot(profile) 
# text_title = 'Image Profile'
# plt.title(text_title) 
# plt.show()    
###############################################################################

############################### Image projection ##############################
# Calculate projection to Oy
if I. ndim == 2:
    ProjI = np. sum(I, 1) / 255
else :
    ProjI = np. sum(I, (1, 2)) / 255 / \
I.shape [2]
# Create graph image
ProjI = np.full((256 , Proj.shape[0] , 3), 255 , dtype = np.uint8 )
DrawGraph(ProjI , Proj , (0, 0, 0), Proj.max())
ProjI = cv2.transpose ( ProjI)
ProjI = cv2.flip (ProjI , 1)
###############################################################################

