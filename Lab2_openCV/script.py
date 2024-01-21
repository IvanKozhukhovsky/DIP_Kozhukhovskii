import cv2
import numpy as np
import matplotlib.pyplot as plt


image_name = 'dog_640_400.jpg'
I = cv2.imread(image_name, cv2.IMREAD_COLOR)

rows, cols = I.shape[0:2]

########################## Conformal transformations ##########################
# Shift_transformation
T_shift = np.float32([[1 , 0 , 50], [0 , 1 , 100]])
I_shift = cv2.warpAffine(I, T_shift, (cols , rows))

# Reflect_transformations
T_reflect = np.float32([[1 , 0 , 0], [0, -1, rows-1]])
I_reflect = cv2.warpAffine(I, T_reflect, (cols, rows))

I_reflect_flip = cv2.flip(I, -1)

# Scale_transformations
scale_x = 2 
scale_y = 2

T_scale = np.float32([[scale_x , 0 , 0], [0, scale_y, 0]])
I_scale = cv2.warpAffine(I, T_scale, (int(cols * scale_x), int(rows * scale_y)))

I_scale_resize = cv2.resize(I, None, fx = scale_x, fy = scale_y, interpolation = cv2.INTER_CUBIC)

# Rotation_transformations
phi_rad = 17.0 * np.pi / 180 
T1 = np. float32 (
    [[1 , 0, -(cols - 1) / 2.0],
      [0, 1, -(rows - 1) / 2.0],
      [0, 0, 1]])
T2 = np. float32 (
    [[np.cos(phi_rad), -np.sin(phi_rad), 0],
      [ np.sin(phi_rad), np.cos(phi_rad), 0],
      [0, 0, 1]])
T3 = np. float32 (
    [[1 , 0, (cols - 1) / 2.0],
      [0, 1, (rows - 1) / 2.0],
      [0, 0, 1]])
T_rotate = np.matmul(T3, np.matmul(T2, T1))[0:2, :]
I_rotate = cv2.warpAffine(I, T_rotate, (cols, rows))

phi_degree = 17.0
T_rotate_rotation_matrix = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -phi_degree, 1)
I_rotate_rotation_matrix = cv2.warpAffine(I, T_rotate_rotation_matrix, (cols, rows))

# Save results
fig, ax = plt.subplots(4, 2, figsize=(10, 12))

ax[0,0].imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
ax[0,0].set_title('Original Image')

ax[0,1].imshow(cv2.cvtColor(I_shift, cv2.COLOR_BGR2RGB))
ax[0,1].set_title('Shift transform')

ax[1,0].imshow(cv2.cvtColor(I_reflect, cv2.COLOR_BGR2RGB))
ax[1,0].set_title('Reflect to the Ox')

ax[1,1].imshow(cv2.cvtColor(I_reflect_flip, cv2.COLOR_BGR2RGB))
ax[1,1].set_title('Reflect to the Ox and Oy with flip')

ax[2,0].imshow(cv2.cvtColor(I_scale, cv2.COLOR_BGR2RGB))
ax[2,0].set_title(f'Scaling at {scale_x}')

ax[2,1].imshow(cv2.cvtColor(I_scale, cv2.COLOR_BGR2RGB))
ax[2,1].set_title(f'Scaling at {scale_x} with resize')

ax[3,0].imshow(cv2.cvtColor(I_rotate, cv2.COLOR_BGR2RGB))
ax[3,0].set_title(f'Rotate at {phi_rad:.3f} radian')

ax[3,1].imshow(cv2.cvtColor(I_rotate_rotation_matrix, cv2.COLOR_BGR2RGB))
ax[3,1].set_title(f'Rotate at {phi_degree} degree')

plt.tight_layout()
plt.savefig('Conformal_Transformations.png')
plt.show()
###############################################################################

############################### Affine mapping ################################
# Bevelling transformations
s = 0.4
T_bevelling = np.float32([[1, s, 0], [0, 1, 0]])
I_bevelling = cv2.warpAffine(I, T_bevelling, (cols, rows))

# Piecewise linear transformations
stretch = 7
T_piecewise_linear = np.float32([[stretch, 0, 0], [0, 1, 0]])
I_piecewise_linear = I.copy()
I_piecewise_linear[:, int(cols/2):, :] = cv2.warpAffine(
    I_piecewise_linear[:, int(cols/2):, :], 
    T_piecewise_linear, (cols-int(cols/2), rows)
)

# Save results
f, ax = plt.subplots(1, 3, figsize=(10,12))
ax[0].imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[1].imshow(cv2.cvtColor(I_bevelling, cv2.COLOR_BGR2RGB))
ax[1].set_title('Bevelling Transformations')
ax[2].imshow(cv2.cvtColor(I_piecewise_linear, cv2.COLOR_BGR2RGB))
ax[2].set_title('Piecewise Linear Transformations')

plt.tight_layout()
plt.savefig('Affine_Mapping_Transformations.png')
plt.show()
# ###############################################################################

# ########################## Nonlinear transformations ##########################
# Projection mapping
T_projection = np.float32([
    [1.2 , 0.1 , 0.00075], 
    [0.3 , 1.1 , 0.0005], 
    [0, 0 , 1]
])
I_projection = cv2.warpPerspective(I, T_projection, (cols, rows))

# Polynomial mapping
T = np.array([[0 , 0], [1, 0], [0, 1],
                [0.00001 , 0], [0.002 , 0], 
                [0.001 , 0]])
I_polynomial = np.zeros(I.shape, I.dtype)
x, y = np.meshgrid(np.arange(cols), np.arange(rows))

# Calculate all new X and Y coordinates
xnew = np.round(T[0, 0] + x * T[1, 0] +
                y * T[2, 0] + x * x * T[3, 0] +
                x * y * T[4, 0] +
                y * y * T[5, 0]).astype(np.float32)
ynew = np.round(T[0, 1] + x * T[1, 1] +
                y * T[2, 1] + x * x * T[3, 1] +
                x * y * T[4, 1] +
                y * y * T[5, 1]).astype(np.float32)

# Calculate mask of valid indexes
mask = np.logical_and(np.logical_and(xnew >= 0, xnew < cols),
                      np.logical_and(ynew >= 0, ynew < rows))

# Apply reindexing
if I.ndim == 2:
    I_polynomial[ynew[mask].astype(int), xnew[mask].astype(int)] =\
        I[y[mask], x[mask]]
else :
    I_polynomial[ynew[mask].astype(int), xnew[mask].astype(int), :] = \
    I[y[mask], x[mask], :]

# Sinusoidal distortion
u, v = np.meshgrid(np.arange(cols), np.arange(rows))
u = u + 20 * np.sin(2 * np.pi * v / 90)
I_sinusoid = cv2.remap(I, u.astype(np.float32), \
                        v.astype(np.float32), cv2.INTER_LINEAR)

# Save results
f, ax = plt.subplots(2, 2, figsize=(10,12))

ax[0,0].imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
ax[0,0].set_title('Original Image')

ax[0,1].imshow(cv2.cvtColor(I_projection, cv2.COLOR_BGR2RGB))
ax[0,1].set_title('Projective Transformation')

ax[1,0].imshow(cv2.cvtColor(I_polynomial, cv2.COLOR_BGR2RGB))
ax[1,0].set_title('Polynomial Transformation')

ax[1,1].imshow(cv2.cvtColor(I_sinusoid, cv2.COLOR_BGR2RGB))
ax[1,1].set_title('Sinusoidal Transformation')

plt.tight_layout()
plt.savefig('Nonlinear_Transformations.png')
plt.show()
###############################################################################

image_name = 'grid_512_512.jpg'
I = cv2.imread(image_name, cv2.IMREAD_COLOR)

rows, cols = I.shape[0:2]

######################### Distortion of Barrel Effect #########################
# Direct transformations
# Create mesh grid for X, Y
xi, yi = np.meshgrid(np.arange(cols), np.arange(rows))

# Shift and normalize grid
xmid = cols / 2.0
ymid = rows / 2.0

xi = xi - xmid
yi = yi - ymid

# Convert to polar and do transformation
r, theta = cv2.cartToPolar(xi / xmid, yi / ymid)
F3 = 0.1
r = r + F3 * r**3

# Undo conversion, normalization and shift
u, v = cv2.polarToCart(r, theta)
u = u * xmid + xmid 
v = v * ymid + ymid 

# Do remapping
I_barrel = cv2.remap(I, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)

# Inverse transformations
# Convert to polar coordinates
r, theta = cv2.cartToPolar(xi / xmid, yi / ymid)

# Apply pincushion distortion
delta_R = r[70, 70] - r[55, 55]
F3 = delta_R / (r[55, 55]**3)
r = r + F3 * r**3 

# Convert back to Cartesian coordinates
u, v = cv2.polarToCart(r, theta)

# Undo normalization and shift
u = u * xmid + xmid 
v = v * ymid + ymid 

# Do remapping for pincushion distortion
I_restored_barrel = cv2.remap(I_barrel, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)

# Save results
f, ax = plt.subplots(1, 3, figsize=(10,12))

ax[0].imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')

ax[1].imshow(cv2.cvtColor(I_barrel, cv2.COLOR_BGR2RGB))
ax[1].set_title('Barrel Distortion')

ax[2].imshow(cv2.cvtColor(I_restored_barrel, cv2.COLOR_BGR2RGB))
ax[2].set_title('Barrel Distortion Correction')

plt.tight_layout()
plt.savefig('Barrel_Distortion_Correction.png')
plt.show()
###############################################################################

######################### Distortion of Pillow Effect #########################
# Direct transformations
# Create mesh grid for X, Y
xi, yi = np.meshgrid(np.arange(cols), np.arange(rows))

# Shift and normalize grid
xmid = cols / 2.0
ymid = rows / 2.0

xi = xi - xmid
yi = yi - ymid

# Convert to polar and do transformation
r, theta = cv2.cartToPolar(xi / xmid, yi / ymid)

# Apply pincushion distortion
F3 = -0.1
r = r + F3 * r**3

# Undo conversion, normalization and shift
u, v = cv2.polarToCart(r, theta)
u = u * xmid + xmid 
v = v * ymid + ymid 

# Do remapping
I_pillow = cv2.remap(I, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)

# Inverse transformations
# Convert to polar coordinates
r, theta = cv2.cartToPolar(xi / xmid, yi / ymid)

# Apply pincushion distortion
delta_R = r[50, 50] - r[85, 85]
F3 = delta_R / (r[85, 85]**3)
r = r + F3 * r**3

# Undo conversion, normalization and shift
u, v = cv2.polarToCart(r, theta)
u = u * xmid + xmid 
v = v * ymid + ymid 

# Do remapping
I_restored_pillow = cv2.remap(I_pillow, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)

# Save results
f, ax = plt.subplots(1, 3, figsize=(10,12))

ax[0].imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')

ax[1].imshow(cv2.cvtColor(I_pillow, cv2.COLOR_BGR2RGB))
ax[1].set_title('Pillow Distortion')

ax[2].imshow(cv2.cvtColor(I_restored_pillow, cv2.COLOR_BGR2RGB))
ax[2].set_title('Pillow Distortion Correction')

plt.tight_layout()
plt.savefig('Pillow_Distortion_Correction.png')
plt.show()
###############################################################################

image_name = 'dog_640_400.jpg' 
I = cv2.imread(image_name, cv2.IMREAD_COLOR)

# Find the centre of the image
rows, cols = I.shape[0:2]
midpoint = rows // 2

# Split the image into top and bottom parts
I_top_half = I[:midpoint, :]
I_bottom_half = I[midpoint:, :]

############################### Stitching images ############################## 
templ_size = 10
templ = I_top_half[-templ_size:,:,:]
res = cv2.matchTemplate(I_bottom_half, templ, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

I_stitch = np.zeros((I_top_half.shape[0] + I_bottom_half.shape[0] - max_loc[1] - 
                        templ_size, I_top_half.shape[1], I_top_half.shape[2]), dtype = np.uint8)
I_stitch[0:I_top_half.shape[0], :, :] = I_top_half
I_stitch[I_top_half.shape[0]:, :, :] = I_bottom_half[max_loc[1] + templ_size:, :, :]

f, ax = plt.subplots(1, 3, figsize=(10,12))
ax[0].imshow(cv2.cvtColor(I_top_half, cv2.COLOR_BGR2RGB))
ax[0].set_title('Top Part')

ax[1].imshow(cv2.cvtColor(I_bottom_half, cv2.COLOR_BGR2RGB))
ax[1].set_title('Bottom Part')

ax[2].imshow(cv2.cvtColor(I_stitch, cv2.COLOR_BGR2RGB))
ax[2].set_title('Stitching Images')

plt.tight_layout()
plt.savefig('Stitching_Images.png')
plt.show()
###############################################################################

############################# AutoStitching images ############################
I_part_1 = cv2.imread('1.jpg', cv2.IMREAD_COLOR)
I_part_2 = cv2.imread('2.jpg', cv2.IMREAD_COLOR)
I_part_3 = cv2.imread('3.jpg', cv2.IMREAD_COLOR)
I_part_4 = cv2.imread('4.jpg', cv2.IMREAD_COLOR)

stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
status, I_stitch = stitcher.stitch([I_part_1, I_part_2, I_part_3, I_part_4])

if status == cv2.STITCHER_OK:
    f, ax = plt.subplots(1, 4, figsize=(14, 12))
    
    ax[0].imshow(cv2.cvtColor(I_part_1, cv2.COLOR_BGR2RGB))
    ax[0].set_title('First Part')
    
    ax[1].imshow(cv2.cvtColor(I_part_2, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Second Part')
    
    ax[2].imshow(cv2.cvtColor(I_part_3, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Third Part')
    
    ax[3].imshow(cv2.cvtColor(I_part_4, cv2.COLOR_BGR2RGB))
    ax[3].set_title('Fourth Part')
    
    plt.tight_layout()
    plt.savefig('Four_Images_for_Stitching.png')
    plt.show()
    
    f, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    ax.imshow(cv2.cvtColor(I_stitch, cv2.COLOR_BGR2RGB))
    ax.set_title('AutoStitching Images')
    
    plt.tight_layout()
    plt.savefig('AutoStitching_Images.png')
    plt.show()
else:
    print("Stitching NOT successful")
###############################################################################