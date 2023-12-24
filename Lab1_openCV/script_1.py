import numpy as np
import cv2
import matplotlib.pyplot as plt


# Get image
# image = cv2.imread('dog.jpg')
# I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

I = cv2.imread('dog.jpg', cv2.IMREAD_COLOR)

for i in range(1, 10):
    for j in range (1, 10):
        # Создание объекта CLAHE
        clahe = cv2.createCLAHE(clipLimit=i*10, tileGridSize=(j, j))
        
        # Применение CLAHE
        I_clahed_b = clahe.apply(I[:, :, 0])
        I_clahed_g = clahe.apply(I[:, :, 1])
        I_clahed_r = clahe.apply(I[:, :, 2])
        
        I_clahed = cv2.merge([I_clahed_b, I_clahed_g, I_clahed_r])
        
        # Сохранение результата
        cv2.imwrite('I_clahed, ' + 'clipLimit=' + str(i*10) + ' tileGridSize=(' + \
                    str(j) + ',' + str(j) + ')' + '.jpg', I_clahed)



