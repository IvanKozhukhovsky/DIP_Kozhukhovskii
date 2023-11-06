clc
clear all


image = imread("dog.jpg");
imshow(image);
imwrite(image, "dog.jpg");
imwrite(image, "dog.png");