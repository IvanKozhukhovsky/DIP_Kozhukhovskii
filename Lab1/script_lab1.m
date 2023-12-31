clc
clear all


image = imread("dog.jpg");
% imshow(image);
% pause
imwrite(image, "dog.jpg");
imwrite(image, "dog.png");

% информация об изображениях
info_jpg = imfinfo("dog.jpg");
info_png = imfinfo("dog.png");

W_jpg = info_jpg.Width;
W_png = info_png.Width;

H_jpg = info_jpg.Height;
H_png = info_png.Height;

Bit_jpg = info_jpg.BitDepth;
Bit_png = info_png.BitDepth;

FileSize_jpg = info_jpg.FileSize;
FileSize_png = info_png.FileSize;

% степень сжатия изображений - больше сжато jpg
K_s_jpg = (W_jpg * H_jpg * Bit_jpg) / (8 * FileSize_jpg);
K_s_png = (W_png * H_png * Bit_png) / (8 * FileSize_png);

% преобразование в полутоновое изображение
gray_jpg = rgb2gray(image);
% imshow(gray_jpg);
% pause

% создадим заранее папку для бинарных изображений
newFolder_Logical = pwd + "\Logical";
if ~exist(newFolder_Logical, "dir")
    mkdir(newFolder_Logical);
end

% TO DO: Image Processing Toolbox
% преобразование в бинарное изображение с порогом 25
binary_jpg = mat2gray(gray_jpg, [0 0.25*double(max(max(gray_jpg)))]);
% imshow(binary_jpg);
% pause
imwrite(binary_jpg, fullfile(newFolder_Logical, "binary_jpg_25.jpg"));
% преобразование в бинарное изображение с порогом 50
binary_jpg = mat2gray(gray_jpg, [0 0.50*double(max(max(gray_jpg)))]);
% imshow(binary_jpg);
% pause
imwrite(binary_jpg, fullfile(newFolder_Logical, "binary_jpg_50.jpg"));
% преобразование в бинарное изображение с порогом 75
binary_jpg = mat2gray(gray_jpg, [0 0.75*double(max(max(gray_jpg)))]);
% imshow(binary_jpg);
% pause
imwrite(binary_jpg, fullfile(newFolder_Logical, "binary_jpg_75.jpg"));

% создадим заранее папку для битовых плоскостей 8-битового изображениЯ
newFolder_BitPlane = pwd + "\BitPlane";
if ~exist(newFolder_BitPlane, "dir")
    mkdir(newFolder_BitPlane);
end

% преобразование в 8-битовое изображение
bitPlanesImage = uint8(gray_jpg);

for i=1:8
    bitPlane = mod(bitPlanesImage, 2^i);
%     imshow(bitPlane);
%     pause
    filename = sprintf('bitPlane%d.jpg', i);
    imwrite(bitPlane, fullfile(newFolder_BitPlane, filename));
end

% создадим заранее папку для дискретизации изображения с определённым ядром
newFolder_Discret = pwd + "\Discret";
if ~exist(newFolder_Discret, "dir")
    mkdir(newFolder_Discret);
end

% TO DO: Image Processing Toolbox
% Применение blkproc с ядром заданного размера
discretImage = blkproc(gray_jpg, [5, 5], 'mean2(x)*ones(size(x))');
% imshow(discretImage);
% pause
imwrite(discretImage, fullfile(newFolder_Discret, "discret_5_5.jpg"));

discretImage = blkproc(gray_jpg, [10, 10], 'mean2(2)*ones(size(x))');
% imshow(discretImage);
% pause
imwrite(discretImage, fullfile(newFolder_Discret, "discret_10_10.jpg"));

discretImage = blkproc(gray_jpg, [20, 20], 'mean2(2)*ones(size(x))');
% imshow(discretImage);
% pause
imwrite(discretImage, fullfile(newFolder_Discret, "discret_20_20.jpg"));

discretImage = blkproc(gray_jpg, [50, 50], 'mean2(2)*ones(size(x))');
% imshow(discretImage);
% pause
imwrite(discretImage, fullfile(newFolder_Discret, "discret_50_50.jpg"));

% создадим заранее папку для квантованных полутоновых изображений по 
% определённому уровню
newFolder_Quantiz = pwd + "\Quantiz";
if ~exist(newFolder_Quantiz, "dir")
    mkdir(newFolder_Quantiz);
end

% TO DO: Image Processing Toolbox
quantizeImage = imquantize(gray_jpg, linspace(0, 255, 4));
imshow(quantizeImage);
pause
imwrite(quantizeImage, fullfile(newFolder_Quantiz, "quantize_4.jpg"));

quantizeImage = imquantize(gray_jpg, linspace(0, 255, 16));
imshow(quantizeImage);
pause
imwrite(quantizeImage, fullfile(newFolder_Quantiz, "quantize_16.jpg"));

quantizeImage = imquantize(gray_jpg, linspace(0, 255, 32));
imshow(quantizeImage);
pause
imwrite(quantizeImage, fullfile(newFolder_Quantiz, "quantize_32.jpg"));

quantizeImage = imquantize(gray_jpg, linspace(0, 255, 64));
imshow(quantizeImage);
pause
imwrite(quantizeImage, fullfile(newFolder_Quantiz, "quantize_64.jpg"));

quantizeImage = imquantize(gray_jpg, linspace(0, 255, 128));
imshow(quantizeImage);
pause
imwrite(quantizeImage, fullfile(newFolder_Quantiz, "quantize_128.jpg"));

% создадим заранее папку для 100 на 100 вырезки из полутонового изображения
newFolder_Crop = pwd + "\Crop";
if ~exist(newFolder_Crop, "dir")
    mkdir(newFolder_Crop);
end

[rows, cols] = size(gray_jpg);
topLeftRow = floor(rows / 2 - 50) + 1;
topLeftCol = floor(cols / 2 - 50) + 1;
croppedImage = gray_jpg(topLeftRow:topLeftRow+99, topLeftCol:topLeftCol+99);
% imshow(croppedImage);
% pause
imwrite(croppedImage, fullfile(newFolder_Crop, "crop.jpg"));


% найдём 4-ки и 8-ки соседей пикселя полутонового изображения
N1 = zeros(4, 1);
N1(1) = gray_jpg(21, 16);
N1(2) = gray_jpg(21, 18);
N1(3) = gray_jpg(20, 17);
N1(4) = gray_jpg(22, 17);

N2 = zeros(4, 1);
N2(1) = gray_jpg(15, 10);
N2(2) = gray_jpg(15, 12);
N2(3) = gray_jpg(14, 11);
N2(4) = gray_jpg(16, 11);

N3 = gray_jpg(18:20, 87:89);
N3 = N3(:);
N3(5) = [];

gray_jpg_mean = mean(mean(gray_jpg));

% наносим на изображение метки в форме квадрата размером 20 на 20 пикселей 
% в центр и по углам изображения белым цветом т.к. gray_jpg_mean < 128

% создадим заранее папку для изображения с чёрными квадратами по углам и в
% центре
newFolder_Marks = pwd + "\Marks";
if ~exist(newFolder_Marks, "dir")
    mkdir(newFolder_Marks);
end

gray_jpg(1:20, 1:20) = ones(20, 20) * 255;
gray_jpg(1:20, cols-19:cols) = ones(20, 20) * 255;
gray_jpg(floor(rows/2)+1 : floor(rows/2)+20, floor(cols/2)+1 : floor(cols/2)+20) = ones(20, 20) * 255;
gray_jpg(rows-19:rows, 1:20) = ones(20, 20) * 255;
gray_jpg(rows-19:rows, cols-19:cols) = ones(20, 20) * 255;

% imshow(gray_jpg);
% pause

imwrite(gray_jpg, fullfile(newFolder_Marks, "marks.jpg"));
