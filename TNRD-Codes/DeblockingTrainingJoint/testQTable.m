clear all; close all;
q = 80;
clean_path = '/home/staff/cheny/DeblockingTestDataSet/quality10/labels/';
idx = 59;
str = num2str(100000+idx);    
image = ['test_' str(2:end) '.png'];
clean = double(imread([clean_path image]));
clean = clean(1:end-1,1:end-1);

imwrite(clean/255,'test.jpg','jpg','Quality',q);
JPEG_header_info = jpeg_read('test.jpg');
Q = JPEG_header_info.quant_tables{1};
cofQ = JPEG_header_info.coef_arrays{1};

load M50.mat;

tQ = min(255,floor(50/q*mtx+0.5));
delta = Q - tQ