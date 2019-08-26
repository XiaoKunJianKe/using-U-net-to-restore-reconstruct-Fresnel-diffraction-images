load('mnist_all.mat');
type = 'train'; %another style:type = 'test'represent test set.because 
% mnist_all.mat contains train and test set.
savePath = 'E:/Mnist_32_32_60000/train_image_';% save  Fresnel diffraction images.
savedir='E:/Mnist_32_32_60000/train_label_'; % save original digit images.
for num = 0:1:9
    numStr = num2str(num);
    tempNumPath = strcat(savePath, numStr);
    tempNumdir1=strcat(savedir,numStr);
    mkdir(tempNumPath);
    mkdir(tempNumdir1);
    tempNumPath = strcat(tempNumPath,'/');
    tempNumdir1=strcat(tempNumdir1,'/');
    tempName = [type, numStr];
    tempFile = eval(tempName);
    [height, width]  = size(tempFile);
    for r = 1:1:500   % number 500 reprsents 500 samples for every digit(i.e. 0 to 9).
        tempImg = reshape(tempFile(r,:),28,28)';
        tempImg=im2double(tempImg);
        tempImg=imresize(tempImg,[32,32]);
        tempNumdir=strcat(tempNumdir1,numStr,'_',num2str(r-1),'.jpg');
        imwrite(tempImg,tempNumdir);
        tempImg=fun_mnist(tempImg);
        tempImg=(tempImg-min(min(tempImg)))/(  max(max(tempImg))-min(min(tempImg)) )*255;
        tempImgPath = strcat(tempNumPath,numStr,'_',num2str(r-1));
        tempImgPath = strcat(tempImgPath,'.jpg');
        imwrite(uint8(tempImg),tempImgPath);
    end
end