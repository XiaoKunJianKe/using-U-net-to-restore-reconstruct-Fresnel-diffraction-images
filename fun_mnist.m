% Fresnel theory of diffraction, create diffraction images.
% in the convert_jpg file,we will use this function.
function [Image] = fun_mnist(img)
[w,t]=size(img);
lamda=600e-6;
k=2*pi/lamda;
z=90000;
t1=linspace(-t/2,t/2,t);
t2=linspace(-w/2,w/2,w);
[xi,yi]=meshgrid(t1,t2);
Ui=img;
Ui=double(Ui);
[xo,yo]=meshgrid(t1,t2);
h=exp(1i*k*z)*exp(1i*k*(xo.^2+yo.^2)/(2*z))/(1i*lamda*z);
u=exp(1i*k*(xo.^2+yo.^2)/(2*z));
Uie=Ui.*u;
H=fftshift(fft2(Uie));
Uo=h.*H;
Image=abs(Uo);
end

