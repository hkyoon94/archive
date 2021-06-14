clear; clc; close all;

D = 32; % pixel length of image (dimension of image = D x D)
dim = D*D;
n = 5; % actual number of memory components
n_ = 5; % must be equal or bigger than n, dimension of role space
N = dim*(n_); % dimension of memory space

mem_comp = zeros(N,n); % initialization for memory components m_i
mem_input = zeros(dim,n+1); % initialization for memory inputs f_i

% loading images
im(:,:,1) = imresize(rgb2gray(imread('violin.png')),[D,D]);
im(:,:,2) = imresize(rgb2gray(imread('trumpet.png')),[D,D]);
im(:,:,3) = imresize(rgb2gray(imread('harp.png')),[D,D]);
im(:,:,4) = imresize(rgb2gray(imread('piano.png')),[D,D]);
im(:,:,5) = imresize(rgb2gray(imread('timpani.png')),[D,D]);
im(:,:,6) = imresize(rgb2gray(imread('forest.png')),[D,D]);

sigma = 0.05; % brightness threshold
scr_siz = get(0,'ScreenSize');
fig1 = figure(1); fig1.Position = floor([scr_siz(3)/4 2.5*scr_siz(4)/4 scr_siz(3)/2 scr_siz(4)/4]);
for i=1:n+1
	subplot(1,6,i); imshow(mat2gray(im(:,:,i)));
	mem_input(:,i) = reshape(im2double(im(:,:,i)),[dim 1]);
	mem_input(:,i) = 2*(mem_input(:,i)-0.5)*sigma; % Assigning each pixel in interval [-sigma +sigma]
	if i ~= n+1
		title(sprintf('Original image %d',i));
	else
		title(sprintf('Irrelevant image'));
	end
end

tag = randn(n_,n); % initializing tagging vectors r_i
tag = gsprocess(tag,n_,n); % orthonormalizing tagging vectors

% building memory components with flattened tensor product
for i=1:n
	mem_comp(:,i) = reshape(mem_input(:,i)*tag(:,i)',[N,1]); 
end

%% STORAGE PHASE

% parameters
omega = 1.5;
tau = (pi/2)/omega;
gamma = 1;
rho = 1;

f1 = @(x,W,b) -x + W*x + b; % model's evolution eq. of x(t)
f2 = @(x,xd,W) -gamma*W + rho*(x*xd'-xd*x'); % model's evolution eq. of W(t)

tfin = 30; % final integration time
dt = 0.1; % learning timestep
xi = linspace(0,pi,n+1)'; xi = xi(1:end-1); % building xi_i with n-equivalent spacing on interval [0,pi]

dt = tau/ceil(tau/dt);
T = round(tau/dt);
t = (-tau:dt:tfin); % integration interval: [-tau,tfin]

x_mat = 1e-4*randn(N,T+1); % initialization of x([-tau 0])
W = 1e-6*randn(N,N); % initialization of W(0)

% building input harmonic pulse ensemble b(t)
b = 0;
for i=1:n
	b = b + 1*sin(omega*t-xi(i)).*mem_comp(:,i);
end

ct = round((length(t)-T)/100);
percent = 0;
fprintf('Now in STORAGE PHASE...\n\n');
f = waitbar(0,'Learning, please wait...');
for i = T+1:length(t)-1
	x_new = x_mat(:,end) + f1(x_mat(:,end),W,b(:,i))*dt;
	W = W + f2(x_mat(:,end),x_mat(:,1),W)*dt;
	x_mat(:,1:end-1) = x_mat(:,2:end);
	x_mat(:,end) = x_new;
	if rem(i-T,ct) == 0
		percent = percent+1;
		waitbar(0.01*percent,f,'Learning, please wait...');
	end
end
waitbar(100,f,'Storage complete! Please proceed to command window...');
fprintf('Storage Phase complete!\n\n');

%% RETRIEVAL PHASE

o1 = randn(dim,1); o1 = o1/norm(o1);
o2 = randn(n_,1); o2 = o2/norm(o2);
rdt = 0.01;
trfin = 15;
tr = (0:rdt:trfin); % retrieval time interval
xr = zeros(N,length(tr)); xr(:,1) = 1e-3*randn(N,1);

fprintf('Now in RETRIEVAL PHASE...\n');

contam_method = input('\nSelect the type of the cue.\n   1. Original \n   2. Relevant, lightly noise-contaminated  \n   3. Relevant, heavily noise-contaminated \n   4. Relevant, partially obstructed \n   5. Irrelevant \n   (Ex: 2)\n   Input:  ');
if contam_method ~= 5
	ret_pick = input(sprintf('\nYou selected a relavent(original) type of cue.\nPlease input an index 1<=I<=%d of the original input for the construction of cue. \n   (Ex: 1)\n   Input:  ',n));
end

I = ret_pick;
switch contam_method
	case 1 % original
		f_c = mem_input(:,I);
		m_c = mem_comp(:,I); 
	case 2 % lightly noise-contaminated
		O1 = norm(mem_input(:,I))*o1;
		noise = 0.1;
		f_c = sqrt(1-noise^2)*mem_input(:,I)+noise*O1;
		r_c = sqrt(1-0.2^2)*tag(:,I)+0.2*o2;
		m_c = reshape(f_c*r_c',[N,1]);
	case 3 % heavily noise-contaminated
		O1 = norm(mem_input(:,I))*o1;
		noise = 0.5;
		f_c = sqrt(1-noise^2)*mem_input(:,I)+noise*O1;
		r_c = sqrt(1-0.2^2)*tag(:,I)+0.2*o2;
		m_c = reshape(f_c*r_c',[N,1]);
	case 4 % partially obstructed
		f_c = reshape(mem_temp(:,I),[D D]);
		a = round(D/3);
		for i=1:a
			for j=a
				f_c(a:a+i,a:a+j) = 0;
			end
		end
		f_c = reshape(f_c,[dim 1]);
		r_c = sqrt(1-0.2^2)*tag(:,I)+0.2*o2;
		m_c = reshape(f_c*r_c',[N,1]);
	case 5 % irrelevant
		f_c = reshape(mem_input(:,n+1),[D D]);
		noise = 0.2;
		r_c = sqrt(1-noise^2)*tag(:,I)+noise*o2;
		add_im = reshape(mem_input(:,n+1)*r_c',[N,1]);
		m_c = add_im;
	otherwise
		error('invalid option number');
end

b_c = 1*sin(omega*tr).*m_c; % building cue input b_c(t)

figure(2);
imshow(kron((1/sigma)*reshape(f_c,[D D]),ones(3,3)),[-1 1]);
title('Cue Visualization');

fprintf('\nNow performing retrieval with selected orders...\nPlease wait...\n');

for i=1:length(tr)-1
	xr(:,i+1) = xr(:,i) + rdt*(-xr(:,i) + W*xr(:,i) + b_c(:,i)); % retrieval equation
end
fprintf('\nRetrieval phase complete!')

a = 1/omega*(atan(omega));
A = round([a+pi/omega, a+2*pi/omega, a+3*pi/omega, a+4*pi/omega, a+5*pi/omega, a+6*pi/omega],2);
fprintf('\n\nNow specify a time instant 0<t<%.1f for observing decoded images.\n   (Try t = %.2f, %.2f, %.2f, %.2f, %.2f, %.2f(recommended))',...
	trfin,A(1),A(2),A(3),A(4),A(5),A(6));
while 0~=1
	ret_t = input('\n\n   Input: t =   ');
	if ret_t > trfin
		while ret_t > trfin
			ret_t = input(sprintf('\nERROR: must be smaller than %.1f, please try again.\n(Put "E" to stop)\n\n   Input:  ',trfin));
			if ret_t <= trfin
				break
			end
		end
	end
	for i=1:n
		R = reshape(xr(:,100*ret_t+1),[dim n_])*tag(:,i);
		R = reshape(R,[D D]);
		f3 = figure(3);
		f3.Position = floor([scr_siz(3)/4 1*scr_siz(4)/4 scr_siz(3)/2 scr_siz(4)/4]);
		subplot(1,5,i); imshow(kron(3*(1/sigma)*R,ones(4,4)),[-1 1]); 
		title(sprintf('recovered image %d',i));
	end
end
