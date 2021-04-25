%% For task "D"
%
clear; clc; close all;

% abstract orthonormal vector fillers
%
n1 = 8; % number of words
n2 = 4; % number of roles
n = 12; % number of memory components
dim1 = 8; % dimension of words
dim2 = 4; % dimension of roles
N = dim1*dim2; % dimension of memory space

mem_input = zeros(dim1,n1); % initializing words representations
% word lists: 1.Mary, 2.John, 3.dog, 4.calling 5.chasing 6.looking 7.living room 8. garden
for i=1:n1
	mem_input(:,i) = randn(dim1,1);
	mem_input(:,i) = mem_input(:,i)/norm(mem_input(:,i));
end
mem_input = gsprocess(mem_input,dim1,n1); % orthonormalizing word representations

role = randn(dim2,dim2); % initializing role lists
role = gsprocess(role,dim2,dim2); % role lists: 1.subject, 2.predicate, 3.object, 4.modifier

% Building semantic memory components
mem_comp = zeros(N,n);
mem_comp(:,1) = reshape(mem_input(:,1)*role(:,1)',[N,1]); % Mary_S
mem_comp(:,2) = reshape(mem_input(:,4)*role(:,2)',[N,1]); % calling_P
mem_comp(:,3) = reshape(mem_input(:,2)*role(:,3)',[N,1]); % John_O
mem_comp(:,4) = reshape(mem_input(:,7)*role(:,4)',[N,1]); % living room_M
% S1 := Mary is calling John in the living room
mem_comp(:,5) = reshape(mem_input(:,2)*role(:,1)',[N,1]); % John_S
mem_comp(:,6) = reshape(mem_input(:,5)*role(:,2)',[N,1]); % chasing_P
mem_comp(:,7) = reshape(mem_input(:,3)*role(:,3)',[N,1]); % dog_O
mem_comp(:,8) = reshape(mem_input(:,8)*role(:,4)',[N,1]); % garden_M
% S2 := John is chasing a dog in the garden
mem_comp(:,9) = reshape(mem_input(:,2)*role(:,1)',[N,1]); % John_S
mem_comp(:,10) = reshape(mem_input(:,6)*role(:,2)',[N,1]); % looking_P
mem_comp(:,11) = reshape(mem_input(:,1)*role(:,3)',[N,1]); % Mary_O
mem_comp(:,12) = reshape(mem_input(:,8)*role(:,4)',[N,1]); % garden_M
% S2 := John is looking at Mary in the garden

% Constructing theoretical value of W
fprintf('Constructing theoretical value of W*...\n\n');
% parameters
omega = 1.5; 
tau = (pi/2)/omega; 
r = 1; 
g = 1;
xi = linspace(0,pi,n2+1)'; xi = xi(1:end-1); % building xi_i with n-equivalent spacing on interval [0,pi]

u = zeros(N,3); v = zeros(N,3); % constructing u, v vectors
u(:,1) = -mem_comp(:,1:4)*sin(xi); v(:,1) = mem_comp(:,1:4)*cos(xi);
u(:,2) = -mem_comp(:,5:8)*sin(xi); v(:,2) = mem_comp(:,5:8)*cos(xi);
u(:,3) = -mem_comp(:,9:12)*sin(xi); v(:,3) = mem_comp(:,9:12)*cos(xi);
L = zeros(1,3); W_temp = zeros(N,N,3);

eta1 = zeros(1,3); eta2 = zeros(1,3); mu = zeros(1,3);
% finding theoretical value of lambda_0 (in supplements)
for i=1:3
	eta1(i) = norm(u(:,i)); eta2(i) = norm(v(:,i));
	mu(i) = u(:,i)'*v(:,i)/(norm(u(:,i))*norm(v(:,i)));
	P5 = 1;
	P4 = 0;
	P3 = 2-2*omega^2;
	P2 = -r*sin(omega*tau)*eta1(i)*eta2(i)*sqrt(1-mu(i)^2)/g;
	P1 = omega^4+2*omega^2+1 - (eta1(i)^2+eta2(i)^2)*r*sin(omega*tau)*omega/g;
	P0 = -r*sin(omega*tau)*eta1(i)*eta2(i)*sqrt(1-mu(i)^2)/g*(omega^2+1);
	R = roots([P5 P4 P3 P2 P1 P0]);
	for j=1:5
		if imag(R(j)) == 0
			L(i) = real(R(j));
		end
	end
end
for i=1:3
	W_temp(:,:,i) = (L(i)/(eta1(i)*eta2(i)*sqrt(1-mu(i)^2)))*(v(:,i)*u(:,i)'-u(:,i)*v(:,i)');
end
for i=1:3
	W_temp(:,:,i) = v(:,i)*u(:,i)'-u(:,i)*v(:,i)';
end

W = (W_temp(:,:,1)+W_temp(:,:,2)+W_temp(:,:,3))/3; % W = (W1+W2+W3)/3
fprintf('Construction complete!\n\n');

%% Retrieval Phase

dt = 0.1;
rdt = 0.01;
K = round(0.5/dt);
trfin = 40;
tr = (0:rdt:trfin);
xr = zeros(N,length(tr)); xr(:,1) = randn(N,1); xr(:,1) = xr(:,1)/norm(xr(:,1));

fprintf('\nNow on RETRIEVAL PHASE...\n\n');

% cue construction
pair = input('\nInput an index of word-role pair the cue.\nInput [i j] indicates retrieval cue of f_i x r_j. \n\n   (Ex: try [1 1]( = Mary_S), or [2 1;1 3]( = John_S + Mary_O), etc.\n   Input:  ');
[n,m] = size(pair);
if n == 1 % single-component cue
	w_r = reshape(mem_input(:,pair(1,1))*role(:,pair(1,2))',[N,1]);
	T1 = (pair(1,2)-1)/4*pi;
	br = 1*sin(omega*tr-T1).*w_r;
elseif n == 2 % doule-component cue
	C1 = reshape(mem_input(:,pair(1,1))*role(:,pair(1,2))',[N,1]);
	C2 = reshape(mem_input(:,pair(2,1))*role(:,pair(2,2))',[N,1]);
	T1 = (pair(1,2)-1)/4*pi;
	T2 = (pair(2,2)-1)/4*pi;
	br = 1*sin(omega*tr-T1).*C1 + 1*sin(omega*tr-T2).*C2;
else % triple-component cue
	C1 = reshape(mem_input(:,pair(1,1))*role(:,pair(1,2))',[N,1]);
	C2 = reshape(mem_input(:,pair(2,1))*role(:,pair(2,2))',[N,1]);
	C3 = reshape(mem_input(:,pair(3,1))*role(:,pair(3,2))',[N,1]);
	T1 = (pair(1,2)-1)/4*pi;
	T2 = (pair(2,2)-1)/4*pi;
	T3 = (pair(3,2)-1)/4*pi;
	br = 1*sin(omega*tr-T1).*C1 + 1*sin(omega*tr-T2).*C2 + 1*sin(omega*tr-T3).*C3;
end

fprintf('\nNow performing retrieval with selected orders...\nPlease wait...\n');

for i=1:length(tr)-1
	xr(:,i+1) = xr(:,i) + rdt*(-xr(:,i) + W*xr(:,i) + br(:,i));
end
fprintf('\nRetrieval process complete!')

R = zeros(dim1,length(tr),n2);
for i=1:n2
	for j=1:length(tr)
		R(:,j,i) = reshape(xr(:,j),[dim1 dim2])*role(:,i);
	end
end
comp = zeros(n2,length(tr)); comp_alt = zeros(n2,n1,length(tr));
comp_alt_int = zeros(n2,n1,length(tr)+1);
for i=1:n2
	for k=1:n1
		for j=1:length(tr)
			comp_alt(i,k,j) = abs(R(:,j,i)'*mem_input(:,k));
			comp_alt_int(i,k,j+1) = comp_alt_int(i,k,j) + comp_alt(i,k,j)*dt;
		end
	end
end

scr_siz = get(0,'ScreenSize');
fig1 = figure(1); fig1.Position = floor([scr_siz(3)/4 1.5*scr_siz(4)/4 scr_siz(3)/2 scr_siz(4)/6]);

subplot(1,n2,1); 
plot(tr,squeeze(comp_alt_int(1,1,2:end)),'color','r','linewidth',2); hold on;
plot(tr,squeeze(comp_alt_int(1,2,2:end)),':','color',[0 0.6 0],'linewidth',2.5);
xlim([0 30]); title('Subject'); legend('Mary','John');

subplot(1,n2,2); 
plot(tr,squeeze(comp_alt_int(2,4,2:end)),'color','r','linewidth',2); hold on;
plot(tr,squeeze(comp_alt_int(2,5,2:end)),'-s','color','b','linewidth',1,'markersize',5,'MarkerIndices',1:300:length(tr));
plot(tr,squeeze(comp_alt_int(2,6,2:end)),':','color',[0 0.6 0],'linewidth',2.5);
xlim([0 30]); title('Predicate'); legend('calling','chasing','looking');

subplot(1,n2,3); 
plot(tr,squeeze(comp_alt_int(3,2,2:end)),'color','r','linewidth',2); hold on;
plot(tr,squeeze(comp_alt_int(3,3,2:end)),'-s','color','b','linewidth',1,'markersize',5,'MarkerIndices',1:300:length(tr));
plot(tr,squeeze(comp_alt_int(3,1,2:end)),':','color',[0 0.6 0],'linewidth',2.5);
xlim([0 30]); title('Object'); legend('John','dog','Mary');

subplot(1,n2,4); 
plot(tr,squeeze(comp_alt_int(4,7,2:end)),'color','r','linewidth',2); hold on;
plot(tr,squeeze(comp_alt_int(4,8,2:end)),':','color',[0 0.6 0],'linewidth',2.5);
xlim([0 30]); title('Modifier'); legend('living room','garden');
