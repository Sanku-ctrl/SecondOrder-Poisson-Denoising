clear all;
clc;
filter_size = 5;
m = filter_size^2 - 1;
filter_num = 24;

global F_NOISE G_TRUTH BASIS INPUT PATCH_SIZE KernelPara MFS T SRmtx
R = 4;
path = '/home/staff/cheny/FoETraininigSets/FoETrainingSets180/';
scale = 3;
[G_TRUTH, u0, F_NOISE] = LoadTrainingImages(R, path, scale);
%% pad input images
psz = 180;
bsz = filter_size + 1;
[T, u0, F_NOISE] = PadNoisyPatches(u0, F_NOISE, R, psz, bsz, scale);
PATCH_SIZE = psz + 2*bsz;
INPUT = u0;
load SRmtx.mat;
SRmtx = A;

BASIS   = gen_dct2(filter_size);
BASIS   = BASIS(:,2:end);
cof_beta = eye(m,m)/10;
LEARNED_X = cof_beta(:);

KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
%% training algorithm parameters
KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
stage = 1;
[x_init, MFS] = Equal_Initialization(KernelPara, stage);

x0 = x_init(:,1);
n = length(x0);
x0 = rand(n,1) - 0.5;
% tic;[loss, grad] = loss_with_gradient_unit_filters(x0);toc
tic;[loss_lut, grad_lut] = loss_with_gradient_unit_filters_LUT(x0);toc
% delta_g = grad_lut - grad;
% ratio = delta_g./(abs(grad) + eps);
% delta_l = loss_lut - loss;
% return;
grad = grad_lut;
eps = 1e-6;
check_n = 2089;
grad2 = zeros(check_n,1);
s = 0;
for i = s+1:s+check_n
    xp = x0;
    xp(i) = xp(i) + eps;
    lp = loss_energy_unit_filters(xp);
    
    xm = x0;
    xm(i) = xm(i) - eps;
    lm = loss_energy_unit_filters(xm);
    
    grad2(i) = (lp - lm)/(2*eps);
    fprintf('num: %d, grad = %.3f, difference = %f\n', i, grad(i), grad2(i) - grad(i));
end