function [loss, grad] = loss_with_gradient_joint_stages(vcof)
global F_NOISE G_TRUTH BASIS PATCH_SIZE KernelPara MFS PT T STAGE Q_UPPER Q_LOWER DCT_MTX
tic;
clean = G_TRUTH;
noisy = F_NOISE;
Q_upper = Q_UPPER;
Q_lower = Q_LOWER;
basis = BASIS;
R = size(clean,2);
r = PATCH_SIZE;
c = PATCH_SIZE;

dct_mtx = DCT_MTX{1};
dct_t   = DCT_MTX{2};
mfs = MFS;
NumW = mfs.NumW;
stage = STAGE;
filter_size = KernelPara.fsz;
m = filter_size^2 - 1;
filtN = KernelPara.filtN;
mtxPT = PT;

trained_model = cell(stage,1);
len_cof = filtN*m + NumW*filtN;
vcof = reshape(vcof,len_cof,stage);

cof = vcof;
save('training_temp_cof.mat','cof','stage','MFS');
clear cof;
for s = 1:stage
    cof = vcof(:,s);
    part1 = cof(1:filtN*m);
    cof_beta = reshape(part1,m,filtN);
    part2 = cof(filtN*m+1:end);
    weights = reshape(part2,NumW,filtN);
    
    K = cell(filtN,1);
    f_norms = zeros(filtN,1);
    for i = 1:filtN
        x_cof = cof_beta(:,i);
        filter = basis*x_cof;
        f_norms(i) = norm(filter);
        filter = filter/(norm(filter) + eps);
        K{i} = reshape(filter,filter_size,filter_size);
    end
    %% update mfs
    MFsALL = updateMFs(mfs, weights, filtN);    
    %% construct model for one stage
    model.cof_beta = cof_beta;
    model.MFsALL = MFsALL;
    model.K = K;
    model.mfs = mfs;
    model.f_norms = f_norms;
    trained_model{s} = model;
end
% make sure not to use temporary variables
clear cof_beta MFsALL K p f_norms;
%% do a gradient descent step for all samples
AllNetWorks = cell(stage,1);
for s = 1:stage
    result.u = zeros(size(noisy));
    result.PTu = zeros(size(noisy));
    result.grad_l_u = zeros(size(noisy));
    result.ind = zeros(size(noisy));
    AllNetWorks{s} = result;
end
%% forward step
input = noisy;
for s = 1:stage
    model = trained_model{s};
    MFsALL = model.MFsALL;
    K = model.K;
    out_u = zeros(size(noisy));
    ind_s = zeros(size(noisy));
    parfor samp = 1:R
        u = reshape(input(:,samp),r,c);
        g = 0;
        for i=1:filtN
            Ku = imfilter(u,K{i},'symmetric');
            Ne1 = lut_eval_one_variable(Ku(:)', mfs.offsetD, mfs.step, MFsALL{i}.P);
            Ne1 = reshape(Ne1,r,c);
            g = g + imfilter(Ne1,rot90(rot90(K{i})),'symmetric');
        end
        x = u - g;
        %% projection to QCS, truncate cofs
        dct_x = dct_mtx*x(:);
        t_dct_x = max(Q_lower(:,samp), min(Q_upper(:,samp), dct_x(:)));
        out_u(:,samp) = dct_t*t_dct_x;        
        ind_s(:,samp) = (abs(t_dct_x - dct_x(:)) < 1e-12);
    end
    AllNetWorks{s}.ind = ind_s;
    AllNetWorks{s}.u = out_u;
    AllNetWorks{s}.PTu = PT*out_u;
    input = AllNetWorks{s}.PTu;
end
x_s = AllNetWorks{stage}.u;
loss = sum(sum((T*x_s - clean).^2))/R;
fprintf('current cost value: %.3f\n', loss);
%% show some samples
showsamples2(6, R, F_NOISE, G_TRUTH, x_s, T);
%% derivative of loss wrt x
AllNetWorks{stage}.grad_l_u = 2/R*T'*(T*x_s-clean);
%% caluculate gradients of loss w.r.t us of each stage
for s = stage-1:-1:1
    model = trained_model{s+1};
    MFsALL = model.MFsALL;
    K = model.K;
    
    grad_l_usp1 = AllNetWorks{s+1}.grad_l_u;
    ind_sp1 = AllNetWorks{s+1}.ind;
    PTu_s = AllNetWorks{s}.PTu;
    
    grad_l_us = zeros(size(noisy));
    parfor samp = 1:R
        les = grad_l_usp1(:,samp);
        e = dct_t*(ind_sp1(:,samp).*(dct_mtx*les));
        u = PTu_s(:,samp);
        % part 1
        part1 = e;
        % part 2
        part2 = 0;
        for i=1:filtN
            k = K{i};
            t = convolution_transposeOMP(e,rot90(rot90(k)),r,c);
            
            Ku = imfilter(reshape(u,r,c),k,'symmetric');
            Ne2 = lut_eval_one_variable(Ku(:)', mfs.offsetD, mfs.step, MFsALL{i}.GX);
            
            part2 = part2 + convolution_transposeOMP(t(:).*Ne2(:),k,r,c);
        end
        grad_l_us(:,samp) = mtxPT'*(part1 - part2);
    end
    AllNetWorks{s}.grad_l_u = grad_l_us;
end
%% calculate gradients of loss w.r.t training parameters
grad_l_paras = zeros(len_cof,stage);
for s = 1:stage
    if s == 1
        input = noisy;
    else
        input = AllNetWorks{s-1}.PTu;
    end
    grad_l_paras(:,s) = ...
        grad_l_thetas(AllNetWorks, trained_model, input, basis, R, r, KernelPara, mfs, s, DCT_MTX);
end
grad = grad_l_paras(:);
toc