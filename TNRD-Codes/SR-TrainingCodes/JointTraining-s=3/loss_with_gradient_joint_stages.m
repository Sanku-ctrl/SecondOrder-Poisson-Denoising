function [loss, grad] = loss_with_gradient_joint_stages(vcof)
global F_NOISE G_TRUTH BASIS PATCH_SIZE KernelPara MFS PT T STAGE SRmtx U0 SCALE
tic;
noisy = F_NOISE;
clean = G_TRUTH;
basis = BASIS;
R = size(noisy,2);
r = PATCH_SIZE;
c = PATCH_SIZE;
mfs = MFS;
NumW = mfs.NumW;
stage = STAGE;
filter_size = KernelPara.fsz;
m = filter_size^2 - 1;
filtN = KernelPara.filtN;
mtxPT = PT;
% A = SRmtx;
% A_t = A';
% AA = A'*A;
scale = SCALE;

trained_model = cell(stage,1);
len_cof = filtN*m + 1 + NumW*filtN;
vcof = reshape(vcof,len_cof,stage);

cof = vcof;
save('training_temp_cof.mat','cof','stage','MFS');
clear cof;
for s = 1:stage
    cof = vcof(:,s);
    part1 = cof(1:filtN*m);
    cof_beta = reshape(part1,m,filtN);
    part2 = cof(filtN*m+1);
    p = exp(part2);
    part3 = cof(filtN*m+2:end);
    weights = reshape(part3,NumW,filtN);
    
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
    model.p = p;
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
    AllNetWorks{s} = result;
end
%% forward step
input = U0;
for s = 1:stage
    model = trained_model{s};
    MFsALL = model.MFsALL;
    K = model.K;
    p = model.p;    
    out_u = zeros(size(input));
    parfor samp = 1:R
        u = input(:,samp);
        f = noisy(:,samp);
        tv = imresize(reshape(u,r,c), 1/scale, 'bicubic') - reshape(f,r/scale,c/scale);
        g = p*imresize(tv, scale, 'bicubic')/scale^2;
%         g = p*A_t*(A*u - f);
%         g = reshape(g,r,c);
        for i=1:filtN
            Ku = imfilter(reshape(u,r,c),K{i},'symmetric');
%             Ne1 = lut_eval(Ku(:)', mfs.offsetD, mfs.step, MFsALL{i}.P, 0, 0, 0);
            Ne1 = lut_eval_one_variable(Ku(:)', mfs.offsetD, mfs.step, MFsALL{i}.P);
            Ne1 = reshape(Ne1,r,c);
            g = g + imfilter(Ne1,rot90(rot90(K{i})),'symmetric');
        end
        out_u(:,samp) = u - g(:);
    end
    AllNetWorks{s}.u = out_u;
    AllNetWorks{s}.PTu = PT*out_u;
    input = AllNetWorks{s}.PTu;
end
x_s = AllNetWorks{stage}.u;
loss = sum(sum((T*x_s - clean).^2))/R;
fprintf('current cost value: %.3f\n', loss);
%% derivative of loss wrt x
AllNetWorks{stage}.grad_l_u = 2/R*T'*(T*x_s-clean);
%% show samples
showsamples(6, T*U0, G_TRUTH, T*x_s);
%% caluculate gradients of loss w.r.t us of each stage
for s = stage-1:-1:1
    model = trained_model{s+1};
    MFsALL = model.MFsALL;
    K = model.K;
    p = model.p;
    
    grad_l_usp1 = AllNetWorks{s+1}.grad_l_u;
    PTu_s = AllNetWorks{s}.PTu;
    
    grad_l_us = zeros(size(U0));
    parfor samp = 1:R
        u = PTu_s(:,samp);
        % part 1
%         part1 = grad_l_usp1(:,samp) - p*AA*grad_l_usp1(:,samp);
        tv = imresize(reshape(grad_l_usp1(:,samp),r,c), 1/scale, 'bicubic');
        tv = p*imresize(tv, scale, 'bicubic')/scale^2;
        part1 = grad_l_usp1(:,samp) - tv(:);
        % part 2
        part2 = 0;
        for i=1:filtN
            k = K{i};
            t = convolution_transposeOMP(grad_l_usp1(:,samp),rot90(rot90(k)),r,c);
            
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
        input = U0;
    else
        input = AllNetWorks{s-1}.PTu;
    end
    reaction_term = zeros(size(input));
    parfor samp = 1:R
        u = input(:,samp);
        f = noisy(:,samp);
        tv = imresize(reshape(u,r,c), 1/scale, 'bicubic') - reshape(f,r/scale,c/scale);
        diff = imresize(tv, scale, 'bicubic')/scale^2;
        reaction_term(:,samp) = diff(:);
    end
    grad_l_paras(:,s) = ...
        grad_l_thetas(AllNetWorks, trained_model, input, basis, R, r, KernelPara, mfs, s, reaction_term);
end
grad = grad_l_paras(:);
toc