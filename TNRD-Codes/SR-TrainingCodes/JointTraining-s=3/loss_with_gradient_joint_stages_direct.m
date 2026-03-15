function [loss, grad] = loss_with_gradient_joint_stages_direct(vcof)
global F_NOISE G_TRUTH BASIS PATCH_SIZE KernelPara MFS PT T STAGE
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

trained_model = cell(stage,1);
len_cof = filtN*m + 1 + NumW*filtN;
vcof = reshape(vcof,len_cof,stage);

save('training_temp_cof.mat','vcof');
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
    model.weights = weights;
    trained_model{s} = model;
end
% make sure not to use temporary variables
clear cof_beta MFsALL K p f_norms weights;
%% do a gradient descent step for all samples
AllNetWorks = cell(stage,1);
for s = 1:stage
    result.u = zeros(size(noisy));
    result.PTu = zeros(size(noisy));
    result.grad_l_u = zeros(size(noisy));
    AllNetWorks{s} = result;
end
%% forward step
means = mfs.means;
precision = mfs.precision;
input = noisy;
for s = 1:stage
    model = trained_model{s};
    MFsALL = model.MFsALL;
    K = model.K;
    p = model.p;
    weights = model.weights;
    out_u = zeros(size(noisy));
    parfor samp = 1:R
        u = input(:,samp);
        f = noisy(:,samp);
        g = (u - f)*p;
        g = reshape(g,r,c);
        for i=1:filtN
            Ku = imfilter(reshape(u,r,c),K{i},'symmetric');
%             Ne1 = lut_eval_one_variable(Ku(:)', mfs.offsetD, mfs.step, MFsALL{i}.P);
            Ne1 = Mapping(Ku(:), means, precision, weights(:,i));
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
%% caluculate gradients of loss w.r.t us of each stage
for s = stage-1:-1:1
    model = trained_model{s+1};
    MFsALL = model.MFsALL;
    K = model.K;
    p = model.p;
    weights = model.weights;
    
    grad_l_usp1 = AllNetWorks{s+1}.grad_l_u;
    PTu_s = AllNetWorks{s}.PTu;
    
    grad_l_us   = zeros(size(noisy));
    parfor samp = 1:R
        u = PTu_s(:,samp);
        % part 1
        part1 = (1-p)*grad_l_usp1(:,samp);
        % part 2
        part2 = 0;
        for i=1:filtN
            k = K{i};
            t = convolution_transposeOMP(grad_l_usp1(:,samp),rot90(rot90(k)),r,c);
            
            Ku = imfilter(reshape(u,r,c),k,'symmetric');
%             Ne2 = lut_eval_one_variable(Ku(:)', mfs.offsetD, mfs.step, MFsALL{i}.GX);
            [~, Ne2, ~] = MappingAndGrad(Ku(:), means, precision, weights(:,i));
            
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
        grad_l_thetas(AllNetWorks, trained_model, input, noisy, basis, R, r, KernelPara, mfs, s);
end
grad = grad_l_paras(:);