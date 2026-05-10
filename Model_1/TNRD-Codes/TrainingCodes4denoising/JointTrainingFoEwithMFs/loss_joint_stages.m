function loss = loss_joint_stages(vcof)
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
input = noisy;
for s = 1:stage
    model = trained_model{s};
    MFsALL = model.MFsALL;
    K = model.K;
    p = model.p;    
    out_u = zeros(size(noisy));
    parfor samp = 1:R
        u = input(:,samp);
        f = noisy(:,samp);
        g = (u - f)*p;
        g = reshape(g,r,c);
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