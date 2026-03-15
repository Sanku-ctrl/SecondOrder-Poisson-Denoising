function x_star = denoisingOneStepGMixMFs(noisy, input, model)
mfsAll = model.mfsAll;
K = model.K;
p = model.p;
mfs = model.mfs;
filtN = length(K);
[r,c] = size(input);
%% do a gradient descent step for all samples
u = input;
f = noisy;
g = (u - f)*p;
parfor i=1:filtN
%     Ku = imfilter(u,K{i},'symmetric');
    Ku = conv2(u,rot90(rot90(K{i})),'same');
%     Ne1 = lut_eval(Ku(:)', mfs.offsetD, mfs.step, mfsAll{i}.P, 0, 0, 0);
    Ne1 = lut_eval_one_variable(Ku(:)', mfs.offsetD, mfs.step, mfsAll{i}.P);
    Ne1 = reshape(Ne1,r,c);
%     g = g + imfilter(Ne1,rot90(rot90(K{i})),'symmetric');
    g = g + conv2(Ne1,K{i},'same');
end
x_star = u - g;
x_star = max(0, min(x_star, 255));