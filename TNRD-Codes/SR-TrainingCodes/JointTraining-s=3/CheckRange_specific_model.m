function [loss, x_star] = CheckRange_specific_model(noisy, clean, input, diffterm, model, T)
[N,R] = size(input);
psz = sqrt(N);
r = psz;
c = psz;

mfsAll = model.mfsAll;
K = model.K;
p = model.p;
beta = model.beta;
mfs = model.mfs;

filtN = length(K);
F = cell(filtN,1);
parfor i = 1:filtN
    F{i} = make_F_filter(K{i}(:),r,c);
end
%% do a gradient descent step for all samples
g = (input - noisy)*p;
parfor i=1:filtN
    Ku = F{i}*input;
%     fprintf('Ku range:k%d, [%.3f %.3f]\n',i, min(Ku(:)), max(Ku(:)));
%     Ne1 = Mapping(Ku(:), means, precision, weights(:,i));
    Ne1 = lut_eval(Ku(:)', mfs.offsetD, mfs.step, mfsAll{i}.P, 0, 0, 0);
    Ne1 = reshape(Ne1,N,R);
    g = g + F{i}'*Ne1;
end
x_star = input - g + beta*diffterm;
loss = sum(sum((T*x_star - clean).^2))/R;