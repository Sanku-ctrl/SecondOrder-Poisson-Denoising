function [loss, grad] = loss_with_gradient_unit_filters(vcof)
global F_NOISE G_TRUTH BASIS INPUT PATCH_SIZE KernelPara MFS DIFF_TERM T
noisy = F_NOISE;
clean = G_TRUTH;
input = INPUT;
diffterm = DIFF_TERM;
basis = BASIS;
R = size(noisy,2);
r = PATCH_SIZE;
c = PATCH_SIZE;

means = MFS.means;
precision = MFS.precision;
NumW = MFS.NumW;

% tic;
filter_size = KernelPara.fsz;
m = filter_size^2 - 1;
filtN = KernelPara.filtN;
part1 = vcof(1:filtN*m);
cof_beta = reshape(part1,m,filtN);
% weight theta
% part2 = vcof(filtN*m+1:filtN*(m+1));
% theta = exp(part2);
part3 = vcof(filtN*m+1);
p = exp(part3);
part4 = vcof(filtN*m+2:end-1);
weights = reshape(part4,NumW,filtN);
part5 = vcof(end);
beta = exp(part5);

% construct filters
% filters = basis*cof_beta;
% sfigure(1);
% DisplayFilters(filters',2,12,filter_size,theta);drawnow;
%% unit norm filters
K = cell(filtN,1);
f_norms = zeros(filtN,1);
parfor i = 1:filtN
    x_cof = cof_beta(:,i);
    filter = basis*x_cof;
    f_norms(i) = norm(filter);
    filter = filter/norm(filter);
    K{i} = reshape(filter,filter_size,filter_size);
end
%% do a gradient descent step for all samples
x = zeros(size(input));
parfor samp = 1:R
    u = input(:,samp);
    f = noisy(:,samp);
    g = (u - f)*p;
    g = reshape(g,r,c);
    for i=1:filtN
        Ku = imfilter(reshape(u,r,c),K{i},'symmetric');
        Ne1 = Mapping(Ku(:), means, precision, weights(:,i));
        Ne1 = reshape(Ne1,r,c);
        g = g + imfilter(Ne1,rot90(rot90(K{i})),'symmetric');
    end
    x(:,samp) = u - g(:) + beta*diffterm(:,samp);
end
% t1 = toc
loss = sum(sum((T*x - clean).^2))/R;
%% derivative of loss wrt x
grad_l_x = 2/R*T'*(T*x-clean);
%% loss and gradient w.r.t psnr
% loss = 0;
% grad_l_x = zeros(size(clean));
% parfor samp = 1:R
%     delta = x(:,samp) - clean(:,samp);
%     error = sum(delta.^2);
%     loss = loss + log(error);
%     grad_l_x(:,samp) = 2*delta/error;
% end
% t2 = toc
%% second way
grad_loss_beta = 0;
grad_loss_weights = 0;
psz = PATCH_SIZE;
pd  = (filter_size-1)/2;
parfor samp = 1:R
    grad_ln_beta = zeros(m,filtN);
    grad_ln_ws   = zeros(NumW,filtN);
    x = reshape(input(:,samp),psz,psz);
    xl = padarray(x, [pd,pd], 'both', 'symmetric');
    v = reshape(grad_l_x(:,samp),psz,psz);
    for i=1:filtN
        k = K{i};
        %% part 1
        kx = imfilter(x,k,'symmetric');
        [Nkx, N2kx, GW] = MappingAndGrad(kx(:), means, precision, weights(:,i));
        Nkx = reshape(Nkx,r,c);
        N2kx = reshape(N2kx,r,c);
        t = convolution_transposeOMP(v,rot90(rot90(k)),psz,psz);
        temp = N2kx.*reshape(t,psz,psz);
        p1 = conv2(xl,rot90(rot90(temp)),'valid');
        %% part 2
        Nkxl = padarray(Nkx, [pd pd], 'both', 'symmetric');
        p2 = conv2(Nkxl,rot90(rot90(v)),'valid');

        gk = p1 + rot90(rot90(p2));
        grad_ln_beta(:,i) = -(eye(m) - cof_beta(:,i)*cof_beta(:,i)'/f_norms(i)^2)/f_norms(i)*basis'*gk(:);
        
        grad_ln_ws(:,i) = -GW*t;
    end
    grad_loss_beta = grad_loss_beta + grad_ln_beta;
    grad_loss_weights = grad_loss_weights + grad_ln_ws;
end
grad_loss_p = -p*sum(sum((input-noisy).*grad_l_x));
grad_loss_b = beta*sum(sum(diffterm.*grad_l_x));
grad = [grad_loss_beta(:);grad_loss_p;grad_loss_weights(:);grad_loss_b];
% t3 = toc