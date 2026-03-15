clear all; close all;
load trainingWithMFs_100_circular.mat;

BASIS = gen_dct2(5);
BASIS = BASIS(:,2:end);
for s = 1:stage
    upper = upper_s(s);
    space = space_s(s);
    q = 0:space:upper;
    pbins = opt_pbins{s};
    
    KernelPara.fsz = 5;
    KernelPara.filtN = 24;
    KernelPara.basis = BASIS;
    KernelPara.cof = cof(:,s);
    
    showMFsFilters(pbins, q,KernelPara);
    pause;
    close all;    
end
