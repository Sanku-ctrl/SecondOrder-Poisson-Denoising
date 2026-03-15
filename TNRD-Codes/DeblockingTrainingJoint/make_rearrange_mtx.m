function D = make_rearrange_mtx(M,N)
bsz = 8;
m = M/bsz;
n = N/bsz;
DCT = gen_dct2(bsz)';

Rij = cell(m,n);
R = [];
for j = 1:n
    for i = 1:m
        row = zeros(bsz^2,1);
        col = zeros(bsz^2,1);
        val = zeros(bsz^2,1);
        for c = 1:bsz
            for r = 1:bsz
                ind = (c-1)*bsz + r;
                row(ind) = ind;
                pr = bsz*(i-1) + r;
                pc = bsz*(j-1) + c;
                col(ind) = pr + (pc-1)*M;
                val(ind) = 1;
            end
        end
        Rij{i,j} = sparse(row,col,val,bsz^2,M*N);
        R = [R;Rij{i,j}];
    end
end

pn = m*n;
K = cell(1,pn);
for i=1:pn
    K{i} = DCT;
end
K = blkdiag(K{:});
D = sparse(R'*K*R);