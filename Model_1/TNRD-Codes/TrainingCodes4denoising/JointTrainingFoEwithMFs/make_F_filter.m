function F = make_F_filter(filter,img_row,img_col)
% filter: column vector
n = size(filter,1);
filter_size = sqrt(n);
row = zeros(n*img_row*img_col,1);
col = zeros(n*img_row*img_col,1);
var = repmat(filter,[img_row*img_col,1]);

for j = 1:img_col % column
    for i = 1:img_row % row
        deta = (filter_size - 1)/2;
        index = zeros(n,1);
        for filter_j = -deta:1:deta
            for filter_i = -deta:1:deta
                ii = i + filter_i;
                jj = j + filter_j;
                if(ii <= 0)
%                     ii = 1 - ii;
                    ii = ii + img_row;
                end
                if(ii > img_row)
                    ii = ii - img_row;
                end
                if(jj <= 0)
                    jj = jj + img_col;
                end
                if(jj > img_col)
                    jj = jj - img_col;
                end
                index((filter_j + deta) * filter_size + filter_i + deta + 1) = ...
                    ii + (jj - 1) * img_row;
            end
        end
        idx = i + (j - 1) * img_row;
        row((idx - 1) * n + 1:idx * n) = idx;
        col((idx - 1) * n + 1:idx * n) = index;
    end
end
F = sparse(row,col,var,img_row*img_col,img_row*img_col);