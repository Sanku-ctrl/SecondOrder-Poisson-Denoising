function cofQ = SymmetricPadQcof(Q, sign_ud, sign_lr, sign_lrud, bs)
[r,c] = size(Q);
Nr = r/bs; Nc = c/bs;
cofQ = zeros(r+2*bs, c+2*bs);
cofQ(1+bs:end-bs,1+bs:end-bs) = Q;
cofQ(1:bs,1+bs:end-bs) = cofQ(1+bs:2*bs,1+bs:end-bs).*repmat(sign_ud,[1, Nc]);
cofQ(end-bs+1:end,1+bs:end-bs) = cofQ(end-2*bs+1:end-bs,1+bs:end-bs).*repmat(sign_ud,[1, Nc]);
cofQ(1+bs:end-bs,1:bs) = cofQ(1+bs:end-bs,1+bs:bs+bs).*repmat(sign_lr,[Nr, 1]);
cofQ(1+bs:end-bs,end-bs+1:end) = cofQ(1+bs:end-bs, end-2*bs+1:end-bs).*repmat(sign_lr,[Nr, 1]);
cofQ(1:bs,1:bs) = cofQ(1+bs:bs+bs,1+bs:bs+bs).*sign_lrud;
cofQ(end-bs+1:end,1:bs) = cofQ(end-2*bs+1:end-bs,1+bs:bs*2).*sign_lrud;
cofQ(1:bs,end-bs+1:end) = cofQ(1+bs:2*bs,end-2*bs+1:end-bs).*sign_lrud;
cofQ(end-bs+1:end,end-bs+1:end) = cofQ(end-2*bs+1:end-bs,end-2*bs+1:end-bs).*sign_lrud;