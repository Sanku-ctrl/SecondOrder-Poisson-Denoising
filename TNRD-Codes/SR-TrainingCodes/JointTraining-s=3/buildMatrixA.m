function A = buildMatrixA(data, param)
% calculate pixel size
psize_coarse = param.scale;
% Blur ----------------
% sigma estimate based on scale
sigma = 0.25*sqrt(param.scale.^2 - 1); 
ksize = ceil(3*sigma);

ksize = ksize + 1 - mod(ksize,2); % make kernel size odd
kernel = fspecial('gaussian', [ksize], max(sigma));

% Initialize vectors
cin  = zeros(1,data.size_fine(1)*data.size_fine(2)*size(kernel,1)*size(kernel,2));
cout = zeros(1,data.size_fine(1)*data.size_fine(2)*size(kernel,1)*size(kernel,2));
val  = zeros(1,data.size_fine(1)*data.size_fine(2)*size(kernel,1)*size(kernel,2));
cnt = 1;

% for each pixel
for x=1:data.size_fine(2)
    for y=1:data.size_fine(1)
        hkl = floor(ksize/2);
        
        % compute sum of kernel
        ksum = 0;
        for dx=-hkl(2):hkl(2)
            for dy=-hkl(1):hkl(1)
                cx = x+dx;
                cy = y+dy;
                if (cx>0 && cy>0 && cx<=data.size_fine(2) && cy<=data.size_fine(1))
                    ksum = ksum+kernel(hkl(1)+dy+1, hkl(2)+dx+1);
                end
            end
        end
        
        % add coefficients
        for dx=-hkl(2):hkl(2)
            for dy=-hkl(1):hkl(1)
                cx = x+dx;
                cy = y+dy;
                if (cx>0 && cy>0 && cx<=data.size_fine(2) && cy<=data.size_fine(1))
                    cin(cnt) = y+(x-1)*data.size_fine(1);
                    cout(cnt) = cy+(cx-1)*data.size_fine(1);
                    val(cnt) = kernel(hkl(1)+dy+1, hkl(2)+dx+1)/ksum;
                    cnt = cnt+1;
                end
            end
        end
    end
end

% trim list
cin = cin(1, 1:cnt-1);
cout = cout(1, 1:cnt-1);
val = val(1, 1:cnt-1);

% create matrix
B = sparse(cin,cout,val,data.size_fine(1)*data.size_fine(2),data.size_fine(1)*data.size_fine(2));

% Downsample ----------------

% Initialize vectors
cin  = zeros(1,data.size_fine(1)*data.size_fine(2)*ceil(param.scale(1))*ceil(param.scale(2)));
cout = zeros(1,data.size_fine(1)*data.size_fine(2)*ceil(param.scale(1))*ceil(param.scale(2)));
val  = zeros(1,data.size_fine(1)*data.size_fine(2)*ceil(param.scale(1))*ceil(param.scale(2)));
cnt = 1;

% for each pixel
for x=0:data.size_coarse(2)-1
    for y=0:data.size_coarse(1)-1
        % Calculate coordinates of pixel in coarse image
        lx = x*psize_coarse(2);
        ly = y*psize_coarse(1);
        rx = lx+psize_coarse(2);
        ry = ly+psize_coarse(1);
        
        % compute sum of kernel
        ksum = 0;
        for cx=floor(lx):floor(rx)
            for cy=floor(ly):floor(ry)
                if (cx>=0 && cy>=0 && cx<data.size_fine(2) && cy<data.size_fine(1))
                    ov_x = min(rx, cx+1) - max(lx, cx);
                    ov_y = min(ry, cy+1) - max(ly, cy);
                    ksum = ksum + ov_x*ov_y;
                end
            end
        end
        
        % add coefficients
        for cx=floor(lx):floor(rx)
            for cy=floor(ly):floor(ry)
                if (cx>=0 && cy>=0 && cx<data.size_fine(2) && cy<data.size_fine(1))
                    ov_x = min(rx, cx+1) - max(lx, cx);
                    ov_y = min(ry, cy+1) - max(ly, cy);
                    
                    cin(cnt) = y+x*data.size_coarse(1)+1;
                    cout(cnt) = cy+cx*data.size_fine(1)+1;
                    val(cnt) = ov_x*ov_y/ksum;
                    cnt = cnt+1;
                end
            end
        end
    end
end

% trim list
cin = cin(1, 1:cnt-1);
cout = cout(1, 1:cnt-1);
val = val(1, 1:cnt-1);

% create matrix
D = sparse(cin,cout,val,data.size_coarse(1)*data.size_coarse(2),data.size_fine(1)*data.size_fine(2));
A = D*B;
% A = D;