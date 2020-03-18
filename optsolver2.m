tic

X = readNPY('regions1.npy');
Y = X;

N = length(X)
eta = 0.1;%1e-1;%1e-2;%1e-3;%*1e1;
epsilon1 = 0.01;%0.1;%0.01;%0.1*4/N^(3/2);

z = zeros(N,1);
e = ones(N,1);
E = ones(N,N)/N;
S = find(eye(N,N));


%X = randi([0, 1], [N,M]);
%X = [1,0,1;0,1,1;0,0,1;1,0,0;1,0,1];


Yv = Y*Y';
Yv = reshape(Yv,[N^2,1]);

for iter=1:10
    iter
    cvx_begin quiet
        variable L(N,N) symmetric
        W = L-diag(diag(L));
        %Lvec = reshape(L,[N^2,1]);
        F = eta*sum(sum_square_abs( L ));
        minimize( trace(Y'*L*Y) + F);
        subject to
            trace(L) == N;
            W(:) <= 0;
            L*e == z;
    cvx_end
    
    Y =(eye(N)+L) \ X;
end

writeNPY(L,'Laplacian.npy')

% for iter=1:8
%     Y;
%     cvx_begin quiet
%         variable L(N,N) symmetric
%         W = L-diag(diag(L));
%         
%         F = norm(L,'fro');
%         %F = sum(sum_square(L));
%         
%         minimize( trace(Y'*L*Y) + F);
%         subject to
%             trace(L) == N;
%             W(:) <= 0;
%             L*e == z;
%     cvx_end
%     
%     Y =(eye(N)+L) \ X;
% end


clear
toc