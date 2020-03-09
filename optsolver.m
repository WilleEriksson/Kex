tic
N = 4;
M = 5;
eta = 0.1;%1e-1;%1e-2;%1e-3;%*1e1;
epsilon1 = 0.01;%0.1;%0.01;%0.1*4/N^(3/2);

z = zeros(N,1);
e = ones(N,1);
E = ones(N,N)/N;
S = find(eye(N,N));


X = randi([0, 1], [N,M]);
Y = X;

for iter=1:5
    cvx_begin % quiet
        variable L(N,N) symmetric
        W = L-diag(diag(L));
        minimize( trace(Y'*L*Y) + sum(sum(L.^2)));
        subject to
            trace(L) == N;
            W(:) <= 0;
            L*e == z;
    cvx_end
    
    Y =(eye(N)+L) \ X;
end
clear
toc