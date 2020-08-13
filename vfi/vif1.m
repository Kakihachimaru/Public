%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A stochastic growth model with                      %
%        Value function iteration                     %
%              iterate over state variable            %
%                     F.GAO. @8/10/2020               %
%              Last modify   @8/10/2020               %
%                                                     %
%                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%start
%%%%%%%setting%%%%%%
betta = 0.99;
delta = 0.03;
gamma = 2;
rho = 0.95;
sig = 0.000049;
alpha = 0.4;
nu = 0.36;
%set capital interval
k_inf = 10;
k_sup = 40;
%dimensions
grid_number_k = 501;
grid_number_A = 21;
total_dimension = grid_number_A * grid_number_k;
%AR process to markov process
[P, interval]=AR2Pi(rho,sig,grid_number_A)
%set technology interval
A = exp(interval);

%%%%%%make grids%%%%%%
k_interval = k_inf:(k_sup-k_inf)/(grid_number_k-1):k_sup;
k_grid = repmat(k_interval',[grid_number_A,1,grid_number_k]);
k_grid = reshape(k_grid,total_dimension,grid_number_k);
k_grid = gpuArray(k_grid);

k_nextday_index = repmat(1:grid_number_k,total_dimension,1); 
k_nextday = k_interval(k_nextday_index);
k_nextday = gpuArray(k_nextday);

c_grid = k_grid.^alpha;
for i = 1:grid_number_A
    c_grid(1:500 + (i-1)*500,:) = A(i) .*  c_grid(1:500 + (i-1)*500,:);
end

c_grid = c_grid + (1-delta).* k_grid - k_nextday;
u_grid = (c_grid.^(1-gamma) - 1) / (1-gamma);
u_grid(c_grid<=0)=-inf;
max(c_grid,[],2);
length(find(ans<0))

%%%%%%vif setting%%%%%%
%%%%%%GPU%%%%%%
v = zeros(grid_number_A,grid_number_k);
v = gpuArray(v);
index_k2total = gpuArray(repmat(1:grid_number_k,total_dimension,1));
index_k2 = gpuArray(repmat((1:grid_number_A)',grid_number_k,grid_number_k));

Itry = sub2ind([grid_number_A grid_number_k],index_k2,index_k2total);
Itry = gpuArray(Itry);

dpixold = 0;
dist = 1;

pai = gpuArray(P);




%%%%%%iteration%%%%%%
while dist> 1e-8

ev = pai *  v;

Evptry = ev(Itry);

clear ev

[vnew, dpix] = max(u_grid+betta*Evptry,[],2);

dist = max(abs(vnew(:)-v(:))) + max(abs(dpix(:)-dpixold(:)));

v = vnew;

v = reshape(v,grid_number_A,grid_number_k);
dpix = reshape(dpix,grid_number_A,grid_number_k);

dpixold = dpix;

end 
k=k_interval(dpix);
%%%%%%simulation%%%%%%
k_t = k_interval(round(grid_number_k/2));
A_t = A(round(grid_number_A/2));
Cpai = cumsum(P,2);
X = [];

for n=1:11000
    i = find(A==A_t);
    j = find(k_interval==k_t);

    L = sum(Cpai(i,:)<rand)+1;
    A0 = A(L);
    k_tomorrow = k(i,j);
    c = A_t * k_t ^ alpha + (1-delta) * k_t - k_tomorrow;
    
    k_t = k_tomorrow;
    A_t = A0;
    if n >10000
        X = [X; k_t A_t c];
    end
end
%%
figure
plot(X(:,1));
figure
plot(X(:,2));
figure
plot(X(:,3));
