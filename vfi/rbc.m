%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  A simple RBC model                                 %
%        Value function iteration                     %
%              iterate over state variable            %
%                     F.GAO. @8/7/2020                %
%              Last modify   @8/7/2020                %
%                                                     %
%                                                     %
%   Current problem:                                  %
%1.Discrete grid of capital imples Nans in labor grid,%
%2.function converged but capital on ss different     %
%  from benchmark                                     %
%3.bad efficiency in fzero: deal with nans            %
%4.debug in unefficient way (crucial)                 %
%                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%start%
%%%%%%settings%%%%%%
%assign value to parameters
betta = 0.99;
delta = 0.03;
gamma = 2;
rho = 0.95;
sig = 0.000049;
alpha = 0.4;
nu = 0.36;
%set capital interval
k_inf = 10;
k_sup = 20;
%dimensions
grid_number_k = 101;
grid_number_A = 21;
total_dimension = grid_number_A * grid_number_k;
%AR process to markov process
[P, interval]=AR2Pi(rho,sig,grid_number_A)
%set technology interval
A = exp(interval);

%%%%%%making_grids%%%%%%
%capital
k_interval = k_inf:(k_sup-k_inf)/(grid_number_k-1):k_sup;
k_grid = repmat(k_interval',[grid_number_A,1,grid_number_k]);
k_grid = reshape(k_grid,total_dimension,grid_number_k);
k_nextday_index = repmat(1:grid_number_k,total_dimension,1); 
k_nextday = k_interval(k_nextday_index);
k_nextday = (k_nextday);
%set fzero option
options = optimset('Display','off');
%%
%labor and consumption
%?%
[a,b]=size(k_grid);
l_grid=NaN(a,b);
c_grid=NaN(a,b);
%1%
tic
for ii = 1:grid_number_A
A_t = A(ii);

	for i = 1 + (ii-1)*grid_number_k :  ii*grid_number_k
		parfor j = 1:grid_number_k
            k_t = k_grid(i,j);
            k_t1 = k_nextday(i,j);
            foc_l = @(l_t,k_t,k_t1,A_t,nu,alpha,delta) (A_t * k_t^alpha * l_t ^(1-alpha) - k_t1 + (1-delta)*k_t )...
*(1-nu) - nu * (1-alpha) * (1-l_t) * A_t * k_t^alpha * l_t ^(-alpha);
            find_l = @(l_t)foc_l(l_t,k_t,k_t1,A_t,nu,alpha,delta);
            l_grid(i,j)=fzero(find_l,x0,options);
            c_grid(i,j)=A_t * k_t^alpha * l_grid(i,j) ^(1-alpha) - k_t1 + (1-delta)*k_t; 
        end
	end
end
toc

%2%skip all nan
tic
l_min = 0.000001;
l_max = 0.999999;
k0 = 25;
for ii = 1:grid_number_A
A_t = A(ii);
for i = 1 + (ii-1)*grid_number_k:ii*grid_number_k
    k_t = k_grid(i,1);
    %
    foc_lmin = @(l_min,k_t,k_t1,A_t,nu,alpha,delta) (A_t * k_t^alpha * l_min ^(1-alpha) - k_t1 + (1-delta)*k_t )...
*(1-nu) - nu * (1-alpha) * (1-l_min) * A_t * k_t^alpha * l_min ^(-alpha);
    foc_lmax = @(l_max,k_t,k_t1,A_t,nu,alpha,delta) (A_t * k_t^alpha * l_max ^(1-alpha) - k_t1 + (1-delta)*k_t )...
*(1-nu) - nu * (1-alpha) * (1-l_max) * A_t * k_t^alpha * l_max ^(-alpha);
    find_k_low = @(k_t1)foc_lmin(l_min,k_t,k_t1,A_t,nu,alpha,delta);
    find_k_up = @(k_t1)foc_lmax(l_max,k_t,k_t1,A_t,nu,alpha,delta);
    k_low = max(k_inf,fzero(find_k_low,k0,options)-1);
    k_max = min(k_sup,fzero(find_k_up,k0,options)+1);
    j_start = find(k_interval<= fix(k_low),1,'first');
    j_end = find(k_interval >= round(k_max),1,'first' );
    
    parfor j = j_start:j_end
		k_t1 = k_nextday(i,j);
        foc_l = @(l_t,k_t,k_t1,A_t,nu,alpha,delta) (A_t * k_t^alpha * l_t ^(1-alpha) - k_t1 + (1-delta)*k_t )...
*(1-nu) - nu * (1-alpha) * (1-l_t) * A_t * k_t^alpha * l_t ^(-alpha);
        find_l = @(l_t)foc_l(l_t,k_t,k_t1,A_t,nu,alpha,delta);
		l_grid(i,j)=fzero(find_l,0.5,options);
		c_grid(i,j)=A_t * k_t^alpha * l_grid(i,j) ^(1-alpha) - k_t1 + (1-delta)*k_t;
                %if isnan(l_grid(i,j)) 
                %    continue
                %end
    end
end
end
toc
%%
%4%
At=repmat(A,[grid_number_k,1,1]);
At=reshape(At,[],1);
At=repmat(At,[1,grid_number_k,1]);
%
l_t=1;
k_t1 = k_nextday;
k_t = k_grid;
check_foc = (At .* k_t.^alpha .* l_t ^(1-alpha) - k_t1 + (1-delta)* k_t )...
*(1-nu) - nu * (1-alpha) * (1-l_t) * At .* k_t.^alpha .* l_t .^(-alpha);

check_foc=logical(check_foc>=0);
sum(check_foc)

[a,b] = size(k_grid);
index_foc_labor = [1:a*b];
index_foc_labor = index_foc_labor(check_foc);


l_t_try = zeros(a,b);
l_t_try = l_t_try(check_foc);

l_grid=NaN(a,b);

A_t_try = At(check_foc);
k_t1_try = k_nextday(check_foc);
k_t_try = k_grid(check_foc);

tic 
parfor i=1 : length(A_t_try)
A_t = A_t_try( i );
k_t1 = k_t1_try( i );
k_t = k_t_try( i );

foc_l = @(l_t,k_t,k_t1,A_t,nu,alpha,delta) (A_t .* k_t.^alpha .* l_t .^(1-alpha) - k_t1 + (1-delta)*k_t )...
*(1-nu) - nu * (1-alpha) .* (1-l_t) .* A_t .* k_t.^alpha .* l_t .^(-alpha);
        find_l = @(l_t)foc_l(l_t,k_t,k_t1,A_t,nu,alpha,delta);
        l_t(i,1) = fzero(find_l,.5,options);
end
toc

l_grid(index_foc_labor) = l_t;
c_grid = NaN(a,b);
c_grid=At .* k_grid.^alpha .* l_grid .^(1-alpha) - k_nextday + (1-delta) * k_grid;

l_interval = 0:1/99:1;

%utility
u_grid = (c_grid.^nu.*(1-l_grid).^(1-nu)).^(1-1/gamma) ./ (1-1/gamma);
u_grid(isnan(u_grid))=-inf;
%check
max(c_grid,[],2);
length(find(ans<0))
max(u_grid,[],2);
sum(ans,2);
length(find(ans==grid_number_k))
%%
%%%%%%iteration setting %%%%%%
%value grid
v = zeros(grid_number_A,grid_number_k);
%index
index_k2total = (repmat(1:grid_number_k,total_dimension,1));
index_k = (repmat((1:grid_number_A)',grid_number_k,grid_number_k));
Itry = sub2ind([grid_number_A grid_number_k],index_k,index_k2total);
Itry = (Itry);
dpixold = 0;

%%%%%%gpu computation%%%%%%
v = gpuArray(v);
pai = gpuArray (P);
u_grid = gpuArray(u_grid);

%%%%%%iteration%%%%%%
dist = 1;
while dist> 1e-8

ev = pai *  v;

Evptry = ev(Itry);

clear ev

[vnew, dpix] = max(u_grid+betta*Evptry,[],2);

dist = max(abs(vnew(:)-v(:))) + max(abs(dpix(:)-dpixold(:)))

v = vnew;

v = reshape(v,grid_number_A,grid_number_k);

dpix = reshape(dpix,grid_number_A,grid_number_k);
dpixold = dpix;

end 

%reshape index matrix, 
%dpixx=reshape(dpix,grid_number_A,grid_number_k);
%dpixx=dpixx';
k=k_interval(dpix);
%   %%
%  for i=1:21
%      for j =1:500
%          index = dpixx(i,j); 
%          l_policy(i,j) =  l_grid(j+(i-1)*500,index);
%      end
%  end
%%
%%%%%%simulation%%%%%%
%initials
k_t = k_interval(round(grid_number_k/2));
A_t = A(round(grid_number_A/2));
Cpai = cumsum(P,2);
X = [];
%loop
for n=1:20000
    i = find(A==A_t);
    j = find(k_interval==k_t);
    k_t1 = k(i,j);
    
    L = sum(Cpai(i,:)<rand)+1;
    A0 = A(L);
    
    
%    foc_l = @(l_t,k_t,k_t1,A_t,nu,alpha,delta) (A_t * k_t^alpha * l_t ^(1-alpha) - k_t1 + (1-delta)*k_t )...
% *(1-nu) - nu * (1-alpha) * (1-l_t) * A_t * k_t^alpha * l_t ^(-alpha);
%        find_l = @(l_t)foc_l(l_t,k_t,k_t1,A_t,nu,alpha,delta);
% 		l=fzero(find_l,.5);
    
    jj = j + (i-1) * grid_number_k;
    ii = k_interval==k_t1;
    l = l_grid(jj,ii);
    
    y = A_t * k_t ^ alpha * l ^(1-alpha);
    c = y + (1-delta) * k_t - k_t1;
    i = y - c;
    r = alpha * A_t * k_t ^ alpha * l ^(1-alpha);
    w = (1-alpha) * A_t * k_t ^ alpha * l ^(-alpha);
    k_t = k_t1;
    A_t = A0;
    if n >10000
        X = [X; c i k_t l r w A_t y];
    end
end
%plot
mynames = {'c','i','k','l','r','w','A','y'};
figure('name','simulation');
for ii=1:numel(mynames)
    v=mynames{ii};
    subplot(2,4,ii)
    %plot(X(:,ii))
    plot(X(end-500:end,ii))
    title(v)
end
xrotate(45)
%statistics
mean_X = mean(X);
mean_X(3) = mean_X(3)/4;
mean_X(5) = (1 + mean_X(5))^4 -1;

mean_X(1:3) = mean_X(1:3)/mean_X(end)*100;
mean_X(4) = mean_X(4)*100;
cv_X = std(X)./mean(X);
cv_X_GDP = cv_X./cv_X(end);
for i = 1:7
    corr_X(i)=corr(X(:,i),X(:,end));
end
corr_X(8) = 1;
rows={'mean';'CV';'CV (in % of CV of gdp)';'corr. with gdp'};
table_me =  array2table([mean_X;cv_X;cv_X_GDP;corr_X],'VariableNames',mynames);
table_me2 = cell2table(rows,'VariableNames',{'Moments'});
table = [table_me2 table_me]
%%
bar_A = median(A);
kstar = (1/betta-1+delta)/median(A)/alpha;
kstar = kstar^(1/(alpha-1));
hstar = nu * (1-alpha) * bar_A * kstar^alpha;
hstar = hstar/((1-nu*alpha)*bar_A*kstar^alpha - (1-nu)*delta*kstar);
Cstar = (bar_A*kstar^alpha-delta*kstar)*hstar;
ss = [ kstar hstar Cstar ]

