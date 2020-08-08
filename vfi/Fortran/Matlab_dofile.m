%%%%%%
% Know more about RISE tool box, visit
% https://github.com/jmaih/RISE_toolbox/issues
%%%%%%
%%%start%%%
is = load('simulation_is_t.dat');
eta = load('simulation_eta.dat');
k = load('simulation_k.dat');
c = load('simulation_c.dat');
l = load('simulation_l.dat');
y = load('simulation_y.dat');
i = load('simulation_i.dat');
r = load('simulation_r.dat');
w = load('simulation_w.dat');
%names = {'eta' 'k' 'c' 'l' 'y' 'i' 'r' 'w'};
mynames = {'c','i','k','l','r','w','A','y'};
system_varialbes=[c i k l r w eta y];
%%
E_system_varialbes=compte_mean(system_varialbes)
%% need RISE tool box for this line %%
sims = pages2struct(ts('1',system_varialbes,mynames))
%%
%% plot the data
mynames=fieldnames(sims);

mynames=mynames-mynames(strncmp(mynames,'du9',3));

figure('name','database');
for ii=1:numel(mynames)
    v=mynames{ii};
    subplot(2,4,ii)
    %plot(sims.(v))
    plot(sims.(v).data(end-500:end))
    % if you dont have toolbox, use
    % plot(system_varialbes(end-500:end,ii))
    %
    title(v)
end
xrotate(45)
%%
X = system_varialbes;
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




function E_system_varialbes=compte_mean(system_varialbes)
E_system_varialbes = 0;
for i = 1:length(system_varialbes) 
E_system_varialbes=E_system_varialbes*(i-1)/i + system_varialbes(i,:)/i;
end
E_system_varialbes(2)= E_system_varialbes(2)/4;
E_system_varialbes(7)=(1+E_system_varialbes(7))^4-1;
% ugly
 i = 2;
E_system_varialbes(i)=E_system_varialbes(i)/E_system_varialbes(5)*100;
 i = 3;
E_system_varialbes(i)=E_system_varialbes(i)/E_system_varialbes(5)*100;
 i = 6;
E_system_varialbes(i)=E_system_varialbes(i)/E_system_varialbes(5)*100;
end
