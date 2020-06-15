%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     A try of GMM estimation : gather daat for NKPC  %
%                 F.GAO    Created@ 06/15/2020        %
%                      Last modify@ 06/15/2020        %
%    data to gather:                                  %
%        ls:  labor compensation in GDP  anually      %
%         1950/1/1-2017/1/1                           %
%        pi:  inflation  monthly                      %
%         1957/1/1-2020/5/1                           %
%        gap: first order difference log GDP quart    %
%         1947/1/1-2020/01/01                         %
%    pick one observation in 12 month data (bias?)    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ls = xlsread('Labor_share_raw_from_FED.csv');
pi = xlsread('CPI_raw_from_FED.csv');
gap = xlsread('GDP_fed.csv');
for k = 1:length(gap)
new_pi(k,:) = pi((1992-1957)*12+ 1 + 12*(k-1));
new_ls(k,:) = ls((1992-1950)+1 + (k-1));
end
data = [new_pi gap new_ls];
clearvars -except data;
save('NKPC')

