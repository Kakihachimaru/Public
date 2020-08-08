function [P, interval]=AR2Pi(rho,sig,M)

p = (1+rho)/2;
A = [p 1-p;1-p p];

sig = sig / (1-rho^2);
sig = sqrt(sig);
yN = sig * sqrt(M-1);
interval = -yN:2*yN/(M-1):yN;

for i=3:M
B = zeros(i);
C = zeros(i);
D = zeros(i);
E = zeros(i);

B(1:i-1,1:i-1) = A;
C(1:i-1,2:i) = A;
D(2:i,1:i-1) = A;
E(2:i,2:i) = A;

A = p * B + (1-p)*C + (1-p) * D + p * E;
A(2:end-1,:) = A(2:end-1,:)./2;
end
%[a b] =eig(A');
%distribution = a(:,1);
%distribution = distribution/sum(distribution);
%figure
%plot(interval,distribution)
P=A;
%clearvars -except P interval
end
