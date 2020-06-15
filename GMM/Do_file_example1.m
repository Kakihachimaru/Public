%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     A try of GMM estimation : Example 1             %
%                 F.GAO    Created@ 06/14/2020        %
%                      Last modify@ 06/14/2020        %
%    Parameter to be estimated:                       %
%                              beta                   %
%    Equation to estimate:                            %
%                          Euler Equation             %
%Simple case:                                         %
%     One equation, One parameter                     %
%Goal:                                                %
%     GMM structure, loss function                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


load('dada.mat');

model.data = (log(data));
model.number_equation = 1;
model.MatlabDisp = 'off';
model.initialguess = [0.5];
%%
result = doestimate(model)
%%

function result = doestimate(model)
	if model.number_equation < length(model.initialguess)
		return
	end	
	W = eye(model.number_equation);
	initial_parameter = model.initialguess;
	%options = optimset('LargeScale', 'off', 'MaxIter', 2500, 'MaxFunEvals', 3500, 'Display', model.MatlabDisp, 'TolFun', 1e-40, 'TolX', 1e-40); 
	[parameter, Fval, Exitflag] =  fminsearch(@(parameter) GMM(parameter, model, W), initial_parameter);%, options); 
    result = parameter;
	
	%
	function J = GMM(parameters, model, W)
	data = model.data;
	data_t1 = data(2:end);
	data_t = data(1:end-1);
	number_observations = length(data_t);
	b = parameters(1,1);
    r = 0.04;
	g = 1/(number_observations-1) * sum(b * (data_t1 ./ data_t).^(-1) * (1+r)-1);
	J = g*W*g';
	end
end
