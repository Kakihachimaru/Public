%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     A try of GMM estimation : Example 2             %
%                 F.GAO    Created@ 06/14/2020        %
%                      Last modify@ 06/14/2020        %
%    Parameter to be estimated:                       %
%discounter elasticity of intertemporal substitution  %
%    Equation to estimate:                            %
%                          Euler Equation             %
%     One equation, two parameter                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


load('dada.mat');

model.data = diff(log(data));
model.number_equation = 2;
model.MatlabDisp = 'off';
model.initialguess = [0.5; 1];
%%
result = doestimate(model)
%%

function result = doestimate(model)
	if model.number_equation < length(model.initialguess)
		return
	end	
	
	W = eye(model.number_equation);
	initial_parameter = model.initialguess;
	options = optimset('LargeScale', 'off', 'MaxIter', 2500, 'MaxFunEvals', 3500, 'Display', model.MatlabDisp, 'TolFun', 1e-40, 'TolX', 1e-40); 
	[parameter, Fval, Exitflag] =  fminsearch(@(parameter) GMM(parameter, model, W), initial_parameter, options); 
    result = parameter;
	
	function J = GMM(parameters, model, W)
	data = model.data;
	data_t1 = data(3:end);
	data_t = data(2:end-1);
    data_iv = data(1:end-2);
    
	number_observations = length(data_t);
	b = parameters(1,1);
	gamma = parameters(2,1);
    r = 0.04;
	g1 =  sum(b * (data_t1 ./ data_t).^(-gamma) * (1+r)-1);
	g2 =  sum(data_iv .* (b * (data_t1 ./ data_t).^(-gamma) * (1+r)-1));
	g1 = 1/(number_observations-1) * g1;
	g2 = 1/(number_observations-1) * g2;
    g = [g1 g2];
	J = g*W*g';
	end
end
