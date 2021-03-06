%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     A try of GMM estimation : Example 4             %
%                 F.GAO    Created@ 06/14/2020        %
%                      Last modify@ 06/14/2020        %
%    Parameter to be estimated:                       %
%             discounter factor                       %
%           elasticity of intertemporal substitution  %
%    Equation to estimate:                            %
%                          Euler Equation             %
%    Goal:                                            %
%     Do T-test etc etc                               %
%    To test we need jacobin  (asymptotic nomality)   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


load('dada.mat');

model.data = diff(log(data));
model.number_equation = 2;
model.MatlabDisp = 'off';
model.initialguess = [0.5; 2];
model.number_parameters = length(model.initialguess);
model.iteration = 3;
model.q = 12;
%%
result = doestimate(model);
disp('parameters')
result.parameter
disp('stds')
result.variance_parameter

function result = doestimate(model)
	if model.number_equation < length(model.initialguess)
		return
	end	
	
	W = eye(model.number_equation);
	initial_parameter = model.initialguess;
	options = optimset('LargeScale', 'off', 'MaxIter', 2500, 'MaxFunEvals', 3500, 'Display', model.MatlabDisp, 'TolFun', 1e-40, 'TolX', 1e-40); 
	[parameter, Fval, Exitflag] =  fminsearch(@(parameter) GMM(parameter, model, W), initial_parameter, options); 
    
	if model.iteration > 1
		for iii = 1 : model.iteration
			W = optimal_weighting_matrix(parameter,model);
			initial_parameter = parameter;
			[parameter, Fval, Exitflag] =  fminsearch(@(parameter) GMM(parameter, model, W), initial_parameter, options);
		end
	end	
	
	d = GMMjacobian(parameter, model);
	
	variance_parameter = diag(inv(d'*W*d))./(length(model.data)-1);	
	result.parameter = parameter;
	result.variance_parameter=variance_parameter;

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
	g2 =  sum((data_iv)  .* (b * (data_t1 ./ data_t).^(-gamma) * (1+r)-1));
	g1 = 1/(number_observations-1) * g1;
	g2 = 1/(number_observations-1) * g2;
    
    g = [g1 g2];
	J = g*W*g';
	end
	
	function W = optimal_weighting_matrix(parameters,model)

	
        q = model.q;
		data = model.data;

        data_t1 = data(3:end);
        data_t = data(2:end-1);
        data_iv = data(1:end-2);
        
		number_observations = length(data_t);
		b = parameters(1,1);
		gamma = parameters(2,1);
		r = 0.04;
		Gamma = zeros(model.number_parameters,model.number_parameters,q+1);
		
		%
		
		g1 =  (b * (data_t1./ data_t).^(-gamma) * (1+r)-1);
		g2 =  ((data_iv) .* (b * (data_t1 ./ data_t).^(-gamma) * (1+r)-1));

		g = [g1 g2];
		% Newey-West estimation
		g = g - repmat(mean(g), number_observations, 1);
		for v = 0 : q
		gtF = g(1+v:end,:);
		gtL = g(1:end-v,:);
		Gamma(:,:,v+1) = (gtF'*gtL)./number_observations;
		end
		S = Gamma(:,:,1);
		for v = 1:q
		Snext = (1-v/(q+1))*(Gamma(:,:,v+1)+Gamma(:,:,v+1)');
		S = S + Snext;
		end
		W = inv(S);
	end

	function d = GMMjacobian(parameters, model)
	% jacobin manually
	
	data = model.data;

    data_t1 = data(3:end);
	data_t = data(2:end-1);
    data_iv = data(1:end-2);
    
	number_observations = length(data_t);
	b = parameters(1,1);
	gamma = parameters(2,1);
    r = 0.04;
    
	% g1 =  sum(b * (data_t1 ./ data_t).^(-gamma) * (1+r)-1);
	% g2 =  sum((data_iv).* (b * (data_t1 ./ data_t).^(-gamma) * (1+r)-1));


	g1b = sum((data_t1 ./ data_t).^(-gamma) * (1+r));
	g1gamma = -sum(b * (data_t1 ./ data_t).^(-gamma-1) * (1+r) * gamma);
	
	g2gamma = -sum((data_iv) .* (b * (data_t1 ./ data_t).^(-gamma-1) * (1+r)*gamma));
	g2b = sum((data_iv) .*((data_t1 ./ data_t).^(-gamma) * (1+r)));
    
	d = [g1b g1gamma; g2b g2gamma]/(number_observations-1);
	end
end
