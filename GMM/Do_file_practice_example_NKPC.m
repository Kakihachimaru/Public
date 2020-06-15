%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     A try of GMM estimation : NKPC                  %
%                 F.GAO    Created@ 06/14/2020        %
%                      Last modify@ 06/15/2020        %
%    Parameter to be estimated:                       %
%             discounter factor                       %
%             calvo price possibility                 %
%    Equation to estimate:                            %
%                          Nkpc                       %
%    Goal:repilcate legit result                      %
%    Ref: Gali and Gertler 1999                       %
%    Result from ref:                                 %
%                                                     %
%        pi_t = -0.016 gap_t + 0.988 E_t(Pi_{t+1})    %
%                                                     %
%    Problem:                                         %
%              I haven't read paper carefully         %
%                data set period may different.       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('NKPC.mat');
model=struct;
model.data = log((data));
%%
model.number_equation = 2;
model.initialguess = [0.5; 2];
model.number_parameters = length(model.initialguess);
model.iteration = 15;
model.q = 1;
%%
result = doestimate(model);
disp('parameters')
result.parameter
disp('stds')
result.variance_parameter
%%

function result = doestimate(model)
	if model.number_equation < length(model.initialguess)
		return
	end	
	
	W = eye(model.number_equation);
	initial_parameter = model.initialguess;
	[parameter, Fval, Exitflag] =  fminsearch(@(parameter) GMM(parameter, model, W), initial_parameter);
    
	if model.iteration > 1
		for iii = 1 : model.iteration
			W = optimal_weighting_matrix(parameter,model);
			initial_parameter = parameter;
			[parameter, Fval, Exitflag] =  fminsearch(@(parameter) GMM(parameter, model, W), initial_parameter);
		end
	end	
	
	d = GMMjacobian(parameter, model);
	
	variance_parameter = diag(inv(d'*W*d))./(length(model.data)-1);	
	result.parameter = parameter;
	result.variance_parameter=variance_parameter;

	function J = GMM(parameters, model, W)
	data = model.data;
	
	pi_t1 = data(2:end,1);
	pi_t = data(1:end-1,1);
	mc = data(1:end-1,2);
	
	number_observations = length(pi_t);
	a = parameters(1,1);
	b = parameters(2,1);
	
	g1 =  sum(pi_t - a * pi_t1 - b * mc );
	g2 =  sum((pi_t - a * pi_t1 - b * mc ) .* pi_t1);
	g1 = 1/(number_observations-1) * g1;
	g2 = 1/(number_observations-1) * g2;
    g = [g1 g2];
	J = g*W*g';
	end
	
	function W = optimal_weighting_matrix(parameters,model)

	
		q = model.q;
		data = model.data;
	
			pi_t1 = data(2:end,1);
			pi_t = data(1:end-1,1);
			mc = data(1:end-1,2);
	
			number_observations = length(pi_t);
			a = parameters(1,1);
			b = parameters(2,1);
			Gamma = zeros(model.number_parameters,model.number_parameters,q+1);
		
		%
		
		g1 =  (pi_t - a * pi_t1 - b * mc );
		g2 =  (pi_t - a * pi_t1 - b * mc ).*pi_t1;

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
	
			pi_t1 = data(2:end,1);
			pi_t = data(1:end-1,1);
			mc = data(1:end-1,2);
	
		number_observations = length(pi_t);
		a = parameters(1,1);
		b = parameters(2,1);

%	g1 =  sum(pi_t - a * pi_t1 - b * mc );
%	g2 =  sum((pi_t - a * pi_t1 - b * mc ) .* pi_t1);

	g1a = -sum( pi_t1  );
	g1b = -sum( mc );
	
	g2a = -sum( pi_t1.*pi_t1  );
	g2b = -sum(mc .* pi_t1);
	d = [g1a g1b; g2a g2b]/(number_observations-1);
	end
end
