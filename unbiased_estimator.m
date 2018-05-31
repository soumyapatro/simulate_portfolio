%Not a function file
[estimate_W, estimate_Wargs, W_actual, Wargs_actual] = output_analysis();

function [W_estimator, W_estimator_args]= optimize()
    p = 1 - (0.5)^(1.5);
    0 = 10;
    N = geornd(p) + n0 - 1;
    S0 = 2;
    F0 = 1;
    Jt = rand(1, 2^(N+1)) < 0.4;
    Zt = normrnd(0, 1,1,2^(N+1));
    S1 = risky_asset(Jt, Zt, S0);
    F1 = repelem(risk_free(F0),2^(N+1));
    S_odd = S1(1:2:end);
    F_odd = F1(1:2:end);
    S_even = S1(2:2:end);
    F_even = F1(2:2:end);
    S_n0 = S1(1:2^n0);
    F_n0 = F1(1:2^n0);
    initial_values = [0, 0];
    
    nonlincon = @nlcon;
    [argmax_tot, max_val_tot] = fmincon( @(params)get_mean(params, S1, F1),initial_values, [], [],[],[], zeros(1,2),[], nonlincon);
    [argmax_even, max_val_even] = fmincon( @(params)get_mean(params, S_even, F_even),initial_values, [], [],[],[], zeros(1,2),[],nonlincon);
    [argmax_odd, max_val_odd] = fmincon( @(params)get_mean(params, S_odd, F_odd),initial_values, [], [],[],[], zeros(1,2),[],nonlincon);
    [argmax_burn, max_val_burn] = fmincon( @(params)get_mean(params, S_n0, F_n0),initial_values, [], [],[],[], zeros(1,2),[], nonlincon);
    
    W_max_num =  max_val_tot - (max_val_even+ max_val_odd)/2;
    W_max_args_num =  argmax_tot - (argmax_even + argmax_odd)/2;
    W_den = geopdf(N,p);
    W_estimator = W_max_num/W_den + max_val_burn;
    W_estimator = -W_estimator;
    W_estimator_args = W_max_args_num/W_den + argmax_burn;
    %W_ub_estimator_cash = exp(W_ub_estimator)
end

%Iterate W* until error and confidence interval achieved
function [estimate_W, estimate_Wargs, W_actual, Wargs_actual] = output_analysis()
    num_sample = 200;
    error = 0.05;
    confidence = 0.95;
    delta = 1 - confidence;
    z = norminv(1 - delta/2);
    burn_in = 100;
    running_mean_W = 0;
    running_2moment_W = 0;
    running_mean_Wargs = 0;
    running_2moment_Wargs = 0;
    num_estimator = 0; % count of number of estimators generated
    
    CIs_W = zeros(1, num_sample);
    estimation_W = zeros(1, num_sample);
    W_actual = zeros(1, num_sample);
    CIs_Wargs = zeros(2, num_sample);
    estimation_Wargs = zeros(2, num_sample);
    Wargs_actual = zeros(2, num_sample);
    while num_estimator <= num_sample
        disp("Iteration" + num_estimator)
        [W_star, W_star_args] = optimize();
       
        running_mean_W = (running_mean_W * num_estimator + W_star) / (num_estimator+1);
        running_mean_Wargs = (running_mean_Wargs * num_estimator + W_star_args) / (num_estimator+1);
        running_2moment_W = (running_2moment_W * num_estimator + W_star.^2) / (num_estimator+1);
        running_2moment_Wargs = (running_2moment_Wargs * num_estimator + W_star_args.^2) / (num_estimator+1);
        
        sample_std_W = sqrt(running_2moment_W - running_mean_W.^2);
        sample_std_Wargs = sqrt(running_2moment_Wargs - running_mean_Wargs.^2);
        
        num_estimator = num_estimator + 1;
        
        confidence_interval_W = z * sample_std_W/(sqrt(num_estimator));
        confidence_interval_Wargs = z * sample_std_Wargs/(sqrt(num_estimator));
        
         W_actual(num_estimator) = W_star;
        Wargs_actual(:, num_estimator) = W_star_args;
        estimation_W(num_estimator) = running_mean_W;
        estimation_Wargs(:, num_estimator) = running_mean_Wargs;
        CIs_W(num_estimator) = confidence_interval_W;
        CIs_Wargs(:, num_estimator) = confidence_interval_Wargs;
        
        if(num_estimator>= ((z^2 * sample_std_W^2)/error^2) && num_estimator>=burn_in)
            disp("Number of samples = "+ num_estimator)
            break;
        end
    
    end
    
    lower_W = estimation_W - CIs_W;
    %lower_Wargs = estimation_Wargs - CIs_Wargs;
    upper_W = estimation_W + CIs_W;
    %upper_Wargs = estimation_Wargs + CIs_Wargs;
    fprintf('Generate %i samples \n', num_estimator);
    n_range = 1:num_estimator;
    plot(n_range, estimation_W(n_range), ...
        n_range,lower_W(n_range), n_range, upper_W(n_range), ...
        n_range, ones(size(n_range)));
    legend('estimation_W','lower CI', 'upper CI', 'true value');
    estimate_W = running_mean_W;
    
    %plot(n_range, estimation_Wargs(n_range), ...    n_range,lower_Wargs(n_range), n_range, upper_Wargs(n_range), ...    n_range, ones(size(n_range)));
    %legend('estimation_Wargs','lower CI', 'upper CI', 'true value');
    estimate_Wargs = running_mean_Wargs;
end


% Do one iteration of W*
function mean_U = get_mean(params,S, F)
    rho = 0.01;
    X = cash_out(params, S, F,rho);
    U = log(X);
    mean_U = -mean(U);
end


    
% Budget constraint
function [c,ceq] = nlcon(params)
    b = 200;
    S0 = 2;
    F0 = 1;
    rho = 0.01;
    c = [];
    ceq = params(1) * F0 + params(2) * S0 + params(2)^2 * rho* S0  - b; % = b
end

% Price of risky asset
function St = risky_asset(Jt, Zt, Stprev)
  %p = 0.4;
  mu0 = 0.1;
  sigma0 = 0.15;
  mu1 = 0.25;
  sigma1 = 0.3;
  %Bernoulli Random Variable
  %Jt = rand <= p ;% n is number of simulations. How to generate Bernoulli process?
  %Normal Random Variable
  %Zt = normrnd(0, 1);
  logRt = (mu0 + Zt .* sigma0).* Jt + (mu1 +Zt .* sigma1).* (1-Jt);
  St = Stprev * exp(logRt);
end

% Price of risk-free asset
function Ft = risk_free(Ftprev)
  r = 0.03;
  Ft = exp(r) * Ftprev;
end

% Cash out at t=1
function X = cash_out(params, S1, F1, rho)
  %disp(S1)
  %disp(F1)
  X = params(1) * F1 + params(2) * S1 - params(2)^2 * rho* S1;
end



