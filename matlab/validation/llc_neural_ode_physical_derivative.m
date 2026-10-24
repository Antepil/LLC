function derivative = llc_neural_ode_physical_derivative(timeS, state, model, gates, vinSignal)
    % LLC_NEURAL_ODE_PHYSICAL_DERIVATIVE Convert dz/dtau into physical dx/dt.
    state = double(state(:));
    gates = double(gates(:).');
    if numel(state) ~= 4 || any(~isfinite(state))
        error('llc:InvalidNeuralODEInput', 'state must contain four finite values.');
    end
    if numel(gates) ~= 4 || any(gates ~= 0 & gates ~= 1)
        error('llc:InvalidNeuralODEInput', 'gates must contain four binary values.');
    end
    vin = llc_evaluate_validation_signal(vinSignal, timeS);
    normalized = llc_neural_ode_forward(model, ...
        [state.' ./ model.state_scales, gates, vin / model.vin_reference_v]);
    derivative = (model.state_scales ./ model.time_reference_s .* normalized).';
    if any(~isfinite(derivative))
        error('llc:NonfiniteNeuralODEState', 'Neural ODE produced a nonfinite derivative.');
    end
end
