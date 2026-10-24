function output = llc_neural_ode_forward(model, input)
    % LLC_NEURAL_ODE_FORWARD Evaluate the exported normalized vector field.
    if ~isstruct(model) || ~isscalar(model) || ~isfield(model, 'network')
        error('llc:InvalidNeuralODE', 'model must be a loaded Neural ODE export.');
    end
    input = double(input);
    if ~ismatrix(input) || isempty(input) || size(input, 2) ~= 9 || any(~isfinite(input), 'all')
        error('llc:InvalidNeuralODEInput', 'input must be a finite N-by-9 matrix.');
    end
    output = input;
    network = model.network;
    for index = 1:numel(network.weights)
        output = output * network.weights{index}.' + network.biases{index}.';
        if strcmp(network.activations{index}, 'SiLU')
            output = output ./ (1 + exp(-output));
        elseif ~strcmp(network.activations{index}, 'linear')
            error('llc:InvalidNeuralODE', 'Unsupported activation in layer %d.', index);
        end
    end
    if size(output, 2) ~= 4 || any(~isfinite(output), 'all')
        error('llc:InvalidNeuralODEOutput', 'Network output must be finite and four-dimensional.');
    end
end
