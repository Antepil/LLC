function llc_neural_ode_sfun(block)
    % LLC_NEURAL_ODE_SFUN Continuous four-state Neural ODE for Simulink.
    setup(block);
end

function setup(block)
    block.NumDialogPrms = 1;
    block.NumInputPorts = 1;
    block.NumOutputPorts = 1;
    block.NumContStates = 4;
    block.SetPreCompInpPortInfoToDynamic;
    block.SetPreCompOutPortInfoToDynamic;
    block.InputPort(1).Dimensions = 5;
    block.InputPort(1).DatatypeID = 0;
    block.InputPort(1).Complexity = 'Real';
    block.InputPort(1).DirectFeedthrough = true;
    block.OutputPort(1).Dimensions = 4;
    block.OutputPort(1).DatatypeID = 0;
    block.OutputPort(1).Complexity = 'Real';
    block.SampleTimes = [0, 0];
    block.SimStateCompliance = 'DefaultSimState';
    block.RegBlockMethod('InitializeConditions', @initializeConditions);
    block.RegBlockMethod('Outputs', @outputs);
    block.RegBlockMethod('Derivatives', @derivatives);
end

function initializeConditions(block)
    runtime = validateRuntime(block.DialogPrm(1).Data);
    block.ContStates.Data = double(runtime.initial_state(:));
end

function outputs(block)
    state = double(block.ContStates.Data(:));
    if numel(state) ~= 4 || any(~isfinite(state))
        error('llc:NonfiniteNeuralODEState', ...
              'Neural ODE continuous state must contain four finite values.');
    end
    block.OutputPort(1).Data = state;
end

function derivatives(block)
    runtime = validateRuntime(block.DialogPrm(1).Data);
    input = double(block.InputPort(1).Data(:).');
    if numel(input) ~= 5 || any(~isfinite(input)) || ...
            any(input(1:4) ~= 0 & input(1:4) ~= 1) || ...
            (input(1) ~= 0 && input(2) ~= 0) || ...
            (input(3) ~= 0 && input(4) ~= 0) || input(5) <= 0
        error('llc:InvalidNeuralODEInput', ...
              'Neural ODE input must contain four binary gates and a finite positive Vin.');
    end
    vinSignal = struct('type', 'constant', 'value_v', input(5));
    value = llc_neural_ode_physical_derivative( ...
        block.CurrentTime, block.ContStates.Data, runtime.model, input(1:4), vinSignal);
    block.Derivatives.Data = value;
end

function runtime = validateRuntime(runtime)
    if ~isstruct(runtime) || ~isscalar(runtime) || ...
            ~all(isfield(runtime, {'model', 'initial_state'}))
        error('llc:InvalidNeuralODEInput', ...
              'llc_neural_runtime must contain model and initial_state.');
    end
    if isfield(runtime, 'is_placeholder') && ...
            (~islogical(runtime.is_placeholder) || ~isscalar(runtime.is_placeholder))
        error('llc:InvalidNeuralODEInput', ...
              'is_placeholder must be a logical scalar.');
    end
    if isfield(runtime, 'is_placeholder') && runtime.is_placeholder
        error('llc:MissingNeuralODEModel', ...
              'Inject a validated Neural ODE model before simulating this block.');
    end
    initial = runtime.initial_state;
    if ~isnumeric(initial) || ~isreal(initial) || numel(initial) ~= 4 || ...
            any(~isfinite(initial))
        error('llc:InvalidNeuralODEInput', ...
              'Neural ODE initial_state must contain four finite real values.');
    end
    model = runtime.model;
    required = {'network', 'state_scales', 'vin_reference_v', ...
                'time_reference_s', 'load_resistance_ohm'};
    if ~isstruct(model) || ~isscalar(model) || ~all(isfield(model, required)) || ...
            ~isnumeric(model.state_scales) || ~isreal(model.state_scales) || ...
            numel(model.state_scales) ~= 4 || any(~isfinite(model.state_scales)) || ...
            any(model.state_scales <= 0) || ...
            ~isnumeric(model.vin_reference_v) || ~isscalar(model.vin_reference_v) || ...
            ~isfinite(model.vin_reference_v) || model.vin_reference_v <= 0 || ...
            ~isnumeric(model.time_reference_s) || ~isscalar(model.time_reference_s) || ...
            ~isfinite(model.time_reference_s) || model.time_reference_s <= 0
        error('llc:InvalidNeuralODEInput', ...
              'Injected Neural ODE model has an invalid scaling contract.');
    end
    llc_neural_ode_forward(model, zeros(1, 9));
end
