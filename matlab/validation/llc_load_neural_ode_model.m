function model = llc_load_neural_ode_model(file, expectedLoadOhm, loadToleranceOhm)
    % LLC_LOAD_NEURAL_ODE_MODEL Load and validate a Python-exported vector field.
    if nargin < 2
        expectedLoadOhm = [];
    end
    if nargin < 3
        loadToleranceOhm = 1e-9;
    end
    if ~(ischar(file) || (isstring(file) && isscalar(file))) || ~isfile(file)
        error('llc:MissingModelExport', 'Neural ODE model export does not exist: %s', string(file));
    end
    raw = load(file);
    required = {'schema_version', 'network', 'state_names', 'gate_names', ...
                'network_input_order', 'network_output_order', 'state_scales', ...
                'vin_reference_v', 'time_reference_s', 'load_resistance_ohm', ...
                'dtype', 'consistency_inputs', 'consistency_outputs'};
    missing = required(~isfield(raw, required));
    if ~isempty(missing)
        error('llc:InvalidModelExport', 'Model export is missing fields: %s', strjoin(missing, ', '));
    end
    if ~strcmp(scalarText(raw.schema_version), 'llc-neural-ode-matlab-export-v1')
        error('llc:InvalidModelExport', 'Unsupported model export schema_version.');
    end
    requireNames(raw.state_names, {'i_Lr', 'v_Cr', 'i_Lm', 'v_o'}, 'state_names');
    requireNames(raw.gate_names, ...
        {'Q1_A_high', 'Q2_A_low', 'Q3_B_high', 'Q4_B_low'}, 'gate_names');
    requireNames(raw.network_input_order, ...
        {'z_i_Lr', 'z_v_Cr', 'z_i_Lm', 'z_v_o', 'Q1_A_high', 'Q2_A_low', ...
         'Q3_B_high', 'Q4_B_low', 'Vin_over_V_ref'}, 'network_input_order');
    requireNames(raw.network_output_order, ...
        {'dz_i_Lr_dtau', 'dz_v_Cr_dtau', 'dz_i_Lm_dtau', 'dz_v_o_dtau'}, ...
        'network_output_order');
    scales = double(raw.state_scales(:).');
    if numel(scales) ~= 4 || any(~isfinite(scales)) || any(scales <= 0)
        error('llc:InvalidModelExport', 'state_scales must contain four finite positive values.');
    end
    vinReference = positiveScalar(raw.vin_reference_v, 'vin_reference_v');
    timeReference = positiveScalar(raw.time_reference_s, 'time_reference_s');
    modelLoad = positiveScalar(raw.load_resistance_ohm, 'load_resistance_ohm');
    if ~isempty(expectedLoadOhm) && abs(modelLoad - expectedLoadOhm) > loadToleranceOhm
        error('llc:ModelContractMismatch', ...
              'Model load %.17g ohm does not match requested load %.17g ohm.', ...
              modelLoad, expectedLoadOhm);
    end
    network = validateNetwork(raw.network);
    inputs = double(raw.consistency_inputs);
    outputs = double(raw.consistency_outputs);
    if ~ismatrix(inputs) || size(inputs, 2) ~= 9 || isempty(inputs) || ...
            ~isequal(size(outputs), [size(inputs, 1), 4]) || ...
            any(~isfinite(inputs), 'all') || any(~isfinite(outputs), 'all')
        error('llc:InvalidModelExport', 'Consistency arrays have invalid dimensions or values.');
    end
    model = struct('schema_version', scalarText(raw.schema_version), ...
                   'network', network, 'state_scales', scales, ...
                   'vin_reference_v', vinReference, ...
                   'time_reference_s', timeReference, ...
                   'load_resistance_ohm', modelLoad, ...
                   'dtype', scalarText(raw.dtype), ...
                   'consistency_inputs', inputs, 'consistency_outputs', outputs);
end

function network = validateNetwork(value)
    if ~isstruct(value) || ~isscalar(value) || ...
            ~all(isfield(value, {'input_dim', 'output_dim', 'weights', 'biases', 'activations'}))
        error('llc:InvalidModelExport', 'network must contain dimensions, weights, biases and activations.');
    end
    if value.input_dim ~= 9 || value.output_dim ~= 4
        error('llc:InvalidModelExport', 'Network dimensions must be 9 inputs and 4 outputs.');
    end
    weights = reshape(value.weights, 1, []);
    biases = reshape(value.biases, 1, []);
    activations = cellstr(string(reshape(value.activations, 1, [])));
    if ~iscell(weights)
        weights = {weights};
    end
    if ~iscell(biases)
        biases = {biases};
    end
    if isempty(weights) || numel(weights) ~= numel(biases) || numel(weights) ~= numel(activations)
        error('llc:InvalidModelExport', 'Network layer arrays must have equal nonzero lengths.');
    end
    expectedInput = 9;
    for index = 1:numel(weights)
        weights{index} = double(weights{index});
        biases{index} = double(biases{index}(:));
        if ~ismatrix(weights{index}) || size(weights{index}, 2) ~= expectedInput || ...
                size(weights{index}, 1) ~= numel(biases{index}) || ...
                any(~isfinite(weights{index}), 'all') || any(~isfinite(biases{index}))
            error('llc:InvalidModelExport', 'Layer %d has incompatible weights or biases.', index);
        end
        expectedInput = size(weights{index}, 1);
        expectedActivation = 'SiLU';
        if index == numel(weights)
            expectedActivation = 'linear';
        end
        if ~strcmp(activations{index}, expectedActivation)
            error('llc:InvalidModelExport', 'Layer %d activation must be %s.', index, expectedActivation);
        end
    end
    if expectedInput ~= 4
        error('llc:InvalidModelExport', 'Final network layer must contain four outputs.');
    end
    network = struct('input_dim', 9, 'output_dim', 4, ...
                     'weights', {weights}, 'biases', {biases}, ...
                     'activations', {activations});
end

function value = positiveScalar(value, name)
    value = double(value);
    if ~isscalar(value) || ~isfinite(value) || value <= 0
        error('llc:InvalidModelExport', '%s must be a finite positive scalar.', name);
    end
end

function requireNames(value, expected, name)
    if ~isequal(cellstr(string(value(:))), expected(:))
        error('llc:InvalidModelExport', '%s does not match the model contract.', name);
    end
end

function value = scalarText(value)
    value = string(value);
    if ~isscalar(value) || strlength(value) == 0
        error('llc:InvalidModelExport', 'Expected nonempty scalar text.');
    end
    value = char(value);
end
