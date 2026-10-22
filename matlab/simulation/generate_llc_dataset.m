function files = generate_llc_dataset(hardwareConfigFile, generationConfigFile)
    % GENERATE_LLC_DATASET Run configured constant-condition LLC trajectories.
    matlabRoot = fileparts(fileparts(mfilename('fullpath')));
    projectRoot = fileparts(matlabRoot);
    addpath(matlabRoot);
    setup_paths;
    if nargin < 1 || isempty(hardwareConfigFile)
        hardwareConfigFile = fullfile(projectRoot, 'configs', 'matlab', 'hardware.json');
    end
    if nargin < 2 || isempty(generationConfigFile)
        generationConfigFile = fullfile(projectRoot, 'configs', 'matlab', 'data_generation.json');
    end
    hardware = llc_read_json(hardwareConfigFile);
    generation = llc_read_json(generationConfigFile);
    validateHardware(hardware);
    validateGeneration(generation);

    modelFile = resolveProjectPath(projectRoot, generation.model_file);
    if ~isfile(modelFile)
        error('llc:MissingModel', 'Simulink model does not exist: %s', modelFile);
    end
    outputDirectory = resolveProjectPath(projectRoot, generation.output_directory);
    if ~isfolder(outputDirectory)
        mkdir(outputDirectory);
    end
    [~, modelName] = fileparts(modelFile);
    load_system(modelFile);
    closeModel = onCleanup(@()close_system(modelName, 0));
    workspace = get_param(modelName, 'ModelWorkspace');
    conditions = generation.conditions;
    files = strings(numel(conditions), 1);

    for index = 1:numel(conditions)
        condition = conditions(index);
        validateCondition(condition, generation.modulation);
        parameters = hardware;
        parameters.vin_v = condition.vin_v;
        parameters.load_ohm = condition.load_ohm;
        events = llc_constant_frequency_events(generation.modulation, ...
                                               condition.switching_frequency_hz, ...
                                               condition.stop_time_s);
        override = struct('parameters', parameters, 'events', events, 'solver', generation.solver);
        assignin(workspace, 'llc_run_override', override);

        simulation = Simulink.SimulationInput(modelName);
        simulation = simulation.setBlockParameter([modelName '/States'], 'Decimation', '1');
        simulation = simulation.setBlockParameter([modelName '/States'], 'MaxDataPoints', 'inf');
        simulation = simulation.setBlockParameter([modelName '/Gates'], 'Decimation', '1');
        simulation = simulation.setBlockParameter([modelName '/Gates'], 'MaxDataPoints', 'inf');
        simulation = applyInitialState(simulation, modelName, generation.initial_state);
        result = sim(simulation);
        stateTimes = result.llc_states.Time;
        stateValues = result.llc_states.Data;
        gateTimes = result.llc_gates.Time;
        gateValues = result.llc_gates.Data;
        validateResult(stateTimes, stateValues, gateTimes, gateValues, hardware, condition, events);

        trajectory = struct;
        trajectory.schema_version = 'llc-raw-trajectory-v1';
        trajectory.trajectory_id = condition.trajectory_id;
        trajectory.state_names = cellstr(hardware.state_names);
        trajectory.state_units = cellstr(hardware.state_units);
        trajectory.state_times_s = stateTimes;
        trajectory.state_values = stateValues;
        trajectory.gate_names = cellstr(hardware.gate_names);
        trajectory.gate_times_s = gateTimes;
        trajectory.gate_values = uint8(gateValues);
        trajectory.condition = condition;
        trajectory.effective_config = struct('hardware', hardware, 'generation', generation, ...
                                             'condition', condition);
        trajectory.source_model = generation.model_file;
        trajectory.matlab_version = version;
        outputFile = fullfile(outputDirectory, sprintf('trajectory_%04d_%s.mat', index, condition.trajectory_id));
        if isfile(outputFile)
            error('llc:OutputExists', 'Refusing to overwrite an existing trajectory: %s', outputFile);
        end
        save(outputFile, '-struct', 'trajectory', '-v7');
        files(index) = string(outputFile);
    end
    clear closeModel;
end

function simulation = applyInitialState(simulation, modelName, initialState)
    blocks = {'Lr', 'Cr', 'Lm', 'Co'};
    variables = {'i_L', 'vc', 'i_L', 'vc'};
    for index = 1:numel(blocks)
        block = [modelName '/' blocks{index}];
        simulation = simulation.setBlockParameter(block, variables{index}, num2str(initialState(index), 17));
        simulation = simulation.setBlockParameter(block, [variables{index} '_priority'], 'High');
        simulation = simulation.setBlockParameter(block, [variables{index} '_specify'], 'on');
    end
end

function validateHardware(config)
    if ~isfield(config, 'schema_version') || ~strcmp(config.schema_version, 'llc-hardware-v1')
        error('llc:InvalidConfig', 'Unsupported hardware schema_version.');
    end
    positive = {'turns_ratio_np_ns', 'resonant_inductance_h', 'resonant_capacitance_f', ...
                'magnetizing_inductance_h', 'output_capacitance_f', ...
                'resonant_winding_resistance_ohm', 'magnetizing_series_resistance_ohm', ...
                'switch_on_resistance_ohm', 'switch_off_conductance_s', ...
                'body_diode_forward_v', 'body_diode_on_resistance_ohm', ...
                'rectifier_forward_v', 'rectifier_on_resistance_ohm', 'diode_off_conductance_s'};
    for index = 1:numel(positive)
        requirePositiveScalar(config, positive{index});
    end
    requireNames(config, 'state_names', {'i_Lr', 'v_Cr', 'i_Lm', 'v_o'});
    requireNames(config, 'state_units', {'A', 'V', 'A', 'V'});
    requireNames(config, 'gate_names', {'Q1_A_high', 'Q2_A_low', 'Q3_B_high', 'Q4_B_low'});
end

function validateGeneration(config)
    if ~isfield(config, 'schema_version') || ~strcmp(config.schema_version, 'llc-data-generation-v1')
        error('llc:InvalidConfig', 'Unsupported data-generation schema_version.');
    end
    requireText(config, 'model_file');
    requireText(config, 'output_directory');
    requiredStructures = {'solver', 'modulation'};
    for index = 1:numel(requiredStructures)
        name = requiredStructures{index};
        if ~isfield(config, name) || ~isstruct(config.(name)) || ~isscalar(config.(name))
            error('llc:IncompleteConfig', '%s must be a JSON object.', name);
        end
    end
    requireText(config.solver, 'name');
    if ~strcmp(config.solver.name, 'ode23t')
        error('llc:InvalidConfig', 'This model configuration requires solver.name = ode23t.');
    end
    requirePositiveScalar(config.solver, 'max_step_s');
    requirePositiveScalar(config.solver, 'relative_tolerance');
    requirePositiveScalar(config.solver, 'absolute_tolerance');
    requirePositiveScalar(config.modulation, 'deadtime_s');
    requirePositiveScalar(config.modulation, 'minimum_pulse_s');
    if ~isfield(config.modulation, 'frequency_range_hz')
        error('llc:IncompleteConfig', 'Missing modulation.frequency_range_hz.');
    end
    if ~isfield(config, 'initial_state') || ~isnumeric(config.initial_state) || ...
            numel(config.initial_state) ~= 4 || any(~isfinite(config.initial_state))
        error('llc:InvalidConfig', 'initial_state must contain four finite values.');
    end
    config.initial_state = reshape(config.initial_state, 1, []); %#ok<NASGU>
    if ~isfield(config, 'conditions') || ~isstruct(config.conditions) || isempty(config.conditions)
        error('llc:IncompleteConfig', 'conditions must contain at least one trajectory object.');
    end
    identifiers = strings(numel(config.conditions), 1);
    for index = 1:numel(config.conditions)
        validateCondition(config.conditions(index), config.modulation);
        identifiers(index) = string(config.conditions(index).trajectory_id);
    end
    if numel(unique(identifiers)) ~= numel(identifiers)
        error('llc:InvalidConfig', 'condition trajectory_id values must be unique.');
    end
end

function validateCondition(condition, modulation)
    requireText(condition, 'trajectory_id');
    if isempty(regexp(condition.trajectory_id, '^[A-Za-z0-9][A-Za-z0-9_-]*$', 'once'))
        error('llc:InvalidCondition', 'trajectory_id may contain only letters, digits, underscores and hyphens.');
    end
    fields = {'vin_v', 'load_ohm', 'switching_frequency_hz', 'stop_time_s'};
    for index = 1:numel(fields)
        requirePositiveScalar(condition, fields{index});
    end
    limits = modulation.frequency_range_hz;
    if condition.switching_frequency_hz < limits(1) || condition.switching_frequency_hz > limits(2)
        error('llc:InvalidCondition', 'Condition %s has an out-of-range switching frequency.', condition.trajectory_id);
    end
end

function validateResult(stateTimes, stateValues, gateTimes, gateValues, hardware, condition, events)
    if ~isvector(stateTimes) || isempty(stateTimes) || any(~isfinite(stateTimes)) || any(diff(stateTimes) <= 0)
        error('llc:InvalidOutput', 'State time must be finite and strictly increasing.');
    end
    if size(stateValues, 1) ~= numel(stateTimes) || size(stateValues, 2) ~= numel(hardware.state_names) || ...
            any(~isfinite(stateValues), 'all')
        error('llc:InvalidOutput', 'State output has invalid dimensions or values.');
    end
    if ~isvector(gateTimes) || isempty(gateTimes) || any(~isfinite(gateTimes)) || any(diff(gateTimes) <= 0) || ...
            size(gateValues, 1) ~= numel(gateTimes) || size(gateValues, 2) ~= numel(hardware.gate_names) || ...
            any(gateValues ~= 0 & gateValues ~= 1, 'all')
        error('llc:InvalidOutput', 'Gate output has invalid dimensions or values.');
    end
    if abs(stateTimes(1)) > eps || any(abs(stateValues(1, :)) > 1e-8)
        error('llc:InvalidOutput', 'Trajectory does not start from the requested zero state.');
    end
    tolerance = max(1e-12, 32 * eps(condition.stop_time_s));
    if abs(stateTimes(end) - condition.stop_time_s) > tolerance || abs(gateTimes(end) - condition.stop_time_s) > tolerance
        error('llc:InvalidOutput', 'Trajectory does not reach the configured stop time.');
    end
    if any(gateValues(:, 1) & gateValues(:, 2)) || any(gateValues(:, 3) & gateValues(:, 4))
        error('llc:InvalidOutput', 'Applied gate signals contain bridge-leg overlap.');
    end
    expectedGates = interp1(events.gate_times_s, ...
                            [double(events.gate_values); zeros(1, size(events.gate_values, 2))], ...
                            gateTimes, 'previous');
    if ~isequal(double(gateValues), expectedGates)
        error('llc:InvalidOutput', 'Applied gate signals differ from the configured post-deadtime events.');
    end
end

function path = resolveProjectPath(projectRoot, configuredPath)
    requireText(struct('path', configuredPath), 'path');
    configuredPath = char(configuredPath);
    if startsWith(configuredPath, filesep)
        path = configuredPath;
    else
        path = fullfile(projectRoot, configuredPath);
    end
end

function requirePositiveScalar(value, name)
    if ~isfield(value, name) || ~isnumeric(value.(name)) || ~isscalar(value.(name)) || ...
            ~isfinite(value.(name)) || value.(name) <= 0
        error('llc:IncompleteConfig', '%s must be a finite positive scalar.', name);
    end
end

function requireText(value, name)
    if ~isfield(value, name) || ~(ischar(value.(name)) || (isstring(value.(name)) && isscalar(value.(name)))) || ...
            strlength(string(value.(name))) == 0
        error('llc:IncompleteConfig', '%s must be nonempty text.', name);
    end
end

function requireNames(value, name, expected)
    if ~isfield(value, name) || ~isequal(cellstr(value.(name)), expected(:))
        error('llc:InvalidConfig', '%s does not match the Simulink model channel order.', name);
    end
end
