function files = generate_llc_closed_loop_dataset(hardwareConfigFile, generationConfigFile)
    % GENERATE_LLC_CLOSED_LOOP_DATASET Simulate voltage-feedback PFM trajectories.
    matlabRoot = fileparts(fileparts(mfilename('fullpath')));
    projectRoot = fileparts(matlabRoot);
    addpath(matlabRoot);
    setup_paths;
    if nargin < 1 || isempty(hardwareConfigFile)
        hardwareConfigFile = fullfile(projectRoot, 'configs', 'matlab', 'hardware.json');
    end
    if nargin < 2 || isempty(generationConfigFile)
        generationConfigFile = fullfile(projectRoot, 'configs', 'matlab', 'closed_loop_data_generation.json');
    end
    hardware = llc_read_json(hardwareConfigFile);
    generation = llc_read_json(generationConfigFile);
    validateHardware(hardware);
    validateGeneration(generation);

    modelFile = resolveProjectPath(projectRoot, generation.model_file);
    if ~isfile(modelFile)
        error('llc:MissingModel', 'Closed-loop Simulink model does not exist: %s', modelFile);
    end
    outputDirectory = resolveProjectPath(projectRoot, generation.output_directory);
    if ~isfolder(outputDirectory)
        mkdir(outputDirectory);
    end
    [~, modelName] = fileparts(modelFile);
    load_system(modelFile);
    closeModel = onCleanup(@()close_system(modelName, 0));
    workspace = get_param(modelName, 'ModelWorkspace');
    files = strings(numel(generation.conditions), 1);

    for index = 1:numel(generation.conditions)
        condition = generation.conditions(index);
        validateCondition(condition);
        parameters = hardware;
        parameters.vin_v = condition.vin_v;
        parameters.load_ohm = condition.load_ohm;
        control = generation.control;
        control.reference_voltage_v = condition.reference_voltage_v;
        override = struct('parameters', parameters, 'control', control, ...
                          'solver', generation.solver, 'stop_time_s', condition.stop_time_s);
        assignin(workspace, 'llc_run_override', override);

        simulation = Simulink.SimulationInput(modelName);
        loggedBlocks = {'States', 'Gates', 'Control'};
        for blockIndex = 1:numel(loggedBlocks)
            block = [modelName '/' loggedBlocks{blockIndex}];
            simulation = simulation.setBlockParameter(block, 'Decimation', '1');
            simulation = simulation.setBlockParameter(block, 'MaxDataPoints', 'inf');
        end
        simulation = applyInitialState(simulation, modelName, generation.initial_state);
        result = sim(simulation);
        stateTimes = result.llc_states.Time;
        stateValues = result.llc_states.Data;
        gateTimes = result.llc_gates.Time;
        gateValues = result.llc_gates.Data;
        controlTimes = result.llc_control_log.Time;
        controlValues = result.llc_control_log.Data;
        report = validateResult(stateTimes, stateValues, gateTimes, gateValues, ...
                                controlTimes, controlValues, hardware, generation, condition);

        trajectory = struct;
        trajectory.schema_version = 'llc-closed-loop-raw-trajectory-v1';
        trajectory.trajectory_id = condition.trajectory_id;
        trajectory.state_names = cellstr(hardware.state_names);
        trajectory.state_units = cellstr(hardware.state_units);
        trajectory.state_times_s = stateTimes;
        trajectory.state_values = stateValues;
        trajectory.gate_names = cellstr(hardware.gate_names);
        trajectory.gate_times_s = gateTimes;
        trajectory.gate_values = uint8(gateValues);
        trajectory.control_names = {'frequency_hz', 'filtered_voltage_v', ...
                                    'reference_voltage_v', 'error_v', 'integral_v_s'};
        trajectory.control_units = {'Hz', 'V', 'V', 'V', 'V*s'};
        trajectory.control_times_s = controlTimes;
        trajectory.control_values = controlValues;
        trajectory.condition = condition;
        trajectory.validation_report = report;
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

function report = validateResult(stateTimes, stateValues, gateTimes, gateValues, ...
                                 controlTimes, controlValues, hardware, generation, condition)
    requireTimeSeries(stateTimes, stateValues, numel(hardware.state_names), 'state');
    requireTimeSeries(gateTimes, gateValues, numel(hardware.gate_names), 'gate');
    requireTimeSeries(controlTimes, controlValues, 5, 'control');
    if any(gateValues ~= 0 & gateValues ~= 1, 'all')
        error('llc:InvalidOutput', 'Gate output contains nonbinary values.');
    end
    if any(gateValues(:, 1) & gateValues(:, 2)) || any(gateValues(:, 3) & gateValues(:, 4))
        error('llc:InvalidOutput', 'Applied gates contain bridge-leg overlap.');
    end
    gateChanges = [true; any(diff(double(gateValues), 1, 1) ~= 0, 2)];
    gateEventTimes = gateTimes(gateChanges);
    gateEventValues = double(gateValues(gateChanges, :));
    gateWidths = diff(gateEventTimes);
    gateIntervals = gateEventValues(1:end-1, :);
    offIntervals = ~any(gateIntervals, 2);
    activeIntervals = any(gateIntervals, 2);
    timingTolerance = 1e-12;
    if ~any(offIntervals) || any(gateWidths(offIntervals) < generation.control.deadtime_s - timingTolerance)
        error('llc:InvalidOutput', 'Applied gates violate the configured deadtime.');
    end
    if ~any(activeIntervals) || ...
            any(gateWidths(activeIntervals) < generation.control.minimum_pulse_s - timingTolerance)
        error('llc:InvalidOutput', 'Applied gates violate the configured minimum pulse width.');
    end
    if any(controlValues(:, 1) < generation.control.minimum_frequency_hz - 1) || ...
            any(controlValues(:, 1) > generation.control.maximum_frequency_hz + 1)
        error('llc:InvalidOutput', 'Frequency command exceeded configured limits.');
    end
    if abs(stateTimes(1)) > eps || any(abs(stateValues(1, :)) > 1e-8)
        error('llc:InvalidOutput', 'Trajectory does not start from the requested zero state.');
    end
    tolerance = max(1e-12, 32 * eps(condition.stop_time_s));
    if abs(stateTimes(end) - condition.stop_time_s) > tolerance
        error('llc:InvalidOutput', 'Trajectory does not reach the configured stop time.');
    end
    tailStart = max(condition.stop_time_s - 200 / mean(controlValues(end-100:end, 1)), ...
                    0.8 * condition.stop_time_s);
    tail = stateTimes >= tailStart;
    steadyVoltage = trapz(stateTimes(tail), stateValues(tail, 4)) / ...
                    (stateTimes(find(tail, 1, 'last')) - stateTimes(find(tail, 1, 'first')));
    relativeError = abs(steadyVoltage - condition.reference_voltage_v) / condition.reference_voltage_v;
    if relativeError > generation.steady_state_relative_tolerance
        error('llc:ClosedLoopNotRegulated', ...
              'Final mean output %.6g V differs from %.6g V reference by %.3f%%.', ...
              steadyVoltage, condition.reference_voltage_v, 100 * relativeError);
    end
    afterRamp = stateTimes >= generation.control.reference_ramp_time_s;
    outsideBand = afterRamp & abs(stateValues(:, 4) - condition.reference_voltage_v) > ...
                  generation.steady_state_relative_tolerance * condition.reference_voltage_v;
    lastOutside = find(outsideBand, 1, 'last');
    if isempty(lastOutside)
        settlingTime = generation.control.reference_ramp_time_s;
    elseif lastOutside < numel(stateTimes)
        settlingTime = stateTimes(lastOutside + 1);
    else
        settlingTime = NaN;
    end
    report = struct('final_mean_output_voltage_v', steadyVoltage, ...
                    'final_relative_error', relativeError, ...
                    'final_frequency_hz', mean(controlValues(controlTimes >= tailStart, 1)), ...
                    'settling_time_s', settlingTime, ...
                    'peak_output_voltage_v', max(stateValues(:, 4)), ...
                    'minimum_deadtime_s', min(gateWidths(offIntervals)), ...
                    'minimum_active_pulse_s', min(gateWidths(activeIntervals)), ...
                    'regulated', true);
end

function requireTimeSeries(times, values, width, name)
    if ~isvector(times) || isempty(times) || any(~isfinite(times)) || any(diff(times) <= 0) || ...
            size(values, 1) ~= numel(times) || size(values, 2) ~= width || any(~isfinite(values), 'all')
        error('llc:InvalidOutput', '%s output has invalid time, dimensions or values.', name);
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
    if ~isfield(config, 'schema_version') || ~strcmp(config.schema_version, 'llc-closed-loop-data-generation-v1')
        error('llc:InvalidConfig', 'Unsupported closed-loop data-generation schema_version.');
    end
    requireText(config, 'model_file');
    requireText(config, 'output_directory');
    if ~isfield(config, 'solver') || ~isstruct(config.solver) || ...
            ~isfield(config, 'control') || ~isstruct(config.control)
        error('llc:IncompleteConfig', 'solver and control must be JSON objects.');
    end
    requireText(config.solver, 'name');
    if ~strcmp(config.solver.name, 'ode23t')
        error('llc:InvalidConfig', 'This model requires solver.name = ode23t.');
    end
    for name = {'max_step_s', 'relative_tolerance', 'absolute_tolerance'}
        requirePositiveScalar(config.solver, name{1});
    end
    controlFields = {'control_period_s', 'voltage_filter_time_constant_s', ...
                     'reference_ramp_time_s', 'initial_frequency_hz', 'bias_frequency_hz', ...
                     'minimum_frequency_hz', 'maximum_frequency_hz', ...
                     'frequency_slew_rate_hz_per_s', 'proportional_gain_hz_per_v', ...
                     'integral_gain_hz_per_v_s', 'deadtime_s', 'minimum_pulse_s'};
    for index = 1:numel(controlFields)
        requirePositiveScalar(config.control, controlFields{index});
    end
    control = config.control;
    if control.minimum_frequency_hz >= control.maximum_frequency_hz || ...
            control.initial_frequency_hz < control.minimum_frequency_hz || ...
            control.initial_frequency_hz > control.maximum_frequency_hz || ...
            control.bias_frequency_hz < control.minimum_frequency_hz || ...
            control.bias_frequency_hz > control.maximum_frequency_hz || ...
            control.deadtime_s + control.minimum_pulse_s >= 0.5 / control.maximum_frequency_hz
        error('llc:InvalidConfig', 'Closed-loop timing or frequency limits are inconsistent.');
    end
    if ~isfield(config, 'initial_state') || ~isnumeric(config.initial_state) || ...
            numel(config.initial_state) ~= 4 || any(~isfinite(config.initial_state))
        error('llc:InvalidConfig', 'initial_state must contain four finite values.');
    end
    requirePositiveScalar(config, 'steady_state_relative_tolerance');
    if config.steady_state_relative_tolerance >= 1
        error('llc:InvalidConfig', 'steady_state_relative_tolerance must be smaller than one.');
    end
    if ~isfield(config, 'conditions') || ~isstruct(config.conditions) || isempty(config.conditions)
        error('llc:IncompleteConfig', 'conditions must contain at least one trajectory object.');
    end
    identifiers = strings(numel(config.conditions), 1);
    for index = 1:numel(config.conditions)
        validateCondition(config.conditions(index));
        identifiers(index) = string(config.conditions(index).trajectory_id);
    end
    if numel(unique(identifiers)) ~= numel(identifiers)
        error('llc:InvalidConfig', 'condition trajectory_id values must be unique.');
    end
end

function validateCondition(condition)
    requireText(condition, 'trajectory_id');
    if isempty(regexp(condition.trajectory_id, '^[A-Za-z0-9][A-Za-z0-9_-]*$', 'once'))
        error('llc:InvalidCondition', 'trajectory_id contains unsupported characters.');
    end
    for name = {'vin_v', 'load_ohm', 'reference_voltage_v', 'stop_time_s'}
        requirePositiveScalar(condition, name{1});
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
