function [files, summary] = generate_llc_closed_loop_dataset(hardwareConfigFile, generationConfigFile)
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
    entryTemplate = struct('trajectory_id', '', 'success', false, 'output_file', '', ...
                           'error_identifier', '', 'error_message', '', 'steady_state', []);
    entries = repmat(entryTemplate, numel(generation.conditions), 1);
    simulationStopTime = generation.steady_state_assessment.check_times_s(end);

    for index = 1:numel(generation.conditions)
        condition = generation.conditions(index);
        entries(index).trajectory_id = condition.trajectory_id;
        try
            parameters = hardware;
            parameters.vin_v = condition.vin_v;
            parameters.load_ohm = condition.load_ohm;
            control = generation.control;
            control.reference_voltage_v = condition.reference_voltage_v;
            override = struct('parameters', parameters, 'control', control, ...
                              'solver', generation.solver, 'stop_time_s', simulationStopTime);
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
            signalReport = validateResult(stateTimes, stateValues, gateTimes, gateValues, ...
                                          controlTimes, controlValues, hardware, generation, ...
                                          simulationStopTime);
            assessmentConfig = generation.steady_state_assessment;
            assessmentConfig.switching_frequency_limits_hz = ...
                [generation.control.minimum_frequency_hz, generation.control.maximum_frequency_hz];
            steadyState = llc_assess_periodic_steady_state( ...
                stateTimes, stateValues, gateTimes, gateValues, ...
                condition.reference_voltage_v, assessmentConfig);
            raw = struct('state_times_s', stateTimes, 'state_values', stateValues, ...
                         'gate_times_s', gateTimes, 'gate_values', gateValues, ...
                         'control_times_s', controlTimes, 'control_values', controlValues);
            [fixed, samplingAudit] = llc_resample_fixed_grid( ...
                raw, generation.sampling, steadyState.selected_stop_time_s);

            trajectory = struct;
            trajectory.schema_version = 'llc-closed-loop-fixed-grid-trajectory-v2';
            trajectory.trajectory_id = condition.trajectory_id;
            trajectory.state_names = cellstr(hardware.state_names);
            trajectory.state_units = cellstr(hardware.state_units);
            trajectory.state_times_s = fixed.state_times_s;
            trajectory.state_values = fixed.state_values;
            trajectory.gate_names = cellstr(hardware.gate_names);
            trajectory.control_names = {'frequency_hz', 'filtered_voltage_v', ...
                                        'reference_voltage_v', 'error_v', 'integral_v_s'};
            trajectory.control_units = {'Hz', 'V', 'V', 'V', 'V*s'};
            trajectory.interval_times_s = fixed.interval_times_s;
            trajectory.interval_gate_values = fixed.interval_gate_values;
            trajectory.interval_control_values = fixed.interval_control_values;
            trajectory.raw_state_times_s = stateTimes;
            trajectory.raw_state_values = stateValues;
            trajectory.raw_gate_times_s = gateTimes;
            trajectory.raw_gate_values = uint8(gateValues);
            trajectory.raw_control_times_s = controlTimes;
            trajectory.raw_control_values = controlValues;
            trajectory.raw_simulation_stop_time_s = simulationStopTime;
            trajectory.condition = condition;
            trajectory.validation_report = struct('signal_integrity', signalReport, ...
                                                  'steady_state', steadyState);
            trajectory.sampling_audit = samplingAudit;
            trajectory.effective_config = struct('hardware', hardware, 'generation', generation, ...
                                                 'condition', condition);
            trajectory.source_model = generation.model_file;
            trajectory.matlab_version = version;
            outputFile = fullfile(outputDirectory, ...
                sprintf('trajectory_%04d_%s.mat', index, condition.trajectory_id));
            if isfile(outputFile)
                error('llc:OutputExists', 'Refusing to overwrite an existing trajectory: %s', outputFile);
            end
            save(outputFile, '-struct', 'trajectory', '-v7');
            files(index) = string(outputFile);
            entries(index).success = true;
            entries(index).output_file = outputFile;
            entries(index).steady_state = steadyState;
        catch exception
            entries(index).error_identifier = exception.identifier;
            entries(index).error_message = exception.message;
            warning('llc:ConditionFailed', 'Condition %s failed:\n%s', ...
                    condition.trajectory_id, getReport(exception, 'extended', 'hyperlinks', 'off'));
        end
    end
    summary = struct('entries', entries, ...
                     'successful_count', nnz([entries.success]), ...
                     'failed_count', nnz(~[entries.success]));
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
                                 controlTimes, controlValues, hardware, generation, simulationStopTime)
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
    tolerances = generation.validation_tolerances;
    timingTolerance = tolerances.timing_s;
    if ~any(offIntervals) || any(gateWidths(offIntervals) < generation.control.deadtime_s - timingTolerance)
        error('llc:InvalidOutput', 'Applied gates violate the configured deadtime.');
    end
    if ~any(activeIntervals) || ...
            any(gateWidths(activeIntervals) < generation.control.minimum_pulse_s - timingTolerance)
        error('llc:InvalidOutput', 'Applied gates violate the configured minimum pulse width.');
    end
    if any(controlValues(:, 1) < generation.control.minimum_frequency_hz - tolerances.frequency_hz) || ...
            any(controlValues(:, 1) > generation.control.maximum_frequency_hz + tolerances.frequency_hz)
        error('llc:InvalidOutput', 'Frequency command exceeded configured limits.');
    end
    if abs(stateTimes(1)) > tolerances.timing_s || ...
            any(abs(stateValues(1, :)) > tolerances.initial_state_abs)
        error('llc:InvalidOutput', 'Trajectory does not start from the requested zero state.');
    end
    tolerance = max(tolerances.stop_time_abs_s, 32 * eps(simulationStopTime));
    if abs(stateTimes(end) - simulationStopTime) > tolerance
        error('llc:InvalidOutput', 'Trajectory does not reach the configured stop time.');
    end
    report = struct('simulation_stop_time_s', simulationStopTime, ...
                    'peak_output_voltage_v', max(stateValues(:, 4)), ...
                    'minimum_deadtime_s', min(gateWidths(offIntervals)), ...
                    'minimum_active_pulse_s', min(gateWidths(activeIntervals)));
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
    if ~isfield(config, 'schema_version') || ~strcmp(config.schema_version, 'llc-closed-loop-data-generation-v3')
        error('llc:InvalidConfig', 'Unsupported closed-loop data-generation schema_version.');
    end
    requireText(config, 'model_file');
    requireText(config, 'output_directory');
    if ~isfield(config, 'solver') || ~isstruct(config.solver) || ...
            ~isfield(config, 'control') || ~isstruct(config.control) || ...
            ~isfield(config, 'sampling') || ~isstruct(config.sampling) || ...
            ~isfield(config, 'validation_tolerances') || ~isstruct(config.validation_tolerances)
        error('llc:IncompleteConfig', ...
              'solver, control, sampling and validation_tolerances must be JSON objects.');
    end
    requireText(config.solver, 'name');
    if ~strcmp(config.solver.name, 'ode23t')
        error('llc:InvalidConfig', 'This model requires solver.name = ode23t.');
    end
    for name = {'max_step_s', 'relative_tolerance', 'absolute_tolerance'}
        requirePositiveScalar(config.solver, name{1});
    end
    sampling = config.sampling;
    if ~isfield(sampling, 'method') || ~strcmp(sampling.method, 'uniform_reference_cycle_v1') || ...
            ~isfield(sampling, 'state_interpolation') || ~strcmp(sampling.state_interpolation, 'linear') || ...
            ~isfield(sampling, 'interval_signal_sampling') || ...
            ~strcmp(sampling.interval_signal_sampling, 'midpoint_previous')
        error('llc:InvalidConfig', 'Unsupported fixed-grid sampling contract.');
    end
    requirePositiveScalar(sampling, 'reference_frequency_hz');
    requirePositiveScalar(sampling, 'intervals_per_cycle');
    if sampling.intervals_per_cycle ~= round(sampling.intervals_per_cycle)
        error('llc:InvalidConfig', 'sampling.intervals_per_cycle must be an integer.');
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
            numel(config.initial_state) ~= 4 || any(~isfinite(config.initial_state)) || ...
            any(config.initial_state ~= 0)
        error('llc:InvalidConfig', 'initial_state must contain four zeros.');
    end
    toleranceFields = {'timing_s', 'frequency_hz', 'initial_state_abs', 'stop_time_abs_s'};
    for index = 1:numel(toleranceFields)
        requirePositiveScalar(config.validation_tolerances, toleranceFields{index});
    end
    validateSteadyStateAssessment(config, sampling);
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
    for name = {'vin_v', 'load_ohm', 'reference_voltage_v'}
        requirePositiveScalar(condition, name{1});
    end
end

function validateSteadyStateAssessment(config, sampling)
    if ~isfield(config, 'steady_state_assessment') || ...
            ~isstruct(config.steady_state_assessment)
        error('llc:IncompleteConfig', 'steady_state_assessment must be a JSON object.');
    end
    assessment = config.steady_state_assessment;
    requiredScalars = {'window_duration_s', 'output_relative_tolerance', ...
                       'state_mean_relative_tolerance', 'ripple_relative_tolerance', ...
                       'ripple_floor_relative_scale', 'minimum_cycles_per_half_window'};
    for index = 1:numel(requiredScalars)
        requirePositiveScalar(assessment, requiredScalars{index});
    end
    if assessment.output_relative_tolerance >= 1 || ...
            assessment.state_mean_relative_tolerance >= 1 || ...
            assessment.ripple_relative_tolerance >= 1 || ...
            assessment.ripple_floor_relative_scale >= 1
        error('llc:InvalidConfig', 'Steady-state relative tolerances must be smaller than one.');
    end
    if assessment.minimum_cycles_per_half_window ~= ...
            round(assessment.minimum_cycles_per_half_window)
        error('llc:InvalidConfig', 'minimum_cycles_per_half_window must be an integer.');
    end
    if ~isfield(assessment, 'check_times_s') || ~isnumeric(assessment.check_times_s) || ...
            isempty(assessment.check_times_s) || any(~isfinite(assessment.check_times_s)) || ...
            any(assessment.check_times_s <= 0) || any(diff(assessment.check_times_s) <= 0)
        error('llc:InvalidConfig', 'check_times_s must be a strictly increasing positive vector.');
    end
    if assessment.check_times_s(1) <= assessment.window_duration_s || ...
            assessment.check_times_s(1) < config.control.reference_ramp_time_s
        error('llc:InvalidConfig', 'The first checkpoint must follow the assessment window and reference ramp.');
    end
    if ~isfield(assessment, 'state_reference_scales') || ...
            ~isnumeric(assessment.state_reference_scales) || ...
            numel(assessment.state_reference_scales) ~= 4 || ...
            any(~isfinite(assessment.state_reference_scales)) || ...
            any(assessment.state_reference_scales <= 0)
        error('llc:InvalidConfig', 'state_reference_scales must contain four finite positive values.');
    end
    intervalCounts = assessment.check_times_s * sampling.reference_frequency_hz * ...
                     sampling.intervals_per_cycle;
    if any(abs(intervalCounts - round(intervalCounts)) > 64 * eps(max(intervalCounts, 1)))
        error('llc:SamplingAlignment', 'Every steady-state checkpoint must end on the fixed grid.');
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
