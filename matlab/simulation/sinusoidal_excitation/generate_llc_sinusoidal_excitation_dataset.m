function [files, summary] = generate_llc_sinusoidal_excitation_dataset( ...
        hardwareConfigFile, generationConfigFile, resumeExisting, conditionIndices)
    % GENERATE_LLC_SINUSOIDAL_EXCITATION_DATASET Generate dynamic Vin/Vref data.
    matlabRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    projectRoot = fileparts(matlabRoot);
    addpath(matlabRoot);
    setup_paths;
    if nargin < 1 || isempty(hardwareConfigFile)
        hardwareConfigFile = fullfile(projectRoot, 'configs', 'matlab', 'hardware.json');
    end
    if nargin < 2 || isempty(generationConfigFile)
        generationConfigFile = fullfile(projectRoot, 'configs', 'matlab', ...
            'sinusoidal_excitation', 'screening_data_generation.json');
    end
    if nargin < 3 || isempty(resumeExisting)
        resumeExisting = false;
    end
    if nargin < 4
        conditionIndices = [];
    end
    if ~islogical(resumeExisting) || ~isscalar(resumeExisting)
        error('llc:InvalidArgument', 'resumeExisting must be a logical scalar.');
    end

    hardware = llc_read_json(hardwareConfigFile);
    generation = llc_read_json(generationConfigFile);
    validateHardware(hardware);
    validateGeneration(generation);
    conditionIndices = selectConditionIndices( ...
        conditionIndices, numel(generation.conditions));
    trajectorySchema = outputSchema(generation);
    formalGeneration = isFormalGeneration(generation);
    modelFile = resolveProjectPath(projectRoot, generation.model_file);
    if ~isfile(modelFile)
        error('llc:MissingModel', 'Shared closed-loop model does not exist: %s', modelFile);
    end
    outputDirectory = resolveProjectPath(projectRoot, generation.output_directory);
    if ~isfolder(outputDirectory)
        mkdir(outputDirectory);
    end

    [~, modelName] = fileparts(modelFile);
    load_system(modelFile);
    closeModel = onCleanup(@()close_system(modelName, 0));
    workspace = get_param(modelName, 'ModelWorkspace');
    files = strings(numel(conditionIndices), 1);
    entryTemplate = struct('trajectory_id', '', 'success', false, 'skipped', false, ...
                           'output_file', '', 'error_identifier', '', 'error_message', '');
    entries = repmat(entryTemplate, numel(conditionIndices), 1);

    for workIndex = 1:numel(conditionIndices)
        index = conditionIndices(workIndex);
        condition = generation.conditions(index);
        requestSignature = buildGenerationRequestSignature( ...
            hardware, generation, condition);
        entries(workIndex).trajectory_id = condition.trajectory_id;
        conditionDirectory = outputDirectory;
        if formalGeneration
            conditionDirectory = fullfile(outputDirectory, ...
                condition.dataset_role, condition.excitation_group);
            if ~isfolder(conditionDirectory)
                mkdir(conditionDirectory);
            end
        end
        outputFile = fullfile(conditionDirectory, ...
            sprintf('trajectory_%04d_%s.mat', index, condition.trajectory_id));
        try
            if resumeExisting && isfile(outputFile)
                if formalGeneration
                    requiredFields = {'trajectory_id', 'schema_version', ...
                        'generation_request_signature'};
                else
                    requiredFields = {'trajectory_id', 'schema_version'};
                end
                availableFields = who('-file', outputFile);
                if ~all(ismember(requiredFields, availableFields))
                    error('llc:InvalidExistingOutput', ...
                          'Existing trajectory lacks resume-integrity metadata: %s', ...
                          outputFile);
                end
                existing = load(outputFile, requiredFields{:});
                signatureMismatch = formalGeneration && ...
                    ~strcmp(existing.generation_request_signature, requestSignature);
                if ~strcmp(existing.trajectory_id, condition.trajectory_id) || ...
                        ~strcmp(existing.schema_version, trajectorySchema) || ...
                        signatureMismatch
                    error('llc:InvalidExistingOutput', ...
                          'Existing trajectory does not match the requested condition: %s', outputFile);
                end
                files(workIndex) = string(outputFile);
                entries(workIndex).success = true;
                entries(workIndex).skipped = true;
                entries(workIndex).output_file = outputFile;
                continue;
            end

            excitation = buildExcitation(generation, condition);
            parameters = hardware;
            parameters.vin_v = excitation.vin.center_value;
            parameters.load_ohm = generation.fixed_load_ohm;
            control = generation.control;
            control.reference_voltage_v = excitation.reference.center_value;
            override = struct('parameters', parameters, 'control', control, ...
                              'excitation', excitation, 'solver', generation.solver, ...
                              'stop_time_s', generation.simulation_stop_time_s);
            assignin(workspace, 'llc_run_override', override);

            simulation = Simulink.SimulationInput(modelName);
            loggedBlocks = {'States', 'Gates', 'Control', 'Excitation'};
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
            excitationTimes = result.llc_excitation_log.Time;
            excitationValues = result.llc_excitation_log.Data;
            validateResult(stateTimes, stateValues, gateTimes, gateValues, ...
                controlTimes, controlValues, excitationTimes, excitationValues, ...
                hardware, generation, excitation);

            gateEvents = llc_extract_gate_events(gateTimes, gateValues);
            raw = struct('state_times_s', stateTimes, 'state_values', stateValues, ...
                         'gate_times_s', gateTimes, 'gate_values', gateValues, ...
                         'control_times_s', controlTimes, 'control_values', controlValues);
            [fixed, samplingAudit] = llc_resample_fixed_grid( ...
                raw, generation.sampling, generation.simulation_stop_time_s);
            stateExcitation = llc_evaluate_sinusoidal_excitation( ...
                fixed.state_times_s, excitation);
            intervalExcitation = llc_evaluate_sinusoidal_excitation( ...
                fixed.interval_times_s, excitation);

            trajectory = struct;
            trajectory.schema_version = trajectorySchema;
            trajectory.trajectory_id = condition.trajectory_id;
            trajectory.state_names = cellstr(hardware.state_names);
            trajectory.state_units = cellstr(hardware.state_units);
            trajectory.state_times_s = fixed.state_times_s;
            trajectory.state_values = fixed.state_values;
            trajectory.state_excitation_names = {'vin_v', 'reference_voltage_v'};
            trajectory.state_excitation_units = {'V', 'V'};
            trajectory.state_excitation_values = stateExcitation;
            trajectory.gate_names = cellstr(hardware.gate_names);
            trajectory.interval_times_s = fixed.interval_times_s;
            trajectory.interval_gate_values = fixed.interval_gate_values;
            trajectory.interval_excitation_values = intervalExcitation;
            trajectory.control_names = {'frequency_hz', 'filtered_voltage_v', ...
                                        'reference_voltage_v', 'error_v', 'integral_v_s'};
            trajectory.control_units = {'Hz', 'V', 'V', 'V', 'V*s'};
            trajectory.interval_control_values = fixed.interval_control_values;
            trajectory.initial_gate_values = gateEvents.initial_gate_values;
            trajectory.gate_event_times_s = gateEvents.gate_event_times_s;
            trajectory.gate_values_after = gateEvents.gate_values_after;
            trajectory.raw_state_times_s = stateTimes;
            trajectory.raw_state_values = stateValues;
            trajectory.raw_gate_times_s = gateTimes;
            trajectory.raw_gate_values = uint8(gateValues);
            trajectory.raw_control_times_s = controlTimes;
            trajectory.raw_control_values = controlValues;
            trajectory.raw_excitation_times_s = excitationTimes;
            trajectory.raw_excitation_values = excitationValues;
            trajectory.raw_simulation_stop_time_s = generation.simulation_stop_time_s;
            trajectory.condition = condition;
            trajectory.fixed_load_ohm = generation.fixed_load_ohm;
            trajectory.excitation = excitation;
            if formalGeneration
                trajectory.dataset_split = condition.dataset_role;
                trajectory.random_seed = condition.random_seed;
                trajectory.realized_excitation_segments = excitation.segments;
                trajectory.generation_request_signature = requestSignature;
            end
            inputContract = struct;
            inputContract.state_names = cellstr(hardware.state_names);
            inputContract.exogenous_names = [cellstr(hardware.gate_names); {'vin_v'}];
            inputContract.excluded_control_names = {'reference_voltage_v'};
            trajectory.network_input_contract = inputContract;
            trajectory.sampling_audit = samplingAudit;
            trajectory.effective_config = struct('hardware', hardware, 'generation', generation, ...
                                                 'condition', condition);
            trajectory.source_model = generation.model_file;
            trajectory.matlab_version = version;
            if isfile(outputFile)
                error('llc:OutputExists', ...
                      'Refusing to overwrite an existing trajectory: %s', outputFile);
            end
            if formalGeneration
                save(outputFile, '-struct', 'trajectory', ...
                     char(generation.mat_file_version));
            else
                save(outputFile, '-struct', 'trajectory', '-v7');
            end
            files(workIndex) = string(outputFile);
            entries(workIndex).success = true;
            entries(workIndex).output_file = outputFile;
        catch exception
            entries(workIndex).error_identifier = exception.identifier;
            entries(workIndex).error_message = exception.message;
            warning('llc:ConditionFailed', 'Condition %s failed:\n%s', ...
                    condition.trajectory_id, getReport(exception, 'extended', 'hyperlinks', 'off'));
        end
    end
    summary = struct('entries', entries, ...
                     'successful_count', nnz([entries.success]), ...
                     'failed_count', nnz(~[entries.success]), ...
                     'skipped_count', nnz([entries.skipped]));
    clear closeModel;
end

function indices = selectConditionIndices(requested, conditionCount)
    if isempty(requested)
        indices = (1:conditionCount).';
        return;
    end
    valid = isnumeric(requested) && isreal(requested) && isvector(requested) && ...
        all(isfinite(requested), 'all') && ...
        all(requested == fix(requested), 'all') && ...
        all(requested >= 1 & requested <= conditionCount, 'all') && ...
        numel(unique(requested)) == numel(requested);
    if ~valid
        error('llc:InvalidArgument', ...
              'conditionIndices must contain unique valid condition indices.');
    end
    indices = double(requested(:));
end

function excitation = buildExcitation(generation, condition)
    if isFormalGeneration(generation)
        excitation = llc_build_continuous_random_excitation( ...
            generation.random_excitation, generation.excitation_timing, condition);
    else
        excitation = struct( ...
            'start_time_s', generation.excitation_timing.start_time_s, ...
            'envelope_rise_time_s', generation.excitation_timing.envelope_rise_time_s, ...
            'vin', condition.vin, 'reference', condition.reference);
    end
end

function result = isFormalGeneration(generation)
    result = strcmp(generation.schema_version, ...
                    'llc-continuous-random-excitation-data-generation-v1');
end

function schema = outputSchema(generation)
    if isFormalGeneration(generation)
        schema = 'llc-continuous-random-excitation-fixed-grid-trajectory-v1';
    else
        schema = 'llc-sinusoidal-excitation-fixed-grid-trajectory-v1';
    end
end

function signature = buildGenerationRequestSignature(hardware, generation, condition)
    normalizedGeneration = rmfield(generation, ...
        {'output_directory', 'conditions'});
    request = struct('hardware', hardware, ...
                     'generation', normalizedGeneration, ...
                     'condition', condition);
    signature = jsonencode(request);
end

function simulation = applyInitialState(simulation, modelName, initialState)
    blocks = {'Lr', 'Cr', 'Lm', 'Co'};
    variables = {'i_L', 'vc', 'i_L', 'vc'};
    for index = 1:numel(blocks)
        block = [modelName '/' blocks{index}];
        simulation = simulation.setBlockParameter(block, variables{index}, ...
                                                  num2str(initialState(index), 17));
        simulation = simulation.setBlockParameter(block, ...
            [variables{index} '_priority'], 'High');
        simulation = simulation.setBlockParameter(block, ...
            [variables{index} '_specify'], 'on');
    end
end

function validateResult(stateTimes, stateValues, gateTimes, gateValues, ...
        controlTimes, controlValues, excitationTimes, excitationValues, ...
        hardware, generation, excitation)
    requireTimeSeries(stateTimes, stateValues, numel(hardware.state_names), 'state');
    requireTimeSeries(gateTimes, gateValues, numel(hardware.gate_names), 'gate');
    requireTimeSeries(controlTimes, controlValues, 5, 'control');
    requireTimeSeries(excitationTimes, excitationValues, 2, 'excitation');
    if any(gateValues ~= 0 & gateValues ~= 1, 'all')
        error('llc:InvalidOutput', 'Gate output contains nonbinary values.');
    end
    if any(gateValues(:, 1) & gateValues(:, 2)) || ...
            any(gateValues(:, 3) & gateValues(:, 4))
        error('llc:InvalidOutput', 'Applied gates contain bridge-leg overlap.');
    end
    if any(controlValues(:, 1) <= 0)
        error('llc:InvalidOutput', 'Frequency must be positive.');
    end
    expectedExcitation = llc_evaluate_sinusoidal_excitation(excitationTimes, excitation);
    excitationError = max(abs(excitationValues - expectedExcitation), [], 'all');
    if excitationError > generation.validation_tolerances.excitation_abs_v
        error('llc:InvalidOutput', 'Logged excitation does not match its configured profile.');
    end
    tolerances = generation.validation_tolerances;
    if abs(stateTimes(1)) > tolerances.timing_s || ...
            any(abs(stateValues(1, :)) > tolerances.initial_state_abs)
        error('llc:InvalidOutput', 'Trajectory does not start from the requested zero state.');
    end
    stopTolerance = max(tolerances.stop_time_abs_s, ...
                        32 * eps(generation.simulation_stop_time_s));
    if abs(stateTimes(end) - generation.simulation_stop_time_s) > stopTolerance
        error('llc:InvalidOutput', 'Trajectory does not reach the configured stop time.');
    end
end

function requireTimeSeries(times, values, width, name)
    if ~isvector(times) || isempty(times) || any(~isfinite(times)) || ...
            any(diff(times) <= 0) || size(values, 1) ~= numel(times) || ...
            size(values, 2) ~= width || any(~isfinite(values), 'all')
        error('llc:InvalidOutput', ...
              '%s output has invalid time, dimensions or values.', name);
    end
end

function validateHardware(config)
    if ~isfield(config, 'schema_version') || ...
            ~strcmp(config.schema_version, 'llc-hardware-v1')
        error('llc:InvalidConfig', 'Unsupported hardware schema_version.');
    end
    requireNames(config, 'state_names', {'i_Lr', 'v_Cr', 'i_Lm', 'v_o'});
    requireNames(config, 'state_units', {'A', 'V', 'A', 'V'});
    requireNames(config, 'gate_names', ...
                 {'Q1_A_high', 'Q2_A_low', 'Q3_B_high', 'Q4_B_low'});
    positive = {'turns_ratio_np_ns', 'resonant_inductance_h', ...
                'resonant_capacitance_f', 'magnetizing_inductance_h', ...
                'output_capacitance_f', 'resonant_winding_resistance_ohm', ...
                'magnetizing_series_resistance_ohm', 'switch_on_resistance_ohm', ...
                'switch_off_conductance_s', 'body_diode_forward_v', ...
                'body_diode_on_resistance_ohm', 'rectifier_forward_v', ...
                'rectifier_on_resistance_ohm', 'diode_off_conductance_s'};
    for index = 1:numel(positive)
        requirePositive(config, positive{index});
    end
end

function validateGeneration(config)
    if ~isfield(config, 'schema_version')
        error('llc:InvalidConfig', ...
              'Unsupported sinusoidal-excitation schema_version.');
    end
    screening = strcmp(config.schema_version, ...
                       'llc-sinusoidal-excitation-data-generation-v1');
    formal = strcmp(config.schema_version, ...
                    'llc-continuous-random-excitation-data-generation-v1');
    if ~screening && ~formal
        error('llc:InvalidConfig', ...
              'Unsupported sinusoidal-excitation schema_version.');
    end
    requireText(config, 'model_file');
    requireText(config, 'output_directory');
    requirePositive(config, 'fixed_load_ohm');
    requirePositive(config, 'simulation_stop_time_s');
    requiredStructs = {'solver', 'sampling', 'control', 'excitation_timing', ...
                       'validation_tolerances'};
    for index = 1:numel(requiredStructs)
        name = requiredStructs{index};
        if ~isfield(config, name) || ~isstruct(config.(name)) || ...
                ~isscalar(config.(name))
            error('llc:IncompleteConfig', '%s must be a JSON object.', name);
        end
    end
    if screening
        validateScreeningTiming(config);
    else
        validateFormalTiming(config);
        if ~isfield(config, 'random_excitation') || ...
                ~isstruct(config.random_excitation) || ...
                ~isscalar(config.random_excitation)
            error('llc:IncompleteConfig', ...
                  'random_excitation must be a JSON object.');
        end
        requireText(config, 'mat_file_version');
        if ~strcmp(config.mat_file_version, '-v7')
            error('llc:InvalidConfig', ...
                  'Formal continuous-random data must use MAT -v7.');
        end
    end
    if ~isfield(config, 'initial_state') || ~isnumeric(config.initial_state) || ...
            numel(config.initial_state) ~= 4 || any(config.initial_state ~= 0)
        error('llc:InvalidConfig', 'initial_state must contain four zeros.');
    end
    requireText(config.solver, 'name');
    if ~strcmp(config.solver.name, 'ode23t')
        error('llc:InvalidConfig', 'This model requires solver.name = ode23t.');
    end
    solverFields = {'max_step_s', 'relative_tolerance', 'absolute_tolerance'};
    for index = 1:numel(solverFields)
        requirePositive(config.solver, solverFields{index});
    end
    if ~isfield(config.sampling, 'method') || ...
            ~strcmp(config.sampling.method, 'uniform_reference_cycle_v1') || ...
            ~isfield(config.sampling, 'state_interpolation') || ...
            ~strcmp(config.sampling.state_interpolation, 'linear') || ...
            ~isfield(config.sampling, 'interval_signal_sampling') || ...
            ~strcmp(config.sampling.interval_signal_sampling, 'midpoint_previous')
        error('llc:InvalidConfig', 'Unsupported fixed-grid sampling contract.');
    end
    requirePositive(config.sampling, 'reference_frequency_hz');
    requirePositive(config.sampling, 'intervals_per_cycle');
    if config.sampling.intervals_per_cycle ~= ...
            round(config.sampling.intervals_per_cycle)
        error('llc:InvalidConfig', 'sampling.intervals_per_cycle must be an integer.');
    end
    controlFields = {'control_period_s', 'voltage_filter_time_constant_s', ...
                     'reference_ramp_time_s', 'initial_frequency_hz', ...
                     'bias_frequency_hz', 'frequency_slew_rate_hz_per_s', ...
                     'proportional_gain_hz_per_v', 'integral_gain_hz_per_v_s', ...
                     'deadtime_s', 'minimum_pulse_s'};
    for index = 1:numel(controlFields)
        requirePositive(config.control, controlFields{index});
    end
    requirePositive(config.control, 'minimum_frequency_hz');
    requirePositive(config.control, 'maximum_frequency_hz');
    if config.control.minimum_frequency_hz >= config.control.maximum_frequency_hz || ...
            config.control.initial_frequency_hz < config.control.minimum_frequency_hz || ...
            config.control.initial_frequency_hz > config.control.maximum_frequency_hz
        error('llc:InvalidConfig', ...
              'The initial frequency must lie inside valid frequency bounds.');
    end
    if 0.5 / config.control.maximum_frequency_hz <= ...
            config.control.deadtime_s + config.control.minimum_pulse_s
        error('llc:InvalidConfig', ...
              'maximum_frequency_hz is incompatible with switching timing.');
    end
    toleranceFields = {'timing_s', 'initial_state_abs', ...
                       'stop_time_abs_s', 'excitation_abs_v'};
    for index = 1:numel(toleranceFields)
        requirePositive(config.validation_tolerances, toleranceFields{index});
    end
    intervals = config.simulation_stop_time_s * ...
        config.sampling.reference_frequency_hz * config.sampling.intervals_per_cycle;
    if abs(intervals - round(intervals)) > 64 * eps(max(intervals, 1))
        error('llc:SamplingAlignment', ...
              'simulation_stop_time_s must end on the fixed grid.');
    end
    if ~isfield(config, 'conditions') || ~isstruct(config.conditions) || ...
            isempty(config.conditions)
        error('llc:IncompleteConfig', 'conditions must contain trajectory objects.');
    end
    identifiers = strings(numel(config.conditions), 1);
    seeds = nan(numel(config.conditions), 1);
    for index = 1:numel(config.conditions)
        condition = config.conditions(index);
        requireText(condition, 'trajectory_id');
        requireText(condition, 'dataset_role');
        requireText(condition, 'excitation_group');
        if isempty(regexp(condition.trajectory_id, ...
                          '^[A-Za-z0-9][A-Za-z0-9_-]*$', 'once'))
            error('llc:InvalidCondition', ...
                  'trajectory_id contains unsupported characters.');
        end
        if screening
            validateSignal(condition, 'vin');
            validateSignal(condition, 'reference');
        else
            role = string(condition.dataset_role);
            if ~isscalar(role) || ...
                    ~any(role == ["train", "validation", "test"])
                error('llc:InvalidCondition', 'Unsupported dataset_role.');
            end
            requireNonnegative(condition, 'random_seed');
            if condition.random_seed ~= round(condition.random_seed) || ...
                    condition.random_seed > double(intmax('uint32'))
                error('llc:InvalidCondition', ...
                      'random_seed must be a uint32-compatible integer.');
            end
            seeds(index) = condition.random_seed;
        end
        llc_evaluate_sinusoidal_excitation(0, buildExcitation(config, condition));
        identifiers(index) = string(condition.trajectory_id);
    end
    if numel(unique(identifiers)) ~= numel(identifiers)
        error('llc:InvalidConfig', 'condition trajectory_id values must be unique.');
    end
    if formal && numel(unique(seeds)) ~= numel(seeds)
        error('llc:InvalidConfig', 'condition random_seed values must be unique.');
    end
end

function validateScreeningTiming(config)
    requireNonnegative(config.excitation_timing, 'start_time_s');
    requireNonnegative(config.excitation_timing, 'envelope_rise_time_s');
    if config.excitation_timing.start_time_s + ...
            config.excitation_timing.envelope_rise_time_s >= ...
            config.simulation_stop_time_s
        error('llc:InvalidConfig', ...
              'Excitation envelope must finish before stop time.');
    end
end

function validateFormalTiming(config)
    timing = config.excitation_timing;
    requireNonnegative(timing, 'start_time_s');
    requirePositive(timing, 'segment_duration_s');
    requirePositive(timing, 'transition_time_s');
    requirePositive(timing, 'segment_count');
    if timing.segment_count ~= round(timing.segment_count) || ...
            timing.transition_time_s > timing.segment_duration_s
        error('llc:InvalidConfig', ...
              'Formal segment timing is invalid.');
    end
    expectedStop = timing.start_time_s + ...
        timing.segment_count * timing.segment_duration_s;
    if abs(expectedStop - config.simulation_stop_time_s) > ...
            64 * eps(max(expectedStop, 1))
        error('llc:InvalidConfig', ...
              'Formal stop time must contain startup and all segments exactly.');
    end
end

function validateSignal(condition, name)
    if ~isfield(condition, name) || ~isstruct(condition.(name)) || ...
            ~isscalar(condition.(name))
        error('llc:IncompleteConfig', '%s must be a JSON object.', name);
    end
    signal = condition.(name);
    requirePositive(signal, 'center_value');
    requireNonnegative(signal, 'amplitude');
    requireNonnegative(signal, 'frequency_hz');
    requireFinite(signal, 'phase_rad');
end

function path = resolveProjectPath(projectRoot, configuredPath)
    configuredPath = char(configuredPath);
    isWindowsDrivePath = ~isempty(regexp(configuredPath, '^[A-Za-z]:[\\/]', 'once'));
    if startsWith(configuredPath, filesep) || startsWith(configuredPath, '/') || ...
            isWindowsDrivePath
        path = configuredPath;
    else
        path = fullfile(projectRoot, configuredPath);
    end
end

function requireNames(value, name, expected)
    if ~isfield(value, name) || ~isequal(cellstr(value.(name)), expected(:))
        error('llc:InvalidConfig', ...
              '%s does not match the shared Simulink model channel order.', name);
    end
end

function requirePositive(value, name)
    requireFinite(value, name);
    if value.(name) <= 0
        error('llc:IncompleteConfig', '%s must be positive.', name);
    end
end

function requireNonnegative(value, name)
    requireFinite(value, name);
    if value.(name) < 0
        error('llc:IncompleteConfig', '%s must be nonnegative.', name);
    end
end

function requireFinite(value, name)
    if ~isfield(value, name) || ~isnumeric(value.(name)) || ...
            ~isscalar(value.(name)) || ~isfinite(value.(name))
        error('llc:IncompleteConfig', '%s must be a finite scalar.', name);
    end
end

function requireText(value, name)
    if ~isfield(value, name) || ...
            ~(ischar(value.(name)) || ...
              (isstring(value.(name)) && isscalar(value.(name)))) || ...
            strlength(string(value.(name))) == 0
        error('llc:IncompleteConfig', '%s must be nonempty text.', name);
    end
end
