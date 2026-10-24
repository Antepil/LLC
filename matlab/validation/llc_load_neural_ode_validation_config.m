function config = llc_load_neural_ode_validation_config(configFile, requireModelExport)
    % LLC_LOAD_NEURAL_ODE_VALIDATION_CONFIG Resolve and validate validation inputs.
    if nargin < 2
        requireModelExport = true;
    end
    if ~islogical(requireModelExport) || ~isscalar(requireModelExport)
        error('llc:InvalidArgument', 'requireModelExport must be a logical scalar.');
    end
    config = llc_read_json(configFile);
    if ~isfield(config, 'schema_version') || ...
            ~strcmp(config.schema_version, 'llc-neural-ode-validation-v2')
        error('llc:InvalidConfig', 'Unsupported neural ODE validation schema_version.');
    end
    matlabRoot = fileparts(fileparts(mfilename('fullpath')));
    projectRoot = fileparts(matlabRoot);
    required = {'hardware_config_file', 'control_config', 'steady_state_config', ...
                'model_export', 'reference_plant', 'paths', 'solver', 'comparison_sampling', ...
                'experiments', 'metrics', 'output'};
    missing = required(~isfield(config, required));
    if ~isempty(missing)
        error('llc:IncompleteConfig', 'Validation config is missing: %s', strjoin(missing, ', '));
    end
    config.project_root = projectRoot;
    requireText(config.output, 'directory');
    if ~isstruct(config.paths) || ~isscalar(config.paths)
        error('llc:InvalidConfig', 'paths must be a JSON object.');
    end
    requireText(config.paths, 'simulink_comparison_model_file');
    config.hardware_config_file = resolvePath(projectRoot, config.hardware_config_file);
    config.reference_plant.model_file = resolvePath(projectRoot, config.reference_plant.model_file);
    config.paths.simulink_comparison_model_file = resolvePath( ...
        projectRoot, config.paths.simulink_comparison_model_file);
    config.model_export.file = resolvePath(projectRoot, config.model_export.file);
    config.output.directory = resolvePath(projectRoot, config.output.directory);
    requireFile(config.hardware_config_file, 'hardware config');
    requireFile(config.reference_plant.model_file, 'reference plant');
    requireFile(config.paths.simulink_comparison_model_file, 'Simulink comparison model');
    if requireModelExport
        requireFile(config.model_export.file, 'model export');
    end
    config.control = referencedSection(projectRoot, config.control_config);
    config.steady_state_assessment = referencedSection(projectRoot, config.steady_state_config);
    validateModelExport(config.model_export);
    validateSolver(config.solver);
    validateSampling(config.comparison_sampling);
    validateMetrics(config.metrics);
    validateOutput(config.output);
    config.experiments = normalizeExperiments(config.experiments, projectRoot);
end

function experiments = normalizeExperiments(raw, projectRoot)
    if isstruct(raw)
        values = num2cell(raw);
    elseif iscell(raw)
        values = raw;
    else
        error('llc:InvalidConfig', 'experiments must be a JSON array.');
    end
    template = struct('experiment_id', '', 'enabled', false, 'mode', '', ...
        'reference_trajectory_file', '', 'startup', [], 'condition', [], ...
        'vin', [], 'reference', [], 'preheat_vin', [], 'preheat_reference', []);
    experiments = repmat(template, numel(values), 1);
    identifiers = strings(numel(values), 1);
    for index = 1:numel(values)
        value = values{index};
        requireText(value, 'experiment_id');
        requireText(value, 'mode');
        if ~isfield(value, 'enabled') || ~islogical(value.enabled) || ~isscalar(value.enabled)
            error('llc:InvalidConfig', 'experiment.enabled must be a logical scalar.');
        end
        names = intersect(fieldnames(template), fieldnames(value));
        for nameIndex = 1:numel(names)
            experiments(index).(names{nameIndex}) = value.(names{nameIndex});
        end
        identifiers(index) = string(value.experiment_id);
        if strcmp(value.mode, 'recorded_gate_replay')
            requireText(value, 'reference_trajectory_file');
            experiments(index).reference_trajectory_file = ...
                resolvePath(projectRoot, value.reference_trajectory_file);
        elseif strcmp(value.mode, 'independent_closed_loop')
            validateClosedLoopExperiment(value);
        else
            error('llc:InvalidConfig', 'Unsupported experiment mode: %s', value.mode);
        end
    end
    if numel(unique(identifiers)) ~= numel(identifiers)
        error('llc:InvalidConfig', 'experiment_id values must be unique.');
    end
end

function validateClosedLoopExperiment(value)
    if ~all(isfield(value, {'condition', 'startup', 'vin', 'reference'}))
        error('llc:InvalidConfig', 'Closed-loop experiment is incomplete.');
    end
    if ~isstruct(value.condition) || ...
            ~isfield(value.condition, 'load_ohm') || value.condition.load_ohm ~= 2.304
        error('llc:InvalidConfig', 'Closed-loop validation requires fixed load 2.304 ohm.');
    end
    if ~isfield(value, 'startup') || ~isstruct(value.startup) || ...
            ~isfield(value.startup, 'mode') || ...
            ~ismember(string(value.startup.mode), ["cold", "warm"])
        error('llc:InvalidConfig', 'startup.mode must be cold or warm.');
    end
    requirePositive(value.startup, 'evaluation_duration_s');
    validateSignal(value.vin, 'linear');
    validateSignal(value.reference, 'previous');
    validateSignalCoverage(value.vin, value.startup.evaluation_duration_s);
    validateSignalCoverage(value.reference, value.startup.evaluation_duration_s);
    if strcmp(value.startup.mode, 'warm')
        if ~all(isfield(value, {'preheat_vin', 'preheat_reference'}))
            error('llc:InvalidConfig', 'Warm startup preheat signals are missing.');
        end
        requirePositive(value.startup, 'preheat_duration_s');
        if ~all(isfield(value.startup, {'require_steady_state', 'extend_preheat_on_failure'})) || ...
                ~islogical(value.startup.require_steady_state) || ...
                ~value.startup.require_steady_state || ...
                ~islogical(value.startup.extend_preheat_on_failure) || ...
                value.startup.extend_preheat_on_failure
            error('llc:InvalidConfig', ...
                  'Warm startup must require steady state and use a non-extending preheat policy.');
        end
        validateConstantSignal(value.preheat_vin, 'preheat_vin');
        validateConstantSignal(value.preheat_reference, 'preheat_reference');
    end
end

function validateSignalCoverage(signal, duration)
    llc_evaluate_validation_signal(signal, [0, duration]);
end

function validateConstantSignal(signal, name)
    if ~isstruct(signal) || ~isscalar(signal) || ~isfield(signal, 'type') || ...
            ~strcmp(signal.type, 'constant')
        error('llc:InvalidConfig', '%s must be a constant preheat signal.', name);
    end
    requirePositive(signal, 'value_v');
end

function validateSignal(signal, requiredInterpolation)
    if ~isstruct(signal) || ~isscalar(signal) || ~isfield(signal, 'type')
        error('llc:InvalidConfig', 'Validation signals must be JSON objects with type.');
    end
    if strcmp(signal.type, 'constant')
        requirePositive(signal, 'value_v');
    elseif strcmp(signal.type, 'time_series')
        if ~isfield(signal, 'interpolation') || ~strcmp(signal.interpolation, requiredInterpolation)
            error('llc:InvalidConfig', 'Signal interpolation must be %s.', requiredInterpolation);
        end
        llc_evaluate_validation_signal(signal, signal.times_s);
    else
        error('llc:InvalidConfig', 'Unsupported validation signal type.');
    end
end

function validateSolver(solver)
    requireText(solver, 'name');
    if ~strcmp(solver.name, 'ode23t')
        error('llc:InvalidConfig', 'Validation solver must be ode23t.');
    end
    for name = {'max_step_s', 'relative_tolerance', 'absolute_tolerance', 'event_time_tolerance_s'}
        requirePositive(solver, name{1});
    end
    if ~isfield(solver, 'fail_on_nonfinite_state') || ...
            ~islogical(solver.fail_on_nonfinite_state) || ~isscalar(solver.fail_on_nonfinite_state) || ...
            ~solver.fail_on_nonfinite_state
        error('llc:InvalidConfig', 'solver.fail_on_nonfinite_state must be true.');
    end
    if ~isfield(solver, 'convergence_check') || ~isstruct(solver.convergence_check) || ...
            ~isscalar(solver.convergence_check) || ...
            ~all(isfield(solver.convergence_check, ...
                {'enabled', 'max_step_factor', 'tolerance_factor'})) || ...
            ~islogical(solver.convergence_check.enabled) || ...
            ~isscalar(solver.convergence_check.enabled)
        error('llc:InvalidConfig', 'solver.convergence_check is incomplete.');
    end
    refinement = [solver.convergence_check.max_step_factor, ...
                  solver.convergence_check.tolerance_factor];
    if any(~isfinite(refinement)) || any(refinement <= 0) || any(refinement >= 1)
        error('llc:InvalidConfig', ...
              'Convergence refinement factors must be between zero and one.');
    end
end

function validateModelExport(value)
    requireText(value, 'required_schema_version');
    if ~strcmp(value.required_schema_version, 'llc-neural-ode-matlab-export-v1')
        error('llc:InvalidConfig', 'Unsupported required model export schema.');
    end
    for name = {'load_abs_tolerance_ohm', 'consistency_abs_tolerance', ...
                'consistency_relative_tolerance'}
        requirePositive(value, name{1});
    end
end

function validateSampling(value)
    for name = {'reference_frequency_hz', 'intervals_per_cycle'}
        requirePositive(value, name{1});
    end
    if value.intervals_per_cycle ~= round(value.intervals_per_cycle) || ...
            ~isfield(value, 'state_interpolation') || ...
            ~strcmp(value.state_interpolation, 'linear') || ...
            ~isfield(value, 'gate_interpolation') || ...
            ~strcmp(value.gate_interpolation, 'previous')
        error('llc:InvalidConfig', ...
              'Comparison sampling requires integer intervals and linear/previous interpolation.');
    end
end

function validateMetrics(value)
    expectedStates = ["rmse", "normalized_rmse_by_model_state_scale", ...
                      "maximum_absolute_error", "final_absolute_error"];
    expectedOutput = ["mean_absolute_error", "reference_tracking_rmse", "tail_mean_error"];
    if ~isstruct(value) || ~isscalar(value) || ...
            ~all(isfield(value, {'state_metrics', 'output_voltage_metrics', ...
                                 'tail_window_s', 'acceptance_thresholds'})) || ...
            ~isequal(string(value.state_metrics(:)).', expectedStates) || ...
            ~isequal(string(value.output_voltage_metrics(:)).', expectedOutput) || ...
            ~isempty(value.acceptance_thresholds)
        error('llc:InvalidConfig', 'Metric declarations do not match the supported contract.');
    end
    requirePositive(value, 'tail_window_s');
end

function validateOutput(value)
    required = {'directory', 'save_resolved_config', 'save_waveforms', ...
                'save_metric_tables', 'save_figures'};
    if ~isstruct(value) || ~isscalar(value) || ~all(isfield(value, required)) || ...
            strlength(string(value.directory)) == 0
        error('llc:InvalidConfig', 'Output configuration is incomplete.');
    end
    for index = 2:numel(required)
        flag = value.(required{index});
        if ~islogical(flag) || ~isscalar(flag)
            error('llc:InvalidConfig', 'Output save switches must be logical scalars.');
        end
    end
end

function value = referencedSection(projectRoot, reference)
    requireText(reference, 'file');
    requireText(reference, 'section');
    file = resolvePath(projectRoot, reference.file);
    root = llc_read_json(file);
    if ~isfield(root, reference.section) || ~isstruct(root.(reference.section))
        error('llc:IncompleteConfig', 'Referenced section %s is missing from %s.', reference.section, file);
    end
    value = root.(reference.section);
end

function path = resolvePath(projectRoot, configured)
    configured = char(string(configured));
    if startsWith(configured, filesep) || ~isempty(regexp(configured, '^[A-Za-z]:[\\/]', 'once'))
        path = configured;
    else
        path = fullfile(projectRoot, configured);
    end
end

function requireFile(file, label)
    if ~isfile(file)
        error('llc:MissingValidationInput', '%s does not exist: %s', label, file);
    end
end

function requirePositive(value, name)
    if ~isfield(value, name) || ~isnumeric(value.(name)) || ~isscalar(value.(name)) || ...
            ~isfinite(value.(name)) || value.(name) <= 0
        error('llc:InvalidConfig', '%s must be a finite positive scalar.', name);
    end
end

function requireText(value, name)
    if ~isstruct(value) || ~isfield(value, name) || ...
            ~(ischar(value.(name)) || (isstring(value.(name)) && isscalar(value.(name)))) || ...
            strlength(string(value.(name))) == 0
        error('llc:InvalidConfig', '%s must be nonempty text.', name);
    end
end
