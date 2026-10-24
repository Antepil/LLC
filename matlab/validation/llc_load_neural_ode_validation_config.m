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
                'model_export', 'reference_plant', 'solver', 'comparison_sampling', ...
                'experiments', 'metrics', 'output'};
    missing = required(~isfield(config, required));
    if ~isempty(missing)
        error('llc:IncompleteConfig', 'Validation config is missing: %s', strjoin(missing, ', '));
    end
    config.project_root = projectRoot;
    config.hardware_config_file = resolvePath(projectRoot, config.hardware_config_file);
    config.reference_plant.model_file = resolvePath(projectRoot, config.reference_plant.model_file);
    config.model_export.file = resolvePath(projectRoot, config.model_export.file);
    config.output.directory = resolvePath(projectRoot, config.output.directory);
    requireFile(config.hardware_config_file, 'hardware config');
    requireFile(config.reference_plant.model_file, 'reference plant');
    if requireModelExport
        requireFile(config.model_export.file, 'model export');
    end
    config.control = referencedSection(projectRoot, config.control_config);
    config.steady_state_assessment = referencedSection(projectRoot, config.steady_state_config);
    validateSolver(config.solver);
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
    if ~isfield(value, 'condition') || ~isstruct(value.condition) || ...
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
    if strcmp(value.startup.mode, 'warm')
        requirePositive(value.startup, 'preheat_duration_s');
        if ~all(isfield(value.startup, {'require_steady_state', 'extend_preheat_on_failure'})) || ...
                ~islogical(value.startup.require_steady_state) || ...
                ~islogical(value.startup.extend_preheat_on_failure) || ...
                value.startup.extend_preheat_on_failure
            error('llc:InvalidConfig', 'Warm startup must require explicit non-extending preheat policy.');
        end
        validateSignal(value.preheat_vin, 'linear');
        validateSignal(value.preheat_reference, 'previous');
    end
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
