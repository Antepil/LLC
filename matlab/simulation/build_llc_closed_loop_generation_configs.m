function files = build_llc_closed_loop_generation_configs(planFile, outputDirectory)
    % BUILD_LLC_CLOSED_LOOP_GENERATION_CONFIGS Expand the reviewed condition plan.
    matlabRoot = fileparts(fileparts(mfilename('fullpath')));
    projectRoot = fileparts(matlabRoot);
    addpath(matlabRoot);
    setup_paths;
    if nargin < 1 || isempty(planFile)
        planFile = fullfile(projectRoot, 'configs', 'matlab', ...
                            'closed_loop_condition_plan.json');
    end
    plan = llc_read_json(planFile);
    validatePlan(plan);
    templateFile = resolveProjectPath(projectRoot, plan.template_config);
    template = llc_read_json(templateFile);
    if nargin < 2 || isempty(outputDirectory)
        outputDirectory = fileparts(planFile);
    end
    if ~isfolder(outputDirectory)
        mkdir(outputDirectory);
    end

    checkCount = round((plan.simulation_stop_time_s - ...
                        plan.steady_state_check_start_s) / ...
                        plan.steady_state_check_step_s);
    checkTimes = plan.steady_state_check_start_s + ...
                 (0:checkCount)' * plan.steady_state_check_step_s;
    checkTimes(end) = plan.simulation_stop_time_s;
    files = strings(numel(plan.groups), 1);
    allKeys = strings(0, 1);

    for groupIndex = 1:numel(plan.groups)
        group = plan.groups(groupIndex);
        conditions = expandGroup(group);
        if numel(conditions) ~= group.expected_count
            error('llc:InvalidConditionPlan', ...
                  'Group %s expands to %d conditions, expected %d.', ...
                  group.name, numel(conditions), group.expected_count);
        end
        keys = compose('%.17g|%.17g|%.17g', [conditions.vin_v].', ...
                       [conditions.load_ohm].', [conditions.reference_voltage_v].');
        if numel(unique(keys)) ~= numel(keys) || any(ismember(keys, allKeys))
            error('llc:InvalidConditionPlan', ...
                  'Group %s contains duplicate or cross-group conditions.', group.name);
        end
        allKeys = [allKeys; keys]; %#ok<AGROW>

        config = template;
        config.schema_version = 'llc-closed-loop-data-generation-v4';
        config.output_directory = group.output_directory;
        config.simulation_stop_time_s = plan.simulation_stop_time_s;
        config.steady_state_assessment.check_times_s = checkTimes;
        config.conditions = conditions;
        outputFile = fullfile(outputDirectory, group.config_file);
        writeJson(outputFile, config);
        files(groupIndex) = string(outputFile);
    end
end

function conditions = expandGroup(group)
    vinValues = reshape(group.vin_v, 1, []);
    loadValues = reshape(group.load_ohm, 1, []);
    referenceValues = reshape(group.reference_voltage_v, 1, []);
    count = numel(vinValues) * numel(loadValues) * numel(referenceValues);
    template = struct('trajectory_id', '', 'dataset_role', group.dataset_role, ...
                      'generalization_type', group.generalization_type, ...
                      'vin_v', NaN, 'load_ohm', NaN, 'reference_voltage_v', NaN);
    conditions = repmat(template, count, 1);
    index = 0;
    for vin = vinValues
        for loadOhm = loadValues
            for referenceVoltage = referenceValues
                index = index + 1;
                conditions(index).trajectory_id = sprintf( ...
                    'closed_loop_%s_vin%s_r%s_vref%s', group.name, ...
                    encodeNumber(vin), encodeNumber(loadOhm), ...
                    encodeNumber(referenceVoltage));
                conditions(index).vin_v = vin;
                conditions(index).load_ohm = loadOhm;
                conditions(index).reference_voltage_v = referenceVoltage;
            end
        end
    end
end

function encoded = encodeNumber(value)
    encoded = strrep(strrep(sprintf('%.12g', value), '-', 'm'), '.', 'p');
end

function validatePlan(plan)
    if ~isfield(plan, 'schema_version') || ...
            ~strcmp(plan.schema_version, 'llc-closed-loop-condition-plan-v1')
        error('llc:InvalidConditionPlan', 'Unsupported condition-plan schema_version.');
    end
    requireText(plan, 'template_config');
    requirePositiveScalar(plan, 'simulation_stop_time_s');
    requirePositiveScalar(plan, 'steady_state_check_start_s');
    requirePositiveScalar(plan, 'steady_state_check_step_s');
    if plan.steady_state_check_start_s >= plan.simulation_stop_time_s
        error('llc:InvalidConditionPlan', ...
              'The steady-state check start must precede the simulation stop time.');
    end
    intervalCount = (plan.simulation_stop_time_s - ...
                     plan.steady_state_check_start_s) / ...
                     plan.steady_state_check_step_s;
    if abs(intervalCount - round(intervalCount)) > 64 * eps(max(intervalCount, 1))
        error('llc:InvalidConditionPlan', ...
              'The final checkpoint must align with the checkpoint step.');
    end
    if ~isfield(plan, 'groups') || ~isstruct(plan.groups) || isempty(plan.groups)
        error('llc:InvalidConditionPlan', 'groups must be a nonempty object array.');
    end
    names = strings(numel(plan.groups), 1);
    configFiles = names;
    outputDirectories = names;
    for index = 1:numel(plan.groups)
        group = plan.groups(index);
        for field = {'name', 'dataset_role', 'generalization_type', ...
                     'config_file', 'output_directory'}
            requireText(group, field{1});
        end
        requirePositiveInteger(group, 'expected_count');
        for field = {'vin_v', 'load_ohm', 'reference_voltage_v'}
            requirePositiveVector(group, field{1});
        end
        names(index) = string(group.name);
        configFiles(index) = string(group.config_file);
        outputDirectories(index) = string(group.output_directory);
    end
    if numel(unique(names)) ~= numel(names) || ...
            numel(unique(configFiles)) ~= numel(configFiles) || ...
            numel(unique(outputDirectories)) ~= numel(outputDirectories)
        error('llc:InvalidConditionPlan', ...
              'Group names, config files and output directories must be unique.');
    end
end

function path = resolveProjectPath(projectRoot, configuredPath)
    configuredPath = char(configuredPath);
    if startsWith(configuredPath, filesep)
        path = configuredPath;
    else
        path = fullfile(projectRoot, configuredPath);
    end
end

function writeJson(path, value)
    file = fopen(path, 'w');
    if file < 0
        error('llc:FileWrite', 'Could not open JSON output: %s', path);
    end
    cleanup = onCleanup(@()fclose(file));
    fwrite(file, jsonencode(value, PrettyPrint=true), 'char');
    fwrite(file, newline, 'char');
    clear cleanup;
end

function requireText(value, name)
    if ~isfield(value, name) || ...
            ~(ischar(value.(name)) || (isstring(value.(name)) && isscalar(value.(name)))) || ...
            strlength(string(value.(name))) == 0
        error('llc:InvalidConditionPlan', '%s must be nonempty text.', name);
    end
end

function requirePositiveScalar(value, name)
    if ~isfield(value, name) || ~isnumeric(value.(name)) || ...
            ~isscalar(value.(name)) || ~isfinite(value.(name)) || value.(name) <= 0
        error('llc:InvalidConditionPlan', '%s must be a finite positive scalar.', name);
    end
end

function requirePositiveInteger(value, name)
    requirePositiveScalar(value, name);
    if value.(name) ~= round(value.(name))
        error('llc:InvalidConditionPlan', '%s must be an integer.', name);
    end
end

function requirePositiveVector(value, name)
    if ~isfield(value, name) || ~isnumeric(value.(name)) || ...
            ~isvector(value.(name)) || isempty(value.(name)) || ...
            any(~isfinite(value.(name))) || any(value.(name) <= 0) || ...
            numel(unique(value.(name))) ~= numel(value.(name))
        error('llc:InvalidConditionPlan', ...
              '%s must be a nonempty unique finite positive vector.', name);
    end
end
