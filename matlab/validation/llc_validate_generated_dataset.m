function [report, summaryTable, stateMetrics, controlMetrics] = ...
        llc_validate_generated_dataset(trajectoryInput, configFile, reportDirectory)
    % LLC_VALIDATE_GENERATED_DATASET Validate saved trajectories and write reports.
    %
    % The structural checks deliberately avoid assumptions about PFM, PWM or
    % phase-shift switching sequences. Modulation-specific checks belong in a
    % separate validator and can consume this function's report.
    matlabRoot = fileparts(fileparts(mfilename('fullpath')));
    projectRoot = fileparts(matlabRoot);
    addpath(matlabRoot);
    setup_paths;

    if nargin < 2 || isempty(configFile)
        configFile = fullfile(projectRoot, 'configs', 'matlab', 'dataset_validation.json');
    end
    config = llc_read_json(configFile);
    validateConfig(config);
    if nargin < 1 || isempty(trajectoryInput)
        trajectoryInput = config.trajectory_directory;
    end
    if nargin < 3 || isempty(reportDirectory)
        reportDirectory = config.report_directory;
    end

    files = resolveInputFiles(projectRoot, trajectoryInput, config.file_pattern);
    reportDirectory = resolveProjectPath(projectRoot, reportDirectory);
    if ~isfolder(reportDirectory)
        mkdir(reportDirectory);
    end

    entries = repmat(emptyEntry, numel(files), 1);
    stateMetrics = emptyStateMetrics;
    controlMetrics = emptyControlMetrics;
    for index = 1:numel(files)
        try
            [entries(index), states, controls] = ...
                llc_validate_saved_trajectory(files(index), config);
            stateMetrics = [stateMetrics; states]; %#ok<AGROW>
            controlMetrics = [controlMetrics; controls]; %#ok<AGROW>
        catch exception
            entries(index) = emptyEntry;
            entries(index).file = files(index);
            entries(index).issue_codes = "VALIDATOR_EXCEPTION";
            entries(index).issue_messages = string(exception.message);
            entries(index).issue_count = 1;
        end
    end
    identifiers = string({entries.trajectory_id}).';
    uniqueIdentifiers = unique(identifiers(strlength(identifiers) > 0));
    for identifierIndex = 1:numel(uniqueIdentifiers)
        duplicates = find(identifiers == uniqueIdentifiers(identifierIndex));
        if numel(duplicates) > 1
            for duplicateIndex = duplicates.'
                entries(duplicateIndex).structural_valid = false;
                entries(duplicateIndex).issue_count = entries(duplicateIndex).issue_count + 1;
                entries(duplicateIndex).issue_codes = appendIssueText( ...
                    entries(duplicateIndex).issue_codes, "DUPLICATE_TRAJECTORY_ID");
                entries(duplicateIndex).issue_messages = appendIssueText( ...
                    entries(duplicateIndex).issue_messages, ...
                    "trajectory_id is repeated in the validated dataset.");
            end
        end
    end

    summaryTable = struct2table(entries);
    report = struct;
    report.schema_version = 'llc-dataset-validation-report-v1';
    report.generated_at_utc = char(datetime('now', 'TimeZone', 'UTC', ...
                                             'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
    report.config = config;
    report.input_files = cellstr(files);
    report.trajectory_count = numel(files);
    report.structurally_valid_count = nnz(summaryTable.structural_valid);
    report.structurally_invalid_count = nnz(~summaryTable.structural_valid);
    report.tail_output_within_tolerance_count = ...
        nnz(summaryTable.output_metric_available & summaryTable.output_tail_within_tolerance);
    report.summary = entries;

    prefix = char(config.report_file_prefix);
    summaryFile = fullfile(reportDirectory, [prefix '_summary.csv']);
    stateFile = fullfile(reportDirectory, [prefix '_state_metrics.csv']);
    controlFile = fullfile(reportDirectory, [prefix '_control_metrics.csv']);
    reportFile = fullfile(reportDirectory, [prefix '_report.mat']);
    report.output_files = struct('summary_csv', summaryFile, ...
                                 'state_metrics_csv', stateFile, ...
                                 'control_metrics_csv', controlFile, ...
                                 'report_mat', reportFile);
    writetable(summaryTable, summaryFile);
    writetable(stateMetrics, stateFile);
    writetable(controlMetrics, controlFile);
    save(reportFile, 'report', 'summaryTable', 'stateMetrics', 'controlMetrics', '-v7');
end

function files = resolveInputFiles(projectRoot, input, pattern)
    input = string(input);
    if ~isvector(input) || isempty(input) || any(strlength(input) == 0)
        error('llc:InvalidValidationInput', 'trajectoryInput must contain paths.');
    end
    if isscalar(input)
        candidate = resolveProjectPath(projectRoot, input);
        if isfolder(candidate)
            listing = dir(fullfile(candidate, char(pattern)));
            listing = listing(~[listing.isdir]);
            [~, order] = sort({listing.name});
            listing = listing(order);
            files = string(fullfile({listing.folder}, {listing.name})).';
        elseif isfile(candidate)
            files = string(candidate);
        else
            error('llc:MissingValidationInput', 'Trajectory input does not exist: %s', candidate);
        end
    else
        files = strings(numel(input), 1);
        for index = 1:numel(input)
            files(index) = string(resolveProjectPath(projectRoot, input(index)));
            if ~isfile(files(index))
                error('llc:MissingValidationInput', 'Trajectory file does not exist: %s', files(index));
            end
        end
    end
    if isempty(files)
        error('llc:MissingValidationInput', 'No trajectory files match %s.', pattern);
    end
end

function path = resolveProjectPath(projectRoot, configuredPath)
    configuredPath = char(string(configuredPath));
    if isAbsolutePath(configuredPath)
        path = configuredPath;
    else
        path = fullfile(projectRoot, configuredPath);
    end
end

function tf = isAbsolutePath(path)
    tf = startsWith(path, filesep) || startsWith(path, '/') || ...
         (~isempty(regexp(path, '^[A-Za-z]:[\\/]', 'once'))) || startsWith(path, '\\');
end

function validateConfig(config)
    required = {'schema_version', 'trajectory_directory', 'file_pattern', ...
                'report_directory', 'report_file_prefix', ...
                'accepted_trajectory_schema_versions', 'required_state_names', ...
                'required_state_units', 'required_condition_fields', ...
                'mutually_exclusive_gate_groups', 'optional_positive_control_names', ...
                'initial_state_abs_tolerance', 'time_abs_tolerance_s', ...
                'fixed_grid_relative_tolerance', 'tail_window_s', ...
                'output_state_name', 'reference_condition_field', ...
                'output_mean_relative_tolerance'};
    if ~isstruct(config) || ~isscalar(config) || ~all(isfield(config, required)) || ...
            ~strcmp(config.schema_version, 'llc-dataset-validation-v1')
        error('llc:InvalidValidationConfig', 'Dataset-validation configuration is incomplete.');
    end
    scalarTextFields = {'trajectory_directory', 'file_pattern', 'report_directory', ...
                        'report_file_prefix', 'output_state_name', ...
                        'reference_condition_field'};
    for index = 1:numel(scalarTextFields)
        value = string(config.(scalarTextFields{index}));
        if ~isscalar(value) || strlength(value) == 0
            error('llc:InvalidValidationConfig', '%s must be nonempty text.', scalarTextFields{index});
        end
    end
    textLists = {'accepted_trajectory_schema_versions', 'required_state_names', ...
                 'required_state_units', 'required_condition_fields', ...
                 'optional_positive_control_names'};
    for index = 1:numel(textLists)
        values = string(config.(textLists{index}));
        if isempty(values) || any(strlength(values) == 0)
            error('llc:InvalidValidationConfig', '%s must contain nonempty names.', textLists{index});
        end
    end
    if numel(config.required_state_names) ~= numel(config.required_state_units)
        error('llc:InvalidValidationConfig', 'State names and units must have equal lengths.');
    end
    groups = config.mutually_exclusive_gate_groups;
    if ~iscell(groups) || any(cellfun(@(group) numel(group) < 2, groups))
        error('llc:InvalidValidationConfig', ...
              'mutually_exclusive_gate_groups must contain groups of at least two names.');
    end
    positive = {'initial_state_abs_tolerance', 'time_abs_tolerance_s', ...
                'fixed_grid_relative_tolerance', 'tail_window_s', ...
                'output_mean_relative_tolerance'};
    for index = 1:numel(positive)
        value = config.(positive{index});
        if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value) || value <= 0
            error('llc:InvalidValidationConfig', '%s must be a finite positive scalar.', positive{index});
        end
    end
end

function result = appendIssueText(existing, addition)
    if strlength(existing) == 0
        result = string(addition);
    else
        result = string(existing) + ";" + string(addition);
    end
end

function entry = emptyEntry
    entry = struct('file', "", 'trajectory_id', "", 'schema_version', "", ...
                   'structural_valid', false, 'issue_count', 0, ...
                   'issue_codes', "", 'issue_messages', "", ...
                   'duration_s', NaN, 'state_sample_count', 0, ...
                   'interval_count', 0, 'gate_overlap_count', 0, ...
                   'output_metric_available', false, 'output_tail_mean', NaN, ...
                   'output_tail_error_pct', NaN, 'output_tail_peak_to_peak', NaN, ...
                   'output_tail_within_tolerance', false);
end

function value = emptyStateMetrics
    value = table('Size', [0, 10], ...
        'VariableTypes', {'string', 'string', 'string', 'double', 'double', ...
                          'double', 'double', 'double', 'double', 'double'}, ...
        'VariableNames', {'trajectory_id', 'state_name', 'unit', 'minimum', 'maximum', ...
                          'absolute_peak', 'tail_mean', 'tail_minimum', 'tail_maximum', ...
                          'tail_peak_to_peak'});
end

function value = emptyControlMetrics
    value = table('Size', [0, 7], ...
        'VariableTypes', {'string', 'string', 'string', 'double', 'double', 'double', 'double'}, ...
        'VariableNames', {'trajectory_id', 'control_name', 'unit', 'minimum', 'maximum', ...
                          'final_value', 'tail_mean'});
end
