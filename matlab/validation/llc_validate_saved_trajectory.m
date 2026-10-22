function [entry, stateMetrics, controlMetrics] = ...
        llc_validate_saved_trajectory(trajectoryFile, config)
    % LLC_VALIDATE_SAVED_TRAJECTORY Run modulation-independent checks on one MAT file.
    %
    % This validator checks the saved data contract, time axes, finite values,
    % zero initial state, binary gates and bridge-leg overlap. It reports
    % waveform statistics without deciding whether a hardware stress is safe.
    trajectoryFile = string(trajectoryFile);
    entry = emptyEntry(trajectoryFile);
    stateMetrics = emptyStateMetrics;
    controlMetrics = emptyControlMetrics;
    issues = strings(0, 1);
    messages = strings(0, 1);

    try
        trajectory = load(trajectoryFile);
    catch exception
        addIssue("FILE_LOAD_FAILED", exception.message);
        finishEntry;
        return;
    end

    required = {'schema_version', 'trajectory_id', 'state_names', 'state_units', ...
                'state_times_s', 'state_values', 'gate_names', 'control_names', ...
                'control_units', 'interval_times_s', 'interval_gate_values', ...
                'interval_control_values', 'raw_state_times_s', 'raw_state_values', ...
                'raw_gate_times_s', 'raw_gate_values', 'raw_control_times_s', ...
                'raw_control_values', 'condition'};
    missing = required(~isfield(trajectory, required));
    if ~isempty(missing)
        addIssue("MISSING_REQUIRED_FIELDS", strjoin(string(missing), ', '));
        finishEntry;
        return;
    end

    entry.trajectory_id = scalarText(trajectory.trajectory_id);
    entry.schema_version = scalarText(trajectory.schema_version);
    if strlength(entry.trajectory_id) == 0
        addIssue("INVALID_TRAJECTORY_ID", "trajectory_id must be nonempty scalar text.");
    end
    if ~any(entry.schema_version == string(config.accepted_trajectory_schema_versions))
        addIssue("UNSUPPORTED_SCHEMA", entry.schema_version);
    end

    stateNames = textVector(trajectory.state_names);
    stateUnits = textVector(trajectory.state_units);
    gateNames = textVector(trajectory.gate_names);
    controlNames = textVector(trajectory.control_names);
    controlUnits = textVector(trajectory.control_units);
    expectedStateNames = string(config.required_state_names(:));
    expectedStateUnits = string(config.required_state_units(:));
    if ~isequal(stateNames, expectedStateNames)
        addIssue("STATE_NAMES_MISMATCH", strjoin(stateNames, ', '));
    end
    if ~isequal(stateUnits, expectedStateUnits)
        addIssue("STATE_UNITS_MISMATCH", strjoin(stateUnits, ', '));
    end
    if numel(unique(gateNames)) ~= numel(gateNames) || any(strlength(gateNames) == 0)
        addIssue("INVALID_GATE_NAMES", "Gate names must be unique and nonempty.");
    end
    if numel(unique(controlNames)) ~= numel(controlNames) || any(strlength(controlNames) == 0) || ...
            numel(controlNames) ~= numel(controlUnits)
        addIssue("INVALID_CONTROL_METADATA", "Control names and units are inconsistent.");
    end

    [fixedStateValid, fixedStateIssues] = validateSeries( ...
        trajectory.state_times_s, trajectory.state_values, numel(stateNames), "FIXED_STATE");
    appendIssues(fixedStateIssues);
    [rawStateValid, rawStateIssues] = validateSeries( ...
        trajectory.raw_state_times_s, trajectory.raw_state_values, numel(stateNames), "RAW_STATE");
    appendIssues(rawStateIssues);
    [rawGateValid, rawGateIssues] = validateSeries( ...
        trajectory.raw_gate_times_s, trajectory.raw_gate_values, numel(gateNames), "RAW_GATE");
    appendIssues(rawGateIssues);
    [rawControlValid, rawControlIssues] = validateSeries( ...
        trajectory.raw_control_times_s, trajectory.raw_control_values, numel(controlNames), "RAW_CONTROL");
    appendIssues(rawControlIssues);

    intervalTimes = trajectory.interval_times_s;
    fixedGates = trajectory.interval_gate_values;
    fixedControls = trajectory.interval_control_values;
    intervalValid = isvector(intervalTimes) && ~isempty(intervalTimes) && ...
                    all(isfinite(intervalTimes)) && all(diff(intervalTimes) > 0);
    if ~intervalValid
        addIssue("INVALID_INTERVAL_TIME", "Interval times must be finite and strictly increasing.");
    end
    gateMatrixValid = isnumeric(fixedGates) || islogical(fixedGates);
    gateMatrixValid = gateMatrixValid && ismatrix(fixedGates) && ...
        size(fixedGates, 1) == numel(intervalTimes) && size(fixedGates, 2) == numel(gateNames);
    if ~gateMatrixValid || any(~isfinite(double(fixedGates)), 'all')
        addIssue("INVALID_FIXED_GATE_SHAPE", "Fixed-grid gates have invalid dimensions or values.");
        gateMatrixValid = false;
    elseif any(fixedGates ~= 0 & fixedGates ~= 1, 'all')
        addIssue("NONBINARY_FIXED_GATES", "Fixed-grid gates must contain only zero or one.");
    end
    controlMatrixValid = isnumeric(fixedControls) && ismatrix(fixedControls) && ...
        size(fixedControls, 1) == numel(intervalTimes) && size(fixedControls, 2) == numel(controlNames);
    if ~controlMatrixValid || any(~isfinite(fixedControls), 'all')
        addIssue("INVALID_FIXED_CONTROL_SHAPE", "Fixed-grid controls have invalid dimensions or values.");
        controlMatrixValid = false;
    end

    if fixedStateValid
        times = trajectory.state_times_s(:);
        values = trajectory.state_values;
        entry.duration_s = times(end) - times(1);
        entry.state_sample_count = numel(times);
        if abs(times(1)) > config.time_abs_tolerance_s
            addIssue("NONZERO_START_TIME", "The fixed-grid trajectory must start at zero.");
        end
        if any(abs(values(1, :)) > config.initial_state_abs_tolerance)
            addIssue("NONZERO_INITIAL_STATE", "Initial state exceeds the configured tolerance.");
        end
        if numel(times) ~= numel(intervalTimes) + 1
            addIssue("FIXED_GRID_LENGTH_MISMATCH", ...
                     "State nodes must contain one more sample than interval signals.");
        elseif intervalValid
            expectedMidpoints = 0.5 * (times(1:end-1) + times(2:end));
            if any(abs(intervalTimes(:) - expectedMidpoints) > config.time_abs_tolerance_s)
                addIssue("INTERVAL_MIDPOINT_MISMATCH", ...
                         "Interval times are not state-grid midpoints.");
            end
        end
        steps = diff(times);
        nominalStep = median(steps);
        stepTolerance = config.time_abs_tolerance_s + ...
                        config.fixed_grid_relative_tolerance * nominalStep;
        if any(abs(steps - nominalStep) > stepTolerance)
            addIssue("NONUNIFORM_FIXED_GRID", "State samples are not uniformly spaced.");
        end
        stateMetrics = buildStateMetrics(entry.trajectory_id, stateNames, stateUnits, ...
                                         times, values, config.tail_window_s);
    end
    if intervalValid
        entry.interval_count = numel(intervalTimes);
    end

    if gateMatrixValid
        [overlapCount, groupIssue] = countGateOverlap(fixedGates, gateNames, ...
                                                      config.mutually_exclusive_gate_groups);
        if strlength(groupIssue) > 0
            addIssue("GATE_GROUP_NAMES_MISSING", groupIssue);
        end
        entry.gate_overlap_count = overlapCount;
        if overlapCount > 0
            addIssue("GATE_LEG_OVERLAP", sprintf('%d fixed-grid intervals overlap.', overlapCount));
        end
    end
    if rawGateValid
        if any(trajectory.raw_gate_values ~= 0 & trajectory.raw_gate_values ~= 1, 'all')
            addIssue("NONBINARY_RAW_GATES", "Raw gates must contain only zero or one.");
        end
        [rawOverlap, groupIssue] = countGateOverlap( ...
            trajectory.raw_gate_values, gateNames, config.mutually_exclusive_gate_groups);
        if strlength(groupIssue) == 0 && rawOverlap > 0
            addIssue("RAW_GATE_LEG_OVERLAP", sprintf('%d raw samples overlap.', rawOverlap));
        end
    end
    if controlMatrixValid
        controlMetrics = buildControlMetrics(entry.trajectory_id, controlNames, controlUnits, ...
                                             intervalTimes(:), fixedControls, config.tail_window_s);
        positiveNames = string(config.optional_positive_control_names(:));
        for index = 1:numel(positiveNames)
            controlIndex = find(controlNames == positiveNames(index), 1);
            if ~isempty(controlIndex) && any(fixedControls(:, controlIndex) <= 0)
                addIssue("NONPOSITIVE_CONTROL", positiveNames(index));
            end
        end
    end

    validateRawCoverage;
    validateCondition;
    calculateOutputMetric;
    finishEntry;

    function validateRawCoverage
        if ~fixedStateValid
            return;
        end
        fixedStart = trajectory.state_times_s(1);
        fixedEnd = trajectory.state_times_s(end);
        sources = {trajectory.raw_state_times_s, trajectory.raw_gate_times_s, ...
                   trajectory.raw_control_times_s};
        labels = ["RAW_STATE", "RAW_GATE", "RAW_CONTROL"];
        valid = [rawStateValid, rawGateValid, rawControlValid];
        for sourceIndex = 1:numel(sources)
            if valid(sourceIndex) && (sources{sourceIndex}(1) > fixedStart + config.time_abs_tolerance_s || ...
                    sources{sourceIndex}(end) < fixedEnd - config.time_abs_tolerance_s)
                addIssue(labels(sourceIndex) + "_COVERAGE", ...
                         labels(sourceIndex) + " does not cover the fixed grid.");
            end
        end
    end

    function validateCondition
        if ~isstruct(trajectory.condition) || ~isscalar(trajectory.condition)
            addIssue("INVALID_CONDITION", "condition must be a scalar structure.");
            return;
        end
        fields = string(config.required_condition_fields(:));
        for fieldIndex = 1:numel(fields)
            name = char(fields(fieldIndex));
            if ~isfield(trajectory.condition, name) || ...
                    ~isnumeric(trajectory.condition.(name)) || ...
                    ~isscalar(trajectory.condition.(name)) || ...
                    ~isfinite(trajectory.condition.(name)) || trajectory.condition.(name) <= 0
                addIssue("INVALID_CONDITION_FIELD", fields(fieldIndex));
            end
        end
    end

    function calculateOutputMetric
        if ~fixedStateValid || ~isstruct(trajectory.condition)
            return;
        end
        outputIndex = find(stateNames == string(config.output_state_name), 1);
        referenceField = char(config.reference_condition_field);
        if isempty(outputIndex) || ~isfield(trajectory.condition, referenceField)
            return;
        end
        reference = trajectory.condition.(referenceField);
        if ~isnumeric(reference) || ~isscalar(reference) || ~isfinite(reference) || reference <= 0
            return;
        end
        times = trajectory.state_times_s(:);
        values = trajectory.state_values(:, outputIndex);
        tailStart = max(times(1), times(end) - config.tail_window_s);
        inside = times >= tailStart;
        tailTimes = times(inside);
        tailValues = values(inside);
        if numel(tailTimes) < 2
            return;
        end
        entry.output_metric_available = true;
        entry.output_tail_mean = trapz(tailTimes, tailValues) / (tailTimes(end) - tailTimes(1));
        entry.output_tail_error_pct = 100 * (entry.output_tail_mean - reference) / reference;
        entry.output_tail_peak_to_peak = max(tailValues) - min(tailValues);
        entry.output_tail_within_tolerance = ...
            abs(entry.output_tail_error_pct) <= 100 * config.output_mean_relative_tolerance;
    end

    function addIssue(code, message)
        issues(end + 1, 1) = string(code);
        messages(end + 1, 1) = string(message);
    end

    function appendIssues(newIssues)
        for issueIndex = 1:numel(newIssues)
            addIssue(newIssues(issueIndex), newIssues(issueIndex));
        end
    end

    function finishEntry
        entry.issue_count = numel(issues);
        entry.issue_codes = strjoin(unique(issues, 'stable'), ';');
        entry.issue_messages = strjoin(messages, ' | ');
        entry.structural_valid = isempty(issues);
    end
end

function [valid, issues] = validateSeries(times, values, width, prefix)
    issues = strings(0, 1);
    valid = isvector(times) && numel(times) >= 2 && isnumeric(times) && ...
            all(isfinite(times)) && all(diff(times(:)) > 0) && ...
            isnumeric(values) && ismatrix(values) && size(values, 1) == numel(times) && ...
            size(values, 2) == width && all(isfinite(values), 'all');
    if ~valid
        issues(end + 1, 1) = prefix + "_SERIES_INVALID";
    end
end

function [count, issue] = countGateOverlap(gates, gateNames, groups)
    count = 0;
    issue = "";
    for index = 1:numel(groups)
        names = string(groups{index});
        columns = zeros(numel(names), 1);
        for nameIndex = 1:numel(names)
            found = find(gateNames == names(nameIndex), 1);
            if isempty(found)
                issue = "Missing gate " + names(nameIndex) + ".";
                return;
            end
            columns(nameIndex) = found;
        end
        count = count + nnz(sum(double(gates(:, columns)), 2) > 1);
    end
end

function result = buildStateMetrics(id, names, units, times, values, tailWindow)
    result = emptyStateMetrics;
    tailStart = max(times(1), times(end) - tailWindow);
    tail = times >= tailStart;
    for index = 1:numel(names)
        series = values(:, index);
        tailSeries = series(tail);
        tailTimes = times(tail);
        if numel(tailTimes) >= 2
            tailMean = trapz(tailTimes, tailSeries) / (tailTimes(end) - tailTimes(1));
        else
            tailMean = tailSeries(end);
        end
        row = {id, names(index), units(index), min(series), max(series), max(abs(series)), ...
               tailMean, min(tailSeries), max(tailSeries), max(tailSeries) - min(tailSeries)};
        result = [result; row]; %#ok<AGROW>
    end
end

function result = buildControlMetrics(id, names, units, times, values, tailWindow)
    result = emptyControlMetrics;
    tail = times >= max(times(1), times(end) - tailWindow);
    for index = 1:numel(names)
        series = values(:, index);
        row = {id, names(index), units(index), min(series), max(series), series(end), mean(series(tail))};
        result = [result; row]; %#ok<AGROW>
    end
end

function value = scalarText(input)
    input = string(input);
    if isscalar(input)
        value = input;
    else
        value = "";
    end
end

function value = textVector(input)
    value = string(input(:));
end

function entry = emptyEntry(file)
    entry = struct('file', string(file), 'trajectory_id', "", 'schema_version', "", ...
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
