function [fixed, audit] = llc_resample_fixed_grid(raw, sampling, stopTimeS)
    % LLC_RESAMPLE_FIXED_GRID Convert native solver output to a Cuk-style grid.
    validateSampling(sampling, stopTimeS);
    intervalCountExact = stopTimeS * sampling.reference_frequency_hz * ...
                         sampling.intervals_per_cycle;
    intervalCount = round(intervalCountExact);
    alignmentTolerance = 64 * eps(max(intervalCountExact, 1));
    if abs(intervalCountExact - intervalCount) > alignmentTolerance
        error('llc:SamplingAlignment', ...
              'stop_time_s must contain an integer number of configured sampling intervals.');
    end

    sampleIntervalS = 1 / (sampling.reference_frequency_hz * sampling.intervals_per_cycle);
    stateTimes = (0:intervalCount)' * sampleIntervalS;
    stateTimes(end) = stopTimeS;
    intervalTimes = 0.5 * (stateTimes(1:end-1) + stateTimes(2:end));
    stateValues = sample(raw.state_times_s, raw.state_values, stateTimes, 'linear');
    gateValues = sample(raw.gate_times_s, raw.gate_values, intervalTimes, 'previous');
    controlValues = sample(raw.control_times_s, raw.control_values, intervalTimes, 'previous');
    if any(gateValues ~= 0 & gateValues ~= 1, 'all')
        error('llc:InvalidOutput', 'Fixed-grid gate values are not binary.');
    end

    fixed = struct('state_times_s', stateTimes, 'state_values', stateValues, ...
                   'interval_times_s', intervalTimes, ...
                   'interval_gate_values', uint8(gateValues), ...
                   'interval_control_values', controlValues);
    audit = struct('method', sampling.method, ...
                   'reference_frequency_hz', sampling.reference_frequency_hz, ...
                   'intervals_per_cycle', sampling.intervals_per_cycle, ...
                   'sample_interval_s', sampleIntervalS, ...
                   'interval_count', intervalCount, ...
                   'state_sample_count', intervalCount + 1, ...
                   'raw_state_sample_count', numel(raw.state_times_s), ...
                   'raw_gate_sample_count', numel(raw.gate_times_s), ...
                   'raw_control_sample_count', numel(raw.control_times_s));
end

function values = sample(sourceTimes, sourceValues, targetTimes, method)
    [uniqueTimes, indices] = unique(sourceTimes, 'last');
    values = interp1(uniqueTimes, sourceValues(indices, :), targetTimes, method);
    if any(~isfinite(values), 'all')
        error('llc:InvalidOutput', 'Interpolation produced nonfinite values or exceeded raw coverage.');
    end
end

function validateSampling(sampling, stopTimeS)
    if ~isstruct(sampling) || ~isscalar(sampling) || ...
            ~isfield(sampling, 'method') || ~strcmp(sampling.method, 'uniform_reference_cycle_v1') || ...
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
    if ~isnumeric(stopTimeS) || ~isscalar(stopTimeS) || ~isfinite(stopTimeS) || stopTimeS <= 0
        error('llc:InvalidConfig', 'stop_time_s must be a finite positive scalar.');
    end
end

function requirePositiveScalar(value, name)
    if ~isfield(value, name) || ~isnumeric(value.(name)) || ~isscalar(value.(name)) || ...
            ~isfinite(value.(name)) || value.(name) <= 0
        error('llc:IncompleteConfig', '%s must be a finite positive scalar.', name);
    end
end
