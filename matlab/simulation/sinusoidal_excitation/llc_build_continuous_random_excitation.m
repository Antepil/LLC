function excitation = llc_build_continuous_random_excitation( ...
        randomization, timing, condition)
    % LLC_BUILD_CONTINUOUS_RANDOM_EXCITATION Realize seeded segment targets.
    validateInputs(randomization, timing, condition);
    group = char(string(condition.excitation_group));
    vinActive = strcmp(group, 'vin_only') || strcmp(group, 'both');
    referenceActive = strcmp(group, 'reference_only') || strcmp(group, 'both');
    stream = RandStream('mt19937ar', 'Seed', condition.random_seed);

    vinPhase = randomPhase(stream, vinActive);
    referencePhase = randomPhase(stream, referenceActive);
    segmentTemplate = struct('start_time_s', 0, ...
                             'vin', targetSignal(400, 0, 0), ...
                             'reference', targetSignal(48, 0, 0));
    segments = repmat(segmentTemplate, timing.segment_count, 1);
    for index = 1:timing.segment_count
        segments(index).start_time_s = timing.start_time_s + ...
            (index - 1) * timing.segment_duration_s;
        segments(index).vin = drawTarget(stream, randomization, 'vin', vinActive);
        segments(index).reference = drawTarget( ...
            stream, randomization, 'reference', referenceActive);
    end

    excitation = struct;
    excitation.profile_type = 'continuous_random_segments_v1';
    excitation.start_time_s = timing.start_time_s;
    excitation.segment_duration_s = timing.segment_duration_s;
    excitation.transition_time_s = timing.transition_time_s;
    excitation.vin = initialSignal(randomization.vin.initial_center_value, ...
        segments(1).vin.frequency_hz, vinPhase);
    excitation.reference = initialSignal( ...
        randomization.reference.initial_center_value, ...
        segments(1).reference.frequency_hz, referencePhase);
    excitation.segments = segments;
    excitation.random_seed = condition.random_seed;
    excitation.excitation_group = group;
end

function phase = randomPhase(stream, active)
    if active
        phase = 2 * pi * rand(stream);
    else
        phase = 0;
    end
end

function result = drawTarget(stream, randomization, name, active)
    parameters = randomization.(name);
    if ~active
        result = targetSignal(parameters.initial_center_value, 0, 0);
        return;
    end
    center = drawUniform(stream, parameters.center_range);
    amplitude = drawUniform(stream, parameters.amplitude_range);
    frequencies = randomization.frequency_choices_hz;
    frequency = frequencies(randi(stream, numel(frequencies)));
    result = targetSignal(center, amplitude, frequency);
end

function value = drawUniform(stream, range)
    value = range(1) + (range(2) - range(1)) * rand(stream);
end

function result = initialSignal(center, frequency, phase)
    result = struct('center_value', center, 'amplitude', 0, ...
                    'frequency_hz', frequency, 'phase_rad', phase);
end

function result = targetSignal(center, amplitude, frequency)
    result = struct('center_value', center, 'amplitude', amplitude, ...
                    'frequency_hz', frequency);
end

function validateInputs(randomization, timing, condition)
    if ~isstruct(randomization) || ~isscalar(randomization) || ...
            ~isfield(randomization, 'frequency_choices_hz') || ...
            ~isnumeric(randomization.frequency_choices_hz) || ...
            isempty(randomization.frequency_choices_hz) || ...
            any(~isfinite(randomization.frequency_choices_hz)) || ...
            any(randomization.frequency_choices_hz <= 0)
        error('llc:InvalidConfig', ...
              'frequency_choices_hz must contain positive finite values.');
    end
    validateRandomSignal(randomization, 'vin');
    validateRandomSignal(randomization, 'reference');
    if ~isstruct(timing) || ~isscalar(timing)
        error('llc:InvalidConfig', 'excitation timing must be a scalar structure.');
    end
    requireNonnegative(timing, 'start_time_s', 'llc:InvalidConfig');
    requirePositive(timing, 'segment_duration_s', 'llc:InvalidConfig');
    requirePositive(timing, 'transition_time_s', 'llc:InvalidConfig');
    requirePositive(timing, 'segment_count', 'llc:InvalidConfig');
    if timing.segment_count ~= round(timing.segment_count) || ...
            timing.transition_time_s > timing.segment_duration_s
        error('llc:InvalidConfig', ...
              'Segment count must be integral and transition must fit in a segment.');
    end
    if ~isstruct(condition) || ~isscalar(condition) || ...
            ~isfield(condition, 'excitation_group')
        error('llc:InvalidCondition', ...
              'Condition must define excitation_group.');
    end
    group = string(condition.excitation_group);
    if ~isscalar(group) || ...
            ~any(group == ["vin_only", "reference_only", "both"])
        error('llc:InvalidCondition', 'Unsupported excitation_group.');
    end
    requireNonnegative(condition, 'random_seed', 'llc:InvalidCondition');
    if condition.random_seed ~= round(condition.random_seed) || ...
            condition.random_seed > double(intmax('uint32'))
        error('llc:InvalidCondition', ...
              'random_seed must be a uint32-compatible integer.');
    end
end

function validateRandomSignal(randomization, name)
    if ~isfield(randomization, name) || ...
            ~isstruct(randomization.(name)) || ...
            ~isscalar(randomization.(name))
        error('llc:InvalidConfig', '%s randomization must be an object.', name);
    end
    value = randomization.(name);
    requirePositive(value, 'initial_center_value', 'llc:InvalidConfig');
    validateRange(value, 'center_range', true);
    validateRange(value, 'amplitude_range', false);
    if value.center_range(1) - value.amplitude_range(2) <= 0
        error('llc:InvalidConfig', ...
              '%s randomization can produce a nonpositive command.', name);
    end
end

function validateRange(value, name, strictlyPositive)
    if ~isfield(value, name) || ~isnumeric(value.(name)) || ...
            numel(value.(name)) ~= 2 || any(~isfinite(value.(name))) || ...
            value.(name)(1) > value.(name)(2) || ...
            (strictlyPositive && value.(name)(1) <= 0) || ...
            (~strictlyPositive && value.(name)(1) < 0)
        error('llc:InvalidConfig', '%s must be a valid two-value range.', name);
    end
end

function requirePositive(value, name, identifier)
    requireFinite(value, name, identifier);
    if value.(name) <= 0
        error(identifier, '%s must be positive.', name);
    end
end

function requireNonnegative(value, name, identifier)
    requireFinite(value, name, identifier);
    if value.(name) < 0
        error(identifier, '%s must be nonnegative.', name);
    end
end

function requireFinite(value, name, identifier)
    if ~isfield(value, name) || ~isnumeric(value.(name)) || ...
            ~isscalar(value.(name)) || ~isfinite(value.(name))
        error(identifier, '%s must be a finite scalar.', name);
    end
end
