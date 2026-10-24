function values = llc_evaluate_sinusoidal_excitation(time, excitation)
    % LLC_EVALUATE_SINUSOIDAL_EXCITATION Evaluate Vin and Vref profiles.
    validateattributes(time, {'numeric'}, ...
                       {'real', 'finite', 'nonnegative', 'vector'});
    time = time(:);
    if isfield(excitation, 'profile_type')
        if ~strcmp(excitation.profile_type, 'continuous_random_segments_v1')
            error('llc:InvalidExcitation', 'Unsupported excitation profile_type.');
        end
        validateSegmentedExcitation(excitation);
        values = [evaluateSegmentedSignal(time, excitation, 'vin'), ...
                  evaluateSegmentedSignal(time, excitation, 'reference')];
        return;
    end

    validateFixedExcitation(excitation);
    envelope = excitationEnvelope(time, excitation.start_time_s, ...
                                  excitation.envelope_rise_time_s);
    tau = max(time - excitation.start_time_s, 0);
    values = [evaluateFixedSignal(tau, envelope, excitation.vin), ...
              evaluateFixedSignal(tau, envelope, excitation.reference)];
end

function values = evaluateSegmentedSignal(time, excitation, name)
    initial = excitation.(name);
    segments = excitation.segments;
    transition = excitation.transition_time_s;
    values = initial.center_value * ones(size(time));
    old = initial;
    phaseCyclesAtStart = 0;
    for index = 1:numel(segments)
        startTime = segments(index).start_time_s;
        if index < numel(segments)
            nextStart = segments(index + 1).start_time_s;
            active = time >= startTime & time < nextStart;
        else
            nextStart = inf;
            active = time >= startTime;
        end
        target = segments(index).(name);
        localTime = time(active) - startTime;
        [center, amplitude, phaseCycles] = blendedParameters( ...
            localTime, old, target, transition);
        phase = initial.phase_rad + ...
            2 * pi * (phaseCyclesAtStart + phaseCycles);
        values(active) = center + amplitude .* sin(phase);
        if isfinite(nextStart)
            phaseCyclesAtStart = phaseCyclesAtStart + integrateFrequency( ...
                nextStart - startTime, old.frequency_hz, ...
                target.frequency_hz, transition);
        end
        old = target;
    end
end

function [center, amplitude, phaseCycles] = blendedParameters( ...
        localTime, old, target, transition)
    s = min(localTime / transition, 1);
    weight = s.^2 .* (3 - 2 * s);
    center = old.center_value + ...
        (target.center_value - old.center_value) .* weight;
    amplitude = old.amplitude + ...
        (target.amplitude - old.amplitude) .* weight;
    phaseCycles = arrayfun(@(duration)integrateFrequency( ...
        duration, old.frequency_hz, target.frequency_hz, transition), ...
        localTime);
end

function cycles = integrateFrequency(duration, oldFrequency, newFrequency, transition)
    if duration <= transition
        s = duration / transition;
        cycles = oldFrequency * duration + ...
            (newFrequency - oldFrequency) * transition * ...
            (s^3 - 0.5 * s^4);
    else
        transitionCycles = 0.5 * transition * (oldFrequency + newFrequency);
        cycles = transitionCycles + newFrequency * (duration - transition);
    end
end

function values = evaluateFixedSignal(tau, envelope, signal)
    values = signal.center_value + envelope .* signal.amplitude .* ...
             sin(2 * pi * signal.frequency_hz .* tau + signal.phase_rad);
end

function envelope = excitationEnvelope(time, startTime, riseTime)
    tau = time - startTime;
    envelope = zeros(size(time));
    if riseTime == 0
        envelope(tau >= 0) = 1;
        return;
    end
    rising = tau > 0 & tau < riseTime;
    envelope(rising) = 0.5 * (1 - cos(pi * tau(rising) / riseTime));
    envelope(tau >= riseTime) = 1;
end

function validateFixedExcitation(excitation)
    if ~isstruct(excitation) || ~isscalar(excitation)
        error('llc:InvalidExcitation', 'excitation must be a scalar structure.');
    end
    requireNonnegative(excitation, 'start_time_s');
    requireNonnegative(excitation, 'envelope_rise_time_s');
    validateInitialSignal(excitation, 'vin', false);
    validateInitialSignal(excitation, 'reference', false);
end

function validateSegmentedExcitation(excitation)
    if ~isstruct(excitation) || ~isscalar(excitation)
        error('llc:InvalidExcitation', 'excitation must be a scalar structure.');
    end
    requireNonnegative(excitation, 'start_time_s');
    requirePositive(excitation, 'segment_duration_s');
    requirePositive(excitation, 'transition_time_s');
    if excitation.transition_time_s > excitation.segment_duration_s || ...
            ~isfield(excitation, 'segments') || ...
            ~isstruct(excitation.segments) || isempty(excitation.segments)
        error('llc:InvalidExcitation', ...
              'Segment timing or segment collection is invalid.');
    end
    validateInitialSignal(excitation, 'vin', true);
    validateInitialSignal(excitation, 'reference', true);
    starts = [excitation.segments.start_time_s];
    if any(~isfinite(starts)) || any(diff(starts) <= 0) || ...
            abs(starts(1) - excitation.start_time_s) > ...
                32 * eps(max(excitation.start_time_s, 1)) || ...
            (numel(starts) > 1 && ...
             any(diff(starts) < excitation.transition_time_s))
        error('llc:InvalidExcitation', ...
              'Segment start times must be increasing and transition-safe.');
    end
    for index = 1:numel(excitation.segments)
        segment = excitation.segments(index);
        validateTargetSignal(segment, 'vin');
        validateTargetSignal(segment, 'reference');
    end
end

function validateInitialSignal(excitation, name, requireZeroAmplitude)
    if ~isfield(excitation, name) || ~isstruct(excitation.(name)) || ...
            ~isscalar(excitation.(name))
        error('llc:InvalidExcitation', '%s must be a scalar structure.', name);
    end
    signal = excitation.(name);
    validateSignalValues(signal, name, true);
    requireFinite(signal, 'phase_rad');
    if requireZeroAmplitude && signal.amplitude ~= 0
        error('llc:InvalidExcitation', ...
              '%s initial amplitude must be zero.', name);
    end
end

function validateTargetSignal(segment, name)
    if ~isfield(segment, name) || ~isstruct(segment.(name)) || ...
            ~isscalar(segment.(name))
        error('llc:InvalidExcitation', ...
              'Each segment must define scalar %s parameters.', name);
    end
    validateSignalValues(segment.(name), name, false);
end

function validateSignalValues(signal, name, allowPhase)
    requirePositive(signal, 'center_value');
    requireNonnegative(signal, 'amplitude');
    requireNonnegative(signal, 'frequency_hz');
    if signal.amplitude > 0 && signal.frequency_hz <= 0
        error('llc:InvalidExcitation', ...
              '%s.frequency_hz must be positive when amplitude is nonzero.', name);
    end
    if signal.amplitude >= signal.center_value
        error('llc:InvalidExcitation', ...
              '%s amplitude must be smaller than its center value.', name);
    end
    if ~allowPhase && isfield(signal, 'phase_rad')
        error('llc:InvalidExcitation', ...
              '%s segment target must not reset phase.', name);
    end
end

function requirePositive(value, name)
    requireFinite(value, name);
    if value.(name) <= 0
        error('llc:InvalidExcitation', '%s must be positive.', name);
    end
end

function requireNonnegative(value, name)
    requireFinite(value, name);
    if value.(name) < 0
        error('llc:InvalidExcitation', '%s must be nonnegative.', name);
    end
end

function requireFinite(value, name)
    if ~isfield(value, name) || ~isnumeric(value.(name)) || ...
            ~isscalar(value.(name)) || ~isfinite(value.(name))
        error('llc:InvalidExcitation', '%s must be a finite scalar.', name);
    end
end
