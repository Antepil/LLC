function events = llc_constant_frequency_events(modulation, frequencyHz, stopTimeS)
    % LLC_CONSTANT_FREQUENCY_EVENTS Build post-deadtime full-bridge gates.
    requirePositiveScalar(frequencyHz, 'switching_frequency_hz');
    requirePositiveScalar(stopTimeS, 'stop_time_s');
    requirePositiveScalar(modulation.deadtime_s, 'modulation.deadtime_s');
    requirePositiveScalar(modulation.minimum_pulse_s, 'modulation.minimum_pulse_s');
    limits = modulation.frequency_range_hz;
    if ~isnumeric(limits) || numel(limits) ~= 2 || any(~isfinite(limits)) || ...
            limits(1) <= 0 || limits(1) >= limits(2)
        error('llc:InvalidConfig', 'modulation.frequency_range_hz must contain two increasing positive values.');
    end
    if frequencyHz < limits(1) || frequencyHz > limits(2)
        error('llc:InvalidCondition', 'Switching frequency %.17g Hz is outside the configured range.', frequencyHz);
    end
    halfPeriod = 0.5 / frequencyHz;
    onTime = halfPeriod - modulation.deadtime_s;
    if onTime < modulation.minimum_pulse_s
        error('llc:InvalidCondition', 'Frequency and deadtime violate the minimum pulse width.');
    end

    boundaries = 0;
    values = zeros(0, 4);
    edge = 0;
    positiveHalfCycle = true;
    while edge < stopTimeS
        deadtimeEnd = min(edge + modulation.deadtime_s, stopTimeS);
        [boundaries, values] = appendInterval(boundaries, values, edge, zeros(1, 4));
        if deadtimeEnd < stopTimeS
            if positiveHalfCycle
                active = [1 0 0 1];
            else
                active = [0 1 1 0];
            end
            [boundaries, values] = appendInterval(boundaries, values, deadtimeEnd, active);
        end
        edge = min(edge + halfPeriod, stopTimeS);
        positiveHalfCycle = ~positiveHalfCycle;
    end
    boundaries(end + 1, 1) = stopTimeS;
    gateSource = timeseries([values; zeros(1, 4)], boundaries);
    events = struct('gate_times_s', boundaries, 'gate_values', uint8(values), ...
                    'gate_source', setinterpmethod(gateSource, 'zoh'), ...
                    'stop_time_s', stopTimeS, 'switching_frequency_hz', frequencyHz);
end

function [boundaries, values] = appendInterval(boundaries, values, time, value)
    if isempty(values)
        values = value;
        return;
    end
    if isequal(values(end, :), value)
        return;
    end
    boundaries(end + 1, 1) = time;
    values(end + 1, :) = value;
end

function requirePositiveScalar(value, name)
    if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value) || value <= 0
        error('llc:InvalidConfig', '%s must be a finite positive scalar.', name);
    end
end
