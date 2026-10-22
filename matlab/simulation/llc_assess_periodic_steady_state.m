function assessment = llc_assess_periodic_steady_state( ...
        stateTimes, stateValues, gateTimes, gateValues, referenceVoltage, config)
    % LLC_ASSESS_PERIODIC_STEADY_STATE Classify periodic steady state at checkpoints.
    validateInputs(stateTimes, stateValues, gateTimes, gateValues, referenceVoltage, config);
    stateReferenceScales = reshape(config.state_reference_scales, 1, []);

    rising = [gateValues(1, 1) > 0; diff(double(gateValues(:, 1) > 0)) > 0];
    cycleBoundaries = gateTimes(rising);
    checkpointTemplate = struct( ...
        'time_s', NaN, ...
        'window_start_s', NaN, ...
        'complete_cycle_count', 0, ...
        'regulated', false, ...
        'dynamics_stable', false, ...
        'settled', false, ...
        'cycle_mean_minimum', nan(1, 4), ...
        'cycle_mean_maximum', nan(1, 4), ...
        'cycle_ripple_mean', nan(1, 4));
    checkpoints = repmat(checkpointTemplate, numel(config.check_times_s), 1);

    for index = 1:numel(config.check_times_s)
        checkTime = config.check_times_s(index);
        windowStart = checkTime - config.window_duration_s;
        checkpoints(index).time_s = checkTime;
        checkpoints(index).window_start_s = windowStart;
        [cycleStarts, cycleEnds] = completeCycles(cycleBoundaries, windowStart, checkTime);
        checkpoints(index).complete_cycle_count = numel(cycleStarts);
        if isempty(cycleStarts)
            continue;
        end

        [cycleMeans, cycleRipples] = cycleStatistics( ...
            stateTimes, stateValues, cycleStarts, cycleEnds);
        cycleMidpoints = (cycleStarts + cycleEnds) / 2;
        firstHalf = cycleMidpoints < windowStart + config.window_duration_s / 2;
        secondHalf = ~firstHalf;
        if nnz(firstHalf) < config.minimum_cycles_per_half_window || ...
                nnz(secondHalf) < config.minimum_cycles_per_half_window
            continue;
        end

        checkpoints(index).cycle_mean_minimum = min(cycleMeans, [], 1);
        checkpoints(index).cycle_mean_maximum = max(cycleMeans, [], 1);
        checkpoints(index).cycle_ripple_mean = mean(cycleRipples, 1);
        outputError = abs(cycleMeans(:, 4) - referenceVoltage) / referenceVoltage;
        regulated = all(outputError <= config.output_relative_tolerance);

        meanLimit = config.state_mean_relative_tolerance .* stateReferenceScales;
        meanHalfChange = abs(mean(cycleMeans(firstHalf, :), 1) - ...
                             mean(cycleMeans(secondHalf, :), 1));
        meanRange = max(cycleMeans, [], 1) - min(cycleMeans, [], 1);
        meansStable = all(meanHalfChange <= meanLimit & meanRange <= meanLimit);

        rippleFloor = config.ripple_floor_relative_scale .* stateReferenceScales;
        rippleLimit = config.ripple_relative_tolerance .* ...
                      max(mean(cycleRipples, 1), rippleFloor);
        rippleHalfChange = abs(mean(cycleRipples(firstHalf, :), 1) - ...
                               mean(cycleRipples(secondHalf, :), 1));
        rippleRange = max(cycleRipples, [], 1) - min(cycleRipples, [], 1);
        ripplesStable = all(rippleHalfChange <= rippleLimit & rippleRange <= rippleLimit);

        checkpoints(index).regulated = regulated;
        checkpoints(index).dynamics_stable = meansStable && ripplesStable;
        checkpoints(index).settled = regulated && meansStable && ripplesStable;
    end

    firstSettled = find([checkpoints.settled], 1, 'first');
    if isempty(firstSettled)
        selectedIndex = numel(checkpoints);
        firstSettledTime = NaN;
        settled = false;
    else
        selectedIndex = firstSettled;
        firstSettledTime = checkpoints(firstSettled).time_s;
        settled = true;
    end
    assessment = struct( ...
        'checkpoints', checkpoints, ...
        'settled', settled, ...
        'settled_by_2ms', checkpoints(1).settled, ...
        'first_settled_time_s', firstSettledTime, ...
        'selected_stop_time_s', checkpoints(selectedIndex).time_s, ...
        'regulated', checkpoints(selectedIndex).regulated, ...
        'dynamics_stable', checkpoints(selectedIndex).dynamics_stable);
end

function [starts, ends] = completeCycles(boundaries, windowStart, checkTime)
    if numel(boundaries) < 2
        starts = zeros(0, 1);
        ends = zeros(0, 1);
        return;
    end
    starts = boundaries(1:end-1);
    ends = boundaries(2:end);
    tolerance = 64 * eps(max(checkTime, 1));
    complete = starts >= windowStart - tolerance & ends <= checkTime + tolerance;
    starts = starts(complete);
    ends = ends(complete);
end

function [means, ripples] = cycleStatistics(stateTimes, stateValues, starts, ends)
    means = zeros(numel(starts), size(stateValues, 2));
    ripples = zeros(size(means));
    for index = 1:numel(starts)
        inside = stateTimes > starts(index) & stateTimes < ends(index);
        sampleTimes = [starts(index); stateTimes(inside); ends(index)];
        sampleValues = interp1(stateTimes, stateValues, sampleTimes, 'linear');
        duration = ends(index) - starts(index);
        means(index, :) = trapz(sampleTimes, sampleValues, 1) / duration;
        ripples(index, :) = max(sampleValues, [], 1) - min(sampleValues, [], 1);
    end
end

function validateInputs(stateTimes, stateValues, gateTimes, gateValues, referenceVoltage, config)
    if ~isvector(stateTimes) || numel(stateTimes) < 2 || any(~isfinite(stateTimes)) || ...
            any(diff(stateTimes) <= 0) || size(stateValues, 1) ~= numel(stateTimes) || ...
            size(stateValues, 2) ~= 4 || any(~isfinite(stateValues), 'all')
        error('llc:InvalidSteadyStateInput', 'State data must be a finite increasing four-state time series.');
    end
    if ~isvector(gateTimes) || numel(gateTimes) < 2 || any(~isfinite(gateTimes)) || ...
            any(diff(gateTimes) <= 0) || size(gateValues, 1) ~= numel(gateTimes) || ...
            size(gateValues, 2) < 1 || any(gateValues ~= 0 & gateValues ~= 1, 'all')
        error('llc:InvalidSteadyStateInput', 'Gate data must be a binary time series with increasing times.');
    end
    if ~isscalar(referenceVoltage) || ~isfinite(referenceVoltage) || referenceVoltage <= 0
        error('llc:InvalidSteadyStateInput', 'referenceVoltage must be a finite positive scalar.');
    end
    required = {'check_times_s', 'window_duration_s', 'output_relative_tolerance', ...
                'state_reference_scales', 'state_mean_relative_tolerance', ...
                'ripple_relative_tolerance', 'ripple_floor_relative_scale', ...
                'minimum_cycles_per_half_window'};
    if ~isstruct(config) || ~all(isfield(config, required))
        error('llc:InvalidSteadyStateConfig', 'Steady-state assessment configuration is incomplete.');
    end
    positiveVectors = {config.check_times_s, config.state_reference_scales};
    if any(cellfun(@(value) ~isnumeric(value) || isempty(value) || ...
            any(~isfinite(value)) || any(value <= 0), positiveVectors))
        error('llc:InvalidSteadyStateConfig', 'Check times and state scales must be finite and positive.');
    end
    if numel(config.state_reference_scales) ~= 4 || ...
            any(diff(config.check_times_s) <= 0) || ...
            config.check_times_s(end) > stateTimes(end) + 64 * eps(max(stateTimes(end), 1))
        error('llc:InvalidSteadyStateConfig', 'Steady-state checkpoints or state scales are inconsistent.');
    end
    scalarFields = required(2:end);
    scalarFields(strcmp(scalarFields, 'state_reference_scales')) = [];
    for index = 1:numel(scalarFields)
        value = config.(scalarFields{index});
        if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value) || value <= 0
            error('llc:InvalidSteadyStateConfig', '%s must be a finite positive scalar.', scalarFields{index});
        end
    end
    if config.minimum_cycles_per_half_window ~= round(config.minimum_cycles_per_half_window)
        error('llc:InvalidSteadyStateConfig', 'minimum_cycles_per_half_window must be an integer.');
    end
end
