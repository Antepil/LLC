function result = llc_reconstruct_closed_loop_controller( ...
        voltageTimesS, outputVoltageV, expectedInitialGates, expectedEventTimesS, ...
        expectedGatesAfter, expectedControlTimesS, expectedControlValues, ...
        referenceSignal, control, stopTimeS, toleranceS)
    % LLC_RECONSTRUCT_CLOSED_LOOP_CONTROLLER Replay deterministic PI/PFM state.
    times = double(voltageTimesS(:));
    voltage = double(outputVoltageV(:));
    if numel(times) < 2 || numel(times) ~= numel(voltage) || ...
            any(~isfinite(times)) || any(~isfinite(voltage)) || any(diff(times) <= 0) || ...
            stopTimeS < times(1) || stopTimeS > times(end)
        error('llc:InvalidControllerReplay', 'Voltage waveform does not cover replay time.');
    end
    state = llc_initialize_closed_loop_controller(control);
    initialGates = uint8(llc_closed_loop_controller_gates(state));
    if ~isequal(initialGates, uint8(expectedInitialGates(:).'))
        error('llc:ControllerReplayMismatch', 'Initial reconstructed gates differ from reference.');
    end
    eventTimes = zeros(0, 1);
    gatesAfter = zeros(0, 4, 'uint8');
    while true
        nextEvent = llc_closed_loop_controller_next_event(state);
        if nextEvent >= stopTimeS
            break;
        end
        measured = interp1(times, voltage, min(max(nextEvent, times(1)), times(end)), 'linear');
        reference = llc_evaluate_validation_signal(referenceSignal, nextEvent);
        previousGates = llc_closed_loop_controller_gates(state);
        state = llc_advance_closed_loop_controller(state, nextEvent, measured, reference, control);
        currentGates = llc_closed_loop_controller_gates(state);
        if ~isequal(previousGates, currentGates)
            eventTimes(end + 1, 1) = nextEvent; %#ok<AGROW>
            gatesAfter(end + 1, :) = uint8(currentGates); %#ok<AGROW>
        end
    end
    state = advancePhaseToStop(state, stopTimeS);
    expectedTimes = double(expectedEventTimesS(:));
    expectedValues = uint8(expectedGatesAfter);
    selected = expectedTimes < stopTimeS;
    expectedTimes = expectedTimes(selected);
    expectedValues = expectedValues(selected, :);
    if numel(expectedTimes) ~= numel(eventTimes) || ...
            ~isequal(size(expectedValues), size(gatesAfter)) || ...
            (~isempty(eventTimes) && any(abs(expectedTimes - eventTimes) > toleranceS)) || ...
            ~isequal(expectedValues, gatesAfter)
        error('llc:ControllerReplayMismatch', ...
              'Reconstructed PI/PFM gate events differ from the reference log.');
    end
    validateControlState(state, expectedControlTimesS, expectedControlValues, ...
                         control, stopTimeS, toleranceS);
    validateFutureSchedule(state, times, voltage, expectedEventTimesS, expectedGatesAfter, ...
                           referenceSignal, control, stopTimeS, toleranceS);
    if state.last_time_s ~= stopTimeS
        error('llc:ControllerReplayMismatch', ...
              'Reconstructed controller did not stop exactly at the handoff boundary.');
    end
    result = struct('initial_gate_values', initialGates, ...
        'gate_event_times_s', eventTimes, 'gate_values_after', gatesAfter, ...
        'final_state', state);
end

function validateFutureSchedule(state, voltageTimes, voltage, expectedTimesS, ...
        expectedGatesAfter, referenceSignal, control, stopTimeS, toleranceS)
    if voltageTimes(end) <= stopTimeS
        error('llc:ControllerReplayMismatch', ...
              'Reference logs must extend beyond the handoff for scheduler validation.');
    end
    expectedTimes = double(expectedTimesS(:));
    expectedValues = uint8(expectedGatesAfter);
    nextControlTime = state.next_control_time_s;
    actualTimes = zeros(0, 1);
    actualValues = zeros(0, 4, 'uint8');
    replay = state;
    verifiedPostControlGate = false;
    while ~verifiedPostControlGate
        nextEvent = llc_closed_loop_controller_next_event(replay);
        if nextEvent > voltageTimes(end)
            error('llc:ControllerReplayMismatch', ...
                  'Reference logs do not cover a post-handoff control update and gate event.');
        end
        measured = interp1(voltageTimes, voltage, nextEvent, 'linear');
        reference = llc_evaluate_validation_signal(referenceSignal, nextEvent);
        before = llc_closed_loop_controller_gates(replay);
        replay = llc_advance_closed_loop_controller(replay, nextEvent, measured, reference, control);
        after = llc_closed_loop_controller_gates(replay);
        if ~isequal(before, after)
            actualTimes(end + 1, 1) = nextEvent; %#ok<AGROW>
            actualValues(end + 1, :) = uint8(after); %#ok<AGROW>
            verifiedPostControlGate = nextEvent > nextControlTime;
        end
    end
    future = find(expectedTimes >= stopTimeS);
    if numel(future) < numel(actualTimes)
        error('llc:ControllerReplayMismatch', ...
              'Reference gate log lacks the required post-handoff scheduler evidence.');
    end
    expectedTimes = expectedTimes(future(1:numel(actualTimes)));
    expectedValues = expectedValues(future(1:numel(actualTimes)), :);
    if any(abs(actualTimes - expectedTimes) > toleranceS) || ...
            ~isequal(actualValues, expectedValues)
        error('llc:ControllerReplayMismatch', ...
              'Reconstructed scheduler does not reproduce post-handoff gate events.');
    end
end

function validateControlState( ...
        state, expectedTimesS, expectedValues, control, stopTimeS, timeTolerance)
    expectedTimes = double(expectedTimesS(:));
    expectedValues = double(expectedValues);
    if numel(expectedTimes) < 2 || any(~isfinite(expectedTimes)) || ...
            any(diff(expectedTimes) <= 0) || ...
            ~isequal(size(expectedValues), [numel(expectedTimes), 5]) || ...
            any(~isfinite(expectedValues), 'all') || ...
            expectedTimes(1) > stopTimeS + timeTolerance || ...
            expectedTimes(end) < stopTimeS - timeTolerance
        error('llc:InvalidControllerReplay', ...
              'Reference controller log does not cover the handoff time.');
    end
    boundaryTolerance = max(1e-14, ...
        32 * eps(max(stopTimeS, control.control_period_s)));
    controlDueAtHandoff = ...
        abs(state.next_control_time_s - stopTimeS) <= boundaryTolerance;
    beforeHandoff = find(expectedTimes < stopTimeS, 1, 'last');
    if controlDueAtHandoff
        if isempty(beforeHandoff)
            error('llc:InvalidControllerReplay', ...
                  'Controller log lacks a pre-handoff sample for a due boundary event.');
        end
        expected = expectedValues(beforeHandoff, :);
    else
        expected = interp1(expectedTimes, expectedValues, stopTimeS, 'previous');
    end
    actual = [state.frequency_hz, state.filtered_voltage_v, state.reference_voltage_v, ...
              state.error_v, state.integral_v_s];
    absoluteTolerance = [1e-6, 1e-8, 1e-8, 1e-8, 1e-12];
    if any(abs(actual - expected) > absoluteTolerance + 1e-10 .* abs(expected))
        error('llc:ControllerReplayMismatch', ...
              ['Reconstructed PI/filter state differs from the reference controller log. ', ...
               'actual=[%s], expected=[%s].'], num2str(actual, ' %.17g'), ...
              num2str(expected, ' %.17g'));
    end
end

function state = advancePhaseToStop(state, stopTimeS)
    if stopTimeS > state.last_time_s
        elapsed = stopTimeS - state.last_time_s;
        state.phase_rad = mod(state.phase_rad + 2 * pi * state.frequency_hz * ...
                              elapsed, 2 * pi);
        state.reference_ramp_elapsed_time_s = ...
            state.reference_ramp_elapsed_time_s + elapsed;
        state.last_time_s = stopTimeS;
    end
end
