function result = llc_reconstruct_closed_loop_controller( ...
        voltageTimesS, outputVoltageV, expectedInitialGates, expectedEventTimesS, ...
        expectedGatesAfter, referenceSignal, control, stopTimeS, toleranceS)
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
        if nextEvent > stopTimeS + toleranceS
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
    selected = expectedTimes <= stopTimeS + toleranceS;
    expectedTimes = expectedTimes(selected);
    expectedValues = expectedValues(selected, :);
    if numel(expectedTimes) ~= numel(eventTimes) || ...
            ~isequal(size(expectedValues), size(gatesAfter)) || ...
            (~isempty(eventTimes) && any(abs(expectedTimes - eventTimes) > toleranceS)) || ...
            ~isequal(expectedValues, gatesAfter)
        error('llc:ControllerReplayMismatch', ...
              'Reconstructed PI/PFM gate events differ from the reference log.');
    end
    result = struct('initial_gate_values', initialGates, ...
        'gate_event_times_s', eventTimes, 'gate_values_after', gatesAfter, ...
        'final_state', state);
end

function state = advancePhaseToStop(state, stopTimeS)
    if stopTimeS > state.last_time_s
        state.phase_rad = mod(state.phase_rad + 2 * pi * state.frequency_hz * ...
                              (stopTimeS - state.last_time_s), 2 * pi);
        state.last_time_s = stopTimeS;
    end
end
