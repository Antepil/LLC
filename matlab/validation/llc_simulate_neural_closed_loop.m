function result = llc_simulate_neural_closed_loop( ...
        model, initialState, controllerState, outputTimesS, vinSignal, referenceSignal, control, solver)
    % LLC_SIMULATE_NEURAL_CLOSED_LOOP Run one event-driven Neural ODE branch.
    times = double(outputTimesS(:));
    state = double(initialState(:));
    if numel(times) < 2 || any(~isfinite(times)) || any(diff(times) <= 0) || ...
            numel(state) ~= 4 || any(~isfinite(state))
        error('llc:InvalidNeuralODEInput', 'Closed-loop times or initial state are invalid.');
    end
    tolerance = solver.event_time_tolerance_s;
    if abs(controllerState.last_time_s - times(1)) > tolerance
        error('llc:ControllerStateMismatch', 'Controller last_time_s must match first output time.');
    end
    initialGates = llc_closed_loop_controller_gates(controllerState);
    before = initialGates;
    if llc_closed_loop_controller_next_event(controllerState) <= times(1)
        controllerState = llc_advance_closed_loop_controller(controllerState, times(1), ...
            state(4), llc_evaluate_validation_signal(referenceSignal, times(1)), control);
    end
    after = llc_closed_loop_controller_gates(controllerState);
    eventTimes = zeros(0, 1);
    gatesAfter = zeros(0, 4, 'uint8');
    if ~isequal(before, after)
        eventTimes(end + 1, 1) = times(1); %#ok<AGROW>
        gatesAfter(end + 1, :) = uint8(after); %#ok<AGROW>
    end
    states = nan(numel(times), 4);
    states(1, :) = state.';
    controlValues = nan(numel(times), 5);
    controlValues(1, :) = controlRow(controllerState);
    currentTime = times(1);
    sampleIndex = 2;
    options = odeset('MaxStep', solver.max_step_s, 'RelTol', solver.relative_tolerance, ...
                     'AbsTol', solver.absolute_tolerance);
    while sampleIndex <= numel(times)
        nextEvent = llc_closed_loop_controller_next_event(controllerState);
        targetTime = min(nextEvent, times(sampleIndex));
        if targetTime < currentTime - tolerance
            error('llc:ControllerStateMismatch', 'Controller scheduled an event in the past.');
        end
        gates = llc_closed_loop_controller_gates(controllerState);
        if targetTime > currentTime + tolerance
            derivative = @(time, value) llc_neural_ode_physical_derivative( ...
                time, value, model, gates, vinSignal);
            [~, values] = ode23t(derivative, [currentTime, targetTime], state, options);
            state = values(end, :).';
        end
        currentTime = targetTime;
        if nextEvent <= currentTime
            previousGates = llc_closed_loop_controller_gates(controllerState);
            reference = llc_evaluate_validation_signal(referenceSignal, currentTime);
            controllerState = llc_advance_closed_loop_controller( ...
                controllerState, currentTime, state(4), reference, control);
            newGates = llc_closed_loop_controller_gates(controllerState);
            if ~isequal(previousGates, newGates)
                eventTimes(end + 1, 1) = currentTime; %#ok<AGROW>
                gatesAfter(end + 1, :) = uint8(newGates); %#ok<AGROW>
            end
        end
        if times(sampleIndex) <= currentTime
            states(sampleIndex, :) = state.';
            controlValues(sampleIndex, :) = controlRow(controllerState);
            sampleIndex = sampleIndex + 1;
        end
    end
    if any(~isfinite(states), 'all') || any(~isfinite(controlValues), 'all')
        error('llc:NonfiniteNeuralODEState', 'Closed-loop simulation produced nonfinite output.');
    end
    result = struct('times_s', times, 'states', states, ...
        'initial_gate_values', uint8(initialGates), ...
        'gate_event_times_s', eventTimes, 'gate_values_after', gatesAfter, ...
        'control_names', {{'frequency_hz', 'filtered_voltage_v', ...
                           'reference_voltage_v', 'error_v', 'integral_v_s'}}, ...
        'control_values', controlValues, 'final_controller_state', controllerState);
end

function row = controlRow(state)
    row = [state.frequency_hz, state.filtered_voltage_v, state.reference_voltage_v, ...
           state.error_v, state.integral_v_s];
end
