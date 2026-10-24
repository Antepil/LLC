function states = llc_integrate_neural_ode_events( ...
        model, outputTimesS, initialState, initialGates, eventTimesS, gatesAfter, vinSignal, solver)
    % LLC_INTEGRATE_NEURAL_ODE_EVENTS Integrate without crossing gate events.
    times = double(outputTimesS(:));
    state = double(initialState(:));
    gates = double(initialGates(:).');
    eventTimes = double(eventTimesS(:));
    gatesAfter = double(gatesAfter);
    validateInputs(times, state, gates, eventTimes, gatesAfter, solver);
    tolerance = solver.event_time_tolerance_s;
    states = nan(numel(times), 4);
    states(1, :) = state.';
    eventIndex = 1;
    while eventIndex <= numel(eventTimes) && eventTimes(eventIndex) <= times(1) + tolerance
        gates = gatesAfter(eventIndex, :);
        eventIndex = eventIndex + 1;
    end
    interior = eventTimes(eventTimes > times(1) + tolerance & eventTimes < times(end) - tolerance);
    breakpoints = unique([times(1); interior; times(end)]);
    options = odeset('MaxStep', solver.max_step_s, 'RelTol', solver.relative_tolerance, ...
                     'AbsTol', solver.absolute_tolerance);
    for segmentIndex = 1:numel(breakpoints) - 1
        startTime = breakpoints(segmentIndex);
        stopTime = breakpoints(segmentIndex + 1);
        nodeIndices = find(times > startTime + tolerance & times <= stopTime + tolerance);
        requested = times(nodeIndices);
        integrationTimes = unique([startTime; requested; stopTime]);
        derivative = @(time, value) llc_neural_ode_physical_derivative( ...
            time, value, model, gates, vinSignal);
        [solutionTimes, solutionStates] = ode23t(derivative, integrationTimes, state, options);
        if ~isempty(nodeIndices)
            states(nodeIndices, :) = interp1(solutionTimes, solutionStates, requested, 'linear');
        end
        state = solutionStates(end, :).';
        while eventIndex <= numel(eventTimes) && eventTimes(eventIndex) <= stopTime + tolerance
            gates = gatesAfter(eventIndex, :);
            eventIndex = eventIndex + 1;
        end
    end
    if any(~isfinite(states), 'all')
        error('llc:NeuralODEIntegration', 'Integrator did not produce every requested finite state.');
    end
end

function validateInputs(times, state, gates, eventTimes, gatesAfter, solver)
    if numel(times) < 2 || any(~isfinite(times)) || any(diff(times) <= 0) || ...
            numel(state) ~= 4 || any(~isfinite(state)) || ...
            numel(gates) ~= 4 || any(gates ~= 0 & gates ~= 1)
        error('llc:InvalidNeuralODEInput', 'Output times, initial state or gates are invalid.');
    end
    if any(~isfinite(eventTimes)) || any(diff(eventTimes) <= 0) || ...
            ~isequal(size(gatesAfter), [numel(eventTimes), 4]) || ...
            any(gatesAfter ~= 0 & gatesAfter ~= 1, 'all') || ...
            any(eventTimes < times(1)) || any(eventTimes > times(end))
        error('llc:InvalidNeuralODEInput', 'Gate event table is invalid or outside coverage.');
    end
    required = {'name', 'max_step_s', 'relative_tolerance', ...
                'absolute_tolerance', 'event_time_tolerance_s'};
    if ~isstruct(solver) || ~all(isfield(solver, required)) || ~strcmp(solver.name, 'ode23t')
        error('llc:InvalidConfig', 'The event integrator requires a complete ode23t config.');
    end
    numeric = [solver.max_step_s, solver.relative_tolerance, ...
               solver.absolute_tolerance, solver.event_time_tolerance_s];
    if any(~isfinite(numeric)) || any(numeric <= 0)
        error('llc:InvalidConfig', 'Solver values must be finite and positive.');
    end
end
