function result = llc_simulate_neural_ode_comparison( ...
        model, hardware, modelFile, referenceInitialState, neuralInitialState, ...
        controllerState, outputTimesS, vinSignal, referenceSignal, control, solver)
    % LLC_SIMULATE_NEURAL_ODE_COMPARISON Run circuit and Neural ODE together.
    times = validateInputs(model, hardware, modelFile, referenceInitialState, ...
        neuralInitialState, controllerState, outputTimesS, vinSignal, ...
        referenceSignal, control, solver);
    [~, modelName] = fileparts(modelFile);
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end
    load_system(modelFile);
    cleanup = onCleanup(@()closeIfLoaded(modelName));
    requireLogBlocks(modelName);

    workspace = get_param(modelName, 'ModelWorkspace');
    parameters = hardware;
    parameters.vin_v = llc_evaluate_validation_signal(vinSignal, 0);
    parameters.load_ohm = model.load_resistance_ohm;
    excitation = struct('validation_schema_version', 'llc-validation-excitation-v1', ...
                        'vin', vinSignal, 'reference', referenceSignal);
    referenceControl = control;
    referenceControl.initial_state = controllerState;
    neuralControl = control;
    neuralControl.initial_state = controllerState;
    runtime = struct('is_placeholder', false, 'model', model, ...
                     'initial_state', double(neuralInitialState(:)));
    override = struct('parameters', parameters, 'control', referenceControl, ...
        'excitation', excitation, 'solver', solver, 'stop_time_s', times(end));
    assignin(workspace, 'llc_run_override', override);
    assignin(workspace, 'llc_neural_runtime', runtime);
    assignin(workspace, 'llc_neural_control', neuralControl);

    simulation = Simulink.SimulationInput(modelName);
    for name = {'States', 'Gates', 'Control', 'NeuralStates', ...
                'NeuralGates', 'NeuralControl', 'StateErrorLog'}
        block = [modelName '/' name{1}];
        simulation = simulation.setBlockParameter(block, 'Decimation', '1');
        simulation = simulation.setBlockParameter(block, 'MaxDataPoints', 'inf');
    end
    simulation = applyReferenceInitialState( ...
        simulation, modelName, double(referenceInitialState(:).'));
    raw = sim(simulation);

    referenceStates = sampleLog(raw, 'llc_states', times, 4, 'linear');
    predictedStates = sampleLog(raw, 'llc_neural_states', times, 4, 'linear');
    loggedError = sampleLog(raw, 'llc_state_error', times, 4, 'linear');
    referenceGates = sampleLog(raw, 'llc_gates', times, 4, 'previous');
    neuralGates = sampleLog(raw, 'llc_neural_gates', times, 4, 'previous');
    referenceControlValues = sampleLog(raw, 'llc_control_log', times, 5, 'previous');
    neuralControlValues = sampleLog(raw, 'llc_neural_control_log', times, 5, 'previous');
    stateError = referenceStates - predictedStates;
    if any(abs(loggedError - stateError) > 1e-8 + 1e-9 .* abs(stateError), 'all')
        error('llc:InvalidComparisonLog', ...
              'Logged state error does not equal reference minus Neural ODE state.');
    end
    result = struct('times_s', times, 'reference_states', referenceStates, ...
        'predicted_states', predictedStates, 'state_error', stateError, ...
        'reference_gate_values', uint8(referenceGates), ...
        'neural_gate_values', uint8(neuralGates), ...
        'reference_control_values', referenceControlValues, ...
        'neural_control_values', neuralControlValues, ...
        'raw_reference_control_values', raw.llc_control_log, ...
        'raw_neural_control_values', raw.llc_neural_control_log, ...
        'final_reference_control', referenceControlValues(end, :), ...
        'final_neural_control', neuralControlValues(end, :));
    clear cleanup;
    closeIfLoaded(modelName);
end

function times = validateInputs(model, hardware, modelFile, referenceInitial, ...
        neuralInitial, controllerState, outputTimes, vinSignal, referenceSignal, control, solver)
    times = double(outputTimes(:));
    if ~isstruct(model) || ~isscalar(model) || ...
            ~isfield(model, 'load_resistance_ohm') || ...
            model.load_resistance_ohm ~= 2.304 || ...
            ~isstruct(hardware) || ~isscalar(hardware) || ~isfile(modelFile) || ...
            numel(referenceInitial) ~= 4 || any(~isfinite(referenceInitial)) || ...
            numel(neuralInitial) ~= 4 || any(~isfinite(neuralInitial)) || ...
            numel(times) < 2 || times(1) ~= 0 || any(~isfinite(times)) || ...
            any(diff(times) <= 0)
        error('llc:InvalidValidationInput', 'Comparison model inputs are invalid.');
    end
    llc_validate_closed_loop_controller_state(controllerState, control);
    if abs(controllerState.last_time_s) > solver.event_time_tolerance_s
        error('llc:ControllerStateMismatch', ...
              'Comparison controller state must use evaluation-local time.');
    end
    llc_evaluate_validation_signal(vinSignal, [0, times(end)]);
    llc_evaluate_validation_signal(referenceSignal, [0, times(end)]);
    requiredSolver = {'name', 'max_step_s', 'relative_tolerance', ...
                      'absolute_tolerance', 'event_time_tolerance_s'};
    if ~isstruct(solver) || ~isscalar(solver) || ~all(isfield(solver, requiredSolver)) || ...
            ~strcmp(solver.name, 'ode23t')
        error('llc:InvalidValidationInput', 'Comparison solver is invalid.');
    end
end

function requireLogBlocks(model)
    required = {'States', 'Gates', 'Control', 'NeuralStates', ...
                'NeuralGates', 'NeuralControl', 'StateErrorLog'};
    missing = strings(0, 1);
    for index = 1:numel(required)
        if getSimulinkBlockHandle([model '/' required{index}]) == -1
            missing(end + 1, 1) = string(required{index}); %#ok<AGROW>
        end
    end
    if ~isempty(missing)
        error('llc:MissingComparisonLog', ...
              'Comparison model is missing log blocks: %s', strjoin(missing, ', '));
    end
end

function values = sampleLog(simulation, name, targetTimes, width, method)
    if ~ismember(name, simulation.who)
        error('llc:MissingComparisonLog', 'Simulation output is missing %s.', name);
    end
    series = simulation.get(name);
    if ~isa(series, 'timeseries')
        error('llc:InvalidComparisonLog', '%s must be a timeseries.', name);
    end
    sourceTimes = double(series.Time(:));
    sourceValues = double(series.Data);
    if numel(sourceTimes) < 2 || size(sourceValues, 1) ~= numel(sourceTimes) || ...
            size(sourceValues, 2) ~= width || any(~isfinite(sourceTimes)) || ...
            any(~isfinite(sourceValues), 'all') || any(diff(sourceTimes) < 0) || ...
            sourceTimes(1) > targetTimes(1) || sourceTimes(end) < targetTimes(end)
        error('llc:InvalidComparisonLog', ...
              '%s is malformed or does not cover the requested time range.', name);
    end
    [uniqueTimes, indices] = unique(sourceTimes, 'last');
    values = interp1(uniqueTimes, sourceValues(indices, :), targetTimes, method);
    if any(~isfinite(values), 'all')
        error('llc:InvalidComparisonLog', '%s interpolation produced nonfinite values.', name);
    end
end

function simulation = applyReferenceInitialState(simulation, modelName, initialState)
    blocks = {'Lr', 'Cr', 'Lm', 'Co'};
    variables = {'i_L', 'vc', 'i_L', 'vc'};
    for index = 1:numel(blocks)
        block = [modelName '/' blocks{index}];
        simulation = simulation.setBlockParameter( ...
            block, variables{index}, num2str(initialState(index), 17));
        simulation = simulation.setBlockParameter( ...
            block, [variables{index} '_priority'], 'High');
        simulation = simulation.setBlockParameter( ...
            block, [variables{index} '_specify'], 'on');
    end
end

function closeIfLoaded(model)
    if bdIsLoaded(model)
        close_system(model, 0);
    end
end
