function result = llc_simulate_reference_closed_loop( ...
        hardware, modelFile, initialState, outputTimesS, vinSignal, referenceSignal, control, solver)
    % LLC_SIMULATE_REFERENCE_CLOSED_LOOP Run the existing Simulink circuit branch.
    times = double(outputTimesS(:));
    initialState = double(initialState(:).');
    if numel(times) < 2 || any(diff(times) <= 0) || times(1) ~= 0 || ...
            numel(initialState) ~= 4 || any(~isfinite(initialState))
        error('llc:InvalidValidationInput', 'Reference times or initial state are invalid.');
    end
    if ~isfile(modelFile)
        error('llc:MissingModel', 'Reference Simulink model does not exist: %s', modelFile);
    end
    llc_evaluate_validation_signal(vinSignal, [times(1), times(end)]);
    llc_evaluate_validation_signal(referenceSignal, [times(1), times(end)]);
    [~, modelName] = fileparts(modelFile);
    load_system(modelFile);
    cleanup = onCleanup(@()closeIfLoaded(modelName));
    workspace = get_param(modelName, 'ModelWorkspace');
    parameters = hardware;
    parameters.vin_v = llc_evaluate_validation_signal(vinSignal, times(1));
    parameters.load_ohm = 2.304;
    excitation = struct('validation_schema_version', 'llc-validation-excitation-v1', ...
                        'vin', vinSignal, 'reference', referenceSignal);
    override = struct('parameters', parameters, 'control', control, ...
                      'excitation', excitation, 'solver', solver, ...
                      'stop_time_s', times(end));
    assignin(workspace, 'llc_run_override', override);
    simulation = Simulink.SimulationInput(modelName);
    for name = {'States', 'Gates', 'Control'}
        block = [modelName '/' name{1}];
        simulation = simulation.setBlockParameter(block, 'Decimation', '1');
        simulation = simulation.setBlockParameter(block, 'MaxDataPoints', 'inf');
    end
    simulation = applyInitialState(simulation, modelName, initialState);
    raw = sim(simulation);
    stateTimes = double(raw.llc_states.Time(:));
    stateValues = double(raw.llc_states.Data);
    gateTimes = double(raw.llc_gates.Time(:));
    gateValues = double(raw.llc_gates.Data);
    controlTimes = double(raw.llc_control_log.Time(:));
    controlValues = double(raw.llc_control_log.Data);
    events = llc_extract_gate_events(gateTimes, gateValues);
    result = struct('times_s', times, ...
        'states', interp1(stateTimes, stateValues, times, 'linear'), ...
        'initial_gate_values', events.initial_gate_values, ...
        'gate_event_times_s', events.gate_event_times_s, ...
        'gate_values_after', events.gate_values_after, ...
        'control_names', {{'frequency_hz', 'filtered_voltage_v', ...
                           'reference_voltage_v', 'error_v', 'integral_v_s'}}, ...
        'control_values', interp1(controlTimes, controlValues, times, 'previous'), ...
        'raw_state_times_s', stateTimes, 'raw_state_values', stateValues, ...
        'raw_gate_times_s', gateTimes, 'raw_gate_values', uint8(gateValues));
    if any(~isfinite(result.states), 'all') || any(~isfinite(result.control_values), 'all')
        error('llc:InvalidOutput', 'Reference branch interpolation produced nonfinite values.');
    end
    clear cleanup;
end

function simulation = applyInitialState(simulation, modelName, initialState)
    blocks = {'Lr', 'Cr', 'Lm', 'Co'};
    variables = {'i_L', 'vc', 'i_L', 'vc'};
    for index = 1:numel(blocks)
        block = [modelName '/' blocks{index}];
        simulation = simulation.setBlockParameter(block, variables{index}, num2str(initialState(index), 17));
        simulation = simulation.setBlockParameter(block, [variables{index} '_priority'], 'High');
        simulation = simulation.setBlockParameter(block, [variables{index} '_specify'], 'on');
    end
end

function closeIfLoaded(model)
    if bdIsLoaded(model)
        close_system(model, 0);
    end
end
