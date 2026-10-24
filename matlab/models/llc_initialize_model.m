function llc_initialize_model(model)
    % LLC_INITIALIZE_MODEL Apply only the configuration injected for this run.
    workspace = get_param(model, 'ModelWorkspace');
    if ~hasVariable(workspace, 'llc_run_override')
        error('llc:MissingSimulationConfig', ...
              'No run configuration was supplied. Use generate_llc_dataset to simulate this model.');
    end
    override = getVariable(workspace, 'llc_run_override');
    required = {'parameters', 'solver'};
    if ~isstruct(override) || ~isscalar(override) || ~all(isfield(override, required))
        error('llc:InvalidSimulationConfig', 'The injected run configuration is incomplete.');
    end
    assignin(workspace, 'llc_p', override.parameters);
    if isfield(override, 'excitation')
        excitation = override.excitation;
    elseif isfield(override, 'control') && ...
            isfield(override.control, 'reference_voltage_v') && ...
            isfield(override.parameters, 'vin_v')
        excitation = struct( ...
            'start_time_s', 0, 'envelope_rise_time_s', 0, ...
            'vin', constantSignal(override.parameters.vin_v), ...
            'reference', constantSignal(override.control.reference_voltage_v));
    elseif isfield(override, 'events')
        excitation = [];
    else
        error('llc:InvalidSimulationConfig', ...
              'Run configuration does not define a valid excitation.');
    end
    if ~isempty(excitation)
        assignin(workspace, 'llc_excitation', excitation);
    end
    if isfield(override, 'events')
        assignin(workspace, 'llc_gate_source', override.events.gate_source);
        assignin(workspace, 'llc_events', override.events);
        stopTime = override.events.stop_time_s;
    elseif isfield(override, 'control') && isfield(override, 'stop_time_s')
        assignin(workspace, 'llc_control', override.control);
        stopTime = override.stop_time_s;
    else
        error('llc:InvalidSimulationConfig', ...
              'Run configuration must contain either events or closed-loop control.');
    end
    solver = override.solver;
    set_param(model, 'StopTime', num2str(stopTime, 17), ...
              'Solver', solver.name, ...
              'MaxStep', num2str(solver.max_step_s, 17), ...
              'RelTol', num2str(solver.relative_tolerance, 17), ...
              'AbsTol', num2str(solver.absolute_tolerance, 17));
end

function signal = constantSignal(value)
    signal = struct('center_value', value, 'amplitude', 0, ...
                    'frequency_hz', 0, 'phase_rad', 0);
end
