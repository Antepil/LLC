function llc_initialize_model(model)
    % LLC_INITIALIZE_MODEL Apply only the configuration injected for this run.
    workspace = get_param(model, 'ModelWorkspace');
    if ~hasVariable(workspace, 'llc_run_override')
        error('llc:MissingSimulationConfig', ...
              'No run configuration was supplied. Use generate_llc_dataset to simulate this model.');
    end
    override = getVariable(workspace, 'llc_run_override');
    required = {'parameters', 'events', 'solver'};
    if ~isstruct(override) || ~isscalar(override) || ~all(isfield(override, required))
        error('llc:InvalidSimulationConfig', 'The injected run configuration is incomplete.');
    end
    assignin(workspace, 'llc_p', override.parameters);
    assignin(workspace, 'llc_gate_source', override.events.gate_source);
    assignin(workspace, 'llc_events', override.events);
    solver = override.solver;
    set_param(model, 'StopTime', num2str(override.events.stop_time_s, 17), ...
              'Solver', solver.name, ...
              'MaxStep', num2str(solver.max_step_s, 17), ...
              'RelTol', num2str(solver.relative_tolerance, 17), ...
              'AbsTol', num2str(solver.absolute_tolerance, 17));
end
