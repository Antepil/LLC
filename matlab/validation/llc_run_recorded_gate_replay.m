function result = llc_run_recorded_gate_replay(model, trajectoryInput, solver, metricConfig)
    % LLC_RUN_RECORDED_GATE_REPLAY Free-roll the model with recorded exact gates.
    if ischar(trajectoryInput) || (isstring(trajectoryInput) && isscalar(trajectoryInput))
        if ~isfile(trajectoryInput)
            error('llc:MissingValidationInput', 'Reference trajectory does not exist: %s', trajectoryInput);
        end
        trajectory = load(trajectoryInput);
    elseif isstruct(trajectoryInput) && isscalar(trajectoryInput)
        trajectory = trajectoryInput;
    else
        error('llc:InvalidValidationInput', 'trajectoryInput must be one MAT path or scalar struct.');
    end
    required = {'schema_version', 'trajectory_id', 'state_names', 'gate_names', ...
                'state_times_s', 'state_values', 'initial_gate_values', ...
                'gate_event_times_s', 'gate_values_after', 'condition'};
    missing = required(~isfield(trajectory, required));
    if ~isempty(missing)
        error('llc:InvalidValidationInput', 'Reference trajectory is missing: %s', strjoin(missing, ', '));
    end
    if ~isfield(trajectory.condition, 'load_ohm') || ...
            abs(trajectory.condition.load_ohm - model.load_resistance_ohm) > 1e-9
        error('llc:ModelContractMismatch', 'Reference trajectory load does not match model.');
    end
    vinSignal = trajectoryVin(trajectory);
    predicted = llc_integrate_neural_ode_events(model, trajectory.state_times_s, ...
        trajectory.state_values(1, :), trajectory.initial_gate_values, ...
        trajectory.gate_event_times_s, trajectory.gate_values_after, vinSignal, solver);
    metrics = llc_compute_neural_ode_metrics(trajectory.state_times_s, predicted, ...
        trajectory.state_values, model.state_scales, metricConfig);
    convergence = [];
    if isfield(solver, 'convergence_check') && solver.convergence_check.enabled
        refined = solver;
        refined.max_step_s = solver.max_step_s * solver.convergence_check.max_step_factor;
        refined.relative_tolerance = solver.relative_tolerance * solver.convergence_check.tolerance_factor;
        refined.absolute_tolerance = solver.absolute_tolerance * solver.convergence_check.tolerance_factor;
        refined.convergence_check.enabled = false;
        refinedStates = llc_integrate_neural_ode_events(model, trajectory.state_times_s, ...
            trajectory.state_values(1, :), trajectory.initial_gate_values, ...
            trajectory.gate_event_times_s, trajectory.gate_values_after, vinSignal, refined);
        convergence = struct('refined_solver', refined, ...
            'maximum_absolute_state_difference', max(abs(refinedStates - predicted), [], 1));
    end
    result = struct('mode', 'recorded_gate_replay', ...
                    'trajectory_id', char(string(trajectory.trajectory_id)), ...
                    'times_s', double(trajectory.state_times_s(:)), ...
                    'reference_states', double(trajectory.state_values), ...
                    'predicted_states', predicted, ...
                    'state_metrics', metrics, 'convergence', convergence);
end

function signal = trajectoryVin(trajectory)
    if isfield(trajectory, 'state_excitation_names') && ...
            isfield(trajectory, 'state_excitation_values')
        names = string(trajectory.state_excitation_names(:));
        index = find(names == "vin_v", 1);
        if isempty(index)
            error('llc:InvalidValidationInput', 'Reference trajectory excitation omits vin_v.');
        end
        signal = struct('type', 'time_series', 'interpolation', 'linear', ...
            'times_s', double(trajectory.state_times_s(:)), ...
            'values_v', double(trajectory.state_excitation_values(:, index)));
    elseif isfield(trajectory.condition, 'vin_v')
        signal = struct('type', 'constant', 'value_v', double(trajectory.condition.vin_v));
    else
        error('llc:InvalidValidationInput', 'Reference trajectory does not define Vin.');
    end
end
