function [result, context] = llc_run_neural_ode_simulink_comparison(configFile, experimentId)
    % LLC_RUN_NEURAL_ODE_SIMULINK_COMPARISON Run one configured dual-branch comparison.
    if nargin < 1 || strlength(string(configFile)) == 0
        matlabRoot = fileparts(fileparts(mfilename('fullpath')));
        projectRoot = fileparts(matlabRoot);
        configFile = fullfile(projectRoot, 'configs', 'matlab', ...
                              'neural_ode_validation.json');
    end
    if nargin < 2 || ~(ischar(experimentId) || ...
            (isstring(experimentId) && isscalar(experimentId))) || ...
            strlength(string(experimentId)) == 0
        error('llc:InvalidArgument', 'experimentId must be nonempty scalar text.');
    end

    config = llc_load_neural_ode_validation_config(configFile, true);
    experiment = selectExperiment(config.experiments, char(string(experimentId)));
    hardware = llc_read_json(config.hardware_config_file);
    model = llc_load_neural_ode_model(config.model_export.file, ...
        2.304, config.model_export.load_abs_tolerance_ohm, ...
        config.model_export.consistency_abs_tolerance, ...
        config.model_export.consistency_relative_tolerance);
    if ~strcmp(model.schema_version, config.model_export.required_schema_version)
        error('llc:ModelContractMismatch', ...
              'Loaded model schema does not match model_export.required_schema_version.');
    end

    step = 1 / (double(config.comparison_sampling.reference_frequency_hz) * ...
                double(config.comparison_sampling.intervals_per_cycle));
    duration = double(experiment.startup.evaluation_duration_s);
    outputTimes = fixedGrid(duration, step);
    startupMode = string(experiment.startup.mode);
    if startupMode == "cold"
        initialState = zeros(4, 1);
        controllerState = llc_initialize_closed_loop_controller(config.control);
        steadyAssessment = [];
    else
        [initialState, controllerState, steadyAssessment] = warmInitialState( ...
            hardware, config.reference_plant.model_file, experiment, ...
            config.control, config.steady_state_assessment, config.solver, step);
    end

    result = llc_simulate_neural_ode_comparison(model, hardware, ...
        config.paths.simulink_comparison_model_file, initialState, initialState, ...
        controllerState, outputTimes, experiment.vin, experiment.reference, ...
        config.control, config.solver);
    result.startup_mode = char(startupMode);
    result.steady_state_assessment = steadyAssessment;
    context = struct('config_file', char(string(configFile)), ...
        'experiment_id', experiment.experiment_id, ...
        'startup_mode', char(startupMode), ...
        'comparison_model_file', config.paths.simulink_comparison_model_file, ...
        'reference_initial_state', initialState, ...
        'neural_initial_state', initialState, ...
        'controller_state', controllerState, ...
        'steady_state_assessment', steadyAssessment);
end

function experiment = selectExperiment(experiments, identifier)
    identifiers = string({experiments.experiment_id});
    selected = identifiers == string(identifier) & ...
               [experiments.enabled] & ...
               string({experiments.mode}) == "independent_closed_loop";
    if nnz(selected) ~= 1
        error('llc:InvalidConfig', ...
              ['experimentId must select exactly one enabled ', ...
               'independent_closed_loop experiment: %s'], identifier);
    end
    experiment = experiments(selected);
end

function times = fixedGrid(duration, step)
    intervals = duration / step;
    count = round(intervals);
    tolerance = 128 * eps(max(abs(intervals), 1));
    if abs(intervals - count) > tolerance
        error('llc:InvalidComparisonGrid', ...
              'Duration must be an integer multiple of the comparison interval.');
    end
    times = (0:count).' * step;
    times(end) = duration;
end

function [initialState, controllerState, assessment] = warmInitialState( ...
        hardware, modelFile, experiment, control, steadyConfig, solver, step)
    handoffTime = double(experiment.startup.preheat_duration_s);
    evaluationDuration = double(experiment.startup.evaluation_duration_s);
    schedulerCoverage = 4 * double(control.control_period_s) + ...
                        2 / min(double(control.initial_frequency_hz), ...
                                double(control.bias_frequency_hz));
    postHandoffDuration = max(evaluationDuration, schedulerCoverage);
    extendedVin = extendSignal(experiment.vin, postHandoffDuration);
    extendedReference = extendSignal(experiment.reference, postHandoffDuration);
    vinSignal = stagedSignal(handoffTime, experiment.preheat_vin, extendedVin);
    referenceSignal = stagedSignal( ...
        handoffTime, experiment.preheat_reference, extendedReference);
    referenceTimes = simulationGrid(handoffTime, postHandoffDuration, step);
    branch = llc_simulate_reference_closed_loop(hardware, modelFile, zeros(4, 1), ...
        referenceTimes, vinSignal, referenceSignal, control, solver);

    assessment = assessPreheat(branch, steadyConfig, referenceSignal, handoffTime);
    initialState = interp1(branch.raw_state_times_s, branch.raw_state_values, ...
                           handoffTime, 'linear').';
    reconstruction = llc_reconstruct_closed_loop_controller( ...
        branch.raw_state_times_s, branch.raw_state_values(:, 4), ...
        branch.initial_gate_values, branch.gate_event_times_s, ...
        branch.gate_values_after, branch.raw_control_times_s, ...
        branch.raw_control_values, referenceSignal, control, handoffTime, ...
        solver.event_time_tolerance_s);
    controllerState = llc_shift_closed_loop_controller_state( ...
        reconstruction.final_state, handoffTime, solver.event_time_tolerance_s);
end

function assessment = assessPreheat(branch, config, referenceSignal, handoffTime)
    if ~isstruct(config) || isempty(fieldnames(config))
        error('llc:InvalidSteadyStateConfig', ...
              'Warm startup requires steady-state assessment configuration.');
    end
    config.check_times_s = handoffTime;
    beforeHandoff = max(0, handoffTime - 64 * eps(max(handoffTime, 1)));
    preheatReference = llc_evaluate_validation_signal(referenceSignal, beforeHandoff);
    assessment = llc_assess_periodic_steady_state( ...
        branch.raw_state_times_s, branch.raw_state_values, ...
        branch.raw_gate_times_s, branch.raw_gate_values, preheatReference, config);
    if ~assessment.settled
        error('llc:PreheatNotSettled', ...
              'Reference plant was not settled at the configured warm-start handoff.');
    end
end

function signal = stagedSignal(switchTime, before, after)
    signal = struct('type', 'staged', 'switch_time_s', switchTime, ...
                    'before', before, 'after', after);
end

function extended = extendSignal(signal, duration)
    extended = signal;
    if string(signal.type) ~= "time_series"
        return;
    end
    times = double(signal.times_s(:));
    values = double(signal.values_v(:));
    tolerance = 64 * eps(max(duration, 1));
    if times(end) < duration - tolerance
        extended.times_s = [times; duration];
        extended.values_v = [values; values(end)];
    end
end

function times = simulationGrid(handoffTime, postHandoffDuration, step)
    stopTime = handoffTime + postHandoffDuration;
    times = (0:floor(stopTime / step)).' * step;
    boundaries = [handoffTime; stopTime];
    for index = 1:numel(boundaries)
        boundary = boundaries(index);
        if all(abs(times - boundary) > 128 * eps(max(boundary, 1)))
            times(end + 1, 1) = boundary; %#ok<AGROW>
        end
    end
    times = unique(sort(times));
    times(end) = stopTime;
end
