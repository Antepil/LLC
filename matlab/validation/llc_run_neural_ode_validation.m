function [report, files] = llc_run_neural_ode_validation(configFile)
    % LLC_RUN_NEURAL_ODE_VALIDATION Run every enabled configured experiment.
    if nargin < 1 || strlength(string(configFile)) == 0
        matlabRoot = fileparts(fileparts(mfilename('fullpath')));
        projectRoot = fileparts(matlabRoot);
        configFile = fullfile(projectRoot, 'configs', 'matlab', ...
                              'neural_ode_validation.json');
    end
    config = llc_load_neural_ode_validation_config(configFile, true);
    hardware = llc_read_json(config.hardware_config_file);
    model = llc_load_neural_ode_model(config.model_export.file, ...
        2.304, config.model_export.load_abs_tolerance_ohm, ...
        config.model_export.consistency_abs_tolerance, ...
        config.model_export.consistency_relative_tolerance);
    if ~strcmp(model.schema_version, config.model_export.required_schema_version)
        error('llc:ModelContractMismatch', ...
              'Loaded model schema does not match model_export.required_schema_version.');
    end
    enabled = config.experiments([config.experiments.enabled]);
    if isempty(enabled)
        error('llc:InvalidConfig', 'At least one validation experiment must be enabled.');
    end
    results = repmat(emptyResult, numel(enabled), 1);
    for index = 1:numel(enabled)
        experiment = enabled(index);
        if strcmp(experiment.mode, 'recorded_gate_replay')
            value = llc_run_recorded_gate_replay(model, ...
                experiment.reference_trajectory_file, config.solver, config.metrics);
        else
            value = llc_run_independent_closed_loop(model, hardware, ...
                config.reference_plant.model_file, experiment, config.control, ...
                config.steady_state_assessment, config.solver, ...
                config.comparison_sampling, config.metrics);
        end
        value.experiment_id = experiment.experiment_id;
        value.mode = experiment.mode;
        results(index) = normalizeResult(value);
    end
    report = struct('schema_version', 'llc-neural-ode-validation-result-v1', ...
        'generated_at', char(datetime('now', 'TimeZone', 'Asia/Shanghai', ...
                                     'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX')), ...
        'config_file', char(string(configFile)), 'results', results);
    files = llc_save_neural_ode_validation_results(report, config);
end

function value = emptyResult
    value = struct('experiment_id', '', 'mode', '', 'startup_mode', '', ...
        'trajectory_id', '', 'times_s', [], 'evaluation_times_s', [], ...
        'reference_states', [], 'predicted_states', [], 'reference_voltage_v', [], ...
        'tail_window_s', [], ...
        'state_metrics', [], 'tracking_metrics', [], 'convergence', [], ...
        'steady_state_assessment', [], 'reference_branch', [], 'neural_branch', []);
end

function normalized = normalizeResult(value)
    normalized = emptyResult;
    names = intersect(fieldnames(normalized), fieldnames(value));
    for index = 1:numel(names)
        normalized.(names{index}) = value.(names{index});
    end
end
