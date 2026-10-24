function metrics = llc_compute_neural_ode_metrics(timesS, predicted, reference, stateScales, metricConfig)
    % LLC_COMPUTE_NEURAL_ODE_METRICS Compute per-state free-rollout errors.
    times = double(timesS(:));
    predicted = double(predicted);
    reference = double(reference);
    scales = double(stateScales(:).');
    if numel(times) < 2 || any(diff(times) <= 0) || ...
            ~isequal(size(predicted), size(reference), [numel(times), 4]) || ...
            numel(scales) ~= 4 || any(scales <= 0) || ...
            any(~isfinite(predicted), 'all') || any(~isfinite(reference), 'all')
        error('llc:InvalidMetricInput', 'Metric inputs must be aligned finite four-state trajectories.');
    end
    errorValues = predicted - reference;
    rmse = sqrt(mean(errorValues .^ 2, 1));
    metrics = struct('rmse', rmse, ...
                     'normalized_rmse', rmse ./ scales, ...
                     'maximum_absolute_error', max(abs(errorValues), [], 1), ...
                     'final_absolute_error', abs(errorValues(end, :)));
    if isfield(metricConfig, 'tail_window_s') && metricConfig.tail_window_s > 0
        tail = times >= times(end) - metricConfig.tail_window_s;
        metrics.tail_output_mean_error_v = mean(errorValues(tail, 4));
    end
end
