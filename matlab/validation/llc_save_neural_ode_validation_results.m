function files = llc_save_neural_ode_validation_results(report, config)
    % LLC_SAVE_NEURAL_ODE_VALIDATION_RESULTS Persist waveforms, metrics and plots.
    validateOutputConfig(config);
    outputDirectory = char(string(config.output.directory));
    if ~isfolder(outputDirectory)
        mkdir(outputDirectory);
    end
    files = struct('results_mat', "", 'resolved_config_mat', "", ...
                   'state_metrics_csv', "", 'tracking_metrics_csv', "", ...
                   'figure_files', strings(0, 1));
    if config.output.save_waveforms
        files.results_mat = string(fullfile(outputDirectory, 'validation_results.mat'));
        save(files.results_mat, 'report', '-v7');
    end
    if config.output.save_resolved_config
        files.resolved_config_mat = string(fullfile(outputDirectory, 'resolved_config.mat'));
        save(files.resolved_config_mat, 'config', '-v7');
    end
    results = report.results;
    if config.output.save_metric_tables
        stateTable = stateMetricTable(results);
        files.state_metrics_csv = string(fullfile(outputDirectory, 'state_metrics.csv'));
        writetable(stateTable, files.state_metrics_csv);
        trackingTable = trackingMetricTable(results);
        if ~isempty(trackingTable)
            files.tracking_metrics_csv = string(fullfile(outputDirectory, 'tracking_metrics.csv'));
            writetable(trackingTable, files.tracking_metrics_csv);
        end
    end
    if config.output.save_figures
        for index = 1:numel(results)
            files.figure_files(end + 1, 1) = saveComparisonFigure(results(index), outputDirectory); %#ok<AGROW>
        end
    end
end

function tableValue = stateMetricTable(results)
    stateNames = ["i_Lr", "v_Cr", "i_Lm", "v_o"];
    rows = cell(numel(results) * 4, 8);
    row = 0;
    for resultIndex = 1:numel(results)
        metrics = results(resultIndex).state_metrics;
        for stateIndex = 1:4
            row = row + 1;
            rows(row, :) = {string(results(resultIndex).experiment_id), ...
                string(results(resultIndex).mode), stateNames(stateIndex), ...
                metrics.rmse(stateIndex), metrics.normalized_rmse(stateIndex), ...
                metrics.maximum_absolute_error(stateIndex), ...
                metrics.final_absolute_error(stateIndex), tailError(metrics, stateIndex)};
        end
    end
    tableValue = cell2table(rows, 'VariableNames', ...
        {'experiment_id', 'mode', 'state', 'rmse', 'normalized_rmse', ...
         'maximum_absolute_error', 'final_absolute_error', 'tail_mean_error'});
end

function value = tailError(metrics, stateIndex)
    if stateIndex == 4 && isfield(metrics, 'tail_output_mean_error_v')
        value = metrics.tail_output_mean_error_v;
    else
        value = NaN;
    end
end

function tableValue = trackingMetricTable(results)
    rows = cell(0, 8);
    for index = 1:numel(results)
        if ~isfield(results(index), 'tracking_metrics') || isempty(results(index).tracking_metrics)
            continue;
        end
        metrics = results(index).tracking_metrics;
        rows(end + 1, :) = {string(results(index).experiment_id), ... %#ok<AGROW>
            string(results(index).mode), metrics.reference_plant_rmse_v, ...
            metrics.neural_ode_rmse_v, metrics.reference_plant_mean_absolute_error_v, ...
            metrics.neural_ode_mean_absolute_error_v, ...
            metrics.reference_plant_tail_mean_error_v, ...
            metrics.neural_ode_tail_mean_error_v};
    end
    if isempty(rows)
        tableValue = table;
    else
        tableValue = cell2table(rows, 'VariableNames', ...
            {'experiment_id', 'mode', 'reference_plant_rmse_v', ...
             'neural_ode_rmse_v', 'reference_plant_mean_absolute_error_v', ...
             'neural_ode_mean_absolute_error_v', ...
             'reference_plant_tail_mean_error_v', 'neural_ode_tail_mean_error_v'});
    end
end

function file = saveComparisonFigure(result, outputDirectory)
    stateNames = {'i_{Lr} (A)', 'v_{Cr} (V)', 'i_{Lm} (A)', 'v_o (V)'};
    figureHandle = figure('Visible', 'off', 'Color', 'white');
    cleanup = onCleanup(@()close(figureHandle));
    plotTimes = result.times_s;
    xLabel = 'Time (s)';
    if isfield(result, 'evaluation_times_s') && ~isempty(result.evaluation_times_s)
        plotTimes = result.evaluation_times_s;
        xLabel = 'Evaluation time (s)';
    end
    layout = tiledlayout(figureHandle, 5, 1, 'TileSpacing', 'compact');
    for stateIndex = 1:4
        axesHandle = nexttile(layout);
        plot(axesHandle, plotTimes, result.reference_states(:, stateIndex), ...
             'LineWidth', 1.0, 'DisplayName', 'Reference');
        hold(axesHandle, 'on');
        plot(axesHandle, plotTimes, result.predicted_states(:, stateIndex), '--', ...
             'LineWidth', 1.0, 'DisplayName', 'Neural ODE');
        ylabel(axesHandle, stateNames{stateIndex});
        grid(axesHandle, 'on');
    end
    tailAxes = nexttile(layout);
    tailWindow = defaultTailWindow(plotTimes);
    if isfield(result, 'tail_window_s') && ~isempty(result.tail_window_s)
        tailWindow = result.tail_window_s;
    end
    tail = plotTimes >= plotTimes(end) - tailWindow;
    plot(tailAxes, plotTimes(tail), result.reference_states(tail, 4), ...
         'LineWidth', 1.0, 'DisplayName', 'Reference');
    hold(tailAxes, 'on');
    plot(tailAxes, plotTimes(tail), result.predicted_states(tail, 4), '--', ...
         'LineWidth', 1.0, 'DisplayName', 'Neural ODE');
    ylabel(tailAxes, 'tail v_o (V)');
    grid(tailAxes, 'on');
    xlabel(layout, xLabel);
    title(layout, strrep(char(string(result.experiment_id)), '_', '\_'));
    legend(nexttile(layout, 1), 'Location', 'best');
    identifier = regexprep(char(string(result.experiment_id)), '[^A-Za-z0-9_-]', '_');
    file = string(fullfile(outputDirectory, [identifier, '_waveforms.png']));
    exportgraphics(figureHandle, file, 'Resolution', 150);
    clear cleanup;
end

function value = defaultTailWindow(times)
    value = max((times(end) - times(1)) / 10, eps(max(abs(times(end)), 1)));
end

function validateOutputConfig(config)
    fields = {'directory', 'save_resolved_config', 'save_waveforms', ...
              'save_metric_tables', 'save_figures'};
    if ~isstruct(config) || ~isfield(config, 'output') || ~isstruct(config.output) || ...
            ~all(isfield(config.output, fields)) || strlength(string(config.output.directory)) == 0
        error('llc:InvalidOutputConfig', 'Validation output configuration is incomplete.');
    end
    flags = [config.output.save_resolved_config, config.output.save_waveforms, ...
             config.output.save_metric_tables, config.output.save_figures];
    if ~islogical(flags) || numel(flags) ~= 4
        error('llc:InvalidOutputConfig', 'Validation output switches must be logical scalars.');
    end
end
