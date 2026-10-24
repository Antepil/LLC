function modelFile = build_llc_neural_ode_comparison_model(modelDirectory)
    % BUILD_LLC_NEURAL_ODE_COMPARISON_MODEL Add a parallel Neural ODE branch.
    if nargin < 1
        modelDirectory = fileparts(mfilename('fullpath'));
    end
    if ~(ischar(modelDirectory) || (isstring(modelDirectory) && isscalar(modelDirectory))) || ...
            ~isfolder(modelDirectory)
        error('llc:InvalidModelDirectory', 'modelDirectory must identify an existing folder.');
    end
    modelDirectory = char(modelDirectory);
    sourceFile = fullfile(modelDirectory, 'llc_full_bridge_closed_loop.slx');
    modelFile = fullfile(modelDirectory, 'llc_neural_ode_comparison.slx');
    if ~isfile(sourceFile)
        error('llc:MissingModel', 'Closed-loop source model does not exist: %s', sourceFile);
    end
    if isfile(modelFile)
        return;
    end

    sourceModel = 'llc_full_bridge_closed_loop';
    comparisonModel = 'llc_neural_ode_comparison';
    closeIfLoaded(sourceModel);
    closeIfLoaded(comparisonModel);
    load_system(sourceFile);
    sourceCleanup = onCleanup(@()closeIfLoaded(sourceModel));
    save_system(sourceModel, modelFile);
    clear sourceCleanup;
    closeIfLoaded(sourceModel);
    load_system(modelFile);
    comparisonCleanup = onCleanup(@()closeIfLoaded(comparisonModel));

    workspace = get_param(comparisonModel, 'ModelWorkspace');
    placeholder = struct('is_placeholder', true, 'model', struct(), ...
                         'initial_state', zeros(4, 1));
    assignin(workspace, 'llc_neural_runtime', placeholder);

    add_block('simulink/User-Defined Functions/Level-2 MATLAB S-Function', ...
        [comparisonModel '/NeuralClosedLoopPFM'], ...
        'FunctionName', 'llc_closed_loop_pfm', 'Parameters', 'llc_control', ...
        'Position', [2100 1270 2260 1350]);
    add_block('simulink/Signal Routing/Demux', ...
        [comparisonModel '/NeuralClosedLoopOutputDemux'], ...
        'Outputs', '[4 5]', 'Position', [2310 1265 2315 1365]);
    add_block('simulink/Signal Routing/Mux', ...
        [comparisonModel '/NeuralGateVinMux'], ...
        'Inputs', '2', 'Position', [2390 1180 2395 1240]);
    add_block('simulink/User-Defined Functions/Level-2 MATLAB S-Function', ...
        [comparisonModel '/NeuralODE'], ...
        'FunctionName', 'llc_neural_ode_sfun', 'Parameters', 'llc_neural_runtime', ...
        'Position', [2470 1165 2610 1255]);
    add_block('simulink/Signal Routing/Demux', ...
        [comparisonModel '/NeuralStateDemux'], ...
        'Outputs', '4', 'Position', [2660 1155 2665 1265]);
    add_block('simulink/Math Operations/Sum', [comparisonModel '/StateError'], ...
        'Inputs', '+-', 'Position', [2740 910 2780 970]);
    add_block('simulink/Signal Routing/Mux', [comparisonModel '/ComparisonMux'], ...
        'Inputs', '2', 'Position', [2740 790 2745 860]);

    addLog(comparisonModel, 'NeuralStates', 'llc_neural_states', [2690 1300 2820 1335]);
    addLog(comparisonModel, 'NeuralGates', 'llc_neural_gates', [2380 1380 2510 1415]);
    addLog(comparisonModel, 'NeuralControl', 'llc_neural_control_log', [2530 1380 2680 1415]);
    addLog(comparisonModel, 'StateErrorLog', 'llc_state_error', [2830 920 2960 955]);
    add_block('simulink/Sinks/Scope', [comparisonModel '/ComparisonScope'], ...
        'Position', [2820 800 2880 850]);
    add_block('simulink/Sinks/Scope', [comparisonModel '/StateErrorScope'], ...
        'Position', [2830 1000 2890 1050]);

    add_line(comparisonModel, 'NeuralClosedLoopPFM/1', ...
        'NeuralClosedLoopOutputDemux/1', 'autorouting', 'on');
    add_line(comparisonModel, 'NeuralClosedLoopOutputDemux/1', ...
        'NeuralGateVinMux/1', 'autorouting', 'on');
    add_line(comparisonModel, 'ExcitationDemux/1', ...
        'NeuralGateVinMux/2', 'autorouting', 'on');
    add_line(comparisonModel, 'NeuralGateVinMux/1', 'NeuralODE/1', 'autorouting', 'on');
    add_line(comparisonModel, 'NeuralODE/1', 'NeuralStateDemux/1', 'autorouting', 'on');
    add_line(comparisonModel, 'NeuralStateDemux/4', ...
        'NeuralClosedLoopPFM/1', 'autorouting', 'on');
    add_line(comparisonModel, 'ExcitationDemux/2', ...
        'NeuralClosedLoopPFM/2', 'autorouting', 'on');
    add_line(comparisonModel, 'NeuralODE/1', 'NeuralStates/1', 'autorouting', 'on');
    add_line(comparisonModel, 'NeuralClosedLoopOutputDemux/1', ...
        'NeuralGates/1', 'autorouting', 'on');
    add_line(comparisonModel, 'NeuralClosedLoopOutputDemux/2', ...
        'NeuralControl/1', 'autorouting', 'on');
    add_line(comparisonModel, 'StateMux/1', 'StateError/1', 'autorouting', 'on');
    add_line(comparisonModel, 'NeuralODE/1', 'StateError/2', 'autorouting', 'on');
    add_line(comparisonModel, 'StateError/1', 'StateErrorLog/1', 'autorouting', 'on');
    add_line(comparisonModel, 'StateError/1', 'StateErrorScope/1', 'autorouting', 'on');
    add_line(comparisonModel, 'StateMux/1', 'ComparisonMux/1', 'autorouting', 'on');
    add_line(comparisonModel, 'NeuralODE/1', 'ComparisonMux/2', 'autorouting', 'on');
    add_line(comparisonModel, 'ComparisonMux/1', 'ComparisonScope/1', 'autorouting', 'on');

    annotation = Simulink.Annotation(comparisonModel, ...
        'NEURAL ODE BRANCH | independent PI/PFM | runtime model injection required');
    annotation.Position = [2100 1090 2860 1130];
    set_param(comparisonModel, 'Location', [20 20 1900 1050]);
    save_system(comparisonModel, modelFile);
    clear comparisonCleanup;
    closeIfLoaded(comparisonModel);
end

function addLog(model, name, variable, position)
    add_block('simulink/Sinks/To Workspace', [model '/' name], ...
        'VariableName', variable, 'SaveFormat', 'Timeseries', ...
        'MaxDataPoints', 'inf', 'Decimation', '1', 'Position', position);
end

function closeIfLoaded(model)
    if bdIsLoaded(model)
        close_system(model, 0);
    end
end
