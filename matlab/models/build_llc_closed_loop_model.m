function modelFile = build_llc_closed_loop_model(modelDirectory)
    % BUILD_LLC_CLOSED_LOOP_MODEL Derive the feedback model from the circuit model.
    if nargin < 1
        modelDirectory = fileparts(mfilename('fullpath'));
    end
    sourceFile = fullfile(modelDirectory, 'llc_full_bridge.slx');
    modelFile = fullfile(modelDirectory, 'llc_full_bridge_closed_loop.slx');
    sourceModel = 'llc_full_bridge';
    closedLoopModel = 'llc_full_bridge_closed_loop';
    if ~isfile(sourceFile)
        error('llc:MissingModel', 'Open-loop source model does not exist: %s', sourceFile);
    end
    if bdIsLoaded(sourceModel)
        close_system(sourceModel, 0);
    end
    if bdIsLoaded(closedLoopModel)
        close_system(closedLoopModel, 0);
    end
    load_system(sourceFile);
    cleanupSource = onCleanup(@()closeIfLoaded(sourceModel));
    save_system(sourceModel, modelFile);
    clear cleanupSource;
    closeIfLoaded(sourceModel);
    load_system(modelFile);
    cleanupClosedLoop = onCleanup(@()closeIfLoaded(closedLoopModel));

    sourceBlock = [closedLoopModel '/ActualGateEvents'];
    sourcePosition = get_param(sourceBlock, 'Position');
    gateTargets = {'GateDemux', 'Gates', 'GateScope'};
    for targetIndex = 1:numel(gateTargets)
        targetPorts = get_param([closedLoopModel '/' gateTargets{targetIndex}], 'PortHandles');
        existingGateLine = get_param(targetPorts.Inport(1), 'Line');
        if existingGateLine ~= -1
            delete_line(existingGateLine);
        end
    end
    delete_block(sourceBlock);
    controller = [closedLoopModel '/ClosedLoopPFM'];
    workspace = get_param(closedLoopModel, 'ModelWorkspace');
    assignin(workspace, 'llc_control', buildControlPlaceholder());
    add_block('simulink/User-Defined Functions/Level-2 MATLAB S-Function', controller, ...
              'Position', sourcePosition, 'FunctionName', 'llc_closed_loop_pfm', ...
              'Parameters', 'llc_control');
    outputDemux = [closedLoopModel '/ClosedLoopOutputDemux'];
    add_block('simulink/Signal Routing/Demux', outputDemux, ...
              'Position', sourcePosition + [150, 0, 155, 100], 'Outputs', '[4 5]');
    add_line(closedLoopModel, 'Measure_V_o/1', 'ClosedLoopPFM/1', 'autorouting', 'on');
    add_line(closedLoopModel, 'ClosedLoopPFM/1', 'ClosedLoopOutputDemux/1', 'autorouting', 'on');
    add_line(closedLoopModel, 'ClosedLoopOutputDemux/1', 'GateDemux/1', 'autorouting', 'on');
    add_line(closedLoopModel, 'ClosedLoopOutputDemux/1', 'Gates/1', 'autorouting', 'on');
    add_line(closedLoopModel, 'ClosedLoopOutputDemux/1', 'GateScope/1', 'autorouting', 'on');

    controlLog = [closedLoopModel '/Control'];
    add_block('simulink/Sinks/To Workspace', controlLog, ...
              'Position', sourcePosition + [230, 80, 300, 100], ...
              'VariableName', 'llc_control_log', 'SaveFormat', 'Timeseries', ...
              'MaxDataPoints', 'inf', 'Decimation', '1');
    add_line(closedLoopModel, 'ClosedLoopOutputDemux/2', 'Control/1', 'autorouting', 'on');
    save_system(closedLoopModel, modelFile);
    clear cleanupClosedLoop;
    closeIfLoaded(closedLoopModel);
end

function control = buildControlPlaceholder()
    control = struct('control_period_s', 1e-5, ...
                     'voltage_filter_time_constant_s', 5e-5, ...
                     'reference_ramp_time_s', 1e-3, 'reference_voltage_v', 48, ...
                     'initial_frequency_hz', 180e3, 'bias_frequency_hz', 110e3, ...
                     'frequency_slew_rate_hz_per_s', 1e8, ...
                     'proportional_gain_hz_per_v', 1e3, ...
                     'integral_gain_hz_per_v_s', 5e6, ...
                     'deadtime_s', 2e-7, 'minimum_pulse_s', 5e-7);
end

function closeIfLoaded(model)
    if bdIsLoaded(model)
        close_system(model, 0);
    end
end
