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
    assignin(workspace, 'llc_excitation', buildExcitationPlaceholder());
    add_block('simulink/User-Defined Functions/Level-2 MATLAB S-Function', controller, ...
              'Position', sourcePosition, 'FunctionName', 'llc_closed_loop_pfm', ...
              'Parameters', 'llc_control');
    outputDemux = [closedLoopModel '/ClosedLoopOutputDemux'];
    add_block('simulink/Signal Routing/Demux', outputDemux, ...
              'Position', sourcePosition + [150, 0, 155, 100], 'Outputs', '[4 5]');
    add_line(closedLoopModel, 'Measure_V_o/1', 'ClosedLoopPFM/1', 'autorouting', 'on');

    excitation = [closedLoopModel '/ClosedLoopExcitation'];
    add_block('simulink/User-Defined Functions/Level-2 MATLAB S-Function', excitation, ...
              'Position', [50, 650, 190, 690], ...
              'FunctionName', 'llc_closed_loop_excitation', ...
              'Parameters', 'llc_excitation');
    excitationDemux = [closedLoopModel '/ExcitationDemux'];
    add_block('simulink/Signal Routing/Demux', excitationDemux, ...
              'Position', [220, 640, 225, 700], 'Outputs', '2');
    add_line(closedLoopModel, 'ClosedLoopExcitation/1', 'ExcitationDemux/1', 'autorouting', 'on');
    add_line(closedLoopModel, 'ExcitationDemux/2', 'ClosedLoopPFM/2', 'autorouting', 'on');

    replace_block(closedLoopModel, 'Name', 'Vin', ...
        'fl_lib/Electrical/Electrical Sources/Controlled Voltage Source', 'noprompt');
    vinConverter = [closedLoopModel '/VinCommand'];
    add_block('nesl_utility/Simulink-PS Converter', vinConverter, ...
              'Position', [250, 650, 330, 680]);
    set_param(vinConverter, 'Unit', 'V', 'FilteringAndDerivatives', 'zero');
    add_line(closedLoopModel, 'ExcitationDemux/1', 'VinCommand/1', 'autorouting', 'on');
    converterPorts = get_param(vinConverter, 'PortHandles');
    vinPorts = get_param([closedLoopModel '/Vin'], 'PortHandles');
    add_line(closedLoopModel, converterPorts.RConn(1), vinPorts.RConn(1), ...
             'autorouting', 'on');
    referencePorts = get_param([closedLoopModel '/PrimaryReference'], 'PortHandles');
    add_line(closedLoopModel, vinPorts.RConn(2), referencePorts.LConn(1), ...
             'autorouting', 'on');
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
    excitationLog = [closedLoopModel '/Excitation'];
    add_block('simulink/Sinks/To Workspace', excitationLog, ...
              'Position', [250, 700, 330, 730], ...
              'VariableName', 'llc_excitation_log', 'SaveFormat', 'Timeseries', ...
              'MaxDataPoints', 'inf', 'Decimation', '1');
    add_line(closedLoopModel, 'ClosedLoopExcitation/1', 'Excitation/1', 'autorouting', 'on');
    save_system(closedLoopModel, modelFile);
    clear cleanupClosedLoop;
    closeIfLoaded(closedLoopModel);
end

function excitation = buildExcitationPlaceholder()
    constantVin = struct('center_value', 400, 'amplitude', 0, ...
                         'frequency_hz', 0, 'phase_rad', 0);
    constantReference = struct('center_value', 48, 'amplitude', 0, ...
                               'frequency_hz', 0, 'phase_rad', 0);
    excitation = struct('start_time_s', 0, 'envelope_rise_time_s', 0, ...
                        'vin', constantVin, 'reference', constantReference);
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
