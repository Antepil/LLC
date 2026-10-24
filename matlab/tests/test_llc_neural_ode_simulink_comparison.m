function tests = test_llc_neural_ode_simulink_comparison
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    matlabRoot = fileparts(fileparts(mfilename('fullpath')));
    testCase.TestData.matlabRoot = matlabRoot;
    addpath(fullfile(matlabRoot, 'validation'));
    addpath(fullfile(matlabRoot, 'modulation'));
    addpath(fullfile(matlabRoot, 'models'));
    addpath(fullfile(matlabRoot, 'export'));
end

function teardown(testCase) %#ok<INUSD>
    bdclose('all');
end

function testContinuousSFunctionMatchesConstantDerivative(testCase)
    runtime = validRuntime;
    runtime.initial_state = [10; 20; 30; 40];
    logged = simulateNeuralHarness(runtime, [0 0 0 0 400], 0.02);
    times = double(logged.Time(:));
    values = double(logged.Data);
    expected = runtime.initial_state.' + times * [1 2 3 4];

    verifyGreaterThan(testCase, numel(times), 2);
    verifyEqual(testCase, values, expected, 'AbsTol', 1e-8);
end

function testContinuousSFunctionRejectsPlaceholderRuntime(testCase)
    runtime = validRuntime;
    runtime.is_placeholder = true;
    verifySimulationError(testCase, ...
        @()simulateNeuralHarness(runtime, [0 0 0 0 400], 1e-6), ...
        'llc:MissingNeuralODEModel');
end

function testContinuousSFunctionRejectsMalformedInitialState(testCase)
    runtime = validRuntime;
    runtime.initial_state = [0; 0; 0];
    verifySimulationError(testCase, ...
        @()simulateNeuralHarness(runtime, [0 0 0 0 400], 1e-6), ...
        'llc:InvalidNeuralODEInput');
end

function testContinuousSFunctionRejectsInvalidGates(testCase)
    verifySimulationError(testCase, ...
        @()simulateNeuralHarness(validRuntime, [1 1 0 0 400], 1e-6), ...
        'llc:InvalidNeuralODEInput');
end

function testContinuousSFunctionRejectsNonfiniteVin(testCase)
    verifySimulationError(testCase, ...
        @()simulateNeuralHarness(validRuntime, [0 0 0 0 NaN], 1e-6), ...
        'llc:InvalidNeuralODEInput');
end

function testComparisonBuilderCreatesIndependentNeuralBranch(testCase)
    [directory, sourceHash] = comparisonModelFixture(testCase);
    modelFile = build_llc_neural_ode_comparison_model(directory);
    [~, modelName] = fileparts(modelFile);
    load_system(modelFile);
    cleanup = onCleanup(@()closeTestModel(modelName));

    requiredBlocks = {'NeuralClosedLoopPFM', 'NeuralClosedLoopOutputDemux', ...
        'NeuralGateVinMux', 'NeuralODE', 'NeuralStateDemux', 'StateError', ...
        'NeuralStates', 'NeuralGates', 'NeuralControl', 'StateErrorLog', ...
        'ComparisonScope', 'StateErrorScope'};
    for index = 1:numel(requiredBlocks)
        verifyNotEmpty(testCase, getSimulinkBlockHandle( ...
            [modelName '/' requiredBlocks{index}]));
    end
    verifyEqual(testCase, get_param([modelName '/NeuralODE'], 'FunctionName'), ...
        'llc_neural_ode_sfun');
    verifyEqual(testCase, get_param([modelName '/NeuralClosedLoopPFM'], 'FunctionName'), ...
        'llc_closed_loop_pfm');
    verifyEqual(testCase, upstreamBlock(modelName, 'NeuralClosedLoopPFM', 1), ...
        [modelName '/NeuralStateDemux']);
    verifyEqual(testCase, upstreamBlock(modelName, 'NeuralClosedLoopPFM', 2), ...
        [modelName '/ExcitationDemux']);
    verifyEqual(testCase, upstreamBlock(modelName, 'NeuralGateVinMux', 1), ...
        [modelName '/NeuralClosedLoopOutputDemux']);
    verifyEqual(testCase, upstreamBlock(modelName, 'NeuralGateVinMux', 2), ...
        [modelName '/ExcitationDemux']);

    workspace = get_param(modelName, 'ModelWorkspace');
    runtime = getVariable(workspace, 'llc_neural_runtime');
    verifyTrue(testCase, runtime.is_placeholder);
    verifyEqual(testCase, runtime.initial_state, zeros(4, 1));
    verifyEqual(testCase, sha256File(fullfile(directory, ...
        'llc_full_bridge_closed_loop.slx')), sourceHash);
    clear cleanup;
    closeTestModel(modelName);
end

function testComparisonBuilderIsIdempotentAndPreservesSource(testCase)
    [directory, sourceHash] = comparisonModelFixture(testCase);
    first = build_llc_neural_ode_comparison_model(directory);
    firstHash = sha256File(first);
    second = build_llc_neural_ode_comparison_model(directory);

    verifyEqual(testCase, second, first);
    verifyEqual(testCase, sha256File(first), firstHash);
    verifyEqual(testCase, sha256File(fullfile(directory, ...
        'llc_full_bridge_closed_loop.slx')), sourceHash);
end

function testColdComparisonSimulationReturnsAlignedFiniteLogs(testCase)
    times = (0:5e-6:20e-6).';
    control = comparisonControl;
    runtime = validRuntime;
    result = llc_simulate_neural_ode_comparison( ...
        runtime.model, comparisonHardware(testCase), comparisonModelFile(testCase), ...
        zeros(4, 1), zeros(4, 1), ...
        llc_initialize_closed_loop_controller(control), times, ...
        constantSignal(400), constantSignal(48), control, comparisonSolver);

    verifyEqual(testCase, result.times_s, times, 'AbsTol', 0);
    verifySize(testCase, result.reference_states, [numel(times), 4]);
    verifySize(testCase, result.predicted_states, [numel(times), 4]);
    verifyEqual(testCase, result.state_error, ...
        result.reference_states - result.predicted_states, 'AbsTol', 1e-10);
    verifyTrue(testCase, all(isfinite(result.reference_states), 'all'));
    verifyTrue(testCase, all(isfinite(result.predicted_states), 'all'));
    verifySize(testCase, result.reference_control_values, [numel(times), 5]);
    verifySize(testCase, result.neural_control_values, [numel(times), 5]);
    verifySize(testCase, result.reference_gate_values, [numel(times), 4]);
    verifySize(testCase, result.neural_gate_values, [numel(times), 4]);
end

function testWarmComparisonRestoresBothBranchesAndIndependentControllers(testCase)
    times = [0; 1e-6; 2e-6];
    control = comparisonControl;
    state = llc_initialize_closed_loop_controller(control);
    state.reference_ramp_elapsed_time_s = control.reference_ramp_time_s;
    state.reference_voltage_v = 48;
    state.filtered_voltage_v = 47;
    state.error_v = 1;
    state.integral_v_s = 2e-5;
    state.frequency_hz = 125e3;
    state.next_control_time_s = 1e-6;
    state.next_edge_time_s = 3e-6;
    state.pending_on_time_s = Inf;
    state.active_pair = 1;
    state.pending_pair = 0;
    referenceInitial = [1; 2; 3; 4];
    neuralInitial = [5; 6; 7; 8];
    runtime = validRuntime;

    result = llc_simulate_neural_ode_comparison( ...
        runtime.model, comparisonHardware(testCase), comparisonModelFile(testCase), ...
        referenceInitial, neuralInitial, state, times, ...
        constantSignal(400), constantSignal(48), control, comparisonSolver);

    verifyEqual(testCase, result.reference_states(1, :), referenceInitial.', 'AbsTol', 1e-8);
    verifyEqual(testCase, result.predicted_states(1, :), neuralInitial.', 'AbsTol', 1e-12);
    expectedControl = [125e3 47 48 1 2e-5];
    verifyEqual(testCase, result.reference_control_values(1, :), expectedControl, ...
        'AbsTol', 1e-10);
    verifyEqual(testCase, result.neural_control_values(1, :), expectedControl, ...
        'AbsTol', 1e-10);
    verifyFalse(testCase, isequal(result.raw_reference_control_values, ...
                                  result.raw_neural_control_values));
end

function testComparisonSimulationRejectsMissingLog(testCase)
    modelFile = modifiedComparisonModel(testCase, 'delete_neural_control');
    verifyError(testCase, @()runShortComparison(testCase, modelFile), ...
        'llc:MissingComparisonLog');
end

function testComparisonSimulationRejectsInsufficientLogCoverage(testCase)
    modelFile = modifiedComparisonModel(testCase, 'truncate_neural_states');
    verifyError(testCase, @()runShortComparison(testCase, modelFile), ...
        'llc:InvalidComparisonLog');
end

function logged = simulateNeuralHarness(runtime, inputValue, stopTime)
    [~, suffix] = fileparts(tempname);
    modelName = matlab.lang.makeValidName(['neural_sfun_' suffix]);
    new_system(modelName);
    cleanup = onCleanup(@()closeTestModel(modelName));
    assignin(get_param(modelName, 'ModelWorkspace'), 'test_runtime', runtime);
    add_block('simulink/Sources/Constant', [modelName '/Input'], ...
        'Value', mat2str(inputValue, 17), 'Position', [30 50 100 80]);
    add_block('simulink/User-Defined Functions/Level-2 MATLAB S-Function', ...
        [modelName '/NeuralODE'], 'FunctionName', 'llc_neural_ode_sfun', ...
        'Parameters', 'test_runtime', 'Position', [160 35 300 95]);
    add_block('simulink/Sinks/To Workspace', [modelName '/States'], ...
        'VariableName', 'neural_states', 'SaveFormat', 'Timeseries', ...
        'MaxDataPoints', 'inf', 'Decimation', '1', ...
        'Position', [360 45 450 85]);
    add_line(modelName, 'Input/1', 'NeuralODE/1');
    add_line(modelName, 'NeuralODE/1', 'States/1');
    set_param(modelName, 'Solver', 'ode23t', 'StopTime', num2str(stopTime, 17), ...
        'MaxStep', num2str(max(stopTime / 4, eps), 17), ...
        'RelTol', '1e-9', 'AbsTol', '1e-11');
    simulation = sim(modelName, 'ReturnWorkspaceOutputs', 'on');
    logged = simulation.get('neural_states');
    clear cleanup;
    closeTestModel(modelName);
end

function [directory, sourceHash] = comparisonModelFixture(testCase)
    directory = tempname;
    mkdir(directory);
    testCase.addTeardown(@()removeDirectoryIfPresent(directory));
    source = fullfile(testCase.TestData.matlabRoot, 'models', ...
        'llc_full_bridge_closed_loop.slx');
    destination = fullfile(directory, 'llc_full_bridge_closed_loop.slx');
    copyfile(source, destination);
    sourceHash = sha256File(destination);
end

function result = runShortComparison(testCase, modelFile)
    control = comparisonControl;
    runtime = validRuntime;
    result = llc_simulate_neural_ode_comparison( ...
        runtime.model, comparisonHardware(testCase), modelFile, ...
        zeros(4, 1), zeros(4, 1), llc_initialize_closed_loop_controller(control), ...
        [0; 5e-6; 10e-6], constantSignal(400), constantSignal(48), ...
        control, comparisonSolver);
end

function file = modifiedComparisonModel(testCase, modification)
    directory = tempname;
    mkdir(directory);
    testCase.addTeardown(@()removeDirectoryIfPresent(directory));
    file = fullfile(directory, 'llc_neural_ode_comparison.slx');
    copyfile(comparisonModelFile(testCase), file);
    bdclose('all');
    load_system(file);
    model = 'llc_neural_ode_comparison';
    if strcmp(modification, 'delete_neural_control')
        delete_block([model '/NeuralControl']);
    elseif strcmp(modification, 'truncate_neural_states')
        set_param([model '/NeuralStates'], 'SampleTime', '1');
    else
        error('llc:TestFixture', 'Unknown model modification.');
    end
    save_system(model, file);
    close_system(model, 0);
end

function file = comparisonModelFile(testCase)
    file = fullfile(testCase.TestData.matlabRoot, 'models', ...
        'llc_neural_ode_comparison.slx');
end

function hardware = comparisonHardware(testCase)
    hardware = llc_read_json(fullfile(fileparts(testCase.TestData.matlabRoot), ...
        'configs', 'matlab', 'hardware.json'));
end

function control = comparisonControl
    control = struct('control_period_s', 1e-5, ...
        'voltage_filter_time_constant_s', 5e-5, ...
        'reference_ramp_time_s', 1e-3, ...
        'initial_frequency_hz', 180e3, 'bias_frequency_hz', 110e3, ...
        'frequency_slew_rate_hz_per_s', 1e8, ...
        'proportional_gain_hz_per_v', 1e3, ...
        'integral_gain_hz_per_v_s', 5e6, ...
        'deadtime_s', 2e-7, 'minimum_pulse_s', 5e-7);
end

function solver = comparisonSolver
    solver = struct('name', 'ode23t', 'max_step_s', 1e-7, ...
        'relative_tolerance', 1e-6, 'absolute_tolerance', 1e-8, ...
        'event_time_tolerance_s', 1e-12);
end

function signal = constantSignal(value)
    signal = struct('type', 'constant', 'value_v', value);
end

function source = upstreamBlock(modelName, blockName, inputIndex)
    ports = get_param([modelName '/' blockName], 'PortHandles');
    line = get_param(ports.Inport(inputIndex), 'Line');
    sourcePort = get_param(line, 'SrcPortHandle');
    source = getfullname(get_param(sourcePort, 'Parent'));
end

function hash = sha256File(file)
    stream = java.io.FileInputStream(java.io.File(file));
    cleanup = onCleanup(@()stream.close());
    digest = java.security.MessageDigest.getInstance('SHA-256');
    buffer = zeros(1, 8192, 'int8');
    while true
        count = stream.read(buffer, 0, numel(buffer));
        if count < 0
            break;
        end
        digest.update(buffer(1:count));
    end
    bytes = typecast(digest.digest(), 'uint8');
    hash = lower(reshape(dec2hex(bytes, 2).', 1, []));
    clear cleanup;
    stream.close();
end

function verifySimulationError(testCase, operation, expectedIdentifier)
    try
        operation();
        verifyFail(testCase, 'Simulation did not reject invalid Neural ODE input.');
    catch exception
        identifiers = exceptionIdentifiers(exception);
        verifyTrue(testCase, any(identifiers == string(expectedIdentifier)), ...
            sprintf('Expected %s in exception chain; got: %s', ...
                    expectedIdentifier, strjoin(identifiers, ', ')));
    end
end

function identifiers = exceptionIdentifiers(exception)
    identifiers = string(exception.identifier);
    for index = 1:numel(exception.cause)
        identifiers = [identifiers; exceptionIdentifiers(exception.cause{index})]; %#ok<AGROW>
    end
    identifiers = unique(identifiers(strlength(identifiers) > 0));
end

function runtime = validRuntime
    runtime = struct('is_placeholder', false, 'model', constantDerivativeModel, ...
                     'initial_state', zeros(4, 1));
end

function model = constantDerivativeModel
    weights = {zeros(4, 9)};
    biases = {[1; 2; 3; 4]};
    model = struct;
    model.network = struct('input_dim', 9, 'output_dim', 4, ...
        'weights', {weights}, 'biases', {biases}, 'activations', {{'linear'}});
    model.state_scales = [1, 1, 1, 1];
    model.vin_reference_v = 400;
    model.time_reference_s = 1;
    model.load_resistance_ohm = 2.304;
end

function closeTestModel(modelName)
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end
end

function removeDirectoryIfPresent(directory)
    if isfolder(directory)
        rmdir(directory, 's');
    end
end
