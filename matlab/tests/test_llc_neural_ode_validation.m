function tests = test_llc_neural_ode_validation
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    matlabRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(matlabRoot);
    testCase.TestData.projectRoot = setup_paths;
end

function testExportedModelLoadsAndMatchesReferenceForward(testCase)
    file = writeModelFixture(testCase);
    model = llc_load_neural_ode_model(file, 2.304, 1e-9);
    actual = llc_neural_ode_forward(model, model.consistency_inputs);
    verifyEqual(testCase, actual, model.consistency_outputs, ...
                'AbsTol', 1e-14, 'RelTol', 1e-14);
end

function testModelLoadRejectsDifferentFixedLoad(testCase)
    file = writeModelFixture(testCase);
    verifyError(testCase, @()llc_load_neural_ode_model(file, 3.0, 1e-9), ...
                'llc:ModelContractMismatch');
end

function testModelLoadRejectsFailedConsistencyFixture(testCase)
    file = writeModelFixture(testCase);
    raw = load(file);
    raw.consistency_outputs(1, 1) = raw.consistency_outputs(1, 1) + 1e-3;
    save(file, '-struct', 'raw');
    verifyError(testCase, @()llc_load_neural_ode_model(file, 2.304, 1e-9, 1e-12, 1e-12), ...
                'llc:ModelConsistencyMismatch');
end

function testPhysicalDerivativeAppliesStateAndTimeScales(testCase)
    model = constantDerivativeModel;
    model.state_scales = [2, 3, 4, 5];
    model.time_reference_s = 1e-5;
    signal = struct('type', 'constant', 'value_v', 400);
    derivative = llc_neural_ode_physical_derivative(0, [2; 3; 4; 5], ...
        model, [1, 0, 0, 1], signal);
    verifyEqual(testCase, derivative, [2; 6; 12; 20] * 1e5, 'AbsTol', 1e-8);
end

function testVinUsesLinearInterpolationWithoutExtrapolation(testCase)
    signal = struct('type', 'time_series', 'interpolation', 'linear', ...
                    'times_s', [0, 1, 2], 'values_v', [380, 400, 440]);
    verifyEqual(testCase, llc_evaluate_validation_signal(signal, [0, 0.5, 2]), ...
                [380, 390, 440], 'AbsTol', 1e-14);
    verifyError(testCase, @()llc_evaluate_validation_signal(signal, 2.1), ...
                'llc:ValidationSignalCoverage');
end

function testReferenceUsesPreviousValueAtSteps(testCase)
    signal = struct('type', 'time_series', 'interpolation', 'previous', ...
                    'times_s', [0, 1, 2], 'values_v', [48, 50, 46]);
    verifyEqual(testCase, llc_evaluate_validation_signal(signal, [0.5, 1, 1.9]), ...
                [48, 50, 50]);
end

function testWarmSignalUsesLocalEvaluationTimeAfterHandoff(testCase)
    before = struct('type', 'constant', 'value_v', 400);
    after = struct('type', 'time_series', 'interpolation', 'linear', ...
                   'times_s', [0, 1], 'values_v', [420, 440]);
    signal = struct('type', 'staged', 'switch_time_s', 5, ...
                    'before', before, 'after', after);
    verifyEqual(testCase, llc_evaluate_validation_signal(signal, [4.9, 5, 5.5, 6]), ...
                [400, 420, 430, 440], 'AbsTol', 1e-14);
end

function testEventIntegratorChangesDerivativeAtExactGateEvent(testCase)
    model = gateDrivenModel;
    solver = struct('name', 'ode23t', 'max_step_s', 0.02, ...
                    'relative_tolerance', 1e-9, 'absolute_tolerance', 1e-11, ...
                    'event_time_tolerance_s', 1e-12);
    signal = struct('type', 'constant', 'value_v', 400);
    states = llc_integrate_neural_ode_events(model, [0; 0.5; 1], ...
        zeros(4, 1), [0, 0, 0, 0], 0.5, [1, 0, 0, 0], signal, solver);
    verifyEqual(testCase, states(:, 1), [0; 0; 0.5], 'AbsTol', 1e-7);
end

function testValidationConfigResolvesColdAndWarmExperiments(testCase)
    root = testCase.TestData.projectRoot;
    file = fullfile(root, 'configs', 'matlab', 'neural_ode_validation.json');
    config = llc_load_neural_ode_validation_config(file, false);
    verifyEqual(testCase, config.schema_version, 'llc-neural-ode-validation-v2');
    verifyEqual(testCase, string({config.experiments.mode}), ...
                ["recorded_gate_replay", "independent_closed_loop", "independent_closed_loop"]);
    verifyEqual(testCase, config.experiments(2).startup.mode, 'cold');
    verifyEqual(testCase, config.experiments(3).startup.mode, 'warm');
    verifyEqual(testCase, config.experiments(2).reference.value_v, 48);
    verifyEqual(testCase, config.experiments(3).startup.preheat_duration_s, 5e-3);
end

function testRecordedGateReplayUsesFreeRolloutAndComputesMetrics(testCase)
    model = gateDrivenModel;
    trajectory = struct( ...
        'schema_version', 'llc-closed-loop-fixed-grid-trajectory-v3', ...
        'trajectory_id', 'synthetic', ...
        'state_names', {{'i_Lr', 'v_Cr', 'i_Lm', 'v_o'}}, ...
        'gate_names', {{'Q1_A_high', 'Q2_A_low', 'Q3_B_high', 'Q4_B_low'}}, ...
        'state_times_s', [0; 0.5; 1], ...
        'state_values', [0 0 0 0; 0 0 0 0; 0.5 0 0 0], ...
        'initial_gate_values', uint8([0 0 0 0]), ...
        'gate_event_times_s', 0.5, ...
        'gate_values_after', uint8([1 0 0 0]), ...
        'condition', struct('vin_v', 400, 'load_ohm', 2.304, 'reference_voltage_v', 48));
    solver = struct('name', 'ode23t', 'max_step_s', 0.02, ...
                    'relative_tolerance', 1e-9, 'absolute_tolerance', 1e-11, ...
                    'event_time_tolerance_s', 1e-12, ...
                    'fail_on_nonfinite_state', true, ...
                    'convergence_check', struct('enabled', false));
    metricConfig = struct('tail_window_s', 0.4, 'acceptance_thresholds', []);
    result = llc_run_recorded_gate_replay(model, trajectory, solver, metricConfig);
    verifyEqual(testCase, result.predicted_states, trajectory.state_values, 'AbsTol', 1e-7);
    verifyEqual(testCase, result.state_metrics.rmse, zeros(1, 4), 'AbsTol', 1e-7);
    verifyEqual(testCase, result.state_metrics.normalized_rmse, zeros(1, 4), 'AbsTol', 1e-7);
end

function testControllerStartsOffThenAppliesDeadtimeAndAlternatingPairs(testCase)
    control = controllerFixture;
    state = llc_initialize_closed_loop_controller(control);
    verifyEqual(testCase, llc_closed_loop_controller_gates(state), [0 0 0 0]);
    state = llc_advance_closed_loop_controller(state, 0, 0, 48, control);
    state = llc_advance_closed_loop_controller(state, control.deadtime_s, 0, 48, control);
    verifyEqual(testCase, llc_closed_loop_controller_gates(state), [1 0 0 1]);
    edge = state.next_edge_time_s;
    state = llc_advance_closed_loop_controller(state, edge, 0, 48, control);
    verifyEqual(testCase, llc_closed_loop_controller_gates(state), [0 0 0 0]);
    state = llc_advance_closed_loop_controller(state, edge + control.deadtime_s, 0, 48, control);
    verifyEqual(testCase, llc_closed_loop_controller_gates(state), [0 1 1 0]);
end

function testNeuralClosedLoopReturnsCompleteControllerState(testCase)
    model = constantDerivativeModel;
    model.network.biases = {zeros(4, 1)};
    control = controllerFixture;
    solver = struct('name', 'ode23t', 'max_step_s', 1e-7, ...
                    'relative_tolerance', 1e-7, 'absolute_tolerance', 1e-9, ...
                    'event_time_tolerance_s', 1e-12);
    vin = struct('type', 'constant', 'value_v', 400);
    reference = struct('type', 'constant', 'value_v', 48);
    result = llc_simulate_neural_closed_loop(model, zeros(4, 1), ...
        llc_initialize_closed_loop_controller(control), [0; 1e-6; 5e-6; 10e-6], ...
        vin, reference, control, solver);
    verifyEqual(testCase, result.states, zeros(4, 4), 'AbsTol', 1e-12);
    verifyTrue(testCase, all(isfield(result.final_controller_state, ...
        {'integral_v_s', 'filtered_voltage_v', 'frequency_hz', 'phase_rad', ...
         'next_control_time_s', 'next_edge_time_s', 'pending_on_time_s'})));
    verifyGreaterThan(testCase, numel(result.gate_event_times_s), 1);
end

function testControllerStateCanBeReconstructedFromReferenceWaveform(testCase)
    model = constantDerivativeModel;
    model.network.biases = {zeros(4, 1)};
    control = controllerFixture;
    solver = shortSolver;
    vin = struct('type', 'constant', 'value_v', 400);
    reference = struct('type', 'constant', 'value_v', 48);
    times = (0:1e-6:20e-6).';
    source = llc_simulate_neural_closed_loop(model, zeros(4, 1), ...
        llc_initialize_closed_loop_controller(control), times, vin, reference, control, solver);
    replay = llc_reconstruct_closed_loop_controller(times, source.states(:, 4), ...
        source.initial_gate_values, source.gate_event_times_s, source.gate_values_after, ...
        reference, control, times(end), solver.event_time_tolerance_s);
    verifyEqual(testCase, replay.gate_event_times_s, source.gate_event_times_s, 'AbsTol', 1e-12);
    verifyEqual(testCase, replay.gate_values_after, source.gate_values_after);
    verifyEqual(testCase, replay.final_state.frequency_hz, ...
                source.final_controller_state.frequency_hz, 'AbsTol', 1e-9);
    verifyEqual(testCase, replay.final_state.integral_v_s, ...
                source.final_controller_state.integral_v_s, 'AbsTol', 1e-14);
end

function testReferenceClosedLoopRunsWithValidationSignals(testCase)
    root = testCase.TestData.projectRoot;
    hardware = llc_read_json(fullfile(root, 'configs', 'matlab', 'hardware.json'));
    modelFile = fullfile(root, 'matlab', 'models', 'llc_full_bridge_closed_loop.slx');
    vin = struct('type', 'constant', 'value_v', 400);
    reference = struct('type', 'constant', 'value_v', 48);
    result = llc_simulate_reference_closed_loop(hardware, modelFile, zeros(1, 4), ...
        (0:1e-6:20e-6).', vin, reference, controllerFixture, shortSolver);
    verifySize(testCase, result.states, [21, 4]);
    verifyEqual(testCase, result.times_s([1, end]), [0; 20e-6], 'AbsTol', 1e-12);
    verifyGreaterThan(testCase, numel(result.gate_event_times_s), 1);
    replay = llc_reconstruct_closed_loop_controller( ...
        result.raw_state_times_s, result.raw_state_values(:, 4), ...
        result.initial_gate_values, result.gate_event_times_s, result.gate_values_after, ...
        reference, controllerFixture, 20e-6, 1e-12);
    verifyEqual(testCase, replay.gate_event_times_s, result.gate_event_times_s, 'AbsTol', 1e-12);
end

function testColdIndependentClosedLoopComparesSeparateBranches(testCase)
    root = testCase.TestData.projectRoot;
    hardware = llc_read_json(fullfile(root, 'configs', 'matlab', 'hardware.json'));
    modelFile = fullfile(root, 'matlab', 'models', 'llc_full_bridge_closed_loop.slx');
    model = constantDerivativeModel;
    model.network.biases = {zeros(4, 1)};
    experiment = struct('startup', struct('mode', 'cold', ...
        'evaluation_duration_s', 20e-6), 'condition', struct('load_ohm', 2.304), ...
        'vin', struct('type', 'constant', 'value_v', 400), ...
        'reference', struct('type', 'constant', 'value_v', 48));
    sampling = struct('reference_frequency_hz', 1e6, 'intervals_per_cycle', 1);
    metrics = struct('tail_window_s', 5e-6, 'acceptance_thresholds', []);
    result = llc_run_independent_closed_loop(model, hardware, modelFile, experiment, ...
        controllerFixture, struct, shortSolver, sampling, metrics);
    verifyEqual(testCase, result.startup_mode, 'cold');
    verifySize(testCase, result.reference_states, [21, 4]);
    verifySize(testCase, result.predicted_states, [21, 4]);
    verifyTrue(testCase, all(isfinite(result.state_metrics.rmse)));
end

function testWarmIndependentClosedLoopHandsOffCompleteState(testCase)
    root = testCase.TestData.projectRoot;
    hardware = llc_read_json(fullfile(root, 'configs', 'matlab', 'hardware.json'));
    modelFile = fullfile(root, 'matlab', 'models', 'llc_full_bridge_closed_loop.slx');
    model = constantDerivativeModel;
    model.network.biases = {zeros(4, 1)};
    experiment = warmExperiment(false);
    result = llc_run_independent_closed_loop(model, hardware, modelFile, experiment, ...
        controllerFixture, shortSteadyConfig, shortSolver, shortSampling, ...
        struct('tail_window_s', 2e-6, 'acceptance_thresholds', []));
    verifyEqual(testCase, result.startup_mode, 'warm');
    verifyEqual(testCase, result.times_s([1, end]), [20e-6; 25e-6], 'AbsTol', 1e-12);
    verifyEqual(testCase, result.evaluation_times_s([1, end]), [0; 5e-6], 'AbsTol', 1e-12);
    verifySize(testCase, result.reference_states, [6, 4]);
    verifySize(testCase, result.predicted_states, [6, 4]);
    verifyFalse(testCase, result.steady_state_assessment.settled);
end

function testWarmIndependentClosedLoopRejectsUnsettledPreheat(testCase)
    root = testCase.TestData.projectRoot;
    hardware = llc_read_json(fullfile(root, 'configs', 'matlab', 'hardware.json'));
    modelFile = fullfile(root, 'matlab', 'models', 'llc_full_bridge_closed_loop.slx');
    model = constantDerivativeModel;
    model.network.biases = {zeros(4, 1)};
    verifyError(testCase, @()llc_run_independent_closed_loop(model, hardware, modelFile, ...
        warmExperiment(true), controllerFixture, shortSteadyConfig, shortSolver, ...
        shortSampling, struct('tail_window_s', 2e-6, 'acceptance_thresholds', [])), ...
        'llc:PreheatNotSettled');
end

function testValidationResultsAreSavedAsMatCsvAndFigures(testCase)
    outputDirectory = tempname;
    testCase.addTeardown(@()removeDirectoryIfPresent(outputDirectory));
    times = [0; 1e-6; 2e-6];
    referenceStates = [zeros(3, 3), [0; 1; 2]];
    predictedStates = referenceStates + 0.1;
    metricConfig = struct('tail_window_s', 1e-6);
    item = struct('experiment_id', 'save_test', 'mode', 'recorded_gate_replay', ...
        'times_s', times, 'reference_states', referenceStates, ...
        'predicted_states', predictedStates, ...
        'state_metrics', llc_compute_neural_ode_metrics( ...
            times, predictedStates, referenceStates, ones(1, 4), metricConfig));
    report = struct('schema_version', 'llc-neural-ode-validation-result-v1', ...
                    'results', item);
    config = struct('output', struct('directory', outputDirectory, ...
        'save_resolved_config', true, 'save_waveforms', true, ...
        'save_metric_tables', true, 'save_figures', true));
    files = llc_save_neural_ode_validation_results(report, config);
    verifyTrue(testCase, isfile(files.results_mat));
    verifyTrue(testCase, isfile(files.resolved_config_mat));
    verifyTrue(testCase, isfile(files.state_metrics_csv));
    verifyTrue(testCase, isfile(files.figure_files(1)));
end

function testConfiguredValidationEntryPointRunsEnabledExperiment(testCase)
    modelFile = writeModelFixture(testCase);
    outputDirectory = tempname;
    testCase.addTeardown(@()removeDirectoryIfPresent(outputDirectory));
    configFile = writeValidationConfigFixture(testCase, modelFile, outputDirectory);
    [report, files] = llc_run_neural_ode_validation(configFile);
    verifyEqual(testCase, report.schema_version, 'llc-neural-ode-validation-result-v1');
    verifyEqual(testCase, numel(report.results), 1);
    verifyEqual(testCase, report.results.experiment_id, 'entry_point_cold');
    verifyEqual(testCase, report.results.startup_mode, 'cold');
    verifyEqual(testCase, files.results_mat, "");
end

function experiment = warmExperiment(requireSteady)
    experiment = struct('startup', struct('mode', 'warm', ...
        'preheat_duration_s', 20e-6, 'evaluation_duration_s', 5e-6, ...
        'require_steady_state', requireSteady), ...
        'condition', struct('load_ohm', 2.304), ...
        'preheat_vin', struct('type', 'constant', 'value_v', 400), ...
        'preheat_reference', struct('type', 'constant', 'value_v', 48), ...
        'vin', struct('type', 'constant', 'value_v', 410), ...
        'reference', struct('type', 'constant', 'value_v', 50));
end

function config = shortSteadyConfig
    config = struct('check_times_s', 20e-6, 'window_duration_s', 10e-6, ...
        'output_relative_tolerance', 0.02, 'state_reference_scales', [10, 400, 10, 48], ...
        'state_mean_relative_tolerance', 0.01, 'ripple_relative_tolerance', 0.05, ...
        'ripple_floor_relative_scale', 0.01, 'minimum_cycles_per_half_window', 1);
end

function sampling = shortSampling
    sampling = struct('reference_frequency_hz', 1e6, 'intervals_per_cycle', 1);
end

function file = writeValidationConfigFixture(testCase, modelFile, outputDirectory)
    root = testCase.TestData.projectRoot;
    config = llc_read_json(fullfile(root, 'configs', 'matlab', ...
                                    'neural_ode_validation.json'));
    config.model_export.file = modelFile;
    config.output.directory = outputDirectory;
    config.output.save_resolved_config = false;
    config.output.save_waveforms = false;
    config.output.save_metric_tables = false;
    config.output.save_figures = false;
    config.experiments = struct('experiment_id', 'entry_point_cold', ...
        'enabled', true, 'mode', 'independent_closed_loop', ...
        'startup', struct('mode', 'cold', 'evaluation_duration_s', 20e-6), ...
        'condition', struct('load_ohm', 2.304), ...
        'vin', struct('type', 'constant', 'value_v', 400), ...
        'reference', struct('type', 'constant', 'value_v', 48));
    config.comparison_sampling = shortSampling;
    config.solver = shortSolver;
    config.solver.fail_on_nonfinite_state = true;
    config.solver.convergence_check = struct('enabled', false);
    file = [tempname, '.json'];
    testCase.addTeardown(@()deleteIfPresent(file));
    stream = fopen(file, 'w');
    if stream < 0
        error('llc:TestFixture', 'Could not create temporary validation config.');
    end
    cleanup = onCleanup(@()fclose(stream));
    fprintf(stream, '%s', jsonencode(config, 'PrettyPrint', true));
    clear cleanup;
end

function file = writeModelFixture(testCase)
    file = [tempname, '.mat'];
    testCase.addTeardown(@()deleteIfPresent(file));
    model = gateDrivenModel;
    schema_version = 'llc-neural-ode-matlab-export-v1'; %#ok<NASGU>
    network = model.network; %#ok<NASGU>
    state_names = {'i_Lr', 'v_Cr', 'i_Lm', 'v_o'}; %#ok<NASGU>
    gate_names = {'Q1_A_high', 'Q2_A_low', 'Q3_B_high', 'Q4_B_low'}; %#ok<NASGU>
    network_input_order = {'z_i_Lr', 'z_v_Cr', 'z_i_Lm', 'z_v_o', ...
        'Q1_A_high', 'Q2_A_low', 'Q3_B_high', 'Q4_B_low', 'Vin_over_V_ref'}; %#ok<NASGU>
    network_output_order = {'dz_i_Lr_dtau', 'dz_v_Cr_dtau', ...
        'dz_i_Lm_dtau', 'dz_v_o_dtau'}; %#ok<NASGU>
    state_scales = model.state_scales; %#ok<NASGU>
    vin_reference_v = model.vin_reference_v; %#ok<NASGU>
    time_reference_s = model.time_reference_s; %#ok<NASGU>
    load_resistance_ohm = model.load_resistance_ohm; %#ok<NASGU>
    dtype = 'float64'; %#ok<NASGU>
    consistency_inputs = [zeros(1, 8), 1; zeros(1, 4), 1, 0, 0, 0, 1]; %#ok<NASGU>
    consistency_outputs = consistency_inputs * network.weights{1}.' + network.biases{1}.'; %#ok<NASGU>
    save(file, 'schema_version', 'network', 'state_names', 'gate_names', ...
         'network_input_order', 'network_output_order', 'state_scales', ...
         'vin_reference_v', 'time_reference_s', 'load_resistance_ohm', ...
         'dtype', 'consistency_inputs', 'consistency_outputs');
end

function model = constantDerivativeModel
    weights = {zeros(4, 9)};
    biases = {[1; 2; 3; 4]};
    model = baseModel(weights, biases);
end

function model = gateDrivenModel
    weights = {zeros(4, 9)};
    weights{1}(1, 5) = 1;
    biases = {zeros(4, 1)};
    model = baseModel(weights, biases);
end

function model = baseModel(weights, biases)
    model = struct;
    model.network = struct('input_dim', 9, 'output_dim', 4, ...
        'weights', {weights}, 'biases', {biases}, 'activations', {{'linear'}});
    model.state_scales = [1, 1, 1, 1];
    model.vin_reference_v = 400;
    model.time_reference_s = 1;
    model.load_resistance_ohm = 2.304;
end

function control = controllerFixture
    control = struct('control_period_s', 1e-5, ...
        'voltage_filter_time_constant_s', 5e-5, ...
        'reference_ramp_time_s', 1e-3, ...
        'initial_frequency_hz', 180e3, 'bias_frequency_hz', 110e3, ...
        'frequency_slew_rate_hz_per_s', 1e8, ...
        'proportional_gain_hz_per_v', 1e3, ...
        'integral_gain_hz_per_v_s', 5e6, ...
        'deadtime_s', 2e-7, 'minimum_pulse_s', 5e-7);
end

function solver = shortSolver
    solver = struct('name', 'ode23t', 'max_step_s', 1e-7, ...
        'relative_tolerance', 1e-6, 'absolute_tolerance', 1e-8, ...
        'event_time_tolerance_s', 1e-12);
end

function deleteIfPresent(file)
    if isfile(file)
        delete(file);
    end
end


function removeDirectoryIfPresent(directory)
    if isfolder(directory)
        rmdir(directory, 's');
    end
end
