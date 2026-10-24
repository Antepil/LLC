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

function deleteIfPresent(file)
    if isfile(file)
        delete(file);
    end
end
