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

function testValidationConfigRejectsInactiveOrMisleadingDeclarations(testCase)
    root = testCase.TestData.projectRoot;
    base = llc_read_json(fullfile(root, 'configs', 'matlab', 'neural_ode_validation.json'));
    invalid = base;
    invalid.solver.convergence_check.max_step_factor = 1;
    verifyInvalidConfig(testCase, invalid);
    invalid = base;
    invalid.model_export.required_schema_version = 'ignored-schema';
    verifyInvalidConfig(testCase, invalid);
    invalid = base;
    invalid.comparison_sampling.state_interpolation = 'previous';
    verifyInvalidConfig(testCase, invalid);
    invalid = base;
    invalid.metrics.state_metrics{1} = 'different_metric';
    verifyInvalidConfig(testCase, invalid);
    invalid = base;
    invalid.output.directory = '';
    verifyInvalidConfig(testCase, invalid);
    invalid = base;
    invalid.output.directory = 42;
    verifyInvalidConfig(testCase, invalid);
end

function testValidationConfigRequiresConstantPreheatAndCoveredEvaluationSignals(testCase)
    root = testCase.TestData.projectRoot;
    base = llc_read_json(fullfile(root, 'configs', 'matlab', 'neural_ode_validation.json'));
    invalid = base;
    invalid.experiments{3}.preheat_vin = struct('type', 'time_series', ...
        'interpolation', 'linear', 'times_s', [0, 5e-3], 'values_v', [400, 410]);
    verifyInvalidConfig(testCase, invalid);
    invalid = base;
    invalid.experiments{2}.vin = struct('type', 'time_series', ...
        'interpolation', 'linear', 'times_s', [0, 1e-3], 'values_v', [400, 410]);
    verifyError(testCase, @()llc_load_neural_ode_validation_config( ...
        writeJsonFixture(testCase, invalid), false), 'llc:ValidationSignalCoverage');
end

function testRecordedGateReplayUsesFreeRolloutAndComputesMetrics(testCase)
    model = gateDrivenModel;
    trajectory = struct( ...
        'schema_version', 'llc-closed-loop-fixed-grid-trajectory-v3', ...
        'trajectory_id', 'synthetic', ...
        'state_names', {{'i_Lr', 'v_Cr', 'i_Lm', 'v_o'}}, ...
        'state_units', {{'A', 'V', 'A', 'V'}}, ...
        'gate_names', {{'Q1_A_high', 'Q2_A_low', 'Q3_B_high', 'Q4_B_low'}}, ...
        'state_times_s', [0; 0.5; 1], ...
        'state_values', [0 0 0 0; 0 0 0 0; 0.5 0 0 0], ...
        'initial_gate_values', uint8([0 0 0 0]), ...
        'gate_event_times_s', 0.5, ...
        'gate_values_after', uint8([1 0 0 0]), ...
        'raw_gate_times_s', [0; 0.5; 1], ...
        'raw_gate_values', uint8([0 0 0 0; 1 0 0 0; 1 0 0 0]), ...
        'raw_state_times_s', [0; 0.5; 1], ...
        'raw_state_values', [0 0 0 0; 0 0 0 0; 0.5 0 0 0], ...
        'raw_simulation_stop_time_s', 1, 'effective_config', effectiveReplayConfig, ...
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

function testRecordedGateReplayRejectsPermutedOrInconsistentContract(testCase)
    trajectory = replayTrajectoryFixture;
    trajectory.state_names = {'v_Cr', 'i_Lr', 'i_Lm', 'v_o'};
    verifyError(testCase, @()llc_run_recorded_gate_replay( ...
        gateDrivenModel, trajectory, replaySolver, replayMetrics), ...
        'llc:InvalidValidationInput');
    trajectory = replayTrajectoryFixture;
    trajectory.raw_gate_values(end, :) = uint8([0 0 0 0]);
    verifyError(testCase, @()llc_run_recorded_gate_replay( ...
        gateDrivenModel, trajectory, replaySolver, replayMetrics), ...
        'llc:InvalidValidationInput');
    trajectory = replayTrajectoryFixture;
    trajectory.raw_state_values(1, 1) = 1;
    verifyError(testCase, @()llc_run_recorded_gate_replay( ...
        gateDrivenModel, trajectory, replaySolver, replayMetrics), ...
        'llc:InvalidValidationInput');
    trajectory = replayTrajectoryFixture;
    trajectory.effective_config.generation.simulation_stop_time_s = 2;
    verifyError(testCase, @()llc_run_recorded_gate_replay( ...
        gateDrivenModel, trajectory, replaySolver, replayMetrics), ...
        'llc:InvalidValidationInput');
    trajectory = replayTrajectoryFixture;
    trajectory.effective_config.generation.simulation_stop_time_s = NaN;
    trajectory.effective_config.condition.vin_v = NaN;
    trajectory.condition.vin_v = NaN;
    verifyError(testCase, @()llc_run_recorded_gate_replay( ...
        gateDrivenModel, trajectory, replaySolver, replayMetrics), ...
        'llc:InvalidValidationInput');
end

function testRecordedGateReplayRejectsInvalidRawStopTimeType(testCase)
    trajectory = replayTrajectoryFixture;
    trajectory.raw_simulation_stop_time_s = true;
    verifyError(testCase, @()llc_run_recorded_gate_replay( ...
        gateDrivenModel, trajectory, replaySolver, replayMetrics), ...
        'llc:InvalidValidationInput');
end

function testRecordedGateReplayRejectsInvalidConfiguredStopTimeType(testCase)
    trajectory = replayTrajectoryFixture;
    trajectory.effective_config.generation.simulation_stop_time_s = true;
    verifyError(testCase, @()llc_run_recorded_gate_replay( ...
        gateDrivenModel, trajectory, replaySolver, replayMetrics), ...
        'llc:InvalidValidationInput');
end

function testRecordedGateReplayRejectsInvalidStateTimeTypes(testCase)
    trajectory = replayTrajectoryFixture;
    trajectory.state_times_s = logical([0; 1]);
    trajectory.state_values = trajectory.state_values([1, 3], :);
    verifyError(testCase, @()llc_run_recorded_gate_replay( ...
        gateDrivenModel, trajectory, replaySolver, replayMetrics), ...
        'llc:InvalidValidationInput');

    trajectory = replayTrajectoryFixture;
    trajectory.raw_state_times_s = logical([0; 1]);
    trajectory.raw_state_values = trajectory.raw_state_values([1, 3], :);
    trajectory.state_values(2, 1) = 0.25;
    verifyError(testCase, @()llc_run_recorded_gate_replay( ...
        gateDrivenModel, trajectory, replaySolver, replayMetrics), ...
        'llc:InvalidValidationInput');
end

function testRecordedGateReplayRejectsNonuniformFixedGrid(testCase)
    trajectory = replayTrajectoryFixture;
    trajectory.state_times_s(2) = 0.4;
    verifyError(testCase, @()llc_run_recorded_gate_replay( ...
        gateDrivenModel, trajectory, replaySolver, replayMetrics), ...
        'llc:InvalidValidationInput');
end

function testRecordedGateReplayRejectsInvalidGateTimeTypes(testCase)
    trajectory = replayTrajectoryFixture;
    trajectory.gate_event_times_s = true;
    trajectory.raw_gate_times_s = [0; 1];
    trajectory.raw_gate_values = uint8([0 0 0 0; 1 0 0 0]);
    verifyError(testCase, @()llc_run_recorded_gate_replay( ...
        gateDrivenModel, trajectory, replaySolver, replayMetrics), ...
        'llc:InvalidValidationInput');

    trajectory = replayTrajectoryFixture;
    trajectory.gate_event_times_s = 1;
    trajectory.raw_gate_times_s = logical([0; 1]);
    trajectory.raw_gate_values = uint8([0 0 0 0; 1 0 0 0]);
    verifyError(testCase, @()llc_run_recorded_gate_replay( ...
        gateDrivenModel, trajectory, replaySolver, replayMetrics), ...
        'llc:InvalidValidationInput');
end

function testRecordedGateReplayRejectsRawStateCoverageGap(testCase)
    trajectory = replayTrajectoryFixture;
    trajectory.raw_state_times_s(1) = eps;
    verifyError(testCase, @()llc_run_recorded_gate_replay( ...
        gateDrivenModel, trajectory, replaySolver, replayMetrics), ...
        'llc:InvalidValidationInput');
end

function testRecordedGateReplayRejectsRawGateCoverageGap(testCase)
    trajectory = replayTrajectoryFixture;
    trajectory.raw_gate_times_s(1) = eps;
    verifyError(testCase, @()llc_run_recorded_gate_replay( ...
        gateDrivenModel, trajectory, replaySolver, replayMetrics), ...
        'llc:InvalidValidationInput');
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

function testControllerRampElapsedTimeSurvivesLocalTimeShift(testCase)
    control = controllerFixture;
    handoff = 0.5 * control.reference_ramp_time_s;
    state = llc_initialize_closed_loop_controller(control);
    state.last_time_s = handoff;
    state.reference_ramp_elapsed_time_s = handoff;
    state.next_control_time_s = handoff;
    state.next_edge_time_s = handoff + 1e-6;
    state.pending_on_time_s = Inf;

    local = llc_shift_closed_loop_controller_state(state, handoff, 1e-12);
    local = llc_advance_closed_loop_controller(local, 0, 24, 48, control);

    verifyEqual(testCase, local.reference_voltage_v, 24, 'AbsTol', 1e-12);
    verifyEqual(testCase, local.reference_ramp_elapsed_time_s, handoff, 'AbsTol', 0);
end

function testControllerShiftKeepsBoundaryEventDue(testCase)
    control = controllerFixture;
    handoff = 5e-4;
    state = warmControllerState(control, handoff);
    state.next_control_time_s = handoff;

    local = llc_shift_closed_loop_controller_state(state, handoff, 1e-12);

    verifyEqual(testCase, local.last_time_s, 0, 'AbsTol', 0);
    verifyEqual(testCase, local.next_control_time_s, 0, 'AbsTol', 0);
end

function testControllerShiftKeepsFutureEventFuture(testCase)
    control = controllerFixture;
    handoff = 5e-4;
    state = warmControllerState(control, handoff);
    state.next_control_time_s = handoff + 2e-6;
    state.next_edge_time_s = handoff + 3e-6;
    state.pending_on_time_s = Inf;

    local = llc_shift_closed_loop_controller_state(state, handoff, 1e-12);

    verifyEqual(testCase, local.next_control_time_s, 2e-6, 'AbsTol', 1e-18);
    verifyEqual(testCase, local.next_edge_time_s, 3e-6, 'AbsTol', 1e-18);
    verifyEqual(testCase, local.pending_on_time_s, Inf);
end

function testControllerStateValidationRejectsIncompleteOrNonfiniteState(testCase)
    control = controllerFixture;
    state = warmControllerState(control, 5e-4);
    verifyWarningFree(testCase, @()llc_validate_closed_loop_controller_state(state, control));

    incomplete = rmfield(state, 'pending_pair');
    verifyError(testCase, ...
        @()llc_validate_closed_loop_controller_state(incomplete, control), ...
        'llc:InvalidControllerState');

    nonfinite = state;
    nonfinite.frequency_hz = NaN;
    verifyError(testCase, ...
        @()llc_validate_closed_loop_controller_state(nonfinite, control), ...
        'llc:InvalidControllerState');
end

function testPfmSFunctionRestoresWarmState(testCase)
    control = controllerFixture;
    state = warmControllerState(control, 5e-4);
    state.active_pair = 2;
    state.pending_pair = 0;
    state.desired_pair = 2;
    state.frequency_hz = 123456;
    state.filtered_voltage_v = 47;
    state.reference_voltage_v = 48;
    state.error_v = 1;
    state.integral_v_s = 2e-5;
    state = llc_shift_closed_loop_controller_state(state, 5e-4, 1e-12);
    control.initial_state = state;

    output = simulatePfmInitialOutput(control);

    verifyEqual(testCase, output, [0 1 1 0 123456 47 48 1 2e-5], ...
        'AbsTol', 1e-12);
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
    times = (0:1e-6:40e-6).';
    source = llc_simulate_neural_closed_loop(model, zeros(4, 1), ...
        llc_initialize_closed_loop_controller(control), times, vin, reference, control, solver);
    replay = llc_reconstruct_closed_loop_controller(times, source.states(:, 4), ...
        source.initial_gate_values, source.gate_event_times_s, source.gate_values_after, ...
        source.times_s, source.control_values, ...
        reference, control, 20e-6, solver.event_time_tolerance_s);
    historical = source.gate_event_times_s <= 20e-6;
    verifyEqual(testCase, replay.gate_event_times_s, ...
                source.gate_event_times_s(historical), 'AbsTol', 1e-12);
    verifyEqual(testCase, replay.gate_values_after, source.gate_values_after(historical, :));
    expectedControl = source.control_values(times == 20e-6, :);
    verifyEqual(testCase, replay.final_state.frequency_hz, expectedControl(1), 'AbsTol', 1e-9);
    verifyEqual(testCase, replay.final_state.integral_v_s, expectedControl(5), 'AbsTol', 1e-14);
end

function testControllerReplayOrdersCoincidentEdgeBeforeControlUpdate(testCase)
    model = constantDerivativeModel;
    model.network.biases = {zeros(4, 1)};
    control = controllerFixture;
    control.initial_frequency_hz = 50e3;
    control.bias_frequency_hz = 50e3;
    control.proportional_gain_hz_per_v = 1e-9;
    control.integral_gain_hz_per_v_s = 1e-9;
    vin = struct('type', 'constant', 'value_v', 400);
    reference = struct('type', 'constant', 'value_v', 48);
    times = [0; 10e-6; 10.2e-6; 20e-6; 20.2e-6; 30e-6];
    source = llc_simulate_neural_closed_loop(model, zeros(4, 1), ...
        llc_initialize_closed_loop_controller(control), times, vin, reference, control, shortSolver);
    replay = llc_reconstruct_closed_loop_controller(times, source.states(:, 4), ...
        source.initial_gate_values, source.gate_event_times_s, source.gate_values_after, ...
        source.times_s, source.control_values, reference, control, 10.2e-6, 1e-12);
    verifyEqual(testCase, replay.gate_event_times_s(end-1:end), [10e-6; 10.2e-6], ...
                'AbsTol', 1e-12);
    verifyEqual(testCase, replay.gate_values_after(end-1, :), uint8([0 0 0 0]));
    verifyEqual(testCase, replay.gate_values_after(end, :), uint8([0 1 1 0]));
    verifyEqual(testCase, replay.final_state.frequency_hz, ...
                source.control_values(3, 1), 'AbsTol', 1e-6);
end

function testControllerReplayNeverConsumesNearFutureHandoffEvent(testCase)
    model = constantDerivativeModel;
    model.network.biases = {zeros(4, 1)};
    control = controllerFixture;
    control.initial_frequency_hz = 50e3;
    control.bias_frequency_hz = 50e3;
    control.proportional_gain_hz_per_v = 1e-9;
    control.integral_gain_hz_per_v_s = 1e-9;
    vin = struct('type', 'constant', 'value_v', 400);
    reference = struct('type', 'constant', 'value_v', 48);
    handoff = 10e-6 - 5e-15;
    times = [0; handoff; 10e-6; 10.2e-6; 15e-6];
    source = llc_simulate_neural_closed_loop(model, zeros(4, 1), ...
        llc_initialize_closed_loop_controller(control), times, vin, reference, control, shortSolver);
    replay = llc_reconstruct_closed_loop_controller(times, source.states(:, 4), ...
        source.initial_gate_values, source.gate_event_times_s, source.gate_values_after, ...
        source.times_s, source.control_values, reference, control, handoff, 1e-12);
    verifyEqual(testCase, replay.final_state.last_time_s, handoff, 'AbsTol', 0);
    verifyEqual(testCase, replay.final_state.next_edge_time_s, 10e-6, 'AbsTol', 1e-14);
    verifyLessThan(testCase, replay.gate_event_times_s(end), handoff);
    continuation = llc_simulate_neural_closed_loop(model, source.states(2, :).', ...
        replay.final_state, times(2:end), vin, reference, control, shortSolver);
    verifyEqual(testCase, continuation.gate_event_times_s(1), 10e-6, 'AbsTol', 1e-14);
    verifyGreaterThan(testCase, continuation.gate_event_times_s(1), handoff);
end

function testReferenceClosedLoopRunsWithValidationSignals(testCase)
    root = testCase.TestData.projectRoot;
    hardware = llc_read_json(fullfile(root, 'configs', 'matlab', 'hardware.json'));
    modelFile = fullfile(root, 'matlab', 'models', 'llc_full_bridge_closed_loop.slx');
    vin = struct('type', 'constant', 'value_v', 400);
    reference = struct('type', 'constant', 'value_v', 48);
    result = llc_simulate_reference_closed_loop(hardware, modelFile, zeros(1, 4), ...
        (0:1e-6:40e-6).', vin, reference, controllerFixture, shortSolver);
    verifySize(testCase, result.states, [41, 4]);
    verifyEqual(testCase, result.times_s([1, end]), [0; 40e-6], 'AbsTol', 1e-12);
    verifyGreaterThan(testCase, numel(result.gate_event_times_s), 1);
    replay = llc_reconstruct_closed_loop_controller( ...
        result.raw_state_times_s, result.raw_state_values(:, 4), ...
        result.initial_gate_values, result.gate_event_times_s, result.gate_values_after, ...
        result.raw_control_times_s, result.raw_control_values, ...
        reference, controllerFixture, 20e-6, 1e-12);
    historical = result.gate_event_times_s <= 20e-6;
    verifyEqual(testCase, replay.gate_event_times_s, ...
                result.gate_event_times_s(historical), 'AbsTol', 1e-12);
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
    verifyTrue(testCase, all(isfinite([ ...
        result.tracking_metrics.reference_plant_tail_mean_error_v, ...
        result.tracking_metrics.neural_ode_tail_mean_error_v])));
end

function testWarmIndependentClosedLoopHandsOffCompleteState(testCase)
    root = testCase.TestData.projectRoot;
    hardware = llc_read_json(fullfile(root, 'configs', 'matlab', 'hardware.json'));
    modelFile = fullfile(root, 'matlab', 'models', 'llc_full_bridge_closed_loop.slx');
    model = constantDerivativeModel;
    model.network.biases = {zeros(4, 1)};
    experiment = warmExperiment(true);
    result = llc_run_independent_closed_loop(model, hardware, modelFile, experiment, ...
        controllerFixture, permissiveShortSteadyConfig, shortSolver, shortSampling, ...
        struct('tail_window_s', 2e-6, 'acceptance_thresholds', []));
    verifyEqual(testCase, result.startup_mode, 'warm');
    verifyEqual(testCase, result.times_s([1, end]), [20e-6; 35e-6], 'AbsTol', 1e-12);
    verifyEqual(testCase, result.evaluation_times_s([1, end]), [0; 15e-6], 'AbsTol', 1e-12);
    verifySize(testCase, result.reference_states, [16, 4]);
    verifySize(testCase, result.predicted_states, [16, 4]);
    verifyTrue(testCase, result.steady_state_assessment.settled);
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
        'preheat_duration_s', 20e-6, 'evaluation_duration_s', 15e-6, ...
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

function config = permissiveShortSteadyConfig
    config = shortSteadyConfig;
    config.window_duration_s = 20e-6;
    config.output_relative_tolerance = 100;
    config.state_mean_relative_tolerance = 1e6;
    config.ripple_relative_tolerance = 1e6;
    config.ripple_floor_relative_scale = 1e6;
end

function sampling = shortSampling
    sampling = struct('reference_frequency_hz', 1e6, 'intervals_per_cycle', 1);
end

function trajectory = replayTrajectoryFixture
    trajectory = struct( ...
        'schema_version', 'llc-closed-loop-fixed-grid-trajectory-v3', ...
        'trajectory_id', 'synthetic', ...
        'state_names', {{'i_Lr', 'v_Cr', 'i_Lm', 'v_o'}}, ...
        'state_units', {{'A', 'V', 'A', 'V'}}, ...
        'gate_names', {{'Q1_A_high', 'Q2_A_low', 'Q3_B_high', 'Q4_B_low'}}, ...
        'state_times_s', [0; 0.5; 1], ...
        'state_values', [0 0 0 0; 0 0 0 0; 0.5 0 0 0], ...
        'initial_gate_values', uint8([0 0 0 0]), ...
        'gate_event_times_s', 0.5, ...
        'gate_values_after', uint8([1 0 0 0]), ...
        'raw_gate_times_s', [0; 0.5; 1], ...
        'raw_gate_values', uint8([0 0 0 0; 1 0 0 0; 1 0 0 0]), ...
        'raw_state_times_s', [0; 0.5; 1], ...
        'raw_state_values', [0 0 0 0; 0 0 0 0; 0.5 0 0 0], ...
        'raw_simulation_stop_time_s', 1, 'effective_config', effectiveReplayConfig, ...
        'condition', struct('vin_v', 400, 'load_ohm', 2.304, ...
                            'reference_voltage_v', 48));
end

function config = effectiveReplayConfig
    condition = struct('vin_v', 400, 'load_ohm', 2.304, ...
                       'reference_voltage_v', 48);
    config = struct('generation', struct('simulation_stop_time_s', 1), ...
                    'condition', condition);
end

function solver = replaySolver
    solver = struct('name', 'ode23t', 'max_step_s', 0.02, ...
        'relative_tolerance', 1e-9, 'absolute_tolerance', 1e-11, ...
        'event_time_tolerance_s', 1e-12, 'fail_on_nonfinite_state', true, ...
        'convergence_check', struct('enabled', false));
end

function metrics = replayMetrics
    metrics = struct('tail_window_s', 0.4, 'acceptance_thresholds', []);
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
    config.comparison_sampling.state_interpolation = 'linear';
    config.comparison_sampling.gate_interpolation = 'previous';
    config.solver = shortSolver;
    config.solver.fail_on_nonfinite_state = true;
    config.solver.convergence_check = struct('enabled', false, ...
        'max_step_factor', 0.5, 'tolerance_factor', 0.1);
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

function verifyInvalidConfig(testCase, config)
    file = writeJsonFixture(testCase, config);
    verifyError(testCase, @()llc_load_neural_ode_validation_config(file, false), ...
                'llc:InvalidConfig');
end

function file = writeJsonFixture(testCase, value)
    file = [tempname, '.json'];
    testCase.addTeardown(@()deleteIfPresent(file));
    stream = fopen(file, 'w');
    if stream < 0
        error('llc:TestFixture', 'Could not create temporary JSON fixture.');
    end
    cleanup = onCleanup(@()fclose(stream));
    fprintf(stream, '%s', jsonencode(value, 'PrettyPrint', true));
    clear cleanup;
end

function file = writeModelFixture(testCase)
    file = [tempname, '.mat'];
    testCase.addTeardown(@()deleteIfPresent(file));
    model = gateDrivenModel;
    schema_version = 'llc-neural-ode-matlab-export-v1';
    network = model.network;
    state_names = {'i_Lr', 'v_Cr', 'i_Lm', 'v_o'};
    gate_names = {'Q1_A_high', 'Q2_A_low', 'Q3_B_high', 'Q4_B_low'};
    network_input_order = {'z_i_Lr', 'z_v_Cr', 'z_i_Lm', 'z_v_o', ...
        'Q1_A_high', 'Q2_A_low', 'Q3_B_high', 'Q4_B_low', 'Vin_over_V_ref'};
    network_output_order = {'dz_i_Lr_dtau', 'dz_v_Cr_dtau', ...
        'dz_i_Lm_dtau', 'dz_v_o_dtau'};
    state_scales = model.state_scales;
    vin_reference_v = model.vin_reference_v;
    time_reference_s = model.time_reference_s;
    load_resistance_ohm = model.load_resistance_ohm;
    dtype = 'float64';
    consistency_inputs = [zeros(1, 8), 1; zeros(1, 4), 1, 0, 0, 0, 1];
    consistency_outputs = consistency_inputs * network.weights{1}.' + network.biases{1}.';
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

function state = warmControllerState(control, handoff)
    state = llc_initialize_closed_loop_controller(control);
    state.reference_ramp_elapsed_time_s = handoff;
    state.last_time_s = handoff;
    state.next_control_time_s = handoff + control.control_period_s;
    state.next_edge_time_s = handoff + 0.25 / control.initial_frequency_hz;
    state.pending_on_time_s = Inf;
end

function output = simulatePfmInitialOutput(control)
    [~, suffix] = fileparts(tempname);
    modelName = matlab.lang.makeValidName(['pfm_warm_' suffix]);
    new_system(modelName);
    cleanup = onCleanup(@()closeTestModel(modelName));
    assignin(get_param(modelName, 'ModelWorkspace'), 'test_control', control);
    add_block('simulink/Sources/Constant', [modelName '/MeasuredVoltage'], ...
        'Value', '47', 'Position', [30 40 80 70]);
    add_block('simulink/Sources/Constant', [modelName '/ReferenceVoltage'], ...
        'Value', '48', 'Position', [30 100 80 130]);
    add_block('simulink/User-Defined Functions/Level-2 MATLAB S-Function', ...
        [modelName '/PFM'], 'FunctionName', 'llc_closed_loop_pfm', ...
        'Parameters', 'test_control', 'Position', [140 45 280 125]);
    add_block('simulink/Sinks/To Workspace', [modelName '/Log'], ...
        'VariableName', 'pfm_output', 'SaveFormat', 'Timeseries', ...
        'Position', [340 65 430 105]);
    add_line(modelName, 'MeasuredVoltage/1', 'PFM/1');
    add_line(modelName, 'ReferenceVoltage/1', 'PFM/2');
    add_line(modelName, 'PFM/1', 'Log/1');
    set_param(modelName, 'Solver', 'VariableStepDiscrete', 'StopTime', '0');
    simulation = sim(modelName, 'ReturnWorkspaceOutputs', 'on');
    logged = simulation.get('pfm_output');
    output = double(logged.Data(1, :));
    clear cleanup;
    closeTestModel(modelName);
end

function closeTestModel(modelName)
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end
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
