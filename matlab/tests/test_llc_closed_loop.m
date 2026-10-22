function tests = test_llc_closed_loop
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    matlabRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(matlabRoot);
    testCase.TestData.projectRoot = setup_paths;
end

function testClosedLoopConfiguration(testCase)
    root = testCase.TestData.projectRoot;
    config = llc_read_json(fullfile(root, 'configs', 'matlab', 'closed_loop_data_generation.json'));
    control = config.control;
    verifyEqual(testCase, config.schema_version, 'llc-closed-loop-data-generation-v2');
    verifyGreaterThan(testCase, control.maximum_frequency_hz, control.minimum_frequency_hz);
    verifyGreaterThan(testCase, 0.5 / control.maximum_frequency_hz, ...
                      control.deadtime_s + control.minimum_pulse_s);
    verifyEqual(testCase, config.conditions.reference_voltage_v, 48);
    verifyEqual(testCase, config.conditions.stop_time_s, 2e-3);
    verifyEqual(testCase, config.sampling.intervals_per_cycle, 100);
    expectedSamples = config.conditions.stop_time_s * config.sampling.reference_frequency_hz * ...
                      config.sampling.intervals_per_cycle + 1;
    verifyEqual(testCase, expectedSamples, 22001);
end

function testFixedGridInterpolation(testCase)
    raw = struct('state_times_s', [0; 0.4; 1], ...
                 'state_values', [0 0; 4 8; 10 20], ...
                 'gate_times_s', [0; 0.5; 1], ...
                 'gate_values', [1 0; 0 1; 0 0], ...
                 'control_times_s', [0; 0.5; 1], ...
                 'control_values', [100 1; 200 2; 300 3]);
    sampling = struct('method', 'uniform_reference_cycle_v1', ...
                      'reference_frequency_hz', 2, 'intervals_per_cycle', 2, ...
                      'state_interpolation', 'linear', ...
                      'interval_signal_sampling', 'midpoint_previous');
    [fixed, audit] = llc_resample_fixed_grid(raw, sampling, 1);
    verifyEqual(testCase, fixed.state_times_s, [0; 0.25; 0.5; 0.75; 1]);
    verifyEqual(testCase, fixed.state_values(:, 1), [0; 2.5; 5; 7.5; 10], 'AbsTol', 1e-14);
    verifyEqual(testCase, double(fixed.interval_gate_values), [1 0; 1 0; 0 1; 0 1]);
    verifyEqual(testCase, fixed.interval_control_values, [100 1; 100 1; 200 2; 200 2]);
    verifyEqual(testCase, audit.state_sample_count, 5);
end

function testClosedLoopModelUsesFeedbackModulator(testCase)
    root = testCase.TestData.projectRoot;
    modelFile = fullfile(root, 'matlab', 'models', 'llc_full_bridge_closed_loop.slx');
    model = 'llc_full_bridge_closed_loop';
    load_system(modelFile);
    cleanup = onCleanup(@()close_system(model, 0));
    verifyEmpty(testCase, find_system(model, 'SearchDepth', 1, 'Name', 'ActualGateEvents'));
    controller = find_system(model, 'SearchDepth', 1, 'Name', 'ClosedLoopPFM');
    verifyNumElements(testCase, controller, 1);
    verifyEqual(testCase, get_param(controller{1}, 'FunctionName'), 'llc_closed_loop_pfm');
    verifyNumElements(testCase, find_system(model, 'SearchDepth', 1, 'Name', 'Control'), 1);
    clear cleanup;
end
