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
    verifyEqual(testCase, config.schema_version, 'llc-closed-loop-data-generation-v3');
    verifyGreaterThan(testCase, control.maximum_frequency_hz, control.minimum_frequency_hz);
    verifyGreaterThan(testCase, 0.5 / control.maximum_frequency_hz, ...
                      control.deadtime_s + control.minimum_pulse_s);
    verifyEqual(testCase, numel(config.conditions), 27);
    combinations = [[config.conditions.vin_v].', ...
                    [config.conditions.load_ohm].', ...
                    [config.conditions.reference_voltage_v].'];
    verifyEqual(testCase, unique(combinations(:, 1)).', [360, 400, 440]);
    verifyEqual(testCase, unique(combinations(:, 2)).', [1.8432, 2.304, 2.7648], 'AbsTol', 1e-12);
    verifyEqual(testCase, unique(combinations(:, 3)).', [43.2, 48, 52.8], 'AbsTol', 1e-12);
    verifyEqual(testCase, size(unique(combinations, 'rows'), 1), 27);
    verifyEqual(testCase, config.initial_state(:).', [0, 0, 0, 0]);
    verifyEqual(testCase, config.sampling.intervals_per_cycle, 100);
    assessment = config.steady_state_assessment;
    verifyEqual(testCase, assessment.check_times_s.', 2e-3:0.1e-3:2.5e-3, 'AbsTol', 1e-14);
    verifyEqual(testCase, assessment.window_duration_s, 0.4e-3);
    verifyEqual(testCase, assessment.output_relative_tolerance, 0.02);
    verifyEqual(testCase, assessment.state_reference_scales(:).', [10, 400, 10, 48]);
    verifyEqual(testCase, assessment.state_mean_relative_tolerance, 0.01);
    verifyEqual(testCase, assessment.ripple_relative_tolerance, 0.05);
    verifyEqual(testCase, assessment.ripple_floor_relative_scale, 0.01);
    expectedSamples = assessment.check_times_s * config.sampling.reference_frequency_hz * ...
                      config.sampling.intervals_per_cycle + 1;
    verifyEqual(testCase, expectedSamples, [22001; 23101; 24201; 25301; 26401; 27501], ...
                'AbsTol', 1e-9);
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
