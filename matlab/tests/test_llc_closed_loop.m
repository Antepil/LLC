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
    verifyEqual(testCase, config.schema_version, 'llc-closed-loop-data-generation-v4');
    verifyFalse(testCase, isfield(control, 'minimum_frequency_hz'));
    verifyFalse(testCase, isfield(control, 'maximum_frequency_hz'));
    verifyGreaterThan(testCase, 0.5 / control.initial_frequency_hz, ...
                      control.deadtime_s + control.minimum_pulse_s);
    verifyEqual(testCase, numel(config.conditions), 144);
    combinations = [[config.conditions.vin_v].', ...
                    [config.conditions.load_ohm].', ...
                    [config.conditions.reference_voltage_v].'];
    verifyEqual(testCase, unique(combinations(:, 1)).', [360, 376, 392, 408, 424, 440]);
    verifyEqual(testCase, unique(combinations(:, 2)).', ...
                [1.8432, 2.1504, 2.4576, 2.7648], 'AbsTol', 1e-12);
    verifyEqual(testCase, unique(combinations(:, 3)).', ...
                [43.2, 45.12, 47.04, 48.96, 50.88, 52.8], 'AbsTol', 1e-12);
    verifyEqual(testCase, size(unique(combinations, 'rows'), 1), 144);
    verifyEqual(testCase, unique(string({config.conditions.dataset_role})), "train");
    verifyEqual(testCase, unique(string({config.conditions.generalization_type})), "training_domain");
    verifyEqual(testCase, config.initial_state(:).', [0, 0, 0, 0]);
    verifyEqual(testCase, config.sampling.intervals_per_cycle, 100);
    assessment = config.steady_state_assessment;
    verifyEqual(testCase, assessment.check_times_s.', 2e-3:0.1e-3:5e-3, 'AbsTol', 1e-14);
    verifyEqual(testCase, config.simulation_stop_time_s, 5e-3);
    verifyEqual(testCase, assessment.window_duration_s, 0.4e-3);
    verifyEqual(testCase, assessment.output_relative_tolerance, 0.02);
    verifyEqual(testCase, assessment.state_reference_scales(:).', [10, 400, 10, 48]);
    verifyEqual(testCase, assessment.state_mean_relative_tolerance, 0.01);
    verifyEqual(testCase, assessment.ripple_relative_tolerance, 0.05);
    verifyEqual(testCase, assessment.ripple_floor_relative_scale, 0.01);
    validation = config.validation_tolerances;
    verifyEqual(testCase, validation.timing_s, 1e-12);
    verifyEqual(testCase, validation.initial_state_abs, 1e-8);
    verifyEqual(testCase, validation.stop_time_abs_s, 1e-12);
    expectedSamples = assessment.check_times_s * config.sampling.reference_frequency_hz * ...
                      config.sampling.intervals_per_cycle + 1;
    verifyEqual(testCase, expectedSamples(end), 55001, 'AbsTol', 1e-9);
end

function testReviewedConditionGroupsAreDisjoint(testCase)
    root = testCase.TestData.projectRoot;
    names = {'closed_loop_data_generation.json', ...
             'closed_loop_interpolation_data_generation.json', ...
             'closed_loop_load_extrapolation_data_generation.json', ...
             'closed_loop_reference_extrapolation_data_generation.json', ...
             'closed_loop_high_voltage_extrapolation_data_generation.json'};
    expectedCounts = [144, 8, 6, 6, 18];
    keys = strings(0, 1);
    for index = 1:numel(names)
        config = llc_read_json(fullfile(root, 'configs', 'matlab', names{index}));
        verifyEqual(testCase, numel(config.conditions), expectedCounts(index));
        verifyEqual(testCase, config.simulation_stop_time_s, 5e-3);
        combinations = [[config.conditions.vin_v].', ...
                        [config.conditions.load_ohm].', ...
                        [config.conditions.reference_voltage_v].'];
        groupKeys = compose('%.17g|%.17g|%.17g', combinations(:, 1), ...
                            combinations(:, 2), combinations(:, 3));
        verifyEqual(testCase, numel(unique(groupKeys)), expectedCounts(index));
        verifyFalse(testCase, any(ismember(groupKeys, keys)));
        keys = [keys; groupKeys]; %#ok<AGROW>
    end
    verifyEqual(testCase, numel(keys), 182);
end

function testConditionConfigBuilderIsReproducible(testCase)
    root = testCase.TestData.projectRoot;
    destination = tempname;
    mkdir(destination);
    cleanup = onCleanup(@()rmdir(destination, 's'));
    planFile = fullfile(root, 'configs', 'matlab', 'closed_loop_condition_plan.json');
    files = build_llc_closed_loop_generation_configs(planFile, destination);
    verifyNumElements(testCase, files, 5);
    for index = 1:numel(files)
        generated = llc_read_json(files(index));
        committed = llc_read_json(fullfile(root, 'configs', 'matlab', ...
            erase(string(files(index)), string(destination) + filesep)));
        verifyEqual(testCase, generated.conditions, committed.conditions);
        verifyEqual(testCase, generated.output_directory, committed.output_directory);
        verifyEqual(testCase, generated.steady_state_assessment.check_times_s, ...
                    committed.steady_state_assessment.check_times_s);
    end
    clear cleanup;
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

function testGateEventExtraction(testCase)
    times = [0; 10e-9; 35e-9; 100e-9; 235e-9];
    values = [1 0 0 1; 1 0 0 1; 0 0 0 0; 0 0 0 0; 0 1 1 0];
    events = llc_extract_gate_events(times, values);
    verifyEqual(testCase, double(events.initial_gate_values), [1 0 0 1]);
    verifyEqual(testCase, events.gate_event_times_s, [35e-9; 235e-9]);
    verifyEqual(testCase, double(events.gate_values_after), [0 0 0 0; 0 1 1 0]);
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
