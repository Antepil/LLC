function tests = test_llc_dataset_validation
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    matlabRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(matlabRoot);
    projectRoot = setup_paths;
    testCase.TestData.config = llc_read_json( ...
        fullfile(projectRoot, 'configs', 'matlab', 'dataset_validation.json'));
    testCase.TestData.temporaryDirectory = tempname;
    mkdir(testCase.TestData.temporaryDirectory);
end

function teardownOnce(testCase)
    directory = testCase.TestData.temporaryDirectory;
    if isfolder(directory)
        rmdir(directory, 's');
    end
end

function testValidTrajectoryProducesGenericMetrics(testCase)
    file = fullfile(testCase.TestData.temporaryDirectory, 'valid.mat');
    saveFixture(file, 48, false, false);

    [entry, states, controls] = ...
        llc_validate_saved_trajectory(file, testCase.TestData.config);

    verifyTrue(testCase, entry.structural_valid);
    verifyEqual(testCase, entry.issue_count, 0);
    verifyTrue(testCase, entry.output_metric_available);
    verifyEqual(testCase, entry.output_tail_error_pct, 0, 'AbsTol', 1e-12);
    verifyEqual(testCase, height(states), 4);
    verifyEqual(testCase, height(controls), 2);
    frequency = controls(controls.control_name == "frequency_hz", :);
    verifyEqual(testCase, frequency.minimum, 90e3);
    verifyEqual(testCase, frequency.maximum, 110e3);
end

function testStructuralDamageIsReportedWithIssueCodes(testCase)
    file = fullfile(testCase.TestData.temporaryDirectory, 'damaged.mat');
    saveFixture(file, 48, true, true);

    entry = llc_validate_saved_trajectory(file, testCase.TestData.config);

    verifyFalse(testCase, entry.structural_valid);
    verifyGreaterThan(testCase, entry.gate_overlap_count, 0);
    verifySubstring(testCase, entry.issue_codes, "NONZERO_INITIAL_STATE");
    verifySubstring(testCase, entry.issue_codes, "GATE_LEG_OVERLAP");
end

function testBatchReportRetainsOutOfToleranceTrajectory(testCase)
    validFile = fullfile(testCase.TestData.temporaryDirectory, 'batch_valid.mat');
    offTargetFile = fullfile(testCase.TestData.temporaryDirectory, 'batch_off_target.mat');
    saveFixture(validFile, 48, false, false);
    saveFixture(offTargetFile, 40, false, false);
    reportDirectory = fullfile(testCase.TestData.temporaryDirectory, 'reports');
    configFile = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
                          'configs', 'matlab', 'dataset_validation.json');

    [report, summary] = llc_validate_generated_dataset( ...
        [string(validFile); string(offTargetFile)], configFile, reportDirectory);

    verifyEqual(testCase, report.trajectory_count, 2);
    verifyEqual(testCase, report.structurally_valid_count, 2);
    verifyEqual(testCase, report.tail_output_within_tolerance_count, 1);
    verifyTrue(testCase, all(summary.structural_valid));
    verifyEqual(testCase, summary.output_tail_within_tolerance, [true; false]);
    verifyTrue(testCase, isfile(report.output_files.summary_csv));
    verifyTrue(testCase, isfile(report.output_files.state_metrics_csv));
    verifyTrue(testCase, isfile(report.output_files.control_metrics_csv));
    verifyTrue(testCase, isfile(report.output_files.report_mat));
    saved = load(report.output_files.report_mat, 'report');
    verifyTrue(testCase, isfield(saved.report, 'output_files'));
end

function testBatchReportRejectsDuplicateTrajectoryIdentifiers(testCase)
    firstFile = fullfile(testCase.TestData.temporaryDirectory, 'duplicate_first.mat');
    secondFile = fullfile(testCase.TestData.temporaryDirectory, 'duplicate_second.mat');
    saveFixture(firstFile, 48, false, false);
    saveFixture(secondFile, 48, false, false);
    second = load(secondFile);
    first = load(firstFile, 'trajectory_id');
    second.trajectory_id = first.trajectory_id;
    save(secondFile, '-struct', 'second', '-v7');
    reportDirectory = fullfile(testCase.TestData.temporaryDirectory, 'duplicate_reports');
    configFile = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
                          'configs', 'matlab', 'dataset_validation.json');

    [report, summary] = llc_validate_generated_dataset( ...
        [string(firstFile); string(secondFile)], configFile, reportDirectory);

    verifyEqual(testCase, report.structurally_valid_count, 0);
    verifyTrue(testCase, all(contains(summary.issue_codes, "DUPLICATE_TRAJECTORY_ID")));
end

function saveFixture(file, tailOutput, overlap, nonzeroInitialState)
    state_times_s = (0:1e-4:1e-3).';
    interval_times_s = 0.5 * (state_times_s(1:end-1) + state_times_s(2:end));
    state_values = zeros(numel(state_times_s), 4);
    state_values(:, 1) = sin(2 * pi * 10e3 * state_times_s);
    state_values(:, 2) = 200 * sin(2 * pi * 10e3 * state_times_s);
    state_values(:, 3) = 0.5 * sin(2 * pi * 10e3 * state_times_s);
    state_values(2:end, 4) = tailOutput;
    if nonzeroInitialState
        state_values(1, 1) = 1;
    end
    interval_gate_values = repmat(uint8([1, 0, 0, 1]), numel(interval_times_s), 1);
    interval_gate_values(2:2:end, :) = repmat(uint8([0, 1, 1, 0]), ...
                                             nnz(2:2:numel(interval_times_s)), 1);
    if overlap
        interval_gate_values(1, 1:2) = 1;
    end
    interval_control_values = [linspace(110e3, 90e3, numel(interval_times_s)).', ...
                               0.5 * ones(numel(interval_times_s), 1)];

    trajectory = struct;
    trajectory.schema_version = 'llc-closed-loop-fixed-grid-trajectory-v2';
    trajectory.trajectory_id = erase(string(file), [string(fileparts(file)) + filesep, ".mat"]);
    trajectory.state_names = {'i_Lr'; 'v_Cr'; 'i_Lm'; 'v_o'};
    trajectory.state_units = {'A'; 'V'; 'A'; 'V'};
    trajectory.state_times_s = state_times_s;
    trajectory.state_values = state_values;
    trajectory.gate_names = {'Q1_A_high'; 'Q2_A_low'; 'Q3_B_high'; 'Q4_B_low'};
    trajectory.control_names = {'frequency_hz'; 'duty_ratio'};
    trajectory.control_units = {'Hz'; '1'};
    trajectory.interval_times_s = interval_times_s;
    trajectory.interval_gate_values = interval_gate_values;
    trajectory.interval_control_values = interval_control_values;
    trajectory.raw_state_times_s = state_times_s;
    trajectory.raw_state_values = state_values;
    trajectory.raw_gate_times_s = state_times_s;
    trajectory.raw_gate_values = [interval_gate_values; interval_gate_values(end, :)];
    trajectory.raw_control_times_s = state_times_s;
    trajectory.raw_control_values = [interval_control_values; interval_control_values(end, :)];
    trajectory.condition = struct('vin_v', 400, 'load_ohm', 2.304, ...
                                  'reference_voltage_v', 48);
    save(file, '-struct', 'trajectory', '-v7');
end
