function tests = test_llc_steady_state_assessment
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    matlabRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(matlabRoot);
    setup_paths;
    testCase.TestData.config = assessmentConfig;
end

function testStablePeriodicTrajectorySettlesAtTwoMilliseconds(testCase)
    [stateTimes, states, gateTimes, gates] = periodicFixture(48, 0);

    result = llc_assess_periodic_steady_state( ...
        stateTimes, states, gateTimes, gates, 48, testCase.TestData.config);

    verifyTrue(testCase, result.settled);
    verifyTrue(testCase, result.settled_by_2ms);
    verifyEqual(testCase, result.first_settled_time_s, 2e-3, 'AbsTol', 1e-12);
    verifyEqual(testCase, result.selected_stop_time_s, 2e-3, 'AbsTol', 1e-12);
    verifyTrue(testCase, result.regulated);
    verifyTrue(testCase, result.dynamics_stable);
    verifyGreaterThanOrEqual(testCase, result.checkpoints(1).complete_cycle_count, 30);
end

function testRegulatedDriftingTrajectoryIsNotDynamicallyStable(testCase)
    [stateTimes, states, gateTimes, gates] = periodicFixture(48, 0.5);

    result = llc_assess_periodic_steady_state( ...
        stateTimes, states, gateTimes, gates, 48, testCase.TestData.config);

    verifyTrue(testCase, result.regulated);
    verifyFalse(testCase, result.dynamics_stable);
    verifyFalse(testCase, result.settled);
    verifyFalse(testCase, result.settled_by_2ms);
    verifyEqual(testCase, result.selected_stop_time_s, 2.5e-3, 'AbsTol', 1e-12);
end

function testMissingCompleteCyclesCannotSettle(testCase)
    [stateTimes, states, gateTimes, gates] = periodicFixture(48, 0);
    gates(:) = 0;

    result = llc_assess_periodic_steady_state( ...
        stateTimes, states, gateTimes, gates, 48, testCase.TestData.config);

    verifyFalse(testCase, result.regulated);
    verifyFalse(testCase, result.dynamics_stable);
    verifyFalse(testCase, result.settled);
    verifyEqual(testCase, result.checkpoints(1).complete_cycle_count, 0);
end

function testUnregulatedTrajectoryIsRetainedThroughFinalCheckpoint(testCase)
    [stateTimes, states, gateTimes, gates] = periodicFixture(45, 0);

    result = llc_assess_periodic_steady_state( ...
        stateTimes, states, gateTimes, gates, 48, testCase.TestData.config);

    verifyFalse(testCase, result.regulated);
    verifyTrue(testCase, result.dynamics_stable);
    verifyFalse(testCase, result.settled);
    verifyTrue(testCase, isnan(result.first_settled_time_s));
    verifyEqual(testCase, result.selected_stop_time_s, 2.5e-3, 'AbsTol', 1e-12);
    verifyEqual(testCase, numel(result.checkpoints), 6);
end

function config = assessmentConfig
    config = struct;
    config.check_times_s = (2.0e-3:0.1e-3:2.5e-3).';
    config.window_duration_s = 0.4e-3;
    config.output_relative_tolerance = 0.02;
    config.state_reference_scales = [10; 400; 10; 48];
    config.state_mean_relative_tolerance = 0.01;
    config.ripple_relative_tolerance = 0.05;
    config.ripple_floor_relative_scale = 0.01;
    config.minimum_cycles_per_half_window = 2;
end

function [times, states, gateTimes, gates] = periodicFixture(outputMean, currentDrift)
    switchingFrequency = 100e3;
    period = 1 / switchingFrequency;
    times = (0:0.5e-6:2.5e-3).';
    phase = 2 * pi * switchingFrequency * times;
    drift = currentDrift * max(times - 1.6e-3, 0) / 0.9e-3;
    states = [5 * sin(phase) + drift, ...
              200 + 50 * sin(phase + 0.2), ...
              2 * sin(phase - 0.3), ...
              outputMean + 0.2 * sin(phase + 0.5)];
    gateTimes = times;
    q1 = mod(times, period) < 0.45 * period;
    q2 = mod(times, period) >= 0.5 * period & mod(times, period) < 0.95 * period;
    gates = double([q1, q2, q2, q1]);
end
