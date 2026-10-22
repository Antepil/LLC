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
    verifyTrue(testCase, result.settled_at_first_checkpoint);
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
    verifyFalse(testCase, result.settled_at_first_checkpoint);
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

function testMalformedBridgeSequenceCannotSettle(testCase)
    [stateTimes, states, gateTimes, gates] = periodicFixture(48, 0);
    gates(:, 2:4) = 0;

    result = llc_assess_periodic_steady_state( ...
        stateTimes, states, gateTimes, gates, 48, testCase.TestData.config);

    verifyFalse(testCase, result.settled);
    verifyFalse(testCase, result.checkpoints(1).gate_cycles_valid);
end

function testDroppedGateCycleCannotSettle(testCase)
    [stateTimes, states, gateTimes, gates] = periodicFixture(48, 0);
    dropped = gateTimes >= 1.80e-3 & gateTimes < 1.81e-3;
    gates(dropped, :) = 0;

    result = llc_assess_periodic_steady_state( ...
        stateTimes, states, gateTimes, gates, 48, testCase.TestData.config);

    verifyFalse(testCase, result.settled_at_first_checkpoint);
    verifyFalse(testCase, result.checkpoints(1).gate_cycles_valid);
end

function testPartialBoundaryCyclesAreExcluded(testCase)
    [stateTimes, states, gateTimes, gates] = periodicFixture(48, 0, 3e-6);

    result = llc_assess_periodic_steady_state( ...
        stateTimes, states, gateTimes, gates, 48, testCase.TestData.config);

    verifyTrue(testCase, result.settled_at_first_checkpoint);
    verifyEqual(testCase, result.checkpoints(1).complete_cycle_count, 39);
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

function testCyclesOutsideFormerFrequencyBoundsAreAccepted(testCase)
    for frequency = [70e3, 250e3]
        [times, ~, ~, ~] = periodicFixture(48, 0);
        states = repmat([0, 200, 0, 48], numel(times), 1);
        [gateTimes, gates] = gateEventFixture(1 / frequency, times(end), 0);
        result = llc_assess_periodic_steady_state( ...
            times, states, gateTimes, gates, 48, testCase.TestData.config);
        verifyTrue(testCase, result.settled_at_first_checkpoint);
    end
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

function [times, states, gateTimes, gates] = periodicFixture(outputMean, currentDrift, gatePhaseShift)
    if nargin < 3
        gatePhaseShift = 0;
    end
    switchingFrequency = 100e3;
    period = 1 / switchingFrequency;
    times = (0:0.5e-6:2.5e-3).';
    phase = 2 * pi * switchingFrequency * times;
    drift = currentDrift * max(times - 1.6e-3, 0) / 0.9e-3;
    states = [5 * sin(phase) + drift, ...
              200 + 50 * sin(phase + 0.2), ...
              2 * sin(phase - 0.3), ...
              outputMean + 0.2 * sin(phase + 0.5)];
    [gateTimes, gates] = gateEventFixture(period, times(end), gatePhaseShift);
end

function [gateTimes, gates] = gateEventFixture(period, stopTime, phaseShift)
    positivePair = [1, 0, 0, 1];
    negativePair = [0, 1, 1, 0];
    offState = zeros(1, 4);
    cycles = -1:ceil(stopTime / period) + 1;
    eventTimes = zeros(4 * numel(cycles), 1);
    eventValues = zeros(4 * numel(cycles), 4);
    for index = 1:numel(cycles)
        cycle = cycles(index);
        positiveStart = phaseShift + cycle * period;
        rows = (4 * index - 3):(4 * index);
        eventTimes(rows) = [positiveStart; positiveStart + 0.45 * period; ...
                            positiveStart + 0.5 * period; positiveStart + 0.95 * period];
        eventValues(rows, :) = [positivePair; offState; negativePair; offState];
    end
    [eventTimes, order] = sort(eventTimes);
    eventValues = eventValues(order, :);
    initialIndex = find(eventTimes <= 0, 1, 'last');
    afterStart = eventTimes > 0 & eventTimes <= stopTime;
    gateTimes = [0; eventTimes(afterStart)];
    gates = [eventValues(initialIndex, :); eventValues(afterStart, :)];
    if gateTimes(end) < stopTime
        gateTimes(end + 1, 1) = stopTime;
        gates(end + 1, :) = gates(end, :);
    end
end
