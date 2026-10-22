function tests = test_llc_data_generation
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    matlabRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(matlabRoot);
    testCase.TestData.projectRoot = setup_paths;
end

function testDefaultJsonFilesParse(testCase)
    root = testCase.TestData.projectRoot;
    hardware = llc_read_json(fullfile(root, 'configs', 'matlab', 'hardware.json'));
    generation = llc_read_json(fullfile(root, 'configs', 'matlab', 'data_generation.json'));
    verifyEqual(testCase, cellstr(hardware.state_names), {'i_Lr'; 'v_Cr'; 'i_Lm'; 'v_o'});
    verifyEqual(testCase, generation.solver.name, 'ode23t');
    verifyEqual(testCase, generation.conditions.switching_frequency_hz, 110000);
end

function testConstantFrequencyGatePattern(testCase)
    modulation = struct('deadtime_s', 2e-7, 'minimum_pulse_s', 5e-7, ...
                        'frequency_range_hz', [70000, 220000]);
    events = llc_constant_frequency_events(modulation, 100000, 12e-6);
    verifyEqual(testCase, events.gate_times_s(1:6), [0; 0.2e-6; 5e-6; 5.2e-6; 10e-6; 10.2e-6], 'AbsTol', 1e-20);
    verifyEqual(testCase, double(events.gate_values(1:6, :)), ...
                [0 0 0 0; 1 0 0 1; 0 0 0 0; 0 1 1 0; 0 0 0 0; 1 0 0 1]);
end

function testRejectsOutOfRangeFrequency(testCase)
    modulation = struct('deadtime_s', 2e-7, 'minimum_pulse_s', 5e-7, ...
                        'frequency_range_hz', [70000, 220000]);
    verifyError(testCase, @()llc_constant_frequency_events(modulation, 300000, 1e-3), ...
                'llc:InvalidCondition');
end
