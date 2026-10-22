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
    verifyEqual(testCase, config.schema_version, 'llc-closed-loop-data-generation-v1');
    verifyGreaterThan(testCase, control.maximum_frequency_hz, control.minimum_frequency_hz);
    verifyGreaterThan(testCase, 0.5 / control.maximum_frequency_hz, ...
                      control.deadtime_s + control.minimum_pulse_s);
    verifyEqual(testCase, config.conditions.reference_voltage_v, 48);
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
