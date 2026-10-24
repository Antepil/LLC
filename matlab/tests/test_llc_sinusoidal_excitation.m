function tests = test_llc_sinusoidal_excitation
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    matlabRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(matlabRoot);
    testCase.TestData.projectRoot = setup_paths;
end

function testScreeningConfigurationIsIndependent(testCase)
    root = testCase.TestData.projectRoot;
    configFile = fullfile(root, 'configs', 'matlab', ...
        'sinusoidal_excitation', 'screening_data_generation.json');
    config = llc_read_json(configFile);
    verifyEqual(testCase, config.schema_version, ...
                'llc-sinusoidal-excitation-data-generation-v1');
    verifyEqual(testCase, config.fixed_load_ohm, 2.304, 'AbsTol', 1e-12);
    verifyEqual(testCase, config.simulation_stop_time_s, 0.057, 'AbsTol', 1e-15);
    verifyEqual(testCase, config.excitation_timing.start_time_s, 0.005);
    verifyEqual(testCase, config.excitation_timing.envelope_rise_time_s, 0.002);
    verifyEqual(testCase, config.control.minimum_frequency_hz, 60000);
    verifyEqual(testCase, config.control.maximum_frequency_hz, 250000);
    verifyNumElements(testCase, config.conditions, 6);
    groups = string({config.conditions.excitation_group});
    verifyEqual(testCase, nnz(groups == "vin_only"), 3);
    verifyEqual(testCase, nnz(groups == "reference_only"), 3);
    vinSignals = [config.conditions(1:3).vin];
    referenceSignals = [config.conditions(4:6).reference];
    verifyEqual(testCase, [vinSignals.frequency_hz], [100, 300, 1000]);
    verifyEqual(testCase, [referenceSignals.frequency_hz], [100, 300, 1000]);
    constantReferences = [config.conditions(1:3).reference];
    constantInputs = [config.conditions(4:6).vin];
    verifyTrue(testCase, all([constantReferences.amplitude] == 0));
    verifyTrue(testCase, all([constantInputs.amplitude] == 0));
    verifyFalse(testCase, contains(config.output_directory, 'llc_closed_loop/train'));
end

function testSinusoidalProfileUsesDelayedHalfCosineEnvelope(testCase)
    excitation = struct( ...
        'start_time_s', 5e-3, 'envelope_rise_time_s', 2e-3, ...
        'vin', signal(400, 10, 100, 0), ...
        'reference', signal(48, 1.2, 300, pi / 3));
    times = [0; 5e-3; 6e-3; 7e-3];
    values = llc_evaluate_sinusoidal_excitation(times, excitation);
    verifyEqual(testCase, values(1:2, :), [400, 48; 400, 48], 'AbsTol', 1e-13);
    vinExpected = 400 + 0.5 * 10 * sin(2 * pi * 100 * 1e-3);
    refExpected = 48 + 0.5 * 1.2 * sin(2 * pi * 300 * 1e-3 + pi / 3);
    verifyEqual(testCase, values(3, :), [vinExpected, refExpected], 'AbsTol', 1e-12);
    verifyEqual(testCase, values(4, 1), ...
                400 + 10 * sin(2 * pi * 100 * 2e-3), 'AbsTol', 1e-12);
end

function testSharedModelHasExcitationInterface(testCase)
    root = testCase.TestData.projectRoot;
    modelFile = fullfile(root, 'matlab', 'models', 'llc_full_bridge_closed_loop.slx');
    model = 'llc_full_bridge_closed_loop';
    load_system(modelFile);
    cleanup = onCleanup(@()close_system(model, 0));
    excitation = find_system(model, 'SearchDepth', 1, 'Name', 'ClosedLoopExcitation');
    verifyNumElements(testCase, excitation, 1);
    verifyEqual(testCase, get_param(excitation{1}, 'FunctionName'), ...
                'llc_closed_loop_excitation');
    verifyNumElements(testCase, find_system(model, 'SearchDepth', 1, 'Name', 'Excitation'), 1);
    verifyNumElements(testCase, find_system(model, 'SearchDepth', 1, 'Name', 'VinCommand'), 1);
    verifyEqual(testCase, get_param([model '/VinCommand'], ...
                'FilteringAndDerivatives'), 'zero');
    controllerPorts = get_param([model '/ClosedLoopPFM'], 'PortHandles');
    verifyNumElements(testCase, controllerPorts.Inport, 2);
    clear cleanup;
end

function testSinusoidalGeneratorSmoke(testCase)
    root = testCase.TestData.projectRoot;
    sourceConfig = llc_read_json(fullfile(root, 'configs', 'matlab', ...
        'sinusoidal_excitation', 'screening_data_generation.json'));
    outputDirectory = tempname;
    mkdir(outputDirectory);
    cleanupDirectory = onCleanup(@()rmdir(outputDirectory, 's'));
    configFile = [tempname, '.json'];
    cleanupConfig = onCleanup(@()deleteIfPresent(configFile));
    sourceConfig.output_directory = outputDirectory;
    sourceConfig.simulation_stop_time_s = 0.003;
    sourceConfig.excitation_timing.start_time_s = 0.001;
    sourceConfig.excitation_timing.envelope_rise_time_s = 0.0005;
    sourceConfig.conditions = sourceConfig.conditions(3);
    writeJson(configFile, sourceConfig);

    [files, summary] = generate_llc_sinusoidal_excitation_dataset([], configFile, false);
    assertEqual(testCase, summary.successful_count, 1, ...
                summary.entries(1).error_message);
    assertEqual(testCase, summary.failed_count, 0);
    assertTrue(testCase, isfile(files(1)));
    trajectory = load(files(1), 'schema_version', 'fixed_load_ohm', ...
                      'state_excitation_values', 'interval_gate_values', ...
                      'interval_control_values', 'network_input_contract');
    verifyEqual(testCase, trajectory.schema_version, ...
                'llc-sinusoidal-excitation-fixed-grid-trajectory-v1');
    verifyEqual(testCase, trajectory.fixed_load_ohm, 2.304, 'AbsTol', 1e-12);
    vinValues = trajectory.state_excitation_values(:, 1);
    verifyGreaterThan(testCase, max(vinValues) - min(vinValues), 1);
    verifyEqual(testCase, size(trajectory.interval_gate_values, 2), 4);
    verifyGreaterThanOrEqual(testCase, min(trajectory.interval_control_values(:, 1)), 60000);
    verifyLessThanOrEqual(testCase, max(trajectory.interval_control_values(:, 1)), 250000);
    verifyEqual(testCase, trajectory.network_input_contract.state_names, ...
                {'i_Lr'; 'v_Cr'; 'i_Lm'; 'v_o'});
    verifyEqual(testCase, trajectory.network_input_contract.exogenous_names, ...
                {'Q1_A_high'; 'Q2_A_low'; 'Q3_B_high'; 'Q4_B_low'; 'vin_v'});
    verifyEqual(testCase, ...
                trajectory.network_input_contract.excluded_control_names, ...
                {'reference_voltage_v'});
    clear cleanupConfig cleanupDirectory;
end

function result = signal(center, amplitude, frequency, phase)
    result = struct('center_value', center, 'amplitude', amplitude, ...
                    'frequency_hz', frequency, 'phase_rad', phase);
end

function writeJson(file, value)
    handle = fopen(file, 'w');
    if handle < 0
        error('llc:TestFile', 'Unable to create temporary JSON configuration.');
    end
    cleanup = onCleanup(@()fclose(handle));
    fwrite(handle, jsonencode(value, PrettyPrint=true), 'char');
    clear cleanup;
end

function deleteIfPresent(file)
    if isfile(file)
        delete(file);
    end
end
