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

function testFormalContinuousRandomConfigurationMatchesScale(testCase)
    root = testCase.TestData.projectRoot;
    configFile = fullfile(root, 'configs', 'matlab', ...
        'sinusoidal_excitation', 'continuous_random_data_generation.json');
    config = llc_read_json(configFile);

    verifyEqual(testCase, config.schema_version, ...
                'llc-continuous-random-excitation-data-generation-v1');
    verifyEqual(testCase, config.mat_file_version, '-v7');
    verifyEqual(testCase, config.fixed_load_ohm, 2.304, 'AbsTol', 1e-12);
    verifyEqual(testCase, config.simulation_stop_time_s, 0.205, ...
                'AbsTol', 1e-15);
    verifyEqual(testCase, config.excitation_timing.start_time_s, 0.005);
    verifyEqual(testCase, config.excitation_timing.segment_duration_s, 0.020);
    verifyEqual(testCase, config.excitation_timing.transition_time_s, 0.002);
    verifyEqual(testCase, config.excitation_timing.segment_count, 10);
    verifyEqual(testCase, config.random_excitation.frequency_choices_hz, ...
                [100; 300; 1000]);
    verifyEqual(testCase, config.random_excitation.vin.center_range, [380; 420]);
    verifyEqual(testCase, config.random_excitation.vin.amplitude_range, [10; 20]);
    verifyEqual(testCase, config.random_excitation.reference.center_range, ...
                [45.6; 50.4]);
    verifyEqual(testCase, config.random_excitation.reference.amplitude_range, ...
                [1.2; 2.4]);
    verifyNumElements(testCase, config.conditions, 24);
    roles = string({config.conditions.dataset_role});
    groups = string({config.conditions.excitation_group});
    verifyEqual(testCase, nnz(roles == "train"), 16);
    verifyEqual(testCase, nnz(roles == "validation"), 4);
    verifyEqual(testCase, nnz(roles == "test"), 4);
    verifyEqual(testCase, nnz(groups == "vin_only"), 6);
    verifyEqual(testCase, nnz(groups == "reference_only"), 6);
    verifyEqual(testCase, nnz(groups == "both"), 12);
    seeds = [config.conditions.random_seed];
    verifyEqual(testCase, numel(unique(seeds)), 24);
    verifyTrue(testCase, all(seeds == round(seeds) & seeds >= 0));
    pointsPerTrajectory = config.simulation_stop_time_s * ...
        config.sampling.reference_frequency_hz * ...
        config.sampling.intervals_per_cycle + 1;
    verifyEqual(testCase, pointsPerTrajectory * numel(config.conditions), ...
                54120024, 'AbsTol', 1e-8);
    verifyFalse(testCase, contains(config.output_directory, 'screening'));
end

function testFormalConfigurationRejectsMalformedContracts(testCase)
    root = testCase.TestData.projectRoot;
    source = llc_read_json(fullfile(root, 'configs', 'matlab', ...
        'sinusoidal_excitation', 'continuous_random_data_generation.json'));
    configFile = [tempname, '.json'];
    cleanupConfig = onCleanup(@()deleteIfPresent(configFile));

    malformed = source;
    malformed.excitation_timing.transition_time_s = 0.021;
    writeJson(configFile, malformed);
    verifyError(testCase, @()generate_llc_sinusoidal_excitation_dataset( ...
        [], configFile, false), 'llc:InvalidConfig');

    malformed = source;
    malformed.random_excitation.vin.center_range = [420; 380];
    writeJson(configFile, malformed);
    verifyError(testCase, @()generate_llc_sinusoidal_excitation_dataset( ...
        [], configFile, false), 'llc:InvalidConfig');

    malformed = source;
    malformed.conditions(1).excitation_group = 'load_only';
    writeJson(configFile, malformed);
    verifyError(testCase, @()generate_llc_sinusoidal_excitation_dataset( ...
        [], configFile, false), 'llc:InvalidCondition');

    malformed = source;
    malformed.conditions(1).random_seed = 1.5;
    writeJson(configFile, malformed);
    verifyError(testCase, @()generate_llc_sinusoidal_excitation_dataset( ...
        [], configFile, false), 'llc:InvalidCondition');

    malformed = source;
    malformed.conditions(1).dataset_role = 'holdout';
    writeJson(configFile, malformed);
    verifyError(testCase, @()generate_llc_sinusoidal_excitation_dataset( ...
        [], configFile, false), 'llc:InvalidCondition');
    clear cleanupConfig;
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

function testContinuousRandomBuilderIsDeterministicAndRangeValid(testCase)
    randomization = formalRandomization();
    timing = formalTiming();
    condition = formalCondition('both', 12031);

    first = llc_build_continuous_random_excitation( ...
        randomization, timing, condition);
    second = llc_build_continuous_random_excitation( ...
        randomization, timing, condition);

    verifyEqual(testCase, first, second);
    verifyEqual(testCase, first.profile_type, ...
                'continuous_random_segments_v1');
    verifyNumElements(testCase, first.segments, timing.segment_count);
    verifyEqual(testCase, [first.segments.start_time_s], ...
        timing.start_time_s + (0:timing.segment_count - 1) * ...
        timing.segment_duration_s, 'AbsTol', 1e-15);
    vin = [first.segments.vin];
    reference = [first.segments.reference];
    verifyGreaterThanOrEqual(testCase, [vin.center_value], 380);
    verifyLessThanOrEqual(testCase, [vin.center_value], 420);
    verifyGreaterThanOrEqual(testCase, [vin.amplitude], 10);
    verifyLessThanOrEqual(testCase, [vin.amplitude], 20);
    verifyTrue(testCase, all(ismember([vin.frequency_hz], [100, 300, 1000])));
    verifyGreaterThanOrEqual(testCase, [reference.center_value], 45.6);
    verifyLessThanOrEqual(testCase, [reference.center_value], 50.4);
    verifyGreaterThanOrEqual(testCase, [reference.amplitude], 1.2);
    verifyLessThanOrEqual(testCase, [reference.amplitude], 2.4);
    verifyTrue(testCase, ...
        all(ismember([reference.frequency_hz], [100, 300, 1000])));
end

function testContinuousRandomBuilderKeepsInactiveSignalConstant(testCase)
    randomization = formalRandomization();
    timing = formalTiming();
    vinOnly = llc_build_continuous_random_excitation( ...
        randomization, timing, formalCondition('vin_only', 12041));
    referenceOnly = llc_build_continuous_random_excitation( ...
        randomization, timing, formalCondition('reference_only', 12042));

    vinOnlyReference = [vinOnly.segments.reference];
    referenceOnlyVin = [referenceOnly.segments.vin];
    verifyEqual(testCase, [vinOnlyReference.center_value], ...
                48 * ones(1, timing.segment_count));
    verifyEqual(testCase, [vinOnlyReference.amplitude], ...
                zeros(1, timing.segment_count));
    verifyEqual(testCase, [vinOnlyReference.frequency_hz], ...
                zeros(1, timing.segment_count));
    verifyEqual(testCase, [referenceOnlyVin.center_value], ...
                400 * ones(1, timing.segment_count));
    verifyEqual(testCase, [referenceOnlyVin.amplitude], ...
                zeros(1, timing.segment_count));
    verifyEqual(testCase, [referenceOnlyVin.frequency_hz], ...
                zeros(1, timing.segment_count));
end

function testContinuousRandomBuilderPreservesGlobalRandomState(testCase)
    rng(9341, 'twister');
    before = rng;
    llc_build_continuous_random_excitation( ...
        formalRandomization(), formalTiming(), formalCondition('both', 12051));
    after = rng;
    verifyEqual(testCase, after, before);
end

function testContinuousRandomBuilderRejectsUnknownGroup(testCase)
    verifyError(testCase, @()llc_build_continuous_random_excitation( ...
        formalRandomization(), formalTiming(), formalCondition('unknown', 12061)), ...
        'llc:InvalidCondition');
end

function testContinuousRandomProfileBlendsParametersAndIntegratesPhase(testCase)
    excitation = llc_build_continuous_random_excitation( ...
        formalRandomization(), formalTiming(), formalCondition('both', 12071));
    transition = excitation.transition_time_s;
    duration = excitation.segment_duration_s;
    firstTime = excitation.segments(1).start_time_s + transition / 2;
    secondTime = excitation.segments(2).start_time_s + transition / 2;

    actual = llc_evaluate_sinusoidal_excitation([firstTime; secondTime], excitation);
    expectedFirst = expectedBlendedValue(excitation.vin, ...
        excitation.segments(1).vin, transition / 2, transition, 0);
    firstFrequency = excitation.segments(1).vin.frequency_hz;
    phaseCyclesAtSecond = firstFrequency * duration;
    expectedSecond = expectedBlendedValue(excitation.segments(1).vin, ...
        excitation.segments(2).vin, transition / 2, transition, ...
        phaseCyclesAtSecond, excitation.vin.phase_rad);

    verifyEqual(testCase, actual(1, 1), expectedFirst, 'AbsTol', 1e-11);
    verifyEqual(testCase, actual(2, 1), expectedSecond, 'AbsTol', 1e-11);
end

function testContinuousRandomProfileIsC1AtEveryBoundary(testCase)
    excitation = llc_build_continuous_random_excitation( ...
        formalRandomization(), formalTiming(), formalCondition('both', 12081));
    segmentStarts = [excitation.segments.start_time_s];
    boundaries = sort([segmentStarts, ...
                       segmentStarts + excitation.transition_time_s]);
    h = 1e-8;
    for boundary = boundaries
        values = llc_evaluate_sinusoidal_excitation( ...
            [boundary - h; boundary; boundary + h], excitation);
        leftDerivative = (values(2, :) - values(1, :)) / h;
        rightDerivative = (values(3, :) - values(2, :)) / h;
        verifyEqual(testCase, values(2, :), ...
            llc_evaluate_sinusoidal_excitation(boundary, excitation), ...
            'AbsTol', 1e-12);
        verifyEqual(testCase, leftDerivative, rightDerivative, ...
                    'AbsTol', 100);
    end
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

function testFormalContinuousRandomGeneratorSmoke(testCase)
    root = testCase.TestData.projectRoot;
    sourceConfig = llc_read_json(fullfile(root, 'configs', 'matlab', ...
        'sinusoidal_excitation', 'continuous_random_data_generation.json'));
    [~, temporaryName] = fileparts(tempname);
    relativeOutputDirectory = fullfile('data', 'raw', temporaryName);
    outputDirectory = fullfile(root, relativeOutputDirectory);
    mkdir(outputDirectory);
    cleanupDirectory = onCleanup(@()rmdir(outputDirectory, 's'));
    configFile = [tempname, '.json'];
    cleanupConfig = onCleanup(@()deleteIfPresent(configFile));
    sourceConfig.output_directory = relativeOutputDirectory;
    sourceConfig.simulation_stop_time_s = 0.004;
    sourceConfig.excitation_timing.start_time_s = 0.001;
    sourceConfig.excitation_timing.segment_duration_s = 0.003;
    sourceConfig.excitation_timing.transition_time_s = 0.0005;
    sourceConfig.excitation_timing.segment_count = 1;
    sourceConfig.conditions = sourceConfig.conditions(9);
    writeJson(configFile, sourceConfig);

    [files, summary] = generate_llc_sinusoidal_excitation_dataset( ...
        [], configFile, false);

    assertEqual(testCase, summary.successful_count, 1, ...
                summary.entries(1).error_message);
    assertEqual(testCase, summary.failed_count, 0);
    trajectory = load(files(1), 'schema_version', 'dataset_split', ...
        'random_seed', 'excitation', 'state_times_s', ...
        'state_excitation_values', 'network_input_contract');
    verifyEqual(testCase, trajectory.schema_version, ...
                'llc-continuous-random-excitation-fixed-grid-trajectory-v1');
    verifyEqual(testCase, trajectory.dataset_split, 'train');
    verifyEqual(testCase, trajectory.random_seed, ...
                sourceConfig.conditions.random_seed);
    verifyEqual(testCase, trajectory.excitation.profile_type, ...
                'continuous_random_segments_v1');
    verifyNumElements(testCase, trajectory.excitation.segments, 1);
    verifyNumElements(testCase, trajectory.state_times_s, 44001);
    verifyGreaterThan(testCase, ...
        range(trajectory.state_excitation_values(:, 1)), 1);
    verifyGreaterThan(testCase, ...
        range(trajectory.state_excitation_values(:, 2)), 0.1);
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

function value = formalRandomization()
    value = struct( ...
        'frequency_choices_hz', [100, 300, 1000], ...
        'vin', struct('initial_center_value', 400, ...
                      'center_range', [380, 420], ...
                      'amplitude_range', [10, 20]), ...
        'reference', struct('initial_center_value', 48, ...
                            'center_range', [45.6, 50.4], ...
                            'amplitude_range', [1.2, 2.4]));
end

function value = formalTiming()
    value = struct('start_time_s', 5e-3, ...
                   'segment_duration_s', 20e-3, ...
                   'transition_time_s', 2e-3, ...
                   'segment_count', 10);
end

function value = formalCondition(group, seed)
    value = struct('trajectory_id', ['test_', group], ...
                   'dataset_role', 'train', ...
                   'excitation_group', group, ...
                   'random_seed', seed);
end

function value = expectedBlendedValue(old, target, localTime, transition, ...
        phaseCyclesAtStart, initialPhase)
    if nargin < 6
        initialPhase = old.phase_rad;
    end
    s = localTime / transition;
    weight = 3 * s^2 - 2 * s^3;
    center = old.center_value + (target.center_value - old.center_value) * weight;
    amplitude = old.amplitude + (target.amplitude - old.amplitude) * weight;
    oldFrequency = old.frequency_hz;
    deltaFrequency = target.frequency_hz - oldFrequency;
    transitionIntegral = oldFrequency * localTime + ...
        deltaFrequency * transition * (s^3 - 0.5 * s^4);
    phase = initialPhase + 2 * pi * (phaseCyclesAtStart + transitionIntegral);
    value = center + amplitude * sin(phase);
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
