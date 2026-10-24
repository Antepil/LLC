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

function testOpenLoopGeneratorDoesNotRequireExcitation(testCase)
    root = testCase.TestData.projectRoot;
    generation = llc_read_json(fullfile(root, 'configs', 'matlab', ...
        'data_generation.json'));
    [~, temporaryName] = fileparts(tempname);
    relativeOutputDirectory = fullfile('data', 'raw', temporaryName);
    outputDirectory = fullfile(root, relativeOutputDirectory);
    mkdir(outputDirectory);
    cleanupDirectory = onCleanup(@()rmdir(outputDirectory, 's'));
    configFile = [tempname, '.json'];
    cleanupConfig = onCleanup(@()deleteIfPresent(configFile));
    generation.output_directory = relativeOutputDirectory;
    generation.conditions.stop_time_s = 20e-6;
    writeJson(configFile, generation);

    files = generate_llc_dataset([], configFile);

    verifyNumElements(testCase, files, 1);
    verifyTrue(testCase, isfile(files(1)));
    trajectory = load(files(1), 'schema_version', 'trajectory_id');
    verifyEqual(testCase, trajectory.schema_version, 'llc-raw-trajectory-v1');
    verifyEqual(testCase, trajectory.trajectory_id, ...
                generation.conditions.trajectory_id);
    clear cleanupConfig cleanupDirectory;
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
