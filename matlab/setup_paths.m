function projectRoot = setup_paths()
    % SETUP_PATHS Add this project's MATLAB implementation directories.
    matlabRoot = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(matlabRoot);
    folders = {'models', 'modulation', 'simulation', 'validation', 'export', 'tests'};
    for index = 1:numel(folders)
        addpath(fullfile(matlabRoot, folders{index}));
    end
end
