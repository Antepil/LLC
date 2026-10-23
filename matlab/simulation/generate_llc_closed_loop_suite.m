function results = generate_llc_closed_loop_suite(groupNames, resumeExisting)
    % GENERATE_LLC_CLOSED_LOOP_SUITE Run reviewed condition groups independently.
    matlabRoot = fileparts(fileparts(mfilename('fullpath')));
    projectRoot = fileparts(matlabRoot);
    addpath(matlabRoot);
    setup_paths;
    planFile = fullfile(projectRoot, 'configs', 'matlab', ...
                        'closed_loop_condition_plan.json');
    plan = llc_read_json(planFile);
    available = string({plan.groups.name});
    if nargin < 1 || isempty(groupNames)
        groupNames = available;
    else
        groupNames = string(groupNames);
    end
    if nargin < 2 || isempty(resumeExisting)
        resumeExisting = true;
    end
    if ~islogical(resumeExisting) || ~isscalar(resumeExisting)
        error('llc:InvalidArgument', 'resumeExisting must be a logical scalar.');
    end
    if any(~ismember(groupNames, available)) || numel(unique(groupNames)) ~= numel(groupNames)
        error('llc:InvalidArgument', ...
              'groupNames must be unique names from: %s.', strjoin(available, ', '));
    end

    template = struct('name', '', 'config_file', '', 'files', strings(0, 1), ...
                      'summary', struct([]));
    results = repmat(template, numel(groupNames), 1);
    for index = 1:numel(groupNames)
        group = plan.groups(strcmp(available, groupNames(index)));
        configFile = fullfile(projectRoot, 'configs', 'matlab', group.config_file);
        if ~isfile(configFile)
            error('llc:MissingConfig', ...
                  'Build the reviewed generation configs first: %s', configFile);
        end
        [files, summary] = generate_llc_closed_loop_dataset([], configFile, resumeExisting);
        results(index).name = char(group.name);
        results(index).config_file = configFile;
        results(index).files = files;
        results(index).summary = summary;
    end
end
