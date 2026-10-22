function value = llc_read_json(file)
    % LLC_READ_JSON Read a required JSON object with actionable errors.
    if ~(ischar(file) || (isstring(file) && isscalar(file)))
        error('llc:InvalidConfig', 'Configuration path must be text.');
    end
    file = char(file);
    if ~isfile(file)
        error('llc:MissingConfig', 'Configuration file does not exist: %s', file);
    end
    [~, ~, extension] = fileparts(file);
    if ~strcmpi(extension, '.json')
        error('llc:InvalidConfig', 'Configuration must be JSON: %s', file);
    end
    try
        value = jsondecode(fileread(file));
    catch cause
        error('llc:InvalidConfig', 'Cannot parse JSON configuration %s: %s', file, cause.message);
    end
    if ~isstruct(value) || ~isscalar(value)
        error('llc:InvalidConfig', 'Configuration root must be a JSON object: %s', file);
    end
end
