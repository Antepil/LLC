function value = llc_evaluate_validation_signal(signal, timeS)
    % LLC_EVALUATE_VALIDATION_SIGNAL Evaluate a constant or covered time series.
    if ~isstruct(signal) || ~isscalar(signal) || ~isfield(signal, 'type')
        error('llc:InvalidValidationSignal', 'Signal must be a scalar struct with a type.');
    end
    if ~isnumeric(timeS) || any(~isfinite(timeS), 'all')
        error('llc:InvalidValidationSignal', 'Signal query times must be finite numeric values.');
    end
    type = string(signal.type);
    if type == "constant"
        if ~isfield(signal, 'value_v') || ~isscalar(signal.value_v) || ...
                ~isfinite(signal.value_v) || signal.value_v <= 0
            error('llc:InvalidValidationSignal', 'Constant signal value_v must be finite and positive.');
        end
        value = repmat(double(signal.value_v), size(timeS));
        return;
    end
    if type ~= "time_series" || ~all(isfield(signal, {'times_s', 'values_v', 'interpolation'}))
        error('llc:InvalidValidationSignal', 'Time series requires times_s, values_v and interpolation.');
    end
    times = double(signal.times_s(:));
    values = double(signal.values_v(:));
    if numel(times) < 2 || numel(times) ~= numel(values) || ...
            any(~isfinite(times)) || any(~isfinite(values)) || any(values <= 0) || any(diff(times) <= 0)
        error('llc:InvalidValidationSignal', 'Time-series samples must be finite, positive and increasing.');
    end
    tolerance = 64 * eps(max(max(abs(times)), 1));
    if any(timeS < times(1) - tolerance, 'all') || any(timeS > times(end) + tolerance, 'all')
        error('llc:ValidationSignalCoverage', 'Signal does not cover every requested time.');
    end
    interpolation = char(string(signal.interpolation));
    if ~ismember(interpolation, {'linear', 'previous'})
        error('llc:InvalidValidationSignal', 'interpolation must be linear or previous.');
    end
    query = min(max(double(timeS), times(1)), times(end));
    value = reshape(interp1(times, values, query(:), interpolation), size(timeS));
end
