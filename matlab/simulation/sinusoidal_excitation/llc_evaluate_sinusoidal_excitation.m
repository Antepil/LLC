function values = llc_evaluate_sinusoidal_excitation(time, excitation)
    % LLC_EVALUATE_SINUSOIDAL_EXCITATION Evaluate Vin and Vref profiles.
    validateattributes(time, {'numeric'}, {'real', 'finite', 'nonnegative'});
    validateExcitation(excitation);
    envelope = excitationEnvelope(time, excitation.start_time_s, ...
                                  excitation.envelope_rise_time_s);
    tau = max(time - excitation.start_time_s, 0);
    values = [evaluateSignal(tau, envelope, excitation.vin), ...
              evaluateSignal(tau, envelope, excitation.reference)];
end

function values = evaluateSignal(tau, envelope, signal)
    values = signal.center_value + envelope .* signal.amplitude .* ...
             sin(2 * pi * signal.frequency_hz .* tau + signal.phase_rad);
end

function envelope = excitationEnvelope(time, startTime, riseTime)
    tau = time - startTime;
    envelope = zeros(size(time));
    if riseTime == 0
        envelope(tau >= 0) = 1;
        return;
    end
    rising = tau > 0 & tau < riseTime;
    envelope(rising) = 0.5 * (1 - cos(pi * tau(rising) / riseTime));
    envelope(tau >= riseTime) = 1;
end

function validateExcitation(excitation)
    if ~isstruct(excitation) || ~isscalar(excitation)
        error('llc:InvalidExcitation', 'excitation must be a scalar structure.');
    end
    requireNonnegative(excitation, 'start_time_s');
    requireNonnegative(excitation, 'envelope_rise_time_s');
    validateSignal(excitation, 'vin');
    validateSignal(excitation, 'reference');
end

function validateSignal(excitation, name)
    if ~isfield(excitation, name) || ~isstruct(excitation.(name)) || ...
            ~isscalar(excitation.(name))
        error('llc:InvalidExcitation', '%s must be a scalar structure.', name);
    end
    signal = excitation.(name);
    requirePositive(signal, 'center_value');
    requireNonnegative(signal, 'amplitude');
    requireNonnegative(signal, 'frequency_hz');
    requireFinite(signal, 'phase_rad');
    if signal.amplitude > 0 && signal.frequency_hz <= 0
        error('llc:InvalidExcitation', ...
              '%s.frequency_hz must be positive when amplitude is nonzero.', name);
    end
    if signal.amplitude >= signal.center_value
        error('llc:InvalidExcitation', ...
              '%s amplitude must be smaller than its center value.', name);
    end
end

function requirePositive(value, name)
    requireFinite(value, name);
    if value.(name) <= 0
        error('llc:InvalidExcitation', '%s must be positive.', name);
    end
end

function requireNonnegative(value, name)
    requireFinite(value, name);
    if value.(name) < 0
        error('llc:InvalidExcitation', '%s must be nonnegative.', name);
    end
end

function requireFinite(value, name)
    if ~isfield(value, name) || ~isnumeric(value.(name)) || ...
            ~isscalar(value.(name)) || ~isfinite(value.(name))
        error('llc:InvalidExcitation', '%s must be a finite scalar.', name);
    end
end
