function state = llc_advance_closed_loop_controller( ...
        state, nowS, measuredVoltageV, referenceTargetV, control)
    % LLC_ADVANCE_CLOSED_LOOP_CONTROLLER Apply due PFM and PI events at nowS.
    tolerance = max(1e-14, 32 * eps(max(nowS, control.deadtime_s)));
    if ~isscalar(nowS) || ~isfinite(nowS) || nowS < state.last_time_s - tolerance || ...
            ~isscalar(measuredVoltageV) || ~isfinite(measuredVoltageV) || ...
            ~isscalar(referenceTargetV) || ~isfinite(referenceTargetV) || referenceTargetV <= 0
        error('llc:InvalidControllerInput', 'Controller time, measurement or reference is invalid.');
    end
    frequency = state.frequency_hz;
    phase = mod(state.phase_rad + 2 * pi * frequency * (nowS - state.last_time_s), 2 * pi);
    edgeDue = nowS >= state.next_edge_time_s - tolerance;
    pendingDue = nowS >= state.pending_on_time_s - tolerance;
    if edgeDue
        state.desired_pair = 3 - state.desired_pair;
        if state.desired_pair == 1
            phase = 0;
        else
            phase = pi;
        end
        state.active_pair = 0;
        state.pending_pair = state.desired_pair;
        state.pending_on_time_s = nowS + control.deadtime_s;
    elseif pendingDue
        state.active_pair = state.pending_pair;
        state.pending_pair = 0;
        state.pending_on_time_s = Inf;
    end
    if nowS >= state.next_control_time_s - tolerance
        alpha = control.control_period_s / ...
            (control.voltage_filter_time_constant_s + control.control_period_s);
        state.filtered_voltage_v = state.filtered_voltage_v + ...
            alpha * (measuredVoltageV - state.filtered_voltage_v);
        state.reference_voltage_v = referenceTargetV * min(nowS / control.reference_ramp_time_s, 1);
        state.error_v = state.reference_voltage_v - state.filtered_voltage_v;
        state.integral_v_s = state.integral_v_s + control.control_period_s * state.error_v;
        desiredFrequency = control.bias_frequency_hz ...
            - control.proportional_gain_hz_per_v * state.error_v ...
            - control.integral_gain_hz_per_v_s * state.integral_v_s;
        maximumChange = control.frequency_slew_rate_hz_per_s * control.control_period_s;
        frequency = frequency + min(max(desiredFrequency - frequency, -maximumChange), maximumChange);
        if ~isfinite(frequency) || frequency <= 0
            error('llc:InvalidFrequency', 'PFM requires a finite positive frequency; got %g Hz.', frequency);
        end
        if 0.5 / frequency <= control.deadtime_s + control.minimum_pulse_s
            error('llc:InvalidSwitchingTiming', ...
                  'Frequency %g Hz cannot accommodate deadtime and minimum pulse.', frequency);
        end
        state.frequency_hz = frequency;
        state.next_control_time_s = state.next_control_time_s + control.control_period_s;
        while state.next_control_time_s <= nowS + tolerance
            state.next_control_time_s = state.next_control_time_s + control.control_period_s;
        end
    end
    if state.desired_pair == 1
        targetPhase = pi;
    else
        targetPhase = 2 * pi;
    end
    state.next_edge_time_s = nowS + (targetPhase - phase) / (2 * pi * frequency);
    state.phase_rad = phase;
    state.last_time_s = nowS;
end
