function localState = llc_shift_closed_loop_controller_state(state, handoffTimeS, toleranceS)
    % LLC_SHIFT_CLOSED_LOOP_CONTROLLER_STATE Move an absolute state to local t=0.
    if ~isnumeric(handoffTimeS) || ~isreal(handoffTimeS) || ...
            ~isscalar(handoffTimeS) || ~isfinite(handoffTimeS) || handoffTimeS < 0 || ...
            ~isnumeric(toleranceS) || ~isreal(toleranceS) || ...
            ~isscalar(toleranceS) || ~isfinite(toleranceS) || toleranceS <= 0
        error('llc:InvalidControllerState', ...
              'Handoff time and tolerance must be finite valid scalars.');
    end
    controlForValidation = struct('reference_ramp_time_s', 1, ...
        'deadtime_s', toleranceS, 'minimum_pulse_s', toleranceS);
    llc_validate_closed_loop_controller_state(state, controlForValidation);
    if abs(state.last_time_s - handoffTimeS) > toleranceS
        error('llc:InvalidControllerState', ...
              'Controller last_time_s does not match the handoff boundary.');
    end
    localState = state;
    localState.last_time_s = 0;
    scheduleFields = {'next_control_time_s', 'next_edge_time_s', 'pending_on_time_s'};
    for index = 1:numel(scheduleFields)
        name = scheduleFields{index};
        value = state.(name);
        if isfinite(value)
            shifted = value - handoffTimeS;
            if shifted < -toleranceS
                error('llc:InvalidControllerState', ...
                      'Controller event %s precedes the handoff boundary.', name);
            end
            if shifted < 0
                shifted = 0;
            end
            localState.(name) = shifted;
        end
    end
end
