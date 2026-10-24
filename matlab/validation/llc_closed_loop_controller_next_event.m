function timeS = llc_closed_loop_controller_next_event(state)
    % LLC_CLOSED_LOOP_CONTROLLER_NEXT_EVENT Return the next control or gate event.
    timeS = min([state.next_control_time_s, state.next_edge_time_s, state.pending_on_time_s]);
    if ~isfinite(timeS)
        error('llc:InvalidControllerState', 'Controller has no finite next event.');
    end
end
