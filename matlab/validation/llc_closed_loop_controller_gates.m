function gates = llc_closed_loop_controller_gates(state)
    % LLC_CLOSED_LOOP_CONTROLLER_GATES Return four applied gates from active_pair.
    if state.active_pair == 1
        gates = [1, 0, 0, 1];
    elseif state.active_pair == 2
        gates = [0, 1, 1, 0];
    elseif state.active_pair == 0
        gates = zeros(1, 4);
    else
        error('llc:InvalidControllerState', 'active_pair must be zero, one or two.');
    end
end
