function llc_closed_loop_pfm(block)
    % LLC_CLOSED_LOOP_PFM Event-driven voltage PI and phase-continuous PFM.
    setup(block);
end

function setup(block)
    block.NumDialogPrms = 1;
    block.NumInputPorts = 1;
    block.NumOutputPorts = 1;
    block.SetPreCompInpPortInfoToDynamic;
    block.SetPreCompOutPortInfoToDynamic;
    block.InputPort(1).Dimensions = 1;
    block.InputPort(1).DatatypeID = 0;
    block.InputPort(1).Complexity = 'Real';
    block.InputPort(1).DirectFeedthrough = false;
    block.OutputPort(1).Dimensions = 9;
    block.OutputPort(1).DatatypeID = 0;
    block.OutputPort(1).Complexity = 'Real';
    block.SampleTimes = [-2, 0];
    block.SimStateCompliance = 'DefaultSimState';
    block.RegBlockMethod('PostPropagationSetup', @postPropagationSetup);
    block.RegBlockMethod('InitializeConditions', @initializeConditions);
    block.RegBlockMethod('Outputs', @outputs);
    block.RegBlockMethod('Update', @update);
end

function postPropagationSetup(block)
    names = {'phase_rad', 'integral_v_s', 'filtered_voltage_v', ...
             'frequency_hz', 'reference_voltage_v', 'error_v', ...
             'last_time_s', 'next_control_time_s', 'next_edge_time_s', ...
             'pending_on_time_s', 'active_pair', 'pending_pair', ...
             'desired_pair', 'next_hit_time_s'};
    block.NumDworks = numel(names);
    for index = 1:numel(names)
        block.Dwork(index).Name = names{index};
        block.Dwork(index).Dimensions = 1;
        block.Dwork(index).DatatypeID = 0;
        block.Dwork(index).Complexity = 'Real';
        block.Dwork(index).UsedAsDiscState = true;
    end
end

function initializeConditions(block)
    control = block.DialogPrm(1).Data;
    setState(block, 'phase_rad', 0);
    setState(block, 'integral_v_s', 0);
    setState(block, 'filtered_voltage_v', 0);
    setState(block, 'frequency_hz', control.initial_frequency_hz);
    setState(block, 'reference_voltage_v', 0);
    setState(block, 'error_v', 0);
    setState(block, 'last_time_s', 0);
    setState(block, 'next_control_time_s', 0);
    setState(block, 'next_edge_time_s', 0.5 / control.initial_frequency_hz);
    setState(block, 'pending_on_time_s', control.deadtime_s);
    setState(block, 'active_pair', 0);
    setState(block, 'pending_pair', 1);
    setState(block, 'desired_pair', 1);
    setState(block, 'next_hit_time_s', min([control.deadtime_s, ...
                                            control.control_period_s, ...
                                            0.5 / control.initial_frequency_hz]));
end

function outputs(block)
    activePair = getState(block, 'active_pair');
    if activePair == 1
        gates = [1, 0, 0, 1];
    elseif activePair == 2
        gates = [0, 1, 1, 0];
    else
        gates = zeros(1, 4);
    end
    block.OutputPort(1).Data = [gates, getState(block, 'frequency_hz'), ...
                               getState(block, 'filtered_voltage_v'), ...
                               getState(block, 'reference_voltage_v'), ...
                               getState(block, 'error_v'), ...
                               getState(block, 'integral_v_s')];
    nextHit = getState(block, 'next_hit_time_s');
    if nextHit <= block.CurrentTime
        nextHit = block.CurrentTime + max(eps(block.CurrentTime), 1e-15);
    end
    block.NextTimeHit = nextHit;
end

function update(block)
    control = block.DialogPrm(1).Data;
    now = block.CurrentTime;
    tolerance = max(1e-14, 32 * eps(max(now, control.deadtime_s)));
    previousTime = getState(block, 'last_time_s');
    frequency = getState(block, 'frequency_hz');
    phase = mod(getState(block, 'phase_rad') + 2 * pi * frequency * (now - previousTime), 2 * pi);
    nextEdge = getState(block, 'next_edge_time_s');
    pendingOn = getState(block, 'pending_on_time_s');
    edgeDue = now >= nextEdge - tolerance;
    pendingDue = now >= pendingOn - tolerance;
    activePair = getState(block, 'active_pair');
    pendingPair = getState(block, 'pending_pair');
    desiredPair = getState(block, 'desired_pair');

    if edgeDue
        desiredPair = 3 - desiredPair;
        if desiredPair == 1
            phase = 0;
        else
            phase = pi;
        end
        activePair = 0;
        pendingPair = desiredPair;
        pendingOn = now + control.deadtime_s;
    elseif pendingDue
        activePair = pendingPair;
        pendingPair = 0;
        pendingOn = Inf;
    end

    nextControl = getState(block, 'next_control_time_s');
    if now >= nextControl - tolerance
        measuredVoltage = block.InputPort(1).Data;
        filteredVoltage = getState(block, 'filtered_voltage_v');
        alpha = control.control_period_s / ...
                (control.voltage_filter_time_constant_s + control.control_period_s);
        filteredVoltage = filteredVoltage + alpha * (measuredVoltage - filteredVoltage);
        reference = control.reference_voltage_v * min(now / control.reference_ramp_time_s, 1);
        errorVoltage = reference - filteredVoltage;
        integral = getState(block, 'integral_v_s');
        candidateIntegral = integral + control.control_period_s * errorVoltage;
        desiredFrequency = piFrequency(control, errorVoltage, candidateIntegral);
        pushingLow = desiredFrequency < control.minimum_frequency_hz && errorVoltage > 0;
        pushingHigh = desiredFrequency > control.maximum_frequency_hz && errorVoltage < 0;
        if pushingLow || pushingHigh
            desiredFrequency = piFrequency(control, errorVoltage, integral);
        else
            integral = candidateIntegral;
        end
        desiredFrequency = min(max(desiredFrequency, control.minimum_frequency_hz), ...
                               control.maximum_frequency_hz);
        maximumChange = control.frequency_slew_rate_hz_per_s * control.control_period_s;
        frequency = frequency + min(max(desiredFrequency - frequency, -maximumChange), maximumChange);
        setState(block, 'filtered_voltage_v', filteredVoltage);
        setState(block, 'reference_voltage_v', reference);
        setState(block, 'error_v', errorVoltage);
        setState(block, 'integral_v_s', integral);
        setState(block, 'frequency_hz', frequency);
        nextControl = nextControl + control.control_period_s;
        while nextControl <= now + tolerance
            nextControl = nextControl + control.control_period_s;
        end
    end

    if desiredPair == 1
        targetPhase = pi;
    else
        targetPhase = 2 * pi;
    end
    nextEdge = now + (targetPhase - phase) / (2 * pi * frequency);
    nextHit = min([nextControl, nextEdge, pendingOn]);
    if nextHit <= now + tolerance
        nextHit = now + max(eps(now), 1e-15);
    end
    setState(block, 'phase_rad', phase);
    setState(block, 'last_time_s', now);
    setState(block, 'next_control_time_s', nextControl);
    setState(block, 'next_edge_time_s', nextEdge);
    setState(block, 'pending_on_time_s', pendingOn);
    setState(block, 'active_pair', activePair);
    setState(block, 'pending_pair', pendingPair);
    setState(block, 'desired_pair', desiredPair);
    setState(block, 'next_hit_time_s', nextHit);
end

function frequency = piFrequency(control, errorVoltage, integral)
    frequency = control.bias_frequency_hz ...
                - control.proportional_gain_hz_per_v * errorVoltage ...
                - control.integral_gain_hz_per_v_s * integral;
end

function value = getState(block, name)
    for index = 1:block.NumDworks
        if strcmp(block.Dwork(index).Name, name)
            value = block.Dwork(index).Data;
            return;
        end
    end
    error('llc:ControllerState', 'Unknown controller state: %s', name);
end

function setState(block, name, value)
    for index = 1:block.NumDworks
        if strcmp(block.Dwork(index).Name, name)
            block.Dwork(index).Data = value;
            return;
        end
    end
    error('llc:ControllerState', 'Unknown controller state: %s', name);
end
