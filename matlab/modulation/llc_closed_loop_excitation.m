function llc_closed_loop_excitation(block)
    % LLC_CLOSED_LOOP_EXCITATION Supply Vin and Vref to the shared model.
    setup(block);
end

function setup(block)
    block.NumDialogPrms = 1;
    block.NumInputPorts = 0;
    block.NumOutputPorts = 1;
    block.SetPreCompOutPortInfoToDynamic;
    block.OutputPort(1).Dimensions = 2;
    block.OutputPort(1).DatatypeID = 0;
    block.OutputPort(1).Complexity = 'Real';
    block.OutputPort(1).SamplingMode = 'Sample';
    block.SampleTimes = [0, 0];
    block.SimStateCompliance = 'DefaultSimState';
    block.RegBlockMethod('Outputs', @outputs);
end

function outputs(block)
    excitation = block.DialogPrm(1).Data;
    block.OutputPort(1).Data = ...
        llc_evaluate_sinusoidal_excitation(block.CurrentTime, excitation);
end
