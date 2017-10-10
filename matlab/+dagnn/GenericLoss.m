classdef GenericLoss < dagnn.ElementWise

  properties
    display = true;
  end

  properties (Transient)
    average = 0
    numAveraged = 0
  end

  methods
    function outputs = account(obj, inputs, outputs)
      if obj.display
        n = obj.numAveraged ;
        m = n + size(inputs{1},4) ;
        obj.average = (n * obj.average + gather(outputs{1})) / m ;
        obj.numAveraged = m ;
      end
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

  end
end