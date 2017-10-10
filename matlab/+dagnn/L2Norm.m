classdef L2Norm < dagnn.ElementWise
  properties
    param = 1e-10;
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnl2norm(inputs{1}, obj.param) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnl2norm(inputs{1}, obj.param, derOutputs{1}) ;
      derParams = {} ;
    end
    
    
    function rfs = getReceptiveFields(obj)
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
    end

    function obj = SpatialNorm(varargin)
      obj.load(varargin) ;
    end
  end
end