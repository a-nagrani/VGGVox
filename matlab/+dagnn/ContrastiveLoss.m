classdef ContrastiveLoss < dagnn.GenericLoss
  properties
    margin = 1;
  end

  methods
    function outputs = forward(obj, inputs, params)
      switch numel(inputs)
        case 3
          outputs{1} = vl_nncontrloss(inputs{:}, 'margin', obj.margin);
        case 4
          outputs{1} = vl_nncontrloss(inputs{1:3}, 'margin', inputs{4});
        otherwise
          error('Invalid number of inputs.');
      end
      if true
        isPos = inputs{3} == 1;
        % posDist = mean(norm(reshape(inputs{1}(:, :, :, isPos) - inputs{2}(:, :, :, isPos), [], sum(isPos))));
        % negDist = mean(norm(reshape(inputs{1}(:, :, :, ~isPos) - inputs{2}(:, :, :, ~isPos), [], sum(~isPos))));
        posDist = sqrt(sum((reshape(inputs{1}(:, :, :, isPos) - inputs{2}(:, :, :, isPos), [], sum(isPos))).^2,1));
        negDist = sqrt(sum((reshape(inputs{1}(:, :, :, ~isPos) - inputs{2}(:, :, :, ~isPos), [], sum(~isPos))).^2,1));
        eer = vl_eer([-1*ones(1,numel(posDist)) 1*ones(1,numel(negDist))],[posDist negDist]);
        fprintf('PD: %.2f ND: %.2f EER: %.3f ', mean(posDist), mean(negDist) ,eer);
      end
      
      logoutputs = outputs;
      logoutputs{1} = numel(inputs{3})*eer;
      obj.account(inputs, logoutputs);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      switch numel(inputs)
        case 3
          [dzdx1, dzdx2] = vl_nncontrloss(inputs{:}, derOutputs{1}, 'margin', obj.margin);
        case 4
          [dzdx1, dzdx2] = vl_nncontrloss(inputs{1:3}, derOutputs{1}, 'margin', inputs{4});
        otherwise
          error('Invalid number of inputs.');
      end
      derInputs = {dzdx1, dzdx2, []};
      derParams = {} ;
    end

    function obj = ContrastiveLoss(varargin)
      obj.load(varargin{:}) ;
    end
  end
end