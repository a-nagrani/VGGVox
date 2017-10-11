function setup_VGGVox()
%SETUP_VGGVOX Sets up VGGVox, by adding its folders 
% to the Matlab path
%
% Copyright (C) 2017 Arsha Nagrani

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab']) ;
