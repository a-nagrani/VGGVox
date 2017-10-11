function setup_VGGVox()
%SETUP_VGGVOX Sets up VGGVox, by adding its folders 
% to the Matlab path
%
% Copyright (C) 2017 Arsha Nagrani

  vl_setupnn ;
  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab']) ;
  addpath(genpath('mfcc'))
