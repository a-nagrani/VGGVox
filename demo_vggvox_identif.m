function demo_vggvox_identif(varargin)
% DEMO_VGGVOX_IDENTIF - minimal demo with the VGGVox model pretrained on
% the VoxCeleb dataset for Speaker Identification

% downloads the VGGVox model and
% prints the class and score of a test speech segment


opts.modelPath = '' ;
opts.gpu = 3;
opts.dataDir = 'testfiles/ident';
opts = vl_argparse(opts, varargin) ;

% Example speech segments for input
inpPath = fullfile(opts.dataDir, 'Y8hIVOBuels_0000002.wav');  %This speech segment belongs to speaker class 1

% Load or download the VGGVox model for Identification
modelName = 'vggvox_ident_net.mat' ;
paths = {opts.modelPath, ...
    modelName, ...
    fullfile(vl_rootnn, 'data', 'models-import', modelName)} ;
ok = find(cellfun(@(x) exist(x, 'file'), paths), 1) ;

if isempty(ok)
    fprintf('Downloading the VGGVox model for Identification ... this may take a while\n') ;
    opts.modelPath = fullfile(vl_rootnn, 'data/models-import', modelName) ;
    mkdir(fileparts(opts.modelPath)) ; base = 'http://www.robots.ox.ac.uk' ;
    url = sprintf('%s/~vgg/data/voxceleb/models/%s', base, modelName) ;
    urlwrite(url, opts.modelPath) ;
else
    opts.modelPath = paths{ok} ;
end
tmp = load(opts.modelPath); net = tmp.net ;

buckets.pool 	= [2 5 8 11 14 17 20 23 27 30];
buckets.width 	= [100 200 300 400 500 600 700 800 900 1000];

% Evaluate network either on CPU or GPU and set up network to be in test
% mode
if ~isempty(opts.gpu),net = vl_simplenn_move(net,'gpu'); end
net.mode = 'test';
net.conserveMemory = false;


% Load input and do a forward pass
inp = test_getinput(inpPath,net.meta,buckets);
s1 = size(inp,2);
p1 = buckets.pool(s1==buckets.width);
net.layers{16}.pool=[1 p1];

res = vl_simplenn(net,inp);
prob 		= gather(squeeze(res(20).x(:,:,:,:)));
prob 		= sum(prob,2);
[score,class]    = max(prob);

% Print score and class for the speech segment
fprintf('Score:%d\nClass:%d\n',score, class);
