function sout = rm_dc_n_dither(sin, fs)
% removes DC component of the signal and add a small dither
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

if fs == 16e3, alpha = 0.99; 
elseif fs == 8e3, alpha = 0.999;
else error('only 8 and 16 kHz data are supported!');
end
    
sin = filter([1 -1], [1 -alpha], sin); % remove DC
dither = rand(size(sin)) + rand(size(sin)) - 1; 
spow = std(sin);
sout = sin + 1e-6 * spow * dither;
