load('0-1782s.mat');
[pks, locs] = findpeaks(Y, T);

%frequency spectra
Spikes = locs;
dt = 0.001;
numBins = 15;
pOverlap = 50;
[f, spec] = welchfft(Spikes, dt, numBins, pOverlap);
plot(f(3:end),spec(3:end));

% %ISI Frequency Distributions 
% Steps = 0:0.005:10;
% [BinCenters, Dist] = ISIFreq(Spikes, Steps);
% figure
% bar(BinCenters, Dist);
% title 'ISI Frequency Distribution'
% xlabel 'Frequencies'
% ylabel 'Distribution'

% %Find Bursts and Pauses
% N_min = 2;
% Steps = -3:0.005:1.5;
% p = 0.05;
% alpha = 0.05;
% [Bursts, Pauses] = RGSDetect(Spikes, N_min, Steps, p, alpha)

