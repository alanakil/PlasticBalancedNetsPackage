% Compute spike count covariance matrix.
% s is a 2x(ns) matrix where ns is the number of spikes
% s(1,:) lists spike times
% and s(2,:) lists corresponding neuron indices
% Neuron indices are assumed to go from 1 to N
%
% Spikes are counts starting at time T1 and ending at 
% time T2.
%
% winsize is the window size over which spikes are counted,
% so winsize is assumed to be much smaller than T2-T1
%
% Covariances are only computed between neurons whose
% indices are listed in the vector Inds. If Inds is not
% passed in then all NxN covariances are computed.

function C=SpikeCountCov(s,N,T1,T2,winsize,Inds)

%     % Default: Use all indices
     if(~exist('Inds','var') || isempty(Inds))
         Inds=1:N;
     end
 
    Inds=round(Inds);
     % Check for out of bounds Inds
     if(min(Inds)<1 || max(Inds)>N)
         error('Out of bounds indicesin Inds!');
     end
     
     % Check for out of bounds times
     if(max(s(1,:))>T2 || min(s(1,:))<T1)
         s=s(:,s(1,:)<=T2 & s(1,:)>=T1);
     end
 
     % Check for out of bounds neuron indices
     s(2,:)=round(s(2,:));
     if(max(s(2,:))>N | min(s(2,:))<1)
         warning('Ignoring out of bound spike times.')
         s=s(:,s(2,:)<=N & s(2,:)>=1);
     end

    % Edges for histogram
    edgest=T1:winsize:T2;
    edgesi=(1:N+1)-.01;
    edges={edgest,edgesi};

    % Get 2D histogram of spike indices and times
    counts=hist3(s','Edges',edges);

    % Get rid of edges, which are 0
    counts=counts(1:end-1,1:end-1);
    
    % Only use counts that are in Inds.
    counts=counts(:,Inds);

    % Compute covariance matrix
    C=cov(counts);

end
