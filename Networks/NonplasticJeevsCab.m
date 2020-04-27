%% Simulation of nonplastic balanced network for different values of Jei
% We want to know how changes in synaptic weights affect rates and
% correlations. So we simulate several networks with different I->E weights
% and compute the rates and correlations and compare them to theory to see
% how changes in weights affects correlated activity.

% Number of neurons in each population
N = 10000;
Ne=0.8*N;
Ni=0.2*N;

% Number of neurons in ffwd layer
Nx=0.2*N;

% Recurrent net connection probabilities
P=[0.1 0.1; 0.1 0.1];

% Ffwd connection probs
Px=[.1; .1];

% Correlation between the spike trains in the ffwd layer
c=0.1;

% Timescale of correlation
taujitter=5;

% Time (in ms) for sim
T=100000;

% Time discretization
dt=.1;

% Proportions
qe=Ne/N;
qi=Ni/N;
qf=Nx/N;

% FFwd spike train rate (in kHz)
rx=10/1000;

% Number of time bins
Nt=round(T/dt);
time=dt:dt:T;

% Extra stimulus: Istim is a time-dependent stimulus
% it is delivered to all neurons with weights given by JIstim.
% Specifically, the stimulus to neuron j at time index i is:
% Istim(i)*JIstim(j)
Istim=zeros(size(time)); 
Istim(time>T/2)=0; 
jestim=0; 
jistim=0;
Jstim=sqrt(N)*[jestim*ones(Ne,1); jistim*ones(Ni,1)]; 

% Synaptic timescales
taux=10;
taue=8;
taui=4;


% Neuron parameters
Cm=1;
gL=1/15;
EL=-72;
Vth=-50;
Vre=-75;
DeltaT=1;
VT=-55;

% Number of time bins to average over when recording
nBinsRecord=1;
dtRecord=nBinsRecord*dt;
timeRecord=dtRecord:dtRecord:T;
Ntrec=numel(timeRecord);

% Integer division function
IntDivide=@(n,k)(floor((n-1)/k)+1);

%%%
Num_J_vals = 10;

eRateT = cell(Num_J_vals,1);
iRateT = cell(Num_J_vals,1);
mC = cell(Num_J_vals,1);
mR = cell(Num_J_vals,1);
spikeTimes = cell(Num_J_vals,1);
spikeIndex = cell(Num_J_vals,1);

dtRate=1000; % ms
Vector_Time_Rate = (dtRate:dtRate:T)/1000;

Jee = linspace(1,20,Num_J_vals);

Jm=[Jee(1) -100; 112.5 -250]/sqrt(N);
Jxm=[180; 135]/sqrt(N);
% Build mean field matrices
Q=[qe qi; qe qi];
Qf=[qf; qf];
W=P.*(Jm*sqrt(N)).*Q;
Wx=Px.*(Jxm*sqrt(N)).*Qf;


parfor k = 1:Num_J_vals
    
    %Random initial voltages
    V0=rand(N,1)*(VT-Vre)+Vre;
    
    % Maximum number of spikes for all neurons
    % in simulation. Make it 50Hz across all neurons
    % If there are more spikes, the simulation will
    % terminate
    maxns=ceil(.05*N*T);
    
    % Indices of neurons to record currents, voltages
    nrecord0=2; % Number to record from each population
    Irecord=[randperm(Ne,nrecord0) randperm(Ni,nrecord0)+Ne];
    numrecord=numel(Irecord); % total number to record
    
    V=V0;
    Ie=zeros(N,1);
    Ii=zeros(N,1);
    Ix=zeros(N,1);
    %     IeRec=zeros(numrecord,Ntrec);
    %     IiRec=zeros(numrecord,Ntrec);
    %     IxRec=zeros(numrecord,Ntrec);
    %     VRec=zeros(numrecord,Ntrec);
    %     wRec=zeros(numrecord,Ntrec);
    
    % Mean connection strengths between each cell type pair
    Jm=[Jee(k) -100; 112.5 -250]/sqrt(N);
    Jxm=[180; 135]/sqrt(N);
    
    % Generate connectivity matrices
    tic
    rng(1);
    J=[Jm(1,1)*binornd(1,P(1,1),Ne,Ne) Jm(1,2)*binornd(1,P(1,2),Ne,Ni); ...
        Jm(2,1)*binornd(1,P(2,1),Ni,Ne) Jm(2,2)*binornd(1,P(2,2),Ni,Ni)];
    Jx=[Jxm(1)*binornd(1,Px(1),Ne,Nx); Jxm(2)*binornd(1,Px(2),Ni,Nx)];
    tGen=toc;
    disp(sprintf('\nTime to generate connections: %.2f sec',tGen))
    
    
    %%% Make (correlated) Poisson spike times for ffwd layer
    %%% See section 5 of SimDescription.pdf for a description of this algorithm
    tic
    if(c<1e-5) % If uncorrelated
        nspikeX=poissrnd(Nx*rx*T);
        st=rand(nspikeX,1)*T;
        sx=zeros(2,numel(st));
        sx(1,:)=sort(st);
        sx(2,:)=randi(Nx,1,numel(st)); % neuron indices
        %clear st;
    else % If correlated
        rm=rx/c; % Firing rate of mother process
        nstm=poissrnd(rm*T); % Number of mother spikes
        stm=rand(nstm,1)*T; % spike times of mother process
        maxnsx=T*rx*Nx*1.2; % Max num spikes
        sx=zeros(2,maxnsx);
        ns=0;
        for j=1:Nx  % For each ffwd spike train
            ns0=binornd(nstm,c); % Number of spikes for this spike train
            st=randsample(stm,ns0); % Sample spike times randomly
            st=st+taujitter*randn(size(st)); % jitter spike times
            st=st(st>0 & st<T); % Get rid of out-of-bounds times
            ns0=numel(st); % Re-compute spike count
            sx(1,ns+1:ns+ns0)=st; % Set the spike times and indices
            sx(2,ns+1:ns+ns0)=j;
            ns=ns+ns0;
        end
        
        % Get rid of padded zeros
        sx = sx(:,sx(1,:)>0);
        
        % Sort by spike time
        [~,I] = sort(sx(1,:));
        sx = sx(:,I);
        
        
        nspikeX=size(sx,2);
        
    end
    tGenx=toc;
    disp(sprintf('\nTime to generate ffwd spikes: %.2f sec',tGenx))
    
    iFspike=1;
    s=zeros(2,maxns);
    nspike=0;
    TooManySpikes=0;
    
    tic
    for i=1:numel(time)
        
        
        % Propogate ffwd spikes
        while(sx(1,iFspike)<=time(i) && iFspike<nspikeX)
            jpre=sx(2,iFspike);
            Ix=Ix+Jx(:,jpre)/taux;
            iFspike=iFspike+1;
        end
        
        % Euler update to V
        V=V+(dt/Cm)*(Istim(i)*Jstim+Ie+Ii+Ix+gL*(EL-V)+gL*DeltaT*exp((V-VT)/DeltaT));
        
        % Find which neurons spiked
        Ispike=find(V>=Vth);
        
        % If there are spikes
        if(~isempty(Ispike))
            
            % Store spike times and neuron indices
            if(nspike+numel(Ispike)<=maxns)
                s(1,nspike+1:nspike+numel(Ispike))=time(i);
                s(2,nspike+1:nspike+numel(Ispike))=Ispike;
            else
                TooManySpikes=1;
                break;
            end
            
            
            % Update synaptic currents
            Ie=Ie+sum(J(:,Ispike(Ispike<=Ne)),2)/taue;
            Ii=Ii+sum(J(:,Ispike(Ispike>Ne)),2)/taui;
             
            % Update cumulative number of spikes
            nspike=nspike+numel(Ispike);
        end
        
        % Euler update to synaptic currents
        Ie=Ie-dt*Ie/taue;
        Ii=Ii-dt*Ii/taui;
        Ix=Ix-dt*Ix/taux;
        
        % This makes plots of V(t) look better.
        % All action potentials reach Vth exactly.
        % This has no real effect on the network sims
        V(Ispike)=Vth;
        
        % Store recorded variables
        ii=IntDivide(i,nBinsRecord);
%         IeRec(:,ii)=IeRec(:,ii)+Ie(Irecord);
%         IiRec(:,ii)=IiRec(:,ii)+Ii(Irecord);
%         IxRec(:,ii)=IxRec(:,ii)+Ix(Irecord);
%         VRec(:,ii)=VRec(:,ii)+V(Irecord);
        
        % Reset mem pot.
        V(Ispike)=Vre;
        
        
    end
%     IeRec=IeRec/nBinsRecord; % Normalize recorded variables by # bins
%     IiRec=IiRec/nBinsRecord;
%     IxRec=IxRec/nBinsRecord;
%     VRec=VRec/nBinsRecord;
    s=s(:,1:nspike); % Get rid of padding in s
    tSim=toc;
    disp(sprintf('\nTime for simulation: %.2f min',tSim/60))
    
    
    %% Make a raster plot of first 500 neurons
    % s(1,:) are the spike times
    % s(2,:) are the associated neuron indices
    % figure
    % plot(s(1,s(2,:)<5000),s(2,s(2,:)<5000),'k.','Markersize',0.25)
    % xlabel('time (ms)')
    % ylabel('Neuron index')
    %xlim([0 2000])
    
    % Mean rate of each neuron (excluding burn-in period)
    Tburn=T/2;
    reSim=hist(s(2,s(1,:)>Tburn & s(2,:)<=Ne),1:Ne)/(T-Tburn);
    riSim=hist(s(2,s(1,:)>Tburn & s(2,:)>Ne)-Ne,1:Ni)/(T-Tburn);
    
    % Mean rate over E and I pops
    reMean=mean(reSim);
    riMean=mean(riSim);
    disp(sprintf('\nMean E and I rates from sims: %.2fHz %.2fHz',1000*reMean,1000*riMean))
    
    % Time-dependent mean rates
    eRateT{k,1}=hist(s(1,s(2,:)<=Ne),1:dtRate:T)/(dtRate*Ne);
    iRateT{k,1}=hist(s(1,s(2,:)>Ne),1:dtRate:T)/(dtRate*Ni);
    
  
    
    %% All the code below computes spike count covariances and correlations
    %%% We want to compare the resulting covariances to what is predicted by
    %%% the theory, first for non-plastic netowrks (eta=0)
    
    % Compute spike count covariances over windows of size
    % winsize starting at time T1 and ending at time T2.
    winsize=250;
    T1=T/2; % Burn-in period of 250 ms
    T2=T;   % Compute covariances until end of simulation
    C=SpikeCountCov(s,N,T1,T2,winsize);
    
    
    % Get mean spike count covariances over each sub-pop
    [II,JJ]=meshgrid(1:N,1:N);
    mCee=mean(C(II<=Ne & JJ<=II));
    mCei=mean(C(II<=Ne & JJ>Ne));
    mCii=mean(C(II>Ne & JJ>II));
    
    % Mean-field spike count cov matrix
    % Compare this to the theoretical prediction
    mC{k,1}=[mCee mCei; mCei mCii]
    
    % Compute spike count correlations
    % This takes a while, so make it optional
    ComputeSpikeCountCorrs=1;
    if(ComputeSpikeCountCorrs)
        
        % Get correlation matrix from cov matrix
        tic
        R=corrcov(C);
        toc
        
        mRee=mean(R(II<=Ne & JJ<=II & isfinite(R)));
        mRei=mean(R(II<=Ne & JJ>Ne & isfinite(R)));
        mRii=mean(R(II>Ne & JJ>II & isfinite(R)));
        
        % Mean-field spike count correlation matrix
        mR{k,1}=[mRee mRei; mRei mRii]
        
        
    end    
    
    if k==Num_J_vals
        spikeTimes{k,1}=s(1,s(2,:)<N+1);
        spikeIndex{k,1}=s(2,s(2,:)<N+1);
    end
end


%% Save theory variables.

winsize=250;
Theory_Cee = zeros(1,Num_J_vals);
Theory_Cei = zeros(1,Num_J_vals);
Theory_Cii = zeros(1,Num_J_vals);
r_e = zeros(1,Num_J_vals);
r_i = zeros(1,Num_J_vals);
Theory_Ree = zeros(1,Num_J_vals);
Theory_Rei = zeros(1,Num_J_vals);
Theory_Rii = zeros(1,Num_J_vals);

for j = 1:Num_J_vals
    
    W(1,1) = Jee(j)*qe*P(1,1); % 0.8 is fraction of exc neurons.
    Ctheory = winsize*inv(W)*Wx*transpose(Wx)*c*rx*inv(transpose(W));
    Theory_Cee(j)  = Ctheory(1,1); % Theoretical ee corrs
    Theory_Cei(j)  = Ctheory(1,2); % theoretical ei corrs
    Theory_Cii(j)  = Ctheory(2,2); % Theoretical ii corrs
    rates = -inv(W)*Wx*rx;
    r_e(j) = 1000*rates(1);
    r_i(j) = 1000*rates(2);
    Theory_Ree(j) = Theory_Cee(j)/r_e(j);
    Theory_Rei(j) = Theory_Cee(j)/sqrt(r_e(j))/sqrt(r_i(j));
    Theory_Rii(j) = Theory_Cee(j)/r_i(j);
end



%% Save all necessary variables.
save('./SavedSimulations/Jee_vs_Cab_Nonplastic.mat', 'eRateT','iRateT','T','dtRate',...
    'timeRecord','N','Theory_Cee','Theory_Cei','Theory_Cii',...
    'c','spikeTimes','spikeIndex','Vector_Time_Rate','Jee',...
    'Num_J_vals','r_e','r_i','Theory_Ree','Theory_Rei',...
    'Theory_Rii','mC','mR','W','Wx','rx');








