%% Simulation of EE plastic balanced network. April 2020.
%%% This code simulates a single run of a balanced network with Hebbian
%%% STDP on EE synapses.

clear

rng(1);

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

%%%
% Correlation between the spike trains in the ffwd layer
c=0.5;

% Timescale of correlation
taujitter=5;

% Mean connection strengths between each cell type pair
Jm=[20 -100; 112.5 -250]/sqrt(N);
Jxm=[180; 135]/sqrt(N);

% Plasticity parameters. Set the upper bound.
Jmax=25/sqrt(N); % Fp: Jmax

% Time (in ms) for sim
T=20000;

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

% Build mean field matrices
Q=[qe qi; qe qi];
Qf=[qf; qf];
W=P.*(Jm*sqrt(N)).*Q;
Wx=Px.*(Jxm*sqrt(N)).*Qf;

% Synaptic timescales
taux=10;
taue=8;
taui=4;

% Generate connectivity matrices
tic
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
    clear st;
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


% Neuron parameters
Cm=1;
gL=1/15;
EL=-72;
Vth=-50;
Vre=-75;
DeltaT=1;
VT=-55;


% Plasticity params
eta=5/1000; % Learning rate 
tauSTDP=200;

% Random initial voltages
V0=rand(N,1)*(VT-Vre)+Vre;

% Maximum number of spikes for all neurons
% in simulation. Make it 50Hz across all neurons
% If there are more spikes, the simulation will
% terminate
maxns=ceil(.05*N*T);

% Indices of neurons to record currents, voltages
nrecord0=100; % Number to record from each population
Irecord=[randperm(Ne,nrecord0) randperm(Ni,nrecord0)+Ne];
numrecord=numel(Irecord); % total number to record

% Synaptic weights to record. 
% The first row of Jrecord is the postsynaptic indices
% The second row is the presynaptic indices
nJrecord0=1000; % Number to record
[II,JJ]=find(J(1:Ne,1:Ne)); % Find non-zero E to E weights
III=randperm(numel(II),nJrecord0); % Choose some at random to record
II=II(III);
JJ=JJ(III);
Jrecord=[II JJ]'; % Record these
numrecordJ=size(Jrecord,2);
if(size(Jrecord,1)~=2)
    error('Jrecord must be 2xnumrecordJ');
end

% Number of time bins to average over when recording
nBinsRecord=100;%20;
dtRecord=nBinsRecord*dt;
timeRecord=dtRecord:dtRecord:T;
Ntrec=numel(timeRecord);

% Integer division function
IntDivide=@(n,k)(floor((n-1)/k)+1);

V=V0;
Ie=zeros(N,1);
Ii=zeros(N,1);
Ix=zeros(N,1);
x=zeros(N,1);
IeRec=zeros(numrecord,Ntrec);
IiRec=zeros(numrecord,Ntrec);
IxRec=zeros(numrecord,Ntrec);
IeRecord=zeros(N,1);
IiRecord=zeros(N,1);
IxRecord=zeros(N,1);
Ie_Record_Wholetime = zeros(1,T/dt);
Ii_Record_Wholetime = zeros(1,T/dt);
conv_Spike=zeros(N,1);
conv_Spike_X=zeros(Nx,1);
sum_conv_Spike = zeros(N,1);
sum_conv_Spike_X = zeros(Nx,1);

VRec=zeros(numrecord,Ntrec);
wRec=zeros(numrecord,Ntrec);
JRec=zeros(1,Ntrec);
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
        Spikes_X = zeros(Nx,1);
        Spikes_X(jpre) = 1;
        conv_Spike_X(:,1) = conv_Spike_X(:,1) + Spikes_X;        
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
        
        % Get actual current coming out of each neuron.
        Spikes = zeros(N,1);
        Spikes(Ispike) = 1;
        conv_Spike(:,1) = conv_Spike(:,1) + Spikes;         
        
        
        % If there is plasticity
        if(eta~=0)
            % Update synaptic weights according to plasticity rules
            % E to E after presynaptic spike    
            J(1:Ne,Ispike(Ispike<=Ne))=J(1:Ne,Ispike(Ispike<=Ne))+ ... 
                -repmat(eta*(x(1:Ne)),1,nnz(Ispike<=Ne)).*(J(1:Ne,Ispike(Ispike<=Ne)));
            % E to E after a postsynaptic spike
            J(Ispike(Ispike<=Ne),1:Ne)=J(Ispike(Ispike<=Ne),1:Ne)+ ... 
                +repmat(eta*x(1:Ne)',nnz(Ispike<=Ne),1).*(Jmax).*(J(Ispike(Ispike<=Ne),1:Ne)~=0);
        end
        
        % Update rate estimates for plasticity rules
        x(Ispike)=x(Ispike)+1;
        
        % Update cumulative number of spikes
        nspike=nspike+numel(Ispike);
    end            
    
    % Euler update to synaptic currents
    Ie=Ie-dt*Ie/taue;
    Ii=Ii-dt*Ii/taui;
    Ix=Ix-dt*Ix/taux;
    conv_Spike(1:Ne,1)=conv_Spike(1:Ne,1)-dt*conv_Spike(1:Ne,1)/taue;
    conv_Spike(Ne+1:N,1)=conv_Spike(Ne+1:N,1)-dt*conv_Spike(Ne+1:N,1)/taui;
    conv_Spike_X(:,1)=conv_Spike_X(:,1)-dt*conv_Spike_X(:,1)/taux;
     
    % Update time-dependent firing rates for plasticity
    x(1:Ne)=x(1:Ne)-dt*x(1:Ne)/tauSTDP;
    x(Ne+1:end)=x(Ne+1:end)-dt*x(Ne+1:end)/tauSTDP;
    
    % This makes plots of V(t) look better.
    % All action potentials reach Vth exactly. 
    % This has no real effect on the network sims
    V(Ispike)=Vth;
    
    % Store recorded variables
    ii=IntDivide(i,nBinsRecord); 
    IeRec(:,ii)=IeRec(:,ii)+Ie(Irecord);
    IiRec(:,ii)=IiRec(:,ii)+Ii(Irecord);
    IxRec(:,ii)=IxRec(:,ii)+Ix(Irecord);
    VRec(:,ii)=VRec(:,ii)+V(Irecord);
    JRec(1,ii)=mean(J(sub2ind(size(J),Jrecord(1,:),Jrecord(2,:))));
    
    % Reset mem pot.
    V(Ispike)=Vre;
    
    if i>=T/2/dt
        sum_conv_Spike = sum_conv_Spike + conv_Spike(:,1);
        sum_conv_Spike_X = sum_conv_Spike_X + conv_Spike_X(:,1);
        IeRecord = IeRecord + Ie;
        IiRecord = IiRecord + Ii;
        IxRecord = IxRecord + Ix;
    end
    
    Ie_Record_Wholetime(1,i) = mean(Ie(1:Ne));
    Ii_Record_Wholetime(1,i) = mean(Ii(1:Ne));

    
end
IeRec=IeRec/nBinsRecord; % Normalize recorded variables by # bins
IiRec=IiRec/nBinsRecord;
IxRec=IxRec/nBinsRecord;
VRec=VRec/nBinsRecord;

% did computation online, so now divide to get the average.
IeRecord = IeRecord/(numel(time)/2);
IiRecord = IiRecord/(numel(time)/2);
IxRecord = IxRecord/(numel(time)/2);

s=s(:,1:nspike); % Get rid of padding in s
tSim=toc;
disp(sprintf('\nTime for simulation: %.2f min',tSim/60))


%% Make a raster plot of first 500 neurons 
% s(1,:) are the spike times
% s(2,:) are the associated neuron indices
figure
plot(s(1,s(2,:)<5000),s(2,s(2,:)<5000),'k.','Markersize',0.25)
xlabel('time (ms)')
ylabel('Neuron index')
xlim([0 2000])

% Mean rate of each neuron (excluding burn-in period)
Tburn=T/2;
reSim=hist(s(2,s(1,:)>Tburn & s(2,:)<=Ne),1:Ne)/(T-Tburn);
riSim=hist(s(2,s(1,:)>Tburn & s(2,:)>Ne)-Ne,1:Ni)/(T-Tburn);

% Mean rate over E and I pops
reMean=mean(reSim);
riMean=mean(riSim);
disp(sprintf('\nMean E and I rates from sims: %.2fHz %.2fHz',1000*reMean,1000*riMean))

% Time-dependent mean rates
dtRate=100; % ms
eRateT=hist(s(1,s(2,:)<=Ne),1:dtRate:T)/(dtRate*Ne);
iRateT=hist(s(1,s(2,:)>Ne),1:dtRate:T)/(dtRate*Ni);

% Plot time-dependent rates
figure
plot((dtRate:dtRate:T)/1000,1000*eRateT)
hold on
plot((dtRate:dtRate:T)/1000,1000*iRateT)
legend('r_e','r_i')
ylabel('rate (Hz)')
xlabel('time (s)')

%% If plastic, plot mean connection weights over time
if(eta~=0)
    figure;
    plot(timeRecord/1000,JRec*sqrt(N), 'linewidth',3)
    xlabel('time (s)')
    ylabel('Mean E to E synaptic weight')
    
    figure;
    histogram(nonzeros(J(1:Ne,1:Ne)*sqrt(N)))
    xlabel('J_{EE}')
    ylabel('Count')

end


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
mC=[mCee mCei; mCei mCii]

% Compute spike count correlations
% This takes a while, so make it optional
ComputeSpikeCountCorrs=0;
if(ComputeSpikeCountCorrs)
    
    % Get correlation matrix from cov matrix
    tic
    R=corrcov(C);
    toc
    
    mRee=mean(R(II<=Ne & JJ<=II & isfinite(R)));
    mRei=mean(R(II<=Ne & JJ>Ne & isfinite(R)));
    mRii=mean(R(II>Ne & JJ>II & isfinite(R)));

    % Mean-field spike count correlation matrix
    mR=[mRee mRei; mRei mRii]


end

%% Mean final E to E synaptic weight: 
mean(nonzeros(J(1:Ne,1:Ne)))*sqrt(N)


%% Compute Cv of ISI.
% Column 1 has spike times and column 2 the neuron index
ComputeCV=0;
if ComputeCV~=0
    spikeTrain = transpose(s);
    % Sort neuron index in ascending order, keeping the correct spike time for
    % each one of them.
    spikeTrain = sortrows(spikeTrain,2);
    
    tic
    BadValue_index = zeros(N-1,1);
    spikes = cell(N,1);
    for i=1:N-1
        
        BadValue_index(i+1) = BadValue_index(i) + sum(spikeTrain(:,2) == i);
        
        spikes{i} = spikeTrain(BadValue_index(i)+1:BadValue_index(i+1),1);
        
    end
    for i=1:N
        if isempty(spikes{i})
            spikes{i} = [];
        end
    end
    ISI=cell(N,1);
    for i=1:N
        ISI{i} = diff(spikes{i}(:),1,1);
        sigma(i) = std(ISI{i}(:));
        mu(i) = mean(ISI{i}(:));
    end
    tSim=toc;
    disp(sprintf('\nTime for CV: %.2f min',tSim/60))
    
    CV_ISI = sigma./mu;
    CV_ISI = CV_ISI(~isnan(CV_ISI));
    
    % Plot distribution of ISI's.
    figure;
    histogram(CV_ISI(CV_ISI~=0),1000)
    
    MEAN_CV_ISI = mean(CV_ISI(CV_ISI~=0))
    
end


%% Compute cov(J,S) to check if this is the issue with weights.
% Use the final steady state weights and the final currents: Ii, Ie.

COV_J_Se = zeros(N,1);
CORR_J_Se = zeros(N,1);
mean_conv_Spike = sum_conv_Spike/(T/dt/2);
mean_conv_Spike_X = sum_conv_Spike_X/(T/dt/2);


%% Compute including the X external input.
% Compute cov(J,S) to check if this is the issue with weights.
% Use the final steady state weights and the final currents: Ii, Ie.

Mean_J_times_Se = zeros(N,1);
Mean_JSe = zeros(N,1);
for i = 1:N
    Mean_J_times_Se(i) = mean( J(i,1:Ne) .* mean_conv_Spike(1:Ne)' ); 
    Mean_JSe(i) =  mean( J(i,1:Ne) ) * mean(mean_conv_Spike(1:Ne)');
end
figure; hold on
plot(Mean_J_times_Se)
plot(Mean_JSe)
plot(Mean_J_times_Se - Mean_JSe, 'linewidth',1)
xlabel('N','fontsize',15)

leg = legend('mean(J*Si)','mean(J)*mean(Si)','cov(J,Si)');
leg.FontSize=11;

mean(Mean_J_times_Se)
mean(Mean_JSe)
mean(Mean_J_times_Se) - mean(Mean_JSe)
-100*(mean(Mean_J_times_Se) - mean(Mean_JSe))/mean(Mean_J_times_Se)

%% Compute correlations between E to E synapses over time.
tic
Corr_EE_syn = corr(JRec(:,T/dt/nBinsRecord/2:end)');
toc
colvect= Corr_EE_syn(find(~tril(ones(size(Corr_EE_syn)))));
Avg_Corr_EE = mean( colvect )


%% Check currents.
% figure; hold on
% plot(mean(IeRec))
% plot(mean(IiRec))
% plot(mean(IxRec))
% plot(mean(IeRec)+mean(IiRec)+mean(IxRec))
% 
% IeRec1 = mean(IeRec);
% IiRec2 = mean(IiRec);
% IxRec3 = mean(IxRec);

%%

% plot(IeRecord(1:N)+IiRecord(1:N))
R_e_sims = mean(IeRecord(1:Ne)+IiRecord(1:Ne))
R_i_sims = mean(IeRecord(Ne+1:N)+IiRecord(Ne+1:N))

W(1,1) = 0.1*(mean(nonzeros(J(1:Ne,1:Ne)))*sqrt(N))*0.8;
eFR = mean(reSim); iFR = mean(riSim);
R_e_theory = (W(1,1) * eFR + W(1,2) * iFR)*sqrt(N)
R_i_theory = (W(2,1) * eFR + W(2,2) * iFR)*sqrt(N)


%% Compute the covariance between weights and rates over time.

for i = 1:(T-1)/dt/1000
    Ie_Record_Wholetime1(i) = mean(Ie_Record_Wholetime(1,1000*i:1000*i+999));
    Ii_Record_Wholetime1(i) = mean(Ii_Record_Wholetime(1,1000*i:1000*i+999));
%     Record_J1(i) = mean(Record_J(1,i:i+1000));
end

% Actual total input to E neurons as a function of time.
R_e_sims_time = Ie_Record_Wholetime1;

% Theoretical total input to E neurons as a function of time.
Wee_time = mean(JRec)*sqrt(N)*0.1*0.8;
R_e_theory_time = zeros(1,T/dt/1000-1);
for i = 1:length(eRateT)-1
    R_e_theory_time(1,i) = ( Wee_time(1,i) * eRateT(1,i))*sqrt(N);
end

Cov_J_r = R_e_sims_time - R_e_theory_time;

Jee_time = (Wee_time/0.1/0.2);
Jee_time = Jee_time(1:999);
Wee_time = Wee_time(1:999);
eRateT = eRateT(1:999);
mean_Jeere = Cov_J_r + Jee_time .* eRateT ;

mean_Jeere = mean_Jeere(2:end-1);
Jee_time = Jee_time(2:end-1);

Jee_IC = Jm(1,1)*sqrt(N);


%%  save('./SavedSimulations/JeiriTrajectories-60.mat', 'Jei_time', ...
%      'iRateT', 'mean_Jeiri','Jei_IC');

 
