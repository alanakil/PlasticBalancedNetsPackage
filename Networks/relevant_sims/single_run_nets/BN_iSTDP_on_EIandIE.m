%% Simulation of a balanced network with plastic I->E and E->I synapses.
%%% Both synapses can change as inhibitory STDP (Vogels et al 2011). 
%%% E->I can also be run as simple Hebbian STDP too.

clear

rng(1);

% Number of neurons in each population
N = 5000;
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
c=0;

% Timescale of correlation
taujitter=5;
% Mean connection strengths between each cell type pair
Jm=[25 -100; 112.5 -250]/sqrt(N);
Jxm=[180; 135]/sqrt(N);

% Time (in ms) for sim
T=1000;

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
Istim(time>0)=0;
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

% Plasticity EI params
Jmax_ei = -200/sqrt(N);
eta_ei=0.0001/Jmax_ei; % Learning rate
tauSTDP=200;
rho1=0.010; % Target E rate 10Hz
alpha1=2*rho1*tauSTDP;

% Plasticity ie parameters. 
Jmax1 = 200/sqrt(N);
eta_ie_homeo=0.0001/Jmax1; % Learning rate
rho2=0.020; % Target I rate 20Hz
alpha2=2*rho2*tauSTDP;

eta_ie_hebb=0; % 0.001; % Learning rate
Jmax_ie = 140/sqrt(N);


% Random initial voltages
V0=rand(N,1)*(VT-Vre)+Vre;

% Maximum number of spikes for all neurons
% in simulation. Make it 50Hz across all neurons
% If there are more spikes, the simulation will
% terminate
maxns=ceil(.05*N*T); % was 0.05.

% Indices of neurons to record currents, voltages
nrecord0=100; % Number to record from each population
Irecord=[randperm(Ne,nrecord0) randperm(Ni,nrecord0)+Ne];
numrecord=numel(Irecord); % total number to record


% Synaptic weights I->E to record.
% The first row of Jrecord is the postsynaptic indices
% The second row is the presynaptic indices
nJrecord0_EI=2000; % Number to record
[II,JJ]=find(J(1:Ne,Ne+1:N)); % Find non-zero I to E weights
III=randperm(numel(II),nJrecord0_EI); % Choose some at random to record
II=II(III);
JJ=JJ(III);
Jrecord_EI=[II JJ+Ne]'; % Record these
numrecordJ1=size(Jrecord_EI,2);
if(size(Jrecord_EI,1)~=2)
    error('Jrecord must be 2xnumrecordJ');
end

% Synaptic weights II to record.
% The first row of Jrecord is the postsynaptic indices
% The second row is the presynaptic indices
nJrecord0_IE=500; % Number of ii synapses to record
[II,JJ]=find(J(Ne+1:N,1:Ne)); % Find non-zero i to i weights
III=randperm(numel(II),nJrecord0_IE); % Choose some at random to record
II=II(III);
JJ=JJ(III);
Jrecord_IE=[II+Ne JJ]'; % Record these
numrecordJ=size(Jrecord_IE,2);
if(size(Jrecord_IE,1)~=2)
    error('Jrecord must be 2xnumrecordJ');
end

% Number of time bins to average over when recording
nBinsRecord=1000;
dtRecord=nBinsRecord*dt;
timeRecord=dtRecord:dtRecord:T;
Ntrec=numel(timeRecord);

% Integer division function
IntDivide=@(n,k)(floor((n-1)/k)+1);

V=V0;
Ie=zeros(N,1);
Ii=zeros(N,1);
Ix=zeros(N,1);
IeRecord=zeros(N,1);
IiRecord=zeros(N,1);
IxRecord=zeros(N,1);
Ie_Record_Wholetime = zeros(1,T/dt);
Ii_Record_Wholetime = zeros(1,T/dt);

x=zeros(N,1);
conv_Spike=zeros(N,1);
conv_Spike_X=zeros(Nx,1);
sum_conv_Spike = zeros(N,1);
sum_conv_Spike_X = zeros(Nx,1);


IeRec=zeros(numrecord,Ntrec);
IiRec=zeros(numrecord,Ntrec);
IxRec=zeros(numrecord,Ntrec);
% VRec=zeros(numrecord,Ntrec);
% wRec=zeros(numrecord,Ntrec);
JRec_EI=zeros(1,Ntrec);
JRec_IE=zeros(1,Ntrec);
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
        
        % If there is I->E plasticity
        if(eta_ei~=0)
            % Update synaptic weights according to plasticity rules
            % I to E after an I spike    
            J(1:Ne,Ispike(Ispike>Ne))=J(1:Ne,Ispike(Ispike>Ne))+ ... 
                -repmat(eta_ei*(x(1:Ne)-alpha1),1,nnz(Ispike>Ne)).*(J(1:Ne,Ispike(Ispike>Ne)));
            % I to E after an E spike
            J(Ispike(Ispike<=Ne),Ne+1:N)=J(Ispike(Ispike<=Ne),Ne+1:N)+ ... 
                -repmat(eta_ei*x(Ne+1:N)',nnz(Ispike<=Ne),1).*(J(Ispike(Ispike<=Ne),Ne+1:N));
        end
        % If there is E->I Homeostatic plasticity
        if(eta_ie_homeo~=0)
            % Update synaptic weights according to plasticity rules
            % after an E spike    
            J(Ne+1:N,Ispike(Ispike<=Ne))=J(Ne+1:N,Ispike(Ispike<=Ne))+ ... 
                +repmat(eta_ie_homeo*(x(Ne+1:N)-alpha2),1,nnz(Ispike<=Ne)).*(J(Ne+1:N,Ispike(Ispike<=Ne)));
            % I to E after an I spike
            J(Ispike(Ispike>Ne),1:Ne)=J(Ispike(Ispike>Ne),1:Ne)+ ... 
                +repmat(eta_ie_homeo*x(1:Ne)',nnz(Ispike>Ne),1).*(J(Ispike(Ispike>Ne),1:Ne));
        end
        % If there is E->I *Hebbian* plasticity
        if(eta_ie_hebb~=0)
            %Update synaptic weights according to plasticity rules
            %E to E after presynaptic spike
            J(Ne+1:N,Ispike(Ispike<=Ne))=J(Ne+1:N,Ispike(Ispike<=Ne))+ ...
                -repmat(eta_ie_hebb*(x(Ne+1:N)),1,nnz(Ispike<=Ne)).*(J(Ne+1:N,Ispike(Ispike<=Ne)));
            %E to E after a postsynaptic spike
            J(Ispike(Ispike>Ne),1:Ne)=J(Ispike(Ispike>Ne),1:Ne)+ ...
                +repmat(eta_ie_hebb*x(1:Ne)',nnz(Ispike>Ne),1).*(Jmax_ie).*(J(Ispike(Ispike>Ne),1:Ne)~=0);
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
    
%     % Store recorded variables
    ii=IntDivide(i,nBinsRecord); 
    IeRec(:,ii)=IeRec(:,ii)+Ie(Irecord);
    IiRec(:,ii)=IiRec(:,ii)+Ii(Irecord);
    IxRec(:,ii)=IxRec(:,ii)+Ix(Irecord);
%     VRec(:,ii)=VRec(:,ii)+V(Irecord);
    JRec_EI(:,ii)= mean(J(sub2ind(size(J),Jrecord_IE(1,:),Jrecord_IE(2,:))));
    JRec_IE(:,ii)= mean(J(sub2ind(size(J),Jrecord_EI(1,:),Jrecord_EI(2,:))));


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
% VRec=VRec/nBinsRecord;

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
plot(s(1,s(2,:)<1000),s(2,s(2,:)<1000),'k.','Markersize',0.5)
xlabel('time (ms)')
ylabel('Neuron index')

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
plot((dtRate:dtRate:T)/1000,1000*eRateT, 'linewidth',3)
hold on
plot((dtRate:dtRate:T)/1000,1000*iRateT, 'linewidth',3)
legend('r_e','r_i')
ylabel('rate (Hz)')
xlabel('time (s)')


%% If plastic, plot mean connection weights over time
if(eta_ei~=0)
    figure
    plot(timeRecord/1000,JRec_EI*sqrt(N), 'linewidth',3)
    
    xlabel('time (s)')
    ylabel('Mean I to E synaptic weight')
   
    figure;
    histogram(nonzeros(J(1:Ne,Ne+1:N)*sqrt(N)))
    xlabel('j_{ei}')
    ylabel('Count')
end
if(eta_ie_homeo~=0)
    figure
    plot(timeRecord/1000,JRec_IE*sqrt(N), 'linewidth',3)
    
    xlabel('time (s)')
    ylabel('Mean E to I synaptic weight')
   
    figure;
    histogram(nonzeros(J(Ne+1:N,1:Ne)*sqrt(N)))
    xlabel('j_{ie}')
    ylabel('Count')
end



%%
%%% All the code below computes spike count covariances and correlations
%%% We want to compare the resulting covariances to what is predicted by
%%% the theory.

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
    mR=[mRee mRei; mRei mRii]


end

%% Compute Cv of ISI.
% Column 1 has spike times and column 2 the neuron index
ComputeCV=1;
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
    
    % CV of ISI of e cells.
    ISI=cell(Ne,1);
    sigma=zeros(Ne,1);
    mu=zeros(Ne,1);
    for i=1:Ne
        ISI{i} = diff(spikes{i}(:),1,1);
        sigma(i) = std(ISI{i}(:));
        mu(i) = mean(ISI{i}(:));
    end
    tSim=toc;
    disp(sprintf('\nTime for CV: %.2f min',tSim/60))
    
    CV_ISI = sigma./mu;
    CV_ISI = CV_ISI(~isnan(CV_ISI));
    CV_ISI_e_cells = CV_ISI(CV_ISI~=0);
    
    MEAN_CV_ISI_e_cells = mean(CV_ISI_e_cells)
    
    % CV ISI of i cells.
    ISI=cell(Ni,1);
    sigma=zeros(Ni,1);
    mu=zeros(Ni,1);
    for i=1:Ni
        ISI{i} = diff(spikes{Ne+i}(:),1,1);
        sigma(i) = std(ISI{i}(:));
        mu(i) = mean(ISI{i}(:));
    end
    tSim=toc;
    disp(sprintf('\nTime for CV: %.2f min',tSim/60))
    
    CV_ISI = sigma./mu;
    CV_ISI = CV_ISI(~isnan(CV_ISI));
    CV_ISI_i_cells = CV_ISI(CV_ISI~=0);
    
%     % Plot distribution of ISI's.
%     figure;
%     histogram(CV_ISI,1000)
    
    MEAN_CV_ISI_i_cells = mean(CV_ISI_i_cells)
    
end


%% Compute including the X external input.
% Compute cov(J,S) to check if this is the issue with weights.
% Use the final steady state weights and the final currents: Ii, Ie.
mean_conv_Spike = sum_conv_Spike/(T/dt/2);
mean_conv_Spike_X = sum_conv_Spike_X/(T/dt/2);

Mean_J_times_Si = zeros(N,1);
Mean_JSi = zeros(N,1);
for i = 1:N
    Mean_J_times_Si(i) = mean( J(i,Ne+1:N) .* mean_conv_Spike(Ne+1:N)' ); %+ mean( Jx(i,:) .* mean_conv_Spike_X(:)' );
    Mean_JSi(i) =  mean( J(i,Ne+1:N) ) * mean(mean_conv_Spike(Ne+1:N)'); %+ mean( Jx(i,:) ) * mean(mean_conv_Spike_X(:)');
end
figure; hold on
plot(Mean_J_times_Si)
plot(Mean_JSi)
plot(Mean_J_times_Si - Mean_JSi, 'linewidth',1)
xlabel('N','fontsize',15)

leg = legend('mean(J*Si)','mean(J)*mean(Si)','cov(J,Si)');
leg.FontSize=11;

mean(Mean_J_times_Si)
mean(Mean_JSi)
mean(Mean_J_times_Si) - mean(Mean_JSi)
-100*(mean(Mean_J_times_Si) - mean(Mean_JSi))/mean(Mean_J_times_Si)

%% Compute correlations between I to E synapses over time.
tic
Corr_IE_syn = corr(JRec_EI(:,T/dt/nBinsRecord/2:end)');
toc
colvect= Corr_IE_syn(find(~tril(ones(size(Corr_IE_syn)))));
Avg_Corr_IE = mean( colvect )


%% Check currents.
figure; hold on
plot(mean(IeRec))
plot(mean(IiRec))
plot(mean(IxRec))
plot(mean(IeRec)+mean(IiRec)+mean(IxRec))

IeRec1 = mean(IeRec);
IiRec2 = mean(IiRec);
IxRec3 = mean(IxRec);
%% Compute <Jr> and <J><r>.
% Accounting for ALL inputs, not only recurrent!!!
plot(IeRecord(1:N)+IiRecord(1:N)+IxRecord(1:N))
R_e_sims = mean(IeRecord(1:Ne)+IiRecord(1:Ne)+IxRecord(1:Ne))
R_i_sims = mean(IeRecord(Ne+1:N)+IiRecord(Ne+1:N)+IxRecord(Ne+1:N))

Jie_sims = sqrt(N)*mean(JRec_EI(length(JRec_EI)/2:end))
Jei_sims = sqrt(N)*mean(JRec_IE(length(JRec_IE)/2:end))

W(2,2) = 0.1*Jie_sims*0.2;
W(1,2) = 0.1*Jei_sims*0.2;
eFR = mean(reSim); iFR = mean(riSim);
R_e_theory = (W(1,1) * eFR + W(1,2) * iFR + Wx(1,1) * rx)*sqrt(N)
R_i_theory = (W(2,1) * eFR + W(2,2) * iFR + Wx(2,1) * rx)*sqrt(N)


Diff_e = -R_e_theory + R_e_sims
Diff_i = -R_i_theory + R_i_sims
