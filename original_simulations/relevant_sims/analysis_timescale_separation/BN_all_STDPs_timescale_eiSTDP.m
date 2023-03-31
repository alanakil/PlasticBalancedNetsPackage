%% Simulation of a balanced network with different types of synaptic plasticity.
%%% This code simulates a tightly balanced spiking network that may
%%% experience STDP on any of its synapses (EE,EI,IE,II). 
%%% Types of STDPs available: Classical Hebbian (hard constraints),
%%% Classical Hebbian (soft constraints), Kohonen's rule, homeostatic 
%%% inhibitory plasticity.
% Authors: Alan Akil, Robert Rosenbaum, and Kre?imir Josi?.
% Publication: "Balanced Networks Under Spike-Timing Dependent Plasticity"
% Date published: March 2021.

clear

seed = 223;
rng(seed);

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

% Correlation between the spike trains in the ffwd layer
c=0;
% Timescale of correlation
taujitter=5;

% Mean connection strengths between each cell type pair
Jm=[25 -150; 112.5 -250]/sqrt(N);
Jxm=[180; 135]/sqrt(N);

% Time (in ms) for sim
T=5000000;

% Time discretization
dt=.1; %ms

% Proportion of neurons in each population.
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


% Neuron parameters
Cm=1;
gL=1/15;
EL=-72;
Vth=-50;
Vre=-75;
DeltaT=1;
VT=-55;

%%%%% Plasticity params %%%%%%
tauSTDP=200; % Timescale of eligibility trace

% EE synapses - hebbian rule with soft constraints
Jmax_ee_soft = 30/sqrt(N); % Soft constraint - also the fixed point of EE weights
eta_ee_soft=0/1000; % Learning rate of EE

% EE synapses - hebbian rule with hard constraints
Jmax_ee_hard = 1000/sqrt(N); % Upper constraint
Jmin_ee_hard = 0/sqrt(N); % Lower constraint
eta_ee_hard=0/1000; % Learning rate of EE

% EE synapses - Kohonen's rule (needs no constraints)
beta = 2/sqrt(N);
eta_ee_kohonen=0/10; % Learning rate of EE

% EI synapses
Jmax_ei = -200/sqrt(N); % arbitrary normalizing constant (same order as Jei)
eta_ei=1/10000/Jmax_ei; % Learning rate of EI
rho0=0.010; % Target rate for e cells
alpha_e=2*rho0*tauSTDP;

% IE synapses - note that this can be hebbian or homeostatic
Jmax_ie = 200/sqrt(N);
eta_ie_homeo=0.000/Jmax_ie; % Learning rate of IE
rho0=0.023; % Target rate for i cells.
alpha_ie=2*rho0*tauSTDP;
% IE Hebbian
eta_ie_hebbian = 0/1000;
Jmax_ie = 112.5/sqrt(N); % only if hebbian

% II synapses
Jmax_ii = -200/sqrt(N);
eta_ii=0.000/Jmax_ii; % Learning rate of EI
rho0=0.023; % Target rate for i cells.
alpha_ii=2*rho0*tauSTDP;

%%% Note: If IE or II homeostatic plasticity are ON, then the network also
%%% needs EI homeostatic STDP ON to maintain stability.

% Number of time bins to average over when recording
nBinsRecord=100;
dtRecord=nBinsRecord*dt;
timeRecord=dtRecord:dtRecord:T;
Ntrec=numel(timeRecord);

eta_ei_iSTDP_vector = [1/10000,1/1000,1/100,1/10,1]/Jmax_ei;

mean_Jei_recorded = zeros(length(eta_ei_iSTDP_vector),Ntrec);
dtRate=100; % ms
eRateT = zeros(length(eta_ei_iSTDP_vector),length(1:dtRate:T));
iRateT = zeros(length(eta_ei_iSTDP_vector),length(1:dtRate:T));
winsize=250; 
T1=T/2; % Burn-in period
T2=T;   % Compute covariances until end of simulation
Ie_current_eta = zeros(length(eta_ei_iSTDP_vector),Ntrec);
Ii_current_eta = zeros(length(eta_ei_iSTDP_vector),Ntrec);
Ix_current_eta = zeros(length(eta_ei_iSTDP_vector),Ntrec);


parfor eta_indx = 1:length(eta_ei_iSTDP_vector)
    
    eta_ei = eta_ei_iSTDP_vector(eta_indx);
    
    J_NeNx = binornd(1,P(1,1),Ne,Nx);
    J_NiNx = binornd(1,P(1,1),Ni,Nx);
    
    % Generate full connectivity matrices
    tic
    J=[Jm(1,1)*binornd(1,P(1,1),Ne,Ne) Jm(1,2)*binornd(1,P(1,1),Ne,Ni); ...
        Jm(2,1)*binornd(1,P(1,1),Ni,Ne) Jm(2,2)*binornd(1,P(1,1),Ni,Ni)];
    Jx=[J_NeNx.*Jxm(1); J_NiNx.*Jxm(2)];
    
    tGen=toc;
    disp(sprintf('\nTime to generate connections: %.2f sec',tGen))
    
    
    %%% Make (correlated) Poisson spike times for ffwd layer
    %%% See Appendix 1 of Akil et al 2020 for more details
    tic
    if(c<1e-5) % If uncorrelated
        nspikeX=poissrnd(Nx*rx*T);
        st=rand(nspikeX,1)*T;
        sx=zeros(2,numel(st));
        sx(1,:)=sort(st);
        sx(2,:)=randi(Nx,1,numel(st)); % neuron indices
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
    
    
    % Random initial voltages
    V0=rand(N,1)*(VT-Vre)+Vre;
    
    % Maximum number of spikes for all neurons
    % in simulation. Make it 50Hz across all neurons
    % If there are more spikes, the simulation will terminate
    maxns=ceil(.05*N*T); % was 0.05.
    
    % Indices of neurons to record currents, voltages
    nrecord0=100; % Number to record from each population
    Irecord=[randperm(Ne,nrecord0) randperm(Ni,nrecord0)+Ne];
    numrecord=numel(Irecord); % total number to record
    
    % Synaptic weights EE to record.
    % The first row of Jrecord is the postsynaptic indices
    % The second row is the presynaptic indices
    nJrecord1=1000; % Number to record
    [II1,JJ1]=find(J(1:Ne,1:Ne)); % Find non-zero E to E weights of stim neurons
    III1=randperm(numel(II1),nJrecord1); % Choose some at random to record
    II1=II1(III1);
    JJ1=JJ1(III1);
    Jrecord_ee=[II1 JJ1]'; % Record these
    numrecordJ_ee=size(Jrecord_ee,2);
    if(size(Jrecord_ee,1)~=2)
        error('Jrecord must be 2xnumrecordJ');
    end
    
    % Synaptic weights EI to record.
    % The first row of Jrecord is the postsynaptic indices
    % The second row is the presynaptic indices
    nJrecord0=1000; % Number to record
    [II,JJ]=find(J(1:Ne,Ne+1:N)); % Find non-zero I to E weights of stim neurons
    III=randperm(numel(II),nJrecord0); % Choose some at random to record
    II=II(III);
    JJ=JJ(III);
    Jrecord_ei=[II JJ+Ne]'; % Record these
    numrecordJ_ei=size(Jrecord_ei,2);
    if(size(Jrecord_ei,1)~=2)
        error('Jrecord must be 2xnumrecordJ');
    end
    
    % Synaptic weights IE to record.
    % The first row of Jrecord is the postsynaptic indices
    % The second row is the presynaptic indices
    nJrecord0=1000; % Number to record
    [II,JJ]=find(J(Ne+1:N,1:Ne)); % Find non-zero E to I weights of stim neurons
    III=randperm(numel(II),nJrecord0); % Choose some at random to record
    II=II(III);
    JJ=JJ(III);
    Jrecord_ie=[II+Ne JJ]'; % Record these
    numrecordJ_ie=size(Jrecord_ie,2);
    if(size(Jrecord_ie,1)~=2)
        error('Jrecord must be 2xnumrecordJ');
    end
    
    % Synaptic weights II to record.
    % The first row of Jrecord is the postsynaptic indices
    % The second row is the presynaptic indices
    nJrecord0=1000; % Number to record
    [II,JJ]=find(J(Ne+1:N,Ne+1:N)); % Find non-zero I to I weights of stim neurons
    III=randperm(numel(II),nJrecord0); % Choose some at random to record
    II=II(III);
    JJ=JJ(III);
    Jrecord_ii=[II+Ne JJ+Ne]'; % Record these
    numrecordJ_ii=size(Jrecord_ii,2);
    if(size(Jrecord_ii,1)~=2)
        error('Jrecord must be 2xnumrecordJ');
    end
    
    % Integer division function
    IntDivide=@(n,k)(floor((n-1)/k)+1);
    
    % Set initial voltage
    V=V0;
    
    % Preallocate memory
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
    JRec_ee=zeros(1,Ntrec);
    JRec_ei=zeros(1,Ntrec);
    JRec_ie=zeros(1,Ntrec);
    JRec_ii=zeros(1,Ntrec);
    JRec_ee_std=zeros(1,Ntrec);
    JRec_ei_std=zeros(1,Ntrec);
    JRec_ie_std=zeros(1,Ntrec);
    JRec_ii_std=zeros(1,Ntrec);
    
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
            
            
            
            % If there is EE Hebbian plasticity with soft constraints
            if(eta_ee_soft~=0)
                %Update synaptic weights according to plasticity rules
                %E to E after presynaptic spike
                J(1:Ne,Ispike(Ispike<=Ne))=J(1:Ne,Ispike(Ispike<=Ne))+ ...
                    -repmat(eta_ee_soft*(x(1:Ne)),1,nnz(Ispike<=Ne)).*(J(1:Ne,Ispike(Ispike<=Ne)));
                %E to E after a postsynaptic spike
                J(Ispike(Ispike<=Ne),1:Ne)=J(Ispike(Ispike<=Ne),1:Ne)+ ...
                    +repmat(eta_ee_soft*x(1:Ne)',nnz(Ispike<=Ne),1).*(Jmax_ee_soft).*(J(Ispike(Ispike<=Ne),1:Ne)~=0);
            end
            
            % If there is EE Hebbian plasticity with hard constraints
            if(eta_ee_hard~=0)
                %Update synaptic weights according to plasticity rules
                %E to E after presynaptic spike
                J(1:Ne,Ispike(Ispike<=Ne))=J(1:Ne,Ispike(Ispike<=Ne))+ ...
                    -repmat(eta_ee_hard*(x(1:Ne)),1,nnz(Ispike<=Ne)).*(J(1:Ne,Ispike(Ispike<=Ne))>Jmin_ee_hard);
                %E to E after a postsynaptic spike
                J(Ispike(Ispike<=Ne),1:Ne)=J(Ispike(Ispike<=Ne),1:Ne)+ ...
                    +repmat(eta_ee_hard*x(1:Ne)',nnz(Ispike<=Ne),1)...
                    .*(J(Ispike(Ispike<=Ne),1:Ne)>Jmin_ee_hard).*(J(Ispike(Ispike<=Ne),1:Ne)<Jmax_ee_hard);
            end
            
            % If there is EE Kohonen plasticity
            if(eta_ee_kohonen~=0)
                % Update synaptic weights according to plasticity rules
                % E to E after presynaptic spike
                J(1:Ne,Ispike(Ispike<=Ne))=J(1:Ne,Ispike(Ispike<=Ne))+ ...
                    +beta*repmat(eta_ee_kohonen*(x(1:Ne)),1,nnz(Ispike<=Ne)).*(J(1:Ne,Ispike(Ispike<=Ne))~=0);
                % E to E after a postsynaptic spike
                J(Ispike(Ispike<=Ne),1:Ne)=J(Ispike(Ispike<=Ne),1:Ne)+ ...
                    -eta_ee_kohonen*(J(Ispike(Ispike<=Ne),1:Ne));
            end
            
            % If there is EI plasticity
            if(eta_ei~=0)
                % Update synaptic weights according to plasticity rules
                % I to E after an I spike
                J(1:Ne,Ispike(Ispike>Ne))=J(1:Ne,Ispike(Ispike>Ne))+ ...
                    -repmat(eta_ei*(x(1:Ne)-alpha_e),1,nnz(Ispike>Ne)).*(J(1:Ne,Ispike(Ispike>Ne)));
                % E to I after an E spike
                J(Ispike(Ispike<=Ne),Ne+1:N)=J(Ispike(Ispike<=Ne),Ne+1:N)+ ...
                    -repmat(eta_ei*x(Ne+1:N)',nnz(Ispike<=Ne),1).*(J(Ispike(Ispike<=Ne),Ne+1:N));
            end
            
            % If there is IE *Homeostatic* plasticity
            if(eta_ie_homeo~=0)
                % Update synaptic weights according to plasticity rules
                % after an E spike
                J(Ne+1:N,Ispike(Ispike<=Ne))=J(Ne+1:N,Ispike(Ispike<=Ne))+ ...
                    -repmat(eta_ie_homeo*(x(Ne+1:N)-alpha_ie),1,nnz(Ispike<=Ne)).*(J(Ne+1:N,Ispike(Ispike<=Ne)));
                % I to E after an I spike
                J(Ispike(Ispike>Ne),1:Ne)=J(Ispike(Ispike>Ne),1:Ne)+ ...
                    -repmat(eta_ie_homeo*x(1:Ne)',nnz(Ispike>Ne),1).*(J(Ispike(Ispike>Ne),1:Ne));
            end
            
            % If there is IE *Hebbian* plasticity
            if(eta_ie_hebbian~=0)
                %Update synaptic weights according to plasticity rules
                %E to E after presynaptic spike
                J(Ne+1:N,Ispike(Ispike<=Ne))=J(Ne+1:N,Ispike(Ispike<=Ne))+ ...
                    -repmat(eta_ie_hebbian*(x(Ne+1:N)),1,nnz(Ispike<=Ne)).*(J(Ne+1:N,Ispike(Ispike<=Ne)));
                %E to E after a postsynaptic spike
                J(Ispike(Ispike>Ne),1:Ne)=J(Ispike(Ispike>Ne),1:Ne)+ ...
                    +repmat(eta_ie_hebbian*x(1:Ne)',nnz(Ispike>Ne),1).*(Jmax_ie).*(J(Ispike(Ispike>Ne),1:Ne)~=0);
            end
            
            % If there is II Homeostatic plasticity
            if(eta_ii~=0)
                % Update synaptic weights according to plasticity rules
                % I to I after a presyanptic spike
                J(Ne+1:N,Ispike(Ispike>Ne))=J(Ne+1:N,Ispike(Ispike>Ne))+ ...
                    -repmat(eta_ii*(x(Ne+1:N)-alpha_ii),1,nnz(Ispike>Ne)).*(J(Ne+1:N,Ispike(Ispike>Ne)));
                % I to I after a postsynaptic spike
                J(Ispike(Ispike>Ne),Ne+1:N)=J(Ispike(Ispike>Ne),Ne+1:N)+ ...
                    -repmat(eta_ii*x(Ne+1:N)',nnz(Ispike>Ne),1).*(J(Ispike(Ispike>Ne),Ne+1:N));
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
        
        % Update time-dependent eligibility traces for plasticity
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
        %     VRec(:,ii)=VRec(:,ii)+V(Irecord);
        JRec_ee(1,ii)=mean(J(sub2ind(size(J),Jrecord_ee(1,:),Jrecord_ee(2,:))));
        JRec_ei(1,ii)=mean(J(sub2ind(size(J),Jrecord_ei(1,:),Jrecord_ei(2,:))));
        JRec_ie(1,ii)=mean(J(sub2ind(size(J),Jrecord_ie(1,:),Jrecord_ie(2,:))));
        JRec_ii(1,ii)=mean(J(sub2ind(size(J),Jrecord_ii(1,:),Jrecord_ii(2,:))));
        JRec_ee_std(1,ii)=1/sqrt(nJrecord0) *sqrt(N)* std(J(sub2ind(size(J),Jrecord_ee(1,:),Jrecord_ee(2,:))));
        JRec_ei_std(1,ii)=1/sqrt(nJrecord0) *sqrt(N)* std(J(sub2ind(size(J),Jrecord_ei(1,:),Jrecord_ei(2,:))));
        JRec_ie_std(1,ii)=1/sqrt(nJrecord0) *sqrt(N)* std(J(sub2ind(size(J),Jrecord_ie(1,:),Jrecord_ie(2,:))));
        JRec_ii_std(1,ii)=1/sqrt(nJrecord0) *sqrt(N)* std(J(sub2ind(size(J),Jrecord_ii(1,:),Jrecord_ii(2,:))));
        
        % Reset membrane potential
        V(Ispike)=Vre;
        
        % Record an average currents to compare <Jr> and <J><r>.
        if i>=T/2/dt
            sum_conv_Spike = sum_conv_Spike + conv_Spike(:,1);
            sum_conv_Spike_X = sum_conv_Spike_X + conv_Spike_X(:,1);
            IeRecord = IeRecord + Ie;
            IiRecord = IiRecord + Ii;
            IxRecord = IxRecord + Ix;
        end
        % Mean time dependent currents
        Ie_Record_Wholetime(1,i) = mean(Ie(1:Ne));
        Ii_Record_Wholetime(1,i) = mean(Ii(1:Ne));
        if mod(i,T/5) == 0 % print every T/5 iterations.
            fprintf('At time %d...\n',i*dt/T);
        end
    end
    % Normalize recorded variables by # bins
    IeRec=IeRec/nBinsRecord;
    IiRec=IiRec/nBinsRecord;
    IxRec=IxRec/nBinsRecord;
    % VRec=VRec/nBinsRecord;
    
    % did computation online, so now divide to get the average.
    IeRecord = IeRecord/(numel(time)/2);
    IiRecord = IiRecord/(numel(time)/2);
    IxRecord = IxRecord/(numel(time)/2);
        
    % Get rid of padding in s
    s=s(:,1:nspike);
    tSim=toc;
    disp(sprintf('\nTime for simulation: %.2f min',tSim/60))
    
    % Record variables needed: J, rate, covs, corrs, currents
    mean_Jei_recorded(eta_indx,:) = JRec_ei*sqrt(N);
    
    % Time-dependent mean rates
    eRateT(eta_indx,:)=hist(s(1,s(2,:)<=Ne),1:dtRate:T)/(dtRate*Ne);
    iRateT(eta_indx,:)=hist(s(1,s(2,:)>Ne),1:dtRate:T)/(dtRate*Ni);
    
    % Compute spike count covariances over windows of size
    % winsize starting at time T1 and ending at time T2.
    C=SpikeCountCov(s,N,T1,T2,winsize);
    
    % Get mean spike count covariances over each sub-pop
    [II,JJ]=meshgrid(1:N,1:N);
    mCee=mean(C(II<=Ne & JJ<=II));
    mCei=mean(C(II<=Ne & JJ>Ne));
    mCii=mean(C(II>Ne & JJ>II));
    
    % Get variances too.
    mVee=mean(C(II<=Ne & JJ==II));
    mVii=mean(C(II>Ne & JJ==II));
    
    % Mean-field spike count cov matrix
    mC{eta_indx}=[mCee mCei; mCei mCii];
    
    % Mean-field spike count variances.
    mV{eta_indx}=[mVee, mVii];
    
    % Compute spike count correlations
    % Get correlation matrix from cov matrix
    tic
    R=corrcov(C);
    toc
    mRee=mean(R(II<=Ne & JJ<=II & isfinite(R)));
    mRei=mean(R(II<=Ne & JJ>Ne & isfinite(R)));
    mRii=mean(R(II>Ne & JJ>II & isfinite(R)));

    % Mean-field spike count correlation matrix
    mR{eta_indx}=[mRee mRei; mRei mRii];

    % Save mean E, I, X currents to check for balance.
    Ie_current_eta(eta_indx,:) = mean(IeRec);
    Ii_current_eta(eta_indx,:) = mean(IiRec);
    Ix_current_eta(eta_indx,:) = mean(IxRec);
    
end
%% Save necessary variables.

save('/scratch/AlanAkil/timescale_analysis_eiSTDP.mat', 'mean_Jei_recorded', 'mC','mV',...
    'mR','Ie_current_eta','Ii_current_eta','Ix_current_eta','eta_ei_iSTDP_vector',...
    'N','T','dt','c','dtRate','Jm','Jxm','seed','T1','T2','tauSTDP',...
    'winsize','W','Wx','rx','eRateT','iRateT','Jmax_ei');

% figure;hold on;
% for i=1:length(eta_ee_kohonen_vector)
%     plot(mean_Jee_recorded(i,:), 'linewidth',2)
% end
% legend('eta1','eta2','eta3','eta4','eta5','eta6','eta7','eta8')

%% Mean rate of each neuron (excluding burn-in period)
% Tburn=T/2;
% reSim=hist(s(2,s(1,:)>Tburn & s(2,:)<=Ne),1:Ne)/(T-Tburn);
% riSim=hist(s(2,s(1,:)>Tburn & s(2,:)>Ne)-Ne,1:Ni)/(T-Tburn);
% 
% % Mean rate over E and I pops
% reMean=mean(reSim);
% riMean=mean(riSim);
% disp(sprintf('\nMean E and I rates from sims: %.2fHz %.2fHz',1000*reMean,1000*riMean))

%% If plastic, plot mean connection weights over time
% if(eta_ee_soft~=0 || eta_ee_hard~=0 || eta_ee_kohonen~=0)
%     figure
%     plot(timeRecord/1000,(JRec_ee)*sqrt(N), 'linewidth',3)
%     xlabel('time (s)')
%     ylabel('Mean E to E synaptic weight')
%    
%     figure;
%     histogram(nonzeros(J(1:Ne,1:Ne)*sqrt(N)))
%     xlabel('j_{EE}')
%     ylabel('Count')
% end
% 
% if(eta_ei~=0)
%     figure
%     plot(timeRecord/1000,(JRec_ei)*sqrt(N), 'linewidth',3)
%     xlabel('time (s)')
%     ylabel('Mean I to E synaptic weight')
%    
%     figure;
%     histogram(nonzeros(J(1:Ne,Ne+1:N)*sqrt(N)))
%     xlabel('j_{EI}')
%     ylabel('Count')
% end
% 
% if(eta_ie_hebbian~=0 || eta_ie_homeo~=0)
%     figure
%     plot(timeRecord/1000,(JRec_ie)*sqrt(N), 'linewidth',3)
%     xlabel('time (s)')
%     ylabel('Mean E to I synaptic weight')
%    
%     figure;
%     histogram(nonzeros(J(Ne+1:N,1:Ne)*sqrt(N)))
%     xlabel('j_{IE}')
%     ylabel('Count')
% end
% 
% if(eta_ii~=0)
%     figure
%     plot(timeRecord/1000,(JRec_ii)*sqrt(N), 'linewidth',3)
%     xlabel('time (s)')
%     ylabel('Mean I to I synaptic weight')
%    
%     figure;
%     histogram(nonzeros(J(Ne+1:N,Ne+1:N)*sqrt(N)))
%     xlabel('j_{II}')
%     ylabel('Count')
% end
%% Compute spike count covariances and correlations


%% Compute CV of ISI, if desired.
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

%% Check E, I, X currents for balance.
% figure; hold on
% plot(mean(IeRec))
% plot(mean(IiRec))
% plot(mean(IxRec))
% plot(mean(IeRec)+mean(IiRec)+mean(IxRec))
% 
% IeRec1 = mean(IeRec);
% IiRec2 = mean(IiRec);
% IxRec3 = mean(IxRec);
