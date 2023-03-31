%% Simulation of a balanced network with different types of synaptic plasticity.
%% Here we analyze dynamics in the limit of large N.
%%% This code simulates a tightly balanced spiking network that may
%%% experience STDP on any of its synapses (EE,EI,IE,II). 
%%% Types of STDPs available: Classical Hebbian (hard constraints),
%%% Classical Hebbian (soft constraints), Kohonen's rule, homeostatic 
%%% inhibitory plasticity.
% Authors: Alan Akil, Robert Rosenbaum, and Kre?imir Josi?.
% Publication: "Balanced Networks Under Spike-Timing Dependent Plasticity"
% Date published: March 2021.

clear

seed = 2538;
rng(seed);

num_network_sizes = 8; % Number of network sizes.
num_realizations = 5; % Number of realizations of networks of fixed size.

% Recurrent net connection probabilities
P=[0.1 0.1; 0.1 0.1];

% Ffwd connection probs
Px=[.1; .1];

% Correlation between the spike trains in the ffwd layer
c=0;
% Timescale of correlation
taujitter=5;

% Time (in ms) for sim
T=3000000;

% Time discretization
dt=.1; % ms

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

% Integer division function
IntDivide=@(n,k)(floor((n-1)/k)+1);

%%%%% Plasticity params %%%%%%
tauSTDP=200; % Timescale of eligibility trace

% Number of time bins to average over when recording
nBinsRecord=100;
dtRecord=nBinsRecord*dt;
timeRecord=dtRecord:dtRecord:T;
Ntrec=numel(timeRecord);

% Size of counting window
winsize=250; 
T1=T/2; % Burn-in period
T2=T;   % Compute covariances until end of simulation
Tburn=T/2; % Burn-in period for rate computation.
dtRate=100; % ms. Window of rates.

% Define vector of network sizes.
N_vector = round( 10.^(linspace(3,4,num_network_sizes)),-1 );

Tburn_raster=T-10000; % Look at last 10 seconds of sims.

% Preallocate memory.
eRateT = cell(num_network_sizes, num_realizations);
iRateT = cell(num_network_sizes, num_realizations);
mean_Jee_recorded = cell(num_network_sizes, num_realizations);
mean_Jei_recorded = cell(num_network_sizes, num_realizations);
mean_Jie_recorded = cell(num_network_sizes, num_realizations);
mean_Jii_recorded = cell(num_network_sizes, num_realizations);
mCee = zeros(num_network_sizes,num_realizations);
mCei = zeros(num_network_sizes,num_realizations);
mCii = zeros(num_network_sizes,num_realizations);
mRee = zeros(num_network_sizes,num_realizations);
mRei = zeros(num_network_sizes,num_realizations);
mRii = zeros(num_network_sizes,num_realizations);
mean_E_current = cell(num_network_sizes,num_realizations);
mean_I_current = cell(num_network_sizes,num_realizations);
mean_X_current = cell(num_network_sizes,num_realizations);
reMean = zeros(num_network_sizes, num_realizations);
riMean = zeros(num_network_sizes, num_realizations);
spikeTimes_E = cell(num_network_sizes,num_realizations);
spikeIndex_E = cell(num_network_sizes,num_realizations);
spikeTimes_I = cell(num_network_sizes,num_realizations);
spikeIndex_I = cell(num_network_sizes,num_realizations);
        

for k = 1:num_network_sizes
    N = N_vector(k); % Total num of neurons
    % Number of neurons in each population
    Ne=0.8*N;
    Ni=0.2*N;
    % Number of neurons in ffwd layer
    Nx=0.2*N;
    % Mean connection strengths between each cell type pair
    Jm=[1 -100; 112.5 -250]/sqrt(N);
    Jxm=[180; 135]/sqrt(N);
    
    % Proportion of neurons in each population.
    qe=Ne/N;
    qi=Ni/N;
    qf=Nx/N;
    
    Jstim=sqrt(N)*[jestim*ones(Ne,1); jistim*ones(Ni,1)]; 
    % Build mean field matrices
    Q=[qe qi; qe qi];
    Qf=[qf; qf];
    W=P.*(Jm*sqrt(N)).*Q;
    Wx=Px.*(Jxm*sqrt(N)).*Qf;
    
    % EE synapses - hebbian rule with soft constraints
    Jmax_ee_soft = 30/sqrt(N); % Soft constraint - also the fixed point of EE weights
    eta_ee_soft=0/1000; % Learning rate of EE

    % EE synapses - hebbian rule with hard constraints
    Jmax_ee_hard = 1000/sqrt(N); % Upper constraint
    Jmin_ee_hard = 0/sqrt(N); % Lower constraint
    eta_ee_hard=0/1000; % Learning rate of EE

    % EE synapses - Kohonen's rule (needs no constraints)
    beta = 2/sqrt(N);
    eta_ee_kohonen=1/1000; % Learning rate of EE

    % EI synapses
    Jmax_ei = -200/sqrt(N); % arbitrary normalizing constant (same order as Jei)
    eta_ei=0/10000/Jmax_ei; % Learning rate of EI
    rho0_e=0.010; % Target rate for e cells
    alpha_e=2*rho0_e*tauSTDP;

    % IE synapses - note that this can be hebbian or homeostatic
    Jmax_ie = 200/sqrt(N);
    eta_ie_homeo=0.000/Jmax_ie; % Learning rate of IE
    rho0_i_homeo=0.023; % Target rate for i cells.
    alpha_ie=2*rho0_i_homeo*tauSTDP;
    % IE Hebbian
    eta_ie_hebbian = 0/1000;
    Jmax_ie = 112.5/sqrt(N); % only if hebbian

    % II synapses
    Jmax_ii = -200/sqrt(N);
    eta_ii=0.000/Jmax_ii; % Learning rate of EI
    rho0_i=0.023; % Target rate for i cells.
    alpha_ii=2*rho0_i*tauSTDP;
    
    %%% Note: 
    %%% If IE or II homeostatic plasticity are ON, then the network also
    %%% needs EI homeostatic STDP ON to maintain stability.
    
    parfor rz = 1:num_realizations

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

        % Random initial voltages
        V0=rand(N,1)*(VT-Vre)+Vre;
        % Set initial voltage
        V=V0;

        % Preallocate memory
        Ie=zeros(N,1);
        Ii=zeros(N,1);
        Ix=zeros(N,1);
        x=zeros(N,1);
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
%             JRec_ei(1,ii)=mean(J(sub2ind(size(J),Jrecord_ei(1,:),Jrecord_ei(2,:))));
    %         JRec_ie(1,ii)=mean(J(sub2ind(size(J),Jrecord_ie(1,:),Jrecord_ie(2,:))));
    %         JRec_ii(1,ii)=mean(J(sub2ind(size(J),Jrecord_ii(1,:),Jrecord_ii(2,:))));
    %         JRec_ee_std(1,ii)=1/sqrt(nJrecord0) *sqrt(N)* std(J(sub2ind(size(J),Jrecord_ee(1,:),Jrecord_ee(2,:))));
    %         JRec_ei_std(1,ii)=1/sqrt(nJrecord0) *sqrt(N)* std(J(sub2ind(size(J),Jrecord_ei(1,:),Jrecord_ei(2,:))));
    %         JRec_ie_std(1,ii)=1/sqrt(nJrecord0) *sqrt(N)* std(J(sub2ind(size(J),Jrecord_ie(1,:),Jrecord_ie(2,:))));
    %         JRec_ii_std(1,ii)=1/sqrt(nJrecord0) *sqrt(N)* std(J(sub2ind(size(J),Jrecord_ii(1,:),Jrecord_ii(2,:))));

            % Reset membrane potential
            V(Ispike)=Vre;

           if mod(i*dt,T/10) == 0 % print every T/5 iterations.
                fprintf('At time %d...\n',i*dt/T);
           end
        end
        % Normalize recorded variables by # bins
        IeRec=mean(IeRec)/nBinsRecord;
        IiRec=mean(IiRec)/nBinsRecord;
        IxRec=mean(IxRec)/nBinsRecord;
        % VRec=VRec/nBinsRecord;

        % Get rid of padding in s
        s=s(:,1:nspike);
        tSim=toc;
        disp(sprintf('\nTime for simulation: %.2f min',tSim/60))

        % Record variables needed: J, rate, covs, corrs, currents
        mean_Jee_recorded{k,rz} = JRec_ee*sqrt(N);
%         mean_Jei_recorded{k,rz} = JRec_ei*sqrt(N);
%         mean_Jie_recorded{k,rz} = JRec_ie*sqrt(N);
%         mean_Jii_recorded{k,rz} = JRec_ii*sqrt(N);

        % Time-dependent mean rates
        eRateT{k,rz}=hist(s(1,s(2,:)<=Ne),dtRate:dtRate:T)/(dtRate*Ne);
        iRateT{k,rz}=hist(s(1,s(2,:)>Ne),dtRate:dtRate:T)/(dtRate*Ni);

        % Mean rate of each neuron (excluding burn-in period)
        reSim=hist(s(2,s(1,:)>Tburn & s(2,:)<=Ne),1:Ne)/(T-Tburn);
        riSim=hist(s(2,s(1,:)>Tburn & s(2,:)>Ne)-Ne,1:Ni)/(T-Tburn);

        % Mean rate over E and I pops
        reMean(k,rz)=mean(reSim);
        riMean(k,rz)=mean(riSim);

        % Compute spike count covariances over windows of size
        % winsize starting at time T1 and ending at time T2.
        C=SpikeCountCov(s,N,T1,T2,winsize);

        % Get mean spike count covariances over each sub-pop
        [II,JJ]=meshgrid(1:N,1:N);
        mCee(k,rz)=mean(C(II<=Ne & JJ<=II));
        mCei(k,rz)=mean(C(II<=Ne & JJ>Ne));
        mCii(k,rz)=mean(C(II>Ne & JJ>II));

        % Get variances too.
        mVee(k,rz)=mean(C(II<=Ne & JJ==II));
        mVii(k,rz)=mean(C(II>Ne & JJ==II));

        % Compute spike count correlations
        % Get correlation matrix from cov matrix
        tic
        R=corrcov(C);
        toc
        mRee(k,rz)=mean(R(II<=Ne & JJ<=II & isfinite(R)));
        mRei(k,rz)=mean(R(II<=Ne & JJ>Ne & isfinite(R)));
        mRii(k,rz)=mean(R(II>Ne & JJ>II & isfinite(R)));

        mean_E_current{k,rz} = IeRec;
        mean_I_current{k,rz} = IiRec;
        mean_X_current{k,rz} = IxRec;
        spikeTimes_E{k,rz} = s(1,s(2,:)<=Ne & s(1,:)>Tburn_raster & s(1,:)<T );
        spikeIndex_E{k,rz} = s(2,s(2,:)<=Ne & s(1,:)>Tburn_raster & s(1,:)<T);
        spikeTimes_I{k,rz} = s(1,s(2,:)>Ne & s(1,:)>Tburn_raster & s(1,:)<T);
        spikeIndex_I{k,rz} = s(2,s(2,:)>Ne & s(1,:)>Tburn_raster & s(1,:)<T);
    end
end
%% Save necessary variables.

spikeTimes_E = spikeTimes_E{num_network_sizes,num_realizations};
spikeIndex_E = spikeIndex_E{num_network_sizes,num_realizations};
spikeTimes_I = spikeTimes_I{num_network_sizes,num_realizations};
spikeIndex_I = spikeIndex_I{num_network_sizes,num_realizations};


save('/scratch/AlanAkil/Kohonen_asNgrows_Sept15.mat',...
    'alpha_e','beta','c','dt','dtRate','dtRecord','eRateT','iRateT',...
    'eta_ee_hard','eta_ee_kohonen','eta_ee_soft','eta_ei','eta_ie_hebbian',...
    'eta_ie_homeo','eta_ii','Jm','Jxm','Jmax_ee_hard',...
    'Jmax_ee_soft','Jmax_ei','Jmax_ie','Jmax_ii',...
    'mCee','mCei','mCii','mean_E_current','mean_I_current',...
    'mean_X_current','mean_Jee_recorded','mRee','mRei','mRii','mVee',...
    'mVii','N_vector','nBinsRecord','Ntrec','num_network_sizes',...
    'num_realizations','P','Px','Q','Qf','reMean','riMean','rho0_e',...
    'rho0_i','rho0_i_homeo','rx','seed','spikeIndex_E',...
    'spikeIndex_I','spikeTimes_E','spikeTimes_I','T','T1','T2','Tburn',...
    'Tburn_raster','tauSTDP','W','winsize','Wx');

