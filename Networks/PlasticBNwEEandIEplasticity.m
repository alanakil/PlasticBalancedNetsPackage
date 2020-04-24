%% Simulation of EE & EI plastic balanced network. 4/20/19


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


% Timescale of correlation
taujitter=5;

% Mean connection strengths between each cell type pair
Jm=[10 -120; 112.5 -250]/sqrt(N);
Jxm=[180; 135]/sqrt(N);
Jmax_EI=-150/sqrt(N);
Jmax_EE=50/sqrt(N);
gamma_plus=1;
gamma_minus=-1;

% Time (in ms) for sim
T=1500000;

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
 
% Neuron parameters
Cm=1;
gL=1/15;
EL=-72;
Vth=-50;
Vre=-75;
DeltaT=1;
VT=-55;

%%% I changed the parameterization so that we specify rho0 and derive alpha
%%% in terms of rho0. Not a big change, but easier to tinker with
%%% parameters this way
% Plasticity params
eta1 = 1/100; % Learning rate for EE.
eta2 = 0;%1/100; % Learning rate for EI.
tauSTDP=50;

% For EI plasticity.
rho0=0.01; % Target rate 10Hz
alpha=2*rho0*tauSTDP;

% Number of time bins to average over when recording
nBinsRecord=100;
dtRecord=nBinsRecord*dt;
timeRecord=dtRecord:dtRecord:T;
Ntrec=numel(timeRecord);

% Integer division function
IntDivide=@(n,k)(floor((n-1)/k)+1);

Num_c_vals = 7;

JRec = cell(Num_c_vals,1);
JRec_IE = cell(Num_c_vals,1);
eRateT = cell(Num_c_vals,1);
iRateT = cell(Num_c_vals,1);
mC = cell(Num_c_vals,1);
mR = cell(Num_c_vals,1);
spikeTimes = cell(Num_c_vals,1);
spikeIndex = cell(Num_c_vals,1);


dtRate=15000; % ms
Vector_Time_Rate = (dtRate:dtRate:T)/1000;

parfor k = 1:Num_c_vals
    
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
    x=zeros(N,1);
%     IeRec=zeros(numrecord,Ntrec);
%     IiRec=zeros(numrecord,Ntrec);
%     IxRec=zeros(numrecord,Ntrec);
%     VRec=zeros(numrecord,Ntrec);
%     wRec=zeros(numrecord,Ntrec);
    
    % Correlation between the spike trains in the ffwd layer
    c=(k-1)/10;
    
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
    
    
    
    % Synaptic weights EE to record.
    % The first row of Jrecord is the postsynaptic indices
    % The second row is the presynaptic indices
    nJrecord0=5000; % Number to record
    [II,JJ]=find(J(1:Ne,1:Ne)); % Find non-zero E to E weights
    III=randperm(numel(II),nJrecord0); % Choose some at random to record
    II=II(III);
    JJ=JJ(III);
    Jrecord=[II JJ]'; % Record these
    numrecordJ=size(Jrecord,2);
    if(size(Jrecord,1)~=2)
        error('Jrecord must be 2xnumrecordJ');
    end
    
    % Synaptic weights IE to record.
    % The first row of Jrecord is the postsynaptic indices
    % The second row is the presynaptic indices
    nJrecord0_EI=5000; % Number to record
    [II1,JJ1]=find(J(1:Ne,Ne+1:N)); % Find non-zero I to E weights
    III1=randperm(numel(II1),nJrecord0_EI); % Choose some at random to record
    II1=II1(III1);
    JJ1=JJ1(III1);
    Jrecord_IE=[II1 JJ1+Ne]'; % Record these
    numrecordJ1=size(Jrecord_IE,2);
    if(size(Jrecord_IE,1)~=2)
        error('Jrecord must be 2xnumrecordJ');
    end
    
    
    JRec{k,1}=zeros(1,Ntrec);
    JRec_IE{k,1}=zeros(1,Ntrec);
    
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
            
            % If there is EE Hebbian plasticity
            if(eta1~=0)
                %Update synaptic weights according to plasticity rules
                %E to E after presynaptic spike
                J(1:Ne,Ispike(Ispike<=Ne))=J(1:Ne,Ispike(Ispike<=Ne))+ ...
                    +gamma_minus*repmat(eta1*(x(1:Ne)),1,nnz(Ispike<=Ne)).*(J(1:Ne,Ispike(Ispike<=Ne)));
                %E to E after a postsynaptic spike
                J(Ispike(Ispike<=Ne),1:Ne)=J(Ispike(Ispike<=Ne),1:Ne)+ ...
                    +gamma_plus*repmat(eta1*x(1:Ne)',nnz(Ispike<=Ne),1).*(Jmax_EE-J(Ispike(Ispike<=Ne),1:Ne)).*(J(Ispike(Ispike<=Ne),1:Ne)~=0);
            end
            
            % If there is EE AntiHebbian plasticity
            %         if(eta1~=0)
            %             % Update synaptic weights according to plasticity rules
            %             % E to E after presynaptic spike
            %             J(1:Ne,Ispike(Ispike<=Ne))=J(1:Ne,Ispike(Ispike<=Ne))+ ...
            %                 -gamma_minus*repmat(eta1*(x(1:Ne)),1,nnz(Ispike<=Ne)).*(J(1:Ne,Ispike(Ispike<=Ne)));
            %             % E to E after a postsynaptic spike
            %             J(Ispike(Ispike<=Ne),1:Ne)=J(Ispike(Ispike<=Ne),1:Ne)+ ...
            %                 -gamma_plus*repmat(eta1*x(1:Ne)',nnz(Ispike<=Ne),1).*(J(Ispike(Ispike<=Ne),1:Ne));
            %         end
            
            % If there is I to E plasticity
            if(eta2~=0)
                % Update synaptic weights according to plasticity rules
                % I to E after an I spike
                J(1:Ne,Ispike(Ispike>Ne))=J(1:Ne,Ispike(Ispike>Ne))+ ...
                    -repmat(eta2*(x(1:Ne)-alpha),1,nnz(Ispike>Ne)).*(J(1:Ne,Ispike(Ispike>Ne)))/Jmax_EI;
                % I to E after an E spike
                J(Ispike(Ispike<=Ne),Ne+1:N)=J(Ispike(Ispike<=Ne),Ne+1:N)+ ...
                    -repmat(eta2*x(Ne+1:N)',nnz(Ispike<=Ne),1).*(J(Ispike(Ispike<=Ne),Ne+1:N))/Jmax_EI;
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
        
        % Update time-dependent firing rates for plasticity
        x(1:Ne)=x(1:Ne)-dt*x(1:Ne)/tauSTDP;
        x(Ne+1:end)=x(Ne+1:end)-dt*x(Ne+1:end)/tauSTDP;
        
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
        JRec{k,1}(1,ii)=mean(J(sub2ind(size(J),Jrecord(1,:),Jrecord(2,:))));
        JRec_IE{k,1}(1,ii)=mean(J(sub2ind(size(J),Jrecord_IE(1,:),Jrecord_IE(2,:))));
        
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
    
%     % Plot time-dependent rates
%     figure
%     plot((dtRate:dtRate:T)/1000,1000*eRateT)
%     hold on
%     plot((dtRate:dtRate:T)/1000,1000*iRateT)
%     legend('r_e','r_i')
%     ylabel('rate (Hz)')
%     xlabel('time (s)')
    
    %% If plastic, plot mean connection weights over time
    % EE
%     if(eta1~=0)
%         figure
%         timeAveragedJ_EE{k,1} = JRec{k,1}*sqrt(N);
% %         plot(timeRecord/1000,timeAveragedJ_EE)
% %         xlabel('time (s)')
% %         ylabel('Mean E to E synaptic weight')
% %         
% %         figure;
% %         histogram(JRec(:,T/dt)*sqrt(N), 40)
% %         xlabel('Jee')
% %         ylabel('Count')
%         
%     end
    % I to E
%     if(eta2~=0)
%         figure
%         timeAveragedJ_IE{k,1} = JRec_IE{k,1}*sqrt(N);
% %         plot(timeRecord/1000,timeAveragedJ_IE)
% %         xlabel('time (s)')
% %         ylabel('Mean I to E synaptic weight')
% %         
% %         figure;
% %         histogram(JRec_IE(:,T/dt)*sqrt(N), 40) % at last time step.
% %         xlabel('Jei')
% %         ylabel('Count')
%         
%     end
    
    timeAveragedJ_EE{k,1} = JRec{k,1}*sqrt(N);
    timeAveragedJ_IE{k,1} = JRec_IE{k,1}*sqrt(N);
    
    %%% All the code below computes spike count covariances and correlations
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
    
    %% Look at final mean synaptic weights.
    mJee(k) = mean(nonzeros(J(1:Ne,1:Ne)))*sqrt(N);
    mJei(k) = mean(nonzeros(J(1:Ne,Ne:N)))*sqrt(N);
    
    
    if k==Num_c_vals
        spikeTimes{k,1}=s(1,s(2,:)<N+1);
        spikeIndex{k,1}=s(2,s(2,:)<N+1);
    end
end

%% Compute Cv of ISI.
% Column 1 has spike times and column 2 the neuron index
Want_CV=0;

if (Want_CV~=0)
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

%% Save theory variables.


Theory_Jee = Jmax_EE*sqrt(N)/(gamma_plus-gamma_minus);
W(1,1) = Theory_Jee*qe*P(1,1);

if eta2 ~=0
    Theory_Jei = [-119.6581, -123.6959,-127.5839,-131.3303,-134.9427,...
        -138.4282,-141.7937];%,-145.044,-148.18665];
%     W(1,2) = Theory_Jei*qi*P(1,1); % 0.2 is fraction of inh neurons.
else
    Theory_Jei=W(1,2)/0.2/P(1,1);
end

c = 0:0.1:(Num_c_vals-1)/10;


% %% Save all necessary variables.
% save('./EE_EI_plasticity_vars1.mat', 'eRateT','iRateT','T','dtRate',...
%     'timeAveragedJ_EE','timeAveragedJ_IE','timeRecord','N',...
%     'c','spikeTimes','spikeIndex','Vector_Time_Rate','mJee','mJei',...
%     'Theory_Jee','Theory_Jei','Num_c_vals',...
%     'mC','mR','W','Wx','rx','eta1','eta2');


%% Save all necessary variables if no EI plasticity.
save('/scratch/AlanAkil/EE_EI_plasticity_vars2_NoEI.mat', 'eRateT','iRateT','dtRate','T',...
    'timeAveragedJ_EE','timeRecord','N',...
    'c','spikeTimes','spikeIndex','Vector_Time_Rate','mJee',...
    'Theory_Jee','Theory_Jei','Num_c_vals',...
    'mC','mR','W','Wx','rx','eta1','eta2');

% %% Save all necessary if there is EI plasticitybut no EE.
% save('./EI_plasticity_vsC.mat', 'eRateT','iRateT','T',...
%     'timeAveragedJ_IE','timeRecord','N','Num_c_vals',...
%     'c','spikeTimes','spikeIndex','Vector_Time_Rate','mJei',...
%     'Theory_Jei','mC','mR','W','Wx','rx','eta1','eta2');
% 





