%% Simulation of a plastic balanced network %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: To compute weights, correlations and rates in the large N limit and compare
% to theory for a plastic I to E and I to I (both iSTDP) balanced network.
clear

%% Define variables that repeat over trials.

% seed=1; % Fix the seed to the rng.
rng('shuffle');


NTrials = 8; % Number of data points, i.e., number of different population sizes used.
c = 0;
N_realizations = 5;

% Recurrent net connection probabilities
P=[0.1 0.1; 0.1 0.1];

% Ffwd connection probs
Px=[.1; .1];

% Timescale of correlation
taujitter=5;
% FFwd spike train rate (in kHz)
rx=10/1000; 

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

% Plasticity EI params
tauSTDP=200;
rho1=0.010; % Target rate 10Hz
alpha1=2*rho1*tauSTDP;

% Plasticity ii parameters. 
rho2=0.020; % Target rate 10Hz
alpha2=2*rho2*tauSTDP;

% Integer division function
IntDivide=@(n,k)(floor((n-1)/k)+1);

jistim=0;
stimProb = 0;
jestim=0;

Tburn = 100000; % burn in period total time (in ms).
% Time discretization
dt=0.1;
timeBP = dt:dt:Tburn;
% Number of time bins to average over when recording
nBinsRecord=100;
dtRecord=nBinsRecord*dt;
timeRecord=dtRecord:dtRecord:Tburn;
Ntrec=numel(timeRecord);
Nt=round(Tburn/dt);

% Preallocate memory.
eRateT = cell(NTrials, N_realizations);
iRateT = cell(NTrials, N_realizations);
meanJei = cell(NTrials, N_realizations);
mCee = zeros(NTrials,N_realizations);
mCei = zeros(NTrials,N_realizations);
mCii = zeros(NTrials,N_realizations);
meanCovs = zeros(NTrials,N_realizations);
mRee = zeros(NTrials,N_realizations);
mRei = zeros(NTrials,N_realizations);
mRii = zeros(NTrials,N_realizations);
meanCorrs = zeros(NTrials,N_realizations);
spikeTimes = cell(NTrials, N_realizations);
spikeIndex = cell(NTrials, N_realizations);
meanEcurrent = cell(NTrials, N_realizations);
meanIcurrent = cell(NTrials, N_realizations);
meanXcurrent = cell(NTrials, N_realizations);
Record_Jeimean = zeros(NTrials,N_realizations);
Record_Jiimean = zeros(NTrials,N_realizations);


N_axis=round( 10.^(linspace(3,4.3,NTrials)),-1 );

for k = 1:NTrials
    
    N = N_axis(1,k);
    Ne=0.8*N;
    Ni=0.2*N;
    
    % Number of neurons in ffwd layer
    Nx=0.2*N;
    
    % Plasticity EI params
    Jmax = -200/sqrt(N);
    eta1=0.01/Jmax; % Learning rate
    
    % Plasticity ii parameters.
    Jmax1 = -500/sqrt(N);
    eta2=-0.01/Jmax1; % Learning rate
    
    % Mean connection strengths between each cell type pair
    Jm=[25 -100; 112.5 -200]/sqrt(N); % Redefine these matrices, because it depends on N.
    Jxm=[180; 135]/sqrt(N);
    
    % Proportions
    qe=Ne/N;
    qi=Ni/N;
    qf=Nx/N;
    
    % Build mean field matrices
    Q=[qe qi; qe qi];
    Qf=[qf; qf];
    W=P.*(Jm*sqrt(N)).*Q;
    Wx=Px.*(Jxm*sqrt(N)).*Qf;
    
    parfor rz = 1:N_realizations
        
        % Generate connectivity matrices
        tic
        J=[Jm(1,1)*binornd(1,P(1,1),Ne,Ne) Jm(1,2)*binornd(1,P(1,2),Ne,Ni); ...
           Jm(2,1)*binornd(1,P(2,1),Ni,Ne) Jm(2,2)*binornd(1,P(2,2),Ni,Ni)];
        Jx=[Jxm(1)*binornd(1,Px(1),Ne,Nx); Jxm(2)*binornd(1,Px(2),Ni,Nx)];
        tGen=toc;
        disp(sprintf('\nTime to generate connections: %.2f sec',tGen))
              
        % Indices of neurons to record currents, voltages
        nrecord0=2; % Number to record from each population
        Irecord=[randperm(Ne,nrecord0) randperm(Ni,nrecord0)+Ne];
        numrecord=numel(Irecord); % total number to record
        
        % Synaptic weights II to record.
        % The first row of Jrecord is the postsynaptic indices
        % The second row is the presynaptic indices
        nJrecord0=2000; % Number to record
        [II,JJ]=find(J(Ne+1:N,Ne+1:N)); % Find non-zero E to E weights
        III=randperm(numel(II),nJrecord0); % Choose some at random to record
        II=II(III);
        JJ=JJ(III);
        Jrecord=[II+Ne JJ+Ne]'; % Record these
        numrecordJ=size(Jrecord,2);
        if(size(Jrecord,1)~=2)
            error('Jrecord must be 2xnumrecordJ');
        end
        
        % Synaptic weights EI to record.
        % The first row of Jrecord is the postsynaptic indices
        % The second row is the presynaptic indices
        nJrecord0_EI=2000; % Number to record
        [II1,JJ1]=find(J(1:Ne,Ne+1:N)); % Find non-zero I to E weights
        III1=randperm(numel(II1),nJrecord0_EI); % Choose some at random to record
        II1=II1(III1);
        JJ1=JJ1(III1);
        Jrecord_IE=[II1 JJ1+Ne]'; % Record these
        numrecordJ1=size(Jrecord_IE,2);
        if(size(Jrecord_IE,1)~=2)
            error('Jrecord must be 2xnumrecordJ');
        end
                
        
        % Maximum number of spikes for all neurons
        % in simulation. Make it 50Hz across all neurons
        % If there are more spikes, the simulation will
        % terminate
        maxns=ceil(.05*N*Tburn);
        StimulatedNeurons = binornd(1,stimProb, Ne,1);
        s=zeros(2,maxns);
        % Stimulate a subpopulation of E neurons only.
        Jstim=sqrt(N)*[jestim*StimulatedNeurons ; jistim*ones(Ni,1)]; % Stimulate only E neurons
        Istim=zeros(size(timeBP));
        
        
        %% Make (correlated) Poisson spike times for ffwd layer
        %%% See section 5 of SimDescription.pdf for a description of this algorithm
        tic
        if(c<1e-5) % If uncorrelated
            nspikeX=poissrnd(Nx*rx*Tburn);
            st=rand(nspikeX,1)*Tburn;
            sx=zeros(2,numel(st));
            sx(1,:)=sort(st);
            sx(2,:)=randi(Nx,1,numel(st)); % neuron indices
%             clear st;
        else % If correlated
            rm=rx/c; % Firing rate of mother process
            nstm=poissrnd(rm*Tburn); % Number of mother spikes
            stm=rand(nstm,1)*Tburn; % spike times of mother process
            maxnsx=Tburn*rx*Nx*1.2; % Max num spikes
            sx=zeros(2,maxnsx);
            ns=0;
            for j=1:Nx  % For each ffwd spike train
                ns0=binornd(nstm,c); % Number of spikes for this spike train
                st=randsample(stm,ns0); % Sample spike times randomly
                st=st+taujitter*randn(size(st)); % jitter spike times use exprnd or randn!
                st=st(st>0 & st<Tburn); % Get rid of out-of-bounds times
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
        disp(sprintf('\nTime to generate ffwd spikes for burn-in period: %.2f sec',tGenx))
        
        
        % Random initial voltages
        V0=rand(N,1)*(VT-Vre)+Vre;
        
        V=V0;
        Ie=zeros(N,1);
        Ii=zeros(N,1);
        Ix=zeros(N,1);
        x=zeros(N,1);
        IeRec=zeros(numrecord,Ntrec);
        IiRec=zeros(numrecord,Ntrec);
        IxRec=zeros(numrecord,Ntrec);
        %VRec=zeros(numrecord,Ntrec);
        %wRec=zeros(numrecord,Ntrec);
        JRec=zeros(1,Ntrec);
        JRec_IE=zeros(1,Ntrec);        
        iFspike=1;
        nspike=0;
        TooManySpikes=0;
        tic
        
        %% Start actual run.
        for i=1:numel(timeBP)
            
            % Propogate ffwd spikes
            while(sx(1,iFspike)<=timeBP(i) && iFspike<nspikeX)
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
                    s(1,nspike+1:nspike+numel(Ispike))=timeBP(i);
                    s(2,nspike+1:nspike+numel(Ispike))=Ispike;
                else
                    TooManySpikes=1;
                    break;
                end
                
                
                % Update synaptic currents
                Ie=Ie+sum(J(:,Ispike(Ispike<=Ne)),2)/taue;
                Ii=Ii+sum(J(:,Ispike(Ispike>Ne)),2)/taui;
                
                
                % If there is EI plasticity
                if(eta1~=0)
                    % Update synaptic weights according to plasticity rules
                    % I to E after an I spike
                    J(1:Ne,Ispike(Ispike>Ne))=J(1:Ne,Ispike(Ispike>Ne))+ ...
                        -repmat(eta1*(x(1:Ne)-alpha1),1,nnz(Ispike>Ne)).*(J(1:Ne,Ispike(Ispike>Ne)));
                    % I to E after an E spike
                    J(Ispike(Ispike<=Ne),Ne+1:N)=J(Ispike(Ispike<=Ne),Ne+1:N)+ ...
                        -repmat(eta1*x(Ne+1:N)',nnz(Ispike<=Ne),1).*(J(Ispike(Ispike<=Ne),Ne+1:N));
                end
                
                
                
                % If there is ii Hebbian plasticity
                if(eta2~=0)
                    % Update synaptic weights according to plasticity rules
                    % I to I after a presyanptic spike
                    J(Ne+1:N,Ispike(Ispike>Ne))=J(Ne+1:N,Ispike(Ispike>Ne))+ ...
                        +repmat(eta2*(x(Ne+1:N)-alpha2),1,nnz(Ispike>Ne)).*(J(Ne+1:N,Ispike(Ispike>Ne)));
                    % I to I after a postsynaptic spike
                    J(Ispike(Ispike>Ne),Ne+1:N)=J(Ispike(Ispike>Ne),Ne+1:N)+ ...
                        +repmat(eta2*x(Ne+1:N)',nnz(Ispike>Ne),1).*(J(Ispike(Ispike>Ne),Ne+1:N));
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
            IeRec(:,ii)=IeRec(:,ii)+Ie(Irecord);
            IiRec(:,ii)=IiRec(:,ii)+Ii(Irecord);
            IxRec(:,ii)=IxRec(:,ii)+Ix(Irecord);
            %VRec(:,ii)=VRec(:,ii)+V(Irecord);
            JRec_IE(:,ii)= mean(J(sub2ind(size(J),Jrecord_IE(1,:),Jrecord_IE(2,:))));
            JRec(:,ii)= mean(J(sub2ind(size(J),Jrecord(1,:),Jrecord(2,:))));

            % Reset mem pot.
            V(Ispike)=Vre;
            
            
        end
        IeRec=IeRec/nBinsRecord; % Normalize recorded variables by # bins
        IiRec=IiRec/nBinsRecord;
        IxRec=IxRec/nBinsRecord;
        %VRec=VRec/nBinsRecord;
        % Record the coupling strengths.
        Record_Jiimean(k,rz) = mean( JRec(length(JRec)/2:end) )*sqrt(N);
        Record_Jeimean(k,rz) = mean( JRec_IE(length(JRec_IE)/2:end) )*sqrt(N);

        
        %% Start analysis of simulation. Calculate rates and correlations.
        
        s=s(:,1:nspike); % Get rid of padding in s
        tSim=toc;
        disp(sprintf('\nTime for simulation: %.2f min',tSim/60))
        
        % Mean rate of each neuron (excluding burn-in period)
        Tburnburn=Tburn/2;
        reSim=hist( s(2,s(1,:)>Tburnburn & s(2,:)<=Ne),1:Ne)/(Tburn-Tburnburn);
        riSim=hist( s(2,s(1,:)>Tburnburn & s(2,:)>Ne)-Ne,1:Ni)/(Tburn-Tburnburn);
        
        % Time-dependent mean rates
        dtRate=5000;
        eRateT{k,rz}=hist(s(1,s(2,:)<=Ne),1:dtRate:Tburn)/(dtRate*Ne);
        iRateT{k,rz}=hist(s(1,s(2,:)>Ne),1:dtRate:Tburn)/(dtRate*Ni);
        
        % Plot time-dependent rates
        %     figure;
        %     plot((dtRate:dtRate:Tburn)/1000,1000*eRateT{trial})
        %     hold on
        %     plot((dtRate:dtRate:Tburn)/1000,1000*iRateT{trial})
        %     legend('r_e','r_i')
        %     ylabel('rate (Hz)')
        %     xlabel('time (s)')
        %     title('Network"s burn in period');
        meanJei{k,rz} = JRec;
        
        %% Computation of correlations.
        
        % Compute spike count covariances over windows of size
        % winsize starting at time T1 and ending at time T2.
        winsize=250;
        T1=Tburnburn; % Burn-in period of 250 ms
        T2=Tburn;   % Compute covariances until end of simulation
        C=SpikeCountCov(s,N,T1,T2,winsize);
        
        % Get mean spike count covariances over each sub-pop
        [II,JJ]=meshgrid(1:N,1:N);
        mCee(k,rz)=mean(C(II<=Ne & JJ<=II));
        mCei(k,rz)=mean(C(II<=Ne & JJ>Ne));
        mCii(k,rz)=mean(C(II>Ne & JJ>II));
        meanCovs(k,rz) = mCee(k,rz) + 2* mCei(k,rz) + mCii(k,rz);
        
        % Get correlation matrix from cov matrix
        tic
        R=corrcov(C);
        toc
        
        mRee(k,rz)=mean(R(II<=Ne & JJ<=II & isfinite(R)));
        mRei(k,rz)=mean(R(II<=Ne & JJ>Ne & isfinite(R)));
        mRii(k,rz)=mean(R(II>Ne & JJ>II & isfinite(R)));
        meanCorrs(k,rz) = mRee(k,rz) + 2* mRei(k,rz) + mRii(k,rz);

        
        
        if k==NTrials-1
            spikeTimes{k,rz} = s( 1 ,s(2,:)>0 & s(2,:)<N+1 ); % Half E and Half I.
            spikeIndex{k,rz} = s( 2 ,s(2,:)>0 & s(2,:)<N+1);
            meanEcurrent{k,rz} = mean(IeRec,1);
            meanIcurrent{k,rz} = mean(IiRec,1);
            meanXcurrent{k,rz} = mean(IxRec,1);
        end
    end 
end

%% Calculate averages and standard errors for all quantities of interest (rate, J, corrs).
% Preallocate memory
eFR = zeros(1,NTrials);
iFR = zeros(1,NTrials);
meanFinalcovs = zeros(1,NTrials);
meanFinalCee = zeros(1,NTrials);
meanFinalCei = zeros(1,NTrials);
meanFinalCii = zeros(1,NTrials);
meanFinalcorrs = zeros(1,NTrials);
meanFinalRee = zeros(1,NTrials);
meanFinalRei = zeros(1,NTrials);
meanFinalRii = zeros(1,NTrials);
eRate_std_error = zeros(1,NTrials);
iRate_std_error = zeros(1,NTrials);
meanCovs_std_error = zeros(1,NTrials);
Cee_std_error = zeros(1,NTrials);
Cei_std_error = zeros(1,NTrials);
Cii_std_error = zeros(1,NTrials);
meanCorrs_std_error = zeros(1,NTrials);
Ree_std_error = zeros(1,NTrials);
Rei_std_error = zeros(1,NTrials);
Rii_std_error = zeros(1,NTrials);
meanERate = zeros(NTrials,N_realizations);
meanIRate = zeros(NTrials,N_realizations);


for i=1:NTrials
    for rz = 1:N_realizations
        meanERate(i,rz) = mean(eRateT{i,rz});
        meanIRate(i,rz) = mean(iRateT{i,rz});
    end
    eFR (i) = mean(meanERate(i,:) ); % Empirical E firing rate
    iFR(i) = mean(meanIRate(i,:) ); % Empirical I firing rate
    meanFinalcovs(i) = mean( meanCovs(i,:) ); % Empirical avg covs
    meanFinalCee(i) = mean( mCee(i,:) ); % Empirical ee covs
    meanFinalCei(i) = mean( mCei(i,:) ); % Empirical ei covs
    meanFinalCii(i) = mean( mCii(i,:) ); % Empirical ii covs
    meanFinalcorrs(i) = mean( meanCorrs(i,:) ); % Empirical avg covs
    meanFinalRee(i) = mean( mRee(i,:) ); % Empirical ee covs
    meanFinalRei(i) = mean( mRei(i,:) ); % Empirical ei covs
    meanFinalRii(i) = mean( mRii(i,:) ); % Empirical ii covs
    % Calculate standard error for every quantity obtained across
    % realizations.       
    eRate_std_error(i) = std( meanERate(i,:) ); %/ sqrt(N_realizations); %E rate std error
    iRate_std_error(i) = std( meanIRate(i,:) ); %/ sqrt(N_realizations); % I rate std error
    meanCovs_std_error(i) = std( meanCovs(i,:) ); %/ sqrt(N_realizations); % mean corrs std error
    Cee_std_error(i) = std( mCee(i,:) ); %/ sqrt(N_realizations); % ee corrs std error
    Cei_std_error(i) = std( mCei(i,:) ); %/ sqrt(N_realizations); % ei corrs std error
    Cii_std_error(i) = std( mCii(i,:) ); %/ sqrt(N_realizations); % ii corrs std error
    meanCorrs_std_error(i) = std( meanCorrs(i,:) ); %/ sqrt(N_realizations); % mean corrs std error
    Ree_std_error(i) = std( mRee(i,:) ); %/ sqrt(N_realizations); % ee corrs std error
    Rei_std_error(i) = std( mRei(i,:) ); %/ sqrt(N_realizations); % ei corrs std error
    Rii_std_error(i) = std( mRii(i,:) ); %/ sqrt(N_realizations); % ii corrs std error
end

%% Calculate theoretical correlations.

% First redefine some variables that are not saved in the parfor loop.
winsize = 250;
N = N_axis(1,NTrials);
eta=0.01; % Learning rate
%%
Ne=0.8*N;
Ni=0.2*N;
Nx=0.2*N;
% Mean connection strengths between each cell type pair
Jm=[25 -100; 112.5 -200]/sqrt(N); % Redefine these matrices, because it depends on N.
Jxm=[180; 135]/sqrt(N);

% Proportions
qe=Ne/N;
qi=Ni/N;
qf=Nx/N;

% Build mean field matrices
Q=[qe qi; qe qi];
Qf=[qf; qf];
W=P.*(Jm*sqrt(N)).*Q;
Wx=Px.*(Jxm*sqrt(N)).*Qf;

if eta~=0
    if c==0
        Theory_Jei = -139.9922;
        Theory_Jii = -292.49415;
    elseif c==0.1
        Theory_Jei = -142.12078;
        Theory_Jii = -294.06975;
    end
    W(1,2) = Theory_Jei*0.2*P(1,1); % 0.2 is fraction of inh neurons.
    W(2,2) = Theory_Jii*0.2*P(1,1);
else
    Theory_Jei=W(1,2)/0.2/P(1,1);
    Theory_Jii=W(2,2)/0.2/P(1,1);
end

Theory_Cee = zeros(1,NTrials);
Theory_Cei = zeros(1,NTrials);
Theory_Cii = zeros(1,NTrials);
if c == 0
    for i=1:NTrials
        Ctheory = 1/N_axis(i) * winsize*inv(W)*Wx*transpose(Wx)*rx /qf *inv(transpose(W));
        Theory_Cee(i)  = Ctheory(1,1); % Theoretical ee corrs
        Theory_Cei(i)  = Ctheory(1,2); % theoretical ei corrs
        Theory_Cii(i)  = Ctheory(2,2); % Theoretical ii corrs
    end
else
    Ctheory = winsize*inv(W)*Wx*transpose(Wx)*c*rx*inv(transpose(W));
    Theory_Cee  = Ctheory(1,1); % Theoretical ee corrs
    Theory_Cei  = Ctheory(1,2); % theoretical ei corrs
    Theory_Cii  = Ctheory(2,2); % Theoretical ii corrs
end


%% Save some variables for plots.

spikeTimes = spikeTimes{NTrials-1,1}; % Pick only one realization to save.
spikeIndex = spikeIndex{NTrials-1,1};
% meanEcurrent = meanEcurrent{6,1}(1:200000);
% meanIcurrent = meanIcurrent{6,1}(1:200000);
% meanXcurrent = meanXcurrent{6,1}(1:200000);
% timeRecord=timeRecord(1:200000);
eRateT= eRateT{NTrials,1};
iRateT= iRateT{NTrials,1};
%%

save('/scratch/AlanAkil/Vogels_ei_ii_Ninf_Corr.mat', 'meanFinalCee','meanFinalCei','meanFinalCii',...
    'Cee_std_error','Cei_std_error','Cii_std_error','Theory_Cee',...
    'Theory_Cei','Theory_Cii','W','Wx','rx','eFR','iFR',...
    'eRate_std_error','iRate_std_error','NTrials',...
    'spikeIndex', 'spikeTimes','meanEcurrent','meanIcurrent',...
    'Tburn','dt','timeRecord','winsize','c','Theory_Jei',...
    'Theory_Jii','meanXcurrent','Record_Jiimean',...
    'NTrials','Record_Jeimean','N_realizations','N_axis',...
    'eRateT', 'iRateT','meanFinalRee','meanFinalRei','meanFinalRii',...
    'Ree_std_error','Rei_std_error','Rii_std_error');

% Note when the sims have finished in the log file.
fprintf('Simulation has finished.\n')

