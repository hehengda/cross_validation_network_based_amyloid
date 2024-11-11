close all;
clc;
clear all;
warning('off','all')

%%

y_idx = 5; % set this to run on cognition in seperate domain or global cognition
y_names = {'reason','memory','vocab','speed','mean_cog'};

model_idx = 7; % set this to run on different pathology models
models = {'PETdata','PETFCNetScore','PETSCNetScore','FCnode','SCnode',...
    'PETFCNetScore_CS','PETSCNetScore_CS'};

disp(['Model ' num2str(model_idx) ' - ' models{model_idx}])
disp(['Cognition #' num2str(y_idx) ' - ' y_names{y_idx}])

subjects = {'P00004225','P00004226','P00004227','P00004229','P00004234','P00004242',...
    'P00004247','P00004250','P00004251','P00004259','P00004261','P00004268','P00004269',...
    'P00004278','P00004280','P00004282','P00004284','P00004287','P00004292','P00004295',...
    'P00004297','P00004300','P00004301','P00004303','P00004304','P00004309','P00004310',...
    'P00004315','P00004317','P00004318','P00004321','P00004325','P00004326','P00004333',...
    'P00004336','P00004347','P00004349','P00004350','P00004352','P00004355','P00004356',...
    'P00004357','P00004358','P00004359','P00004360','P00004361','P00004370','P00004372',...
    'P00004375','P00004376','P00004377','P00004378','P00004380','P00004383','P00004384',...
    'P00004385','P00004386','P00004389','P00004390','P00004391','P00004394','P00004395',...
    'P00004396','P00004397','P00004400','P00004405','P00004407','P00004408','P00004411',...
    'P00004412','P00004414','P00004418','P00004420','P00004423','P00004426','P00004429',...
    'P00004430','P00004436','P00004437','P00004441','P00004443','P00004445','P00004447',...
    'P00004451','P00004454','P00004463','P00004466','P00004472','P00004475',...
    'P00004480','P00004482','P00004484','P00004487','P00004489','P00004492','P00004494',...
    'P00004495','P00004497','P00004504','P00004509','P00004594','P00004595','P00004609',...
    'P00004610','P00004626','P00004660','P00004842','P00004961','P00005001'}; % 109 subjects

Exclude_SC = {'P00004221','P00004397','P00004326','P00004327','P00004428'}; % did not pass QC 
Exclude_FC = {'P00004350','P00004396','P00004411','P00004626','P00004736','P00004300','P00004392'}; % did not pass QC
Exclude_SC_nodata = {'P00004377','P00004389','P00004445','P00004594','P00004660'};

exclude_idx = [];
for i = 1:length(Exclude_SC)
    exclude_idx = [exclude_idx find(contains(subjects,Exclude_SC{i}))];
end
for i = 1:length(Exclude_SC_nodata)
    exclude_idx = [exclude_idx find(contains(subjects,Exclude_SC_nodata{i}))];
end
for i = 1:length(Exclude_FC)
    exclude_idx = [exclude_idx find(contains(subjects,Exclude_FC{i}))];
end

subjects(exclude_idx) = [];

parcroi = 200;

roi_num = parcroi + 14;

load CR_RANN_Data/SCdata.mat % structural connetome: rois x rois x subjects
load CR_RANN_Data/FCdata.mat % functional connetome: rois x rois x subjects
load CR_RANN_Data/PETdata.mat % PET regional SUVR: rois x subjects
load CR_RANN_Data/demographic.mat % age sex(female=1,male=0) education
load CR_RANN_Data/cognition.mat % baseline and followup cognitions
load CR_RANN_Data/corticalthickness.mat % hemispherical cortical thickness

%%
atlas_lookup = 'Schaefer2018_200Parcels_17Networks_ori.txt';
fileID1 = fopen(atlas_lookup,'r') ;
formatSpec = '%f%s%f%f%f%f' ;
atlas_lookup_list = textscan(fileID1 , formatSpec);
fclose(fileID1) ;
roi_names = atlas_lookup_list{2};

%% compute PET net score

PETFCNetScore = zeros(roi_num,length(subjects)); % connectivity weighted 
for j = 1:length(subjects)
    for i = 1:roi_num
        FCdata_pos = FCdata(i,:,j);
        FCdata_pos(i) = 1;
        PETFCNetScore(i,j) = nansum(PETdata(:,j)'.*FCdata_pos)...
            /length(FCdata_pos);
    end
end

PETSCNetScore = zeros(roi_num,length(subjects)); % connectivity weighted 
for j = 1:length(subjects)
    for i = 1:roi_num

        SCdata_pos = SCdata(i,:,j);
        PETSCNetScore(i,j) = (nansum(PETdata(:,j)'.*SCdata_pos)+PETdata(i,j))...
            /(length(find(SCdata_pos>0))+1);
         
    end
end

FCnode = zeros(roi_num,length(subjects)); % centrality only
for j = 1:length(subjects)
    for i = 1:roi_num
        pos = FCdata(i,:,j);
        FCnode(i,j) = nansum(pos);
    end
end

SCnode = zeros(roi_num,length(subjects)); % centrality only
for j = 1:length(subjects)
    for i = 1:roi_num
        SCnode(i,j) = nansum(SCdata(i,:,j))/sum((SCdata(i,:,j)>0));
    end
end


PETFCNetScore_CS = zeros(roi_num,length(subjects)); % centrality-scaled NAP
for j = 1:length(subjects)
    for i = 1:roi_num
        FCdata_pos = FCdata(i,:,j);
        PETFCNetScore_CS(i,j) = nansum(FCdata_pos)*PETdata(i,j);
    end
end

PETSCNetScore_CS = zeros(roi_num,length(subjects)); % centrality-scaled NAP
for j = 1:length(subjects)
    for i = 1:roi_num
        SCdata_pos = SCdata(i,:,j);
        PETSCNetScore_CS(i,j) = nansum(SCdata_pos)*PETdata(i,j)...
            /length(find(SCdata_pos>0));
    end
end

%% cross-validated predictive model 

sub_num = size(reason_blz,1);
y_all = zeros(sub_num,5);

y_all(:,1) = reason_f1z - reason_blz;
y_all(:,2) = memory_f1z - memory_blz;
y_all(:,3) = vocab_f1z - vocab_blz;
y_all(:,4) = speed_f1z - speed_blz;
y_all(:,5) = mean([speed_f1z vocab_f1z memory_f1z reason_f1z]')' - ...
      mean([speed_blz vocab_blz memory_blz reason_blz]')';
  
baseline_all = zeros(sub_num,5);
baseline_all(:,1) = reason_blz;
baseline_all(:,2) = memory_blz;
baseline_all(:,3) = vocab_blz;
baseline_all(:,4) = speed_blz;
baseline_all(:,5) = mean([speed_blz vocab_blz memory_blz reason_blz]')';

y = y_all(:,y_idx);
baseline = baseline_all(:,y_idx);

x = [age sex education lh_Mean_thickness rh_Mean_thickness baseline];

data_all = cell(7,1);

data_all{1} = PETdata;
data_all{2} = PETFCNetScore;
data_all{3} = PETSCNetScore;
data_all{4} = FCnode;
data_all{5} = SCnode;
data_all{6} = PETFCNetScore_CS;
data_all{7} = PETSCNetScore_CS;

data = data_all{model_idx};

nanidx = find(isnan(y));
for i = 1:size(x,2)
    if ismember(find(isnan(x(:,i))),nanidx)
    else
        nanidx = [nanidx;find(isnan(x(:,i)))];
    end
end

y(nanidx) = [];
data(:,nanidx) = [];
x(nanidx,:) = [];

count = 1;
R_iter =[];
P_iter = [];
MSE = [];
AIC = [];

no_sub=length(y);
disp(['Number of subjects ' num2str(no_sub)])

% randinds_record = zeros(500,no_sub);
load('randinds_record.mat') % use the same train-test split partition for proper model comparison

% threshold for feature selection
pthresh = 0.05;
kfolds = 15;

for iter = 1:500
disp(['iter - ' num2str(iter)])
    all_behav = y;
    all_mats  = data;  
    
    no_node=size(all_mats,1);

    behav_pred_neg=zeros(no_sub,1);

%     randinds=randperm(no_sub);
%     randinds_record(iter,:) = randinds;
    
    randinds=randinds_record(iter,:);
    
    ksample=floor(no_sub/kfolds);
    pred_observed_neg=[];

    for leftout=1:kfolds

            %display(['Running fold ' num2str(leftout)]);   
            train_mats = all_mats;
            train_behav = all_behav;

            si=1+((leftout-1)*ksample);
            fi=si+ksample-1;
            testinds=randinds(si:fi);
            traininds=setdiff(randinds,testinds);   
            train_mats(:,testinds)=[];
            train_behav(testinds)=[];

            no_trials_all = length(train_behav);

            x_train = x(traininds,:);

            train_vcts=train_mats;

            b = glmfit(x_train,train_behav);
            y_res = train_behav - x_train*b(2:end) - b(1);
            train_behav = y_res;
            
            edge_no=size(train_vcts,1);

            r_mat=zeros(1,edge_no);
            p_mat=zeros(1,edge_no);
            for edge_i = 1:edge_no
               [~,stats]=robustfit(train_vcts(edge_i,:)',train_behav); 
               cur_t = stats.t(2);
               r_mat(edge_i) = sign(cur_t)*sqrt(cur_t^2/(no_trials_all-1-2+cur_t^2));
               p_mat(edge_i) = 2*(1-tcdf(abs(cur_t), no_trials_all-1-2));  % two-tailed
            end
            
            neg_mask=zeros(no_node,1);
            neg_edges=find(r_mat<0 & p_mat < pthresh);
            neg_mask(neg_edges)=1;

            disp(num2str(sum(neg_mask)))
            
            neg_mask_all(:,count)=neg_mask;
            count = count + 1;
        
            train_sumneg=zeros(size(train_mats,2),1);

            for ss=1:size(train_sumneg)
                train_sumneg(ss) = nansum(train_mats(:,ss).*neg_mask);  
            end

            fit_neg=polyfit(train_sumneg(~isnan(train_behav)),train_behav(~isnan(train_behav)),1);
            
            behav_pred_neg=[]; 

            test_mat=all_mats(:,testinds);  
            test_behav=all_behav(testinds);

            y_res = test_behav - x(testinds,:)*b(2:end) - b(1);
            test_behav = y_res;

            for j=1:length(test_behav)
                test_sumneg(j)=nansum(test_mat(:,j).*neg_mask);
                behav_pred_neg(j,:)=fit_neg(1)*test_sumneg(j) + fit_neg(2);
            end

            pred_observed_neg=[pred_observed_neg; behav_pred_neg test_behav];
           

    end

    [R_neg, P_neg] = corr(pred_observed_neg(:,1),pred_observed_neg(:,2),'rows','pairwise');

    MSE_neg = mean((pred_observed_neg(:,1) - pred_observed_neg(:,2)).^2);

    n = length(pred_observed_neg(:,1));
    sigma2 = var(pred_observed_neg(:,2)-pred_observed_neg(:,1));
    logLikelihood = -0.5*n*log(2*pi) - 0.5*n*log(sigma2) - (1/(2*sigma2))*sum((pred_observed_neg(:,2)-pred_observed_neg(:,1)).^2);
    numParams = 2; 
    AIC_neg = -2*logLikelihood + 2*numParams;

    R_iter = [R_iter;R_neg];
    P_iter = [P_iter;P_neg];
    MSE = [MSE;MSE_neg];
    AIC = [AIC;AIC_neg];
    
end
      
disp(['              Neg-feature'])
disp(['median R value = ' num2str(median(R_iter))])
disp(['median MSE value = ' num2str(median(MSE))])
disp(['median AIC value = ' num2str(median(AIC))])

neg_mask_overall=NaN(size(neg_mask_all,1),1); % extract the features selected for 95% of the time
for i=1:size(neg_mask_all,1)
        if sum(neg_mask_all(i,:))/size(neg_mask_all,2)>=.95
            neg_mask_overall(i)=1;
        else
            neg_mask_overall(i)=0;
        end
end
        
disp([' '])
disp(['Negative feature'])
negidx = find(neg_mask_overall);
for i = 1:length(negidx)
    disp(roi_names{negidx(i)})
end

