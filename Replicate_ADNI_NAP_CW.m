clear all;
clc;
close all;

subjects = {'S1016','S4037','S4176','S4277','S4387','S4469','S4580','S6007',...
    'S6031','S6059','S6085','S6157','S6226','S6281','S6335','S6371','S6462',...
    'S6492','S6709','S6739','S6813','S4043','S4177','S4278','S4400',...
    'S4483','S4644','S6008','S6049','S6062','S6088','S6188','S6233','S6285',...
    'S6318','S6350','S6401','S6465','S6524','S6714','S6751','S1261','S4084',...
    'S4179','S4288','S4401','S4488','S6009','S6051','S6063','S6103',...
    'S6292','S6320','S6351','S6413','S6466','S6551','S6717','S6785','S1280',...
    'S4200','S4365','S4424','S4491','S4951','S6014','S6053','S6065',...
    'S6113','S6200','S6251','S6321','S6354','S6429','S6487','S6644','S6722',...
    'S6786','S4021','S4105','S4213','S4376','S4453','S4576','S4952','S6030',...
    'S6067','S6151','S6207','S6256','S6314','S6328','S6367','S6456','S6488','S6701','S6730','S6801'}; % no converter

parcroi = 200;
roi_num = 214;

load ADNI_Data/corticalthickness.mat % hemispherical cortical thickness, rois x subjects
load ADNI_Data/SCdata.mat % structural connetome: rois x rois x subjects
load ADNI_Data/PETdata.mat % PET regional SUVR: rois x subjects
load ADNI_Data/demographic.mat % age sex(female=1,male=0) education
load ADNI_Data/cognition.mat % baseline and followup cognitions: LDEL ADAS-DWR 

%% ROI specificity tests, random selection of ROIs to predict cognition
iteration = 500;
R_iters = zeros(iteration,1);

for iteri = 1:iteration

    disp(num2str(iteri))

    y = 1/2*([LDELfu_ADNIMERGE+ADASDWRfu_ADNIMERGE] - [LDELbl_ADNIMERGE+ADASDWRbl_ADNIMERGE]);
    baseline = 1/2*([LDELbl_ADNIMERGE+ADASDWRbl_ADNIMERGE]);
    
    outliers=[find(y<(nanmean(y)-3*nanstd(y)));find(y>(nanmean(y)+3*nanstd(y)))];
    y(outliers) = NaN;

    nanidx = find(isnan(y));
    
    x = [age sex education lhmeanthick rhmeanthick baseline PACCfulengh_Mnum-PACCbllengh_Mnum];

    for i = 1:size(x,2)
        if ismember(find(isnan(x(:,i))),nanidx)
        else
            nanidx = [nanidx;find(isnan(x(:,i)))];
        end
    end

    SCnode = zeros(roi_num,length(subjects)); % centrality-only
    for j = 1:length(subjects)
        for i = 1:roi_num
            SCnode(i,j) = nansum(SCdata(i,:,j))/sum((SCdata(i,:,j)>0));
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

    PETSCNetScore = zeros(roi_num,length(subjects));  % connectivity-weighted NAP
    for j = 1:length(subjects)
        for i = 1:roi_num

            SCdata_pos = SCdata(i,:,j);
            PETSCNetScore(i,j) = (nansum(PETdata(:,j)'.*SCdata_pos)+PETdata(i,j))...
                /(length(find(SCdata_pos>0))+1);

        end
    end
    
    data = PETSCNetScore; % test on connectivity-weighted NAP

    y(nanidx) = [];
    data(:,nanidx) = [];
    x(nanidx,:) = [];

    all_behav = y;
    b = glmfit(x,all_behav);
    y_res = all_behav - x*b(2:end) - b(1);

    fit_neg = [-0.4346    0.5590 ]; % same parameters as the actual model
    neg_mask_idx = randsample(roi_num,23); % keep the same number of ROIs as the actual signature pattern

    neg_mask_overall = zeros(size(data,1),1);
    neg_mask_overall(neg_mask_idx) = 1;

    featuresum = zeros(size(data,2),1);
    for ss=1:size(data,2)
        featuresum(ss) = nansum(data(:,ss).*neg_mask_overall);  
    end

    nanidx = find(featuresum==0);
    behav_pred_neg=fit_neg(1)*featuresum + fit_neg(2);

    yfeature = featuresum;
    yfeature(nanidx) = [];
    y_actual = y_res;
    y_actual(nanidx) = [];
    y_pred = behav_pred_neg;
    y_pred(nanidx) = [];

    [R_neg, P_neg] = corr(y_pred,y_actual,'rows','pairwise');

    R_iters(iteri) = R_neg;

end

%% Running the same experienment with the actual features of pathology signature from the CR-RANN data

iteri = iteri + 1;

y = 1/2*([LDELfu_ADNIMERGE+ADASDWRfu_ADNIMERGE] - [LDELbl_ADNIMERGE+ADASDWRbl_ADNIMERGE]);
baseline = 1/2*([LDELbl_ADNIMERGE+ADASDWRbl_ADNIMERGE]);

outliers=[find(y<(nanmean(y)-3*nanstd(y)));find(y>(nanmean(y)+3*nanstd(y)))];
y(outliers) = NaN;

nanidx = find(isnan(y));

x = [age sex education lhmeanthick rhmeanthick baseline PACCfulengh_Mnum-PACCbllengh_Mnum];

for i = 1:size(x,2)
    if ismember(find(isnan(x(:,i))),nanidx)
    else
        nanidx = [nanidx;find(isnan(x(:,i)))];
    end
end

SCnode = zeros(roi_num,length(subjects));
for j = 1:length(subjects)
    for i = 1:roi_num
        SCnode(i,j) = nansum(SCdata(i,:,j))/sum((SCdata(i,:,j)>0));
    end
end

PETSCNetScore_CS = zeros(roi_num,length(subjects));
for j = 1:length(subjects)
    for i = 1:roi_num
        SCdata_pos = SCdata(i,:,j);
        PETSCNetScore_CS(i,j) = nansum(SCdata_pos)*PETdata(i,j)...
            /length(find(SCdata_pos>0));
    end
end

PETSCNetScore = zeros(roi_num,length(subjects));
for j = 1:length(subjects)
    for i = 1:roi_num

        SCdata_pos = SCdata(i,:,j);
        PETSCNetScore(i,j) = (nansum(PETdata(:,j)'.*SCdata_pos)+PETdata(i,j))...
            /(length(find(SCdata_pos>0))+1);

    end
end

data = PETSCNetScore;

y(nanidx) = [];
data(:,nanidx) = [];
x(nanidx,:) = [];

all_behav = y;
b = glmfit(x,all_behav);
y_res = all_behav - x*b(2:end) - b(1);


fit_neg = [-0.4346    0.5590 ];
neg_mask_idx = [15,17,22,24,44,57,63,83,84,95,107,108,109,112,115,117,118,121,122,174,175,176,191]; % 23 region selected in 95% time of the cross-validated model from CR-RANN data

neg_mask_overall = zeros(size(data,1),1);
neg_mask_overall(neg_mask_idx) = 1;

featuresum = zeros(size(data,2),1);
for ss=1:size(data,2)
    featuresum(ss) = nansum(data(:,ss).*neg_mask_overall);  
end

nanidx = find(featuresum==0);
behav_pred_neg=fit_neg(1)*featuresum + fit_neg(2);

yfeature = featuresum;
yfeature(nanidx) = [];
y_actual = y_res;
y_actual(nanidx) = [];
y_pred = behav_pred_neg;
y_pred(nanidx) = [];

[R_neg, P_neg] = corr(y_pred,y_actual,'rows','pairwise');

R_iters(iteri) = R_neg;
R_iters(iteri)

%%

P_random = length(find(R_iters>=R_iters(end)))/length(R_iters)

%%

load ADNI_Data/colormap.mat

map(6,:) = map2(3,:);
map(6,:) = map2(4,:);

xbeta = y_pred;
ybeta = y_actual;
figure,
scatter(xbeta,ybeta,120,map(1,:),'filled','MarkerFaceAlpha',.4)
xlabel(['Precited cognitive change'])
ylabel(['Actual cognitive change'])

set(gca,'FontSize',20)
% grid on

hold on
mdl = fitlm(xbeta,ybeta);
intercept = table2array(mdl.Coefficients(1,1));
beta = table2array(mdl.Coefficients(2,1));
% pValue = table2array(mdl.Coefficients(2,4));
pValue = P_random;
tvalue = table2array(mdl.Coefficients(2,3));

xi = linspace(min(xbeta),max(xbeta)) ;
yi = beta*xi+intercept;
plot(xi,yi,'--','LineWidth',2,'Color',[0.7 0.7 0.7]) ;
[R,P] = corrcoef(xbeta,ybeta);

str = {['Correlation: R = ',num2str(R(1,2),'%.4f'),'; P = ',num2str(P(1,2),'%.4f')],['Permutation P = ',num2str(pValue,'%.4f')]};
a = annotation('textbox', [0.14, 0.81, 0.1, 0.1], 'String', str,'LineStyle','none');
a.FontSize = 18;
a.Color = [0.5 0.5 0.5];

xlim([-0.45 0.2])
ylim([-2 1.8])

