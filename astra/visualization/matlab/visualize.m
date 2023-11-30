function [outputArg1,outputArg2] = visualize(type_in, psize_in, nr_in, met_in, org_in, file)
% Astra visualization

repo_root = "/Users/amithkamath/repo/astra/";
data_root = "/Users/amithkamath/data/DLDP/";
code_root = "/Users/amithkamath/repo/astra/";

% 82 and 90 -> smallest TV, 81, 88
case_nr = ["DLDP_070", "DLDP_071", "DLDP_072", "DLDP_073", "DLDP_074", "DLDP_075", "DLDP_076", "DLDP_078", "DLDP_079", "DLDP_080", "DLDP_081", "DLDP_082", "DLDP_083", "DLDP_084", "DLDP_085", "DLDP_086", "DLDP_087", "DLDP_088", "DLDP_089", "DLDP_090", "DLDP_091", "DLDP_092", "DLDP_093", "DLDP_094",];

%%% Perturbation type
pert_type = ["Erosion", "Dilation"];

%%% Perturbation size
pert_size = [2, 3, 4, 5, 6];

%%% metrics evaluated
metric = ["Max", "Mean", "DMax", "DMean"];

%%% List of organs
% oar_list = ["Brainstem", "Hippocampus_L", "Hippocampus_R", "Eye_L", "Eye_R", "Chiasm", "OpticNerve_R", "OpticNerve_L"];
% target = "Target";
organs = ["Target", "Brainstem", "Hippocampus_L", "Hippocampus_R", "Eye_L", "Eye_R", "Chiasm", "OpticNerve_L", "OpticNerve_R", "Brain"];

%%% avoid min values to be 0
eps = 1e-12;

%%% Choose type (type), case nr (nr), values of which organ (org), and kind
%%% of metric (met)
type = type_in;         % 1,2
psize = psize_in;       % [1, 4]
nr = nr_in;             % [1, 4]
met = met_in;           % [1, 4]
org = org_in;           % [1, 9], if we want the whole brain org = 10




%%% Input Ground truth of all organs as well as the Brain
gt = zeros(length(organs), 128, 128, 128);
for i = 1:length(organs)
    gt(i, :, :, :) = niftiread(fullfile(data_root, "ground_truth/" + case_nr(nr) + "/" + organs(i) + ".nii.gz"));
end


ct_data = niftiread(fullfile(data_root, "ground_truth/" + case_nr(nr) + "/CT.nii.gz"));
organ_size = zeros(1, length(organs));

%%% Calculate Organ sizes
for i = 1:length(organs)
    organ_size(i) = nnz(gt(i,:,:,:));
end

%%% Config for colormaps and scene
groundtruthConfig = load(fullfile(code_root, "astra", "visualization", "groundtruthConfig.mat"));
sceneConfig = load(fullfile(code_root, "astra", "visualization", "sceneConfig.mat"));
heatmapConfig = load(fullfile(code_root, "astra", "visualization", "heatmapConfig2.mat"));
heatmapConfigT = load(fullfile(code_root, "astra", "visualization", "heatmapConfig2.mat"));
objectConfigJet = load(fullfile(code_root, "astra", "visualization", "objectConfigJet.mat"));
objectConfigJetV2 = load(fullfile(code_root, "astra", "visualization", "objectConfigJetV2.mat"));
colormp = load(fullfile(code_root, "astra", "visualization", "colormapBlRd.mat"));
colormapRed = load(fullfile(code_root, "astra", "visualization", "colormapRed.mat"));
colormapBrew = load(fullfile(code_root, "astra", "visualization", "colormapBrew.mat"));
colormapRedBrew = load(fullfile(code_root, "astra", "visualization", "colormapRedBrew.mat"));
colormapRedBrew = load(fullfile(code_root, "astra", "visualization", "colormapRedBrew.mat"));
% alphamapN = load(fullfile(repo_root, "alphamapDiff.mat"));

% objectConfigJetV2.objectConfigJetV2.Alphamap = objectConfig.objectConfig1.Alphamap;

max_ct = max(ct_data(:));
ct_data(1, :, :) = max_ct;
ct_data(:, 1, :) = max_ct;
ct_data(:, :, 1) = max_ct;
ct_data(128, :, :) = max_ct;
ct_data(:, 128, :) = max_ct;
ct_data(:, :, 128) = max_ct;

% vol = volshow(ct_data, RenderingStyle="SlicePlanes");
% vol.SlicePlaneValues = [-1 0 0 97; 0 -1 0 68; 0 0 -1 2];

props = regionprops3(squeeze(gt(1, :, :, :)), "Centroid");
sceneConfig.sceneConfig.CameraTarget = props.Centroid;
sceneConfig.sceneConfig.BackgroundColor = [0, 0, 0];
sceneConfig.sceneConfig.Lighting = 0;

%view_panel = uipanel(figure,'Title', "Target Volume for case:" + string(nr_in));
%view_panel.FontSize=14;
%view_panel.TitlePosition='centertop';
%vol = volshow(squeeze(gt(1, :, :, :)), groundtruthConfig.groundtruthConfig, 'Parent', view_panel);

% groundtruthConfig.groundtruthConfig.Colormap = [0.980, 0.298, 0.337];
vol = volshow(squeeze(gt(1, :, :, :)), groundtruthConfig.groundtruthConfig); %, "Parent", vol.Parent);
scenePropsT = fields(sceneConfig.sceneConfig);
for prop_idx = 1:length(scenePropsT)
    try
        vol.Parent.(scenePropsT{prop_idx}) = sceneConfig.sceneConfig.(scenePropsT{prop_idx});
    catch
    end
end


%%% Dose prediction without perturbation
pred_tv_gt = niftiread(fullfile(data_root, "prediction/" + pert_type(type) + "/" + case_nr(nr) + "/Prediction_NoPert" + organs(1) + ".nii.gz"));

%%% Number of element in the HL and the HR above a threshold
pred_hl_thresh = niftiread(fullfile(data_root, "prediction/" + pert_type(type) + "/" + case_nr(nr) + "/NumAboveThresh_" + organs(3) + ".nii.gz"));
pred_hr_thresh = niftiread(fullfile(data_root, "prediction/" + pert_type(type) + "/" + case_nr(nr) + "/NumAboveThresh_" + organs(4) + ".nii.gz"));

% max_val_hl = max(pred_hl_thresh(pred_hl_thresh > 0))
% min_val_hl = min(pred_hl_thresh(pred_hl_thresh > 0))
% mean_val_hl = mean(pred_hl_thresh(pred_hl_thresh > 0))
% 
% max_val_hr = max(pred_hr_thresh(pred_hr_thresh > 0))
% min_val_hr = min(pred_hr_thresh(pred_hr_thresh > 0))
% mean_val_hr = mean(pred_hr_thresh(pred_hr_thresh > 0))

%%% Input data of TV
pert_tv_organ = zeros(length(organs)-2, 128, 128, 128);
pert_tv_organ(1, :, :, :) = niftiread(fullfile(data_root, "prediction/" + pert_type(type) + "/" + case_nr(nr) + "/Perturbed_" + organs(1) + "_" + metric(met) + ".nii.gz"));

%%% Input data of OARs
for i = 2:length(organs)-1
    pert_tv_organ(i, :, :, :) = niftiread(fullfile(data_root, "prediction/" + pert_type(type) + "/" + case_nr(nr) + "/Perturbed_TV_" + organs(i) + "_" + metric(met) + ".nii.gz"));
end

ind = find(squeeze(pert_tv_organ(1,:,:,:)));
[i1, i2, i3] = ind2sub(size(squeeze(pert_tv_organ(1,:,:,:))), ind);


%%%  Output String
curr_org_str = "Perturbed TV: " + metric(met) + " of " + organs(org) + " on Target - " + case_nr(nr);

%%% store data for depiction for further manipulation
curr_org = squeeze(pert_tv_organ(org, :, :, :));

%%% Gradient calculation
[gradMag, gradDir, gradEl] = imgradient3(curr_org);


%%% not normed min/max/mean values
max_val = max(curr_org(curr_org > 0));
min_val = min(curr_org(curr_org > 0));
mean_val = mean(curr_org(curr_org > 0));

%%% Number of points perturbed
num_pert_pt = nnz(pert_tv_organ(1,:,:,:));

%%% Display Max, Min, Mean
disp("Max of " + metric(met) + " dose in the " + organs(org) + " is: " + string(max_val))
disp("Min of " + metric(met) + " dose in the " + organs(org) + " is: " + string(min_val))
disp("Mean of " + metric(met) + " dose in the " + organs(org) + " is: " + string(mean_val))

% Correlation calculations
ind = find(curr_org);
[i1, i2, i3] = ind2sub(size(curr_org), ind);

pert_tv_el = nonzeros(curr_org);
gt_tv_el = zeros(nnz(curr_org), 1);
for i = 1:nnz(curr_org)
    gt_tv_el(i,1) = pred_tv_gt(i1(i), i2(i), i3(i));
end

corr = corrcoef(gt_tv_el, pert_tv_el)     % [[AA, AB],[BA, BB]]

%%% Restriction calculations
restr = 0;
restrstr = "";
switch metric(met)
    case 'DMax'
        disp("No delta max values restraints defined.")
    case 'Max'
        if isequal(curr_org, squeeze(pert_tv_organ(2, :, :, :))) % BS
            restr = 54;
            num = nnz(curr_org(curr_org >= restr));
            restrstr = string(num / num_pert_pt *100) + "% of the perturbed points produce a dose larger than " + string(restr) + ...
                " Gy in the " + organs(org);
            disp(restrstr);

        elseif isequal(curr_org, squeeze(pert_tv_organ(3, :, :, :))) || isequal(curr_org, squeeze(pert_tv_organ(4, :, :, :))) % HL or HR
            restr = 7.4;
            perc = 0.4;
            numAbove = 0;

            maxAllowedPt = ceil(organ_size(org) * perc)
            if(org == 3)
                numAbove = pred_hl_thresh(pred_hl_thresh > maxAllowedPt);
            elseif(org == 4)
                numAbove = pred_hr_thresh(pred_hr_thresh > maxAllowedPt);
            end

            if (isempty(numAbove))
                numAbove = 0;
            end
            restrstr = string(numAbove / num_pert_pt *100) + "% of the perturbed points produce in more than " + string(perc*100) + ...
                "% of the " + organs(org) + " a dose larger than " + string(restr) + " Gy";
            disp(restrstr)

        elseif isequal(curr_org, squeeze(pert_tv_organ(5, :, :, :))) || isequal(curr_org, squeeze(pert_tv_organ(6, :, :, :))) %EL, ER
            restr = 10;
            num = nnz(curr_org(curr_org >= restr));
            restrstr = string(num / num_pert_pt *100) + "% of the perturbed points produce a dose larger than " + string(restr) + ...
                " Gy in the " + organs(org);
            disp(restrstr)

        elseif isequal(curr_org, squeeze(pert_tv_organ(7, :, :, :))) %Chiasm
            restr = 54;
            num = nnz(curr_org(curr_org >= restr));
            restrstr = string(num / num_pert_pt *100) + "% of the perturbed points produce a dose larger than " + string(restr) + ...
                " Gy in the " + organs(org);
            disp(restrstr)

        elseif isequal(curr_org, squeeze(pert_tv_organ(8, :, :, :))) || isequal(curr_org, squeeze(pert_tv_organ(9, :, :, :))) % OpNL, OpNR
            restr = 54;
            num = nnz(curr_org(curr_org >= restr));
            restrstr = string(num / num_pert_pt *100) + "% of the perturbed points produce a dose larger than " + string(restr) + ...
                " Gy in the " + organs(org);
            disp(restrstr)
        end
    case 'Mean'
        disp("No mean values restraints defined.")

    case 'DMean'
        disp("No delta mean values restraints defined.")
end


objectConfigJetV2.objectConfigJetV2.Colormap = colormp.colompBlRd;
% heatmapConfigT.objectConfig.Colormap = colormapBrew.colormpInv;
%%% limits setting and norming for volshow
max_cap = max(curr_org(curr_org > 0));
temp_max = max_cap;
if(met == 1)
    if((org == 2) ||(org == 5) || (org == 6) || (org == 7) || (org == 8) || (org == 9))
        if((org == 5) || (org == 6))
            max_allowedVal = restr;
        else
            max_allowedVal = (restr * 0.9);
        end
        if(max_allowedVal < min_val)
            curr_org = (curr_org - max_allowedVal + eps);
            curr_org(curr_org < 0) = eps;
            max_cap = max(curr_org(curr_org>0));

            %             colormapRedBrew.colormapRedBrew;
            objectConfigJetV2.objectConfigJetV2.Colormap = colormapRed.colormapRed;
        elseif ((max_allowedVal > min_val) && (max_allowedVal < max_val))
            curr_org = (curr_org - min_val + eps);
            curr_org(curr_org < 0) = eps;
            max_cap = max_allowedVal - min_val;
        elseif(max_allowedVal > max_val)
            curr_org = (curr_org - min_val + eps);
            curr_org(curr_org < 0) = eps;
            max_cap = max_allowedVal - min_val;
        end
    else
        curr_org = (curr_org - min_val + eps);
        curr_org(curr_org < 0) = eps;
        max_cap = max(curr_org(curr_org>0));
    end
else
    curr_org = (curr_org - min_val + eps);
    curr_org(curr_org < 0) = eps;
    max_cap = max(curr_org(curr_org>0));

end

curr_org = curr_org ./ max_cap;

min_scaled = min(curr_org(curr_org > 0));
max_scaled = max(curr_org(curr_org > 0));
min_overall = eps;

curr_org(curr_org == 0) = nan;

% colormp = colormap(brewermap([], 'RdYlBu'));

% objectConfigJetV2.objectConfigJetV2.Colormap = colormapBrew.colormpInv;

% curating data to show smooth volume
dil_organ = imdilate(curr_org, strel('sphere', 4));

surf_curr = zeros(size(squeeze(gt(1, :, :, :))));
surf_curr(squeeze(gt(1, :, :, :)) > 0) = dil_organ(squeeze(gt(1, :, :, :)) > 0);
surf_curr = imdilate(surf_curr, strel('sphere', 4));

myColormap = objectConfigJetV2.objectConfigJetV2.Colormap;
% objectConfigJetV2.objectConfigJetV2.Alphamap = alphamapN.myAlphamap;
middle = min_val +(restr - min_val)/2;

if(met == 1)
    if((org == 2) ||(org == 5) || (org == 6) || (org == 7) || (org == 8) || (org == 9))
        figure(1)
        colormap(myColormap);
        if(max_allowedVal < min_val)
            %             clim([0, max_scaled])
            %             c = colorbar('Ticks',[0 min_scaled max_scaled],...
            %                 'TickLabels',{"90% Restr: " + string(max_allowedVal), "Min: " + string(min_val), " Max: " + string(max_val)});
            c = colorbar('Ticks',[min_scaled max_scaled],...
                'TickLabels',{"Min: " + string(min_val), " Max: " + string(max_val)});
        elseif ((max_allowedVal > min_val) && (max_allowedVal < max_val))
            %             clim([0, max_scaled])
            min_overall = min_scaled;
            %             clim([0, max_scaled]);
            c = colorbar('Ticks',[min_scaled 1],...
                'TickLabels',{"Min: " + string(min_val), "90% Restr: " + string(max_allowedVal) + ",... Max: " + string(max_val)});
        elseif(max_allowedVal > max_val)
            %             clim([0, 1])
            min_overall = min_scaled;
            c = colorbar('Ticks',[min_scaled max_scaled 1],...
                'TickLabels',{"Min: " + string(min_val), "Max: " + string(max_val), "90% Restr: " + string(max_allowedVal)});
        end
        c.Location = 'northoutside';
        c.Label.String = metric(met) + ' [Gy]';
        axis off
        title(curr_org_str)
    else
        figure(1)
        colormap(myColormap);
        min_overall = min_scaled;
        c = colorbar('Ticks',[min_scaled max_scaled],...
            'TickLabels',{"Min: " + string(min_val), "Max: " + string(max_val)});
        c.Location = 'northoutside';
        c.Label.String = metric(met) + ' [Gy]';
        axis off
        title(curr_org_str)
    end
else
    figure(1)
    colormap(myColormap);
    min_overall = min_scaled;
    c = colorbar('Ticks',[min_scaled max_scaled],...
        'TickLabels',{"Min: " + string(min_val), "Max: " + string(max_val)});
    c.Location = 'northoutside';
    c.Label.String = metric(met) + ' [Gy]';
    axis off
    title(curr_org_str)
end
% lims = clim

vol = volshow(surf_curr, objectConfigJetV2.objectConfigJetV2, 'Parent', vol.Parent);

if(met == 1)
    if((org == 2) ||(org == 5) || (org == 6) || (org == 7) || (org == 8) || (org == 9))
        colormp = objectConfigJetV2.objectConfigJetV2.Colormap;
        %         alphamp = heatmapConfigT.objectConfig.Alphamap;


        VAxis = [min_overall, 1];
        newscale = linspace(min(surf_curr(surf_curr > 0)) - min(VAxis(:)),...
            max(surf_curr(surf_curr >0)) - min(VAxis(:)), size(colormp, 1))/diff(VAxis);
        newscale(newscale < 0) = 0;
        newscale(newscale > 1) = 1;

        vol.Colormap = interp1(linspace(0, 1, size(colormap, 1)), colormp, newscale);
    end
end

sceneProps = fields(sceneConfig.sceneConfig);
for prop_idx = 1:length(sceneProps)
    vol.Parent.(sceneProps{prop_idx}) = sceneConfig.sceneConfig.(sceneProps{prop_idx});
end


volshow(squeeze(gt(1, :, :, :)), groundtruthConfig.groundtruthConfig, 'Parent', vol.Parent);
sceneProps = fields(sceneConfig.sceneConfig);
for prop_idx = 1:length(sceneProps)
    vol.Parent.(sceneProps{prop_idx}) = sceneConfig.sceneConfig.(sceneProps{prop_idx});
end


if (org ==1)
    for i = 1:(length(organs)-1)

        volshow(squeeze(gt(i, :, :, :)), groundtruthConfig.groundtruthConfig, 'Parent', vol.Parent);
        sceneProps = fields(sceneConfig.sceneConfig);
        for prop_idx = 1:length(sceneProps)
            vol.Parent.(sceneProps{prop_idx}) = sceneConfig.sceneConfig.(sceneProps{prop_idx});
        end
    end
else
    volshow(squeeze(gt(org, :, :, :)), groundtruthConfig.groundtruthConfig, 'Parent', vol.Parent);
    sceneProps = fields(sceneConfig.sceneConfig);
    for prop_idx = 1:length(sceneProps)
        vol.Parent.(sceneProps{prop_idx}) = sceneConfig.sceneConfig.(sceneProps{prop_idx});
    end
end
%% Output to Excel
% outp = [metric(met), pert_type(type), pert_size(psize), organs(org), max_val, min_val, mean_val, corr(1,2), restr, restrstr];
% T = array2table(outp, 'VariableNames',{'Metric', 'Pert Type', 'Pert Size', 'Organ' 'Max', 'Min', 'Mean', 'Corr', 'Restriction', 'Message'});
%
% if(file == 1)
%     filename = 'MATLAB/data_output/output_val.xlsx';
% elseif(file == 2)
%     filename = 'MATLAB/data_output/output_val_ptv.xlsx';
% end
%
%
% filename = 'MATLAB/data_output/output_val_ptv.xlsx';
% sheetName = case_nr(nr);
% try
%     existingData = readtable(filename, 'Sheet', sheetName, 'VariableNamingRule', 'preserve');
% catch
%     Handle the case when the sheet does not exist or the file is empty
%     existingData = table();
% end
%
%
% Concatenate existing data with the new data
% combinedData = [existingData; T];
%
% writetable(combinedData, filename, "Sheet", sheetName)



%% Output Video
% filename = fullfile(repo_root, "results", "animated_" + case_nr + ".avi");
%
% viewer = vol.Parent;
% hFig = viewer.Parent;
%
% vol_size = size(gt_target);
%
% numberOfFrames = 360/2;
% vec = linspace(0,2*pi,numberOfFrames)';
% dist = sqrt(vol_size(1)^2 + vol_size(2)^2 + vol_size(3)^2);
% myPosition = props.Centroid + ([cos(vec) sin(vec) ones(size(vec))]*dist);
%
% v = VideoWriter(filename,'Uncompressed AVI');
% open(v)
%
% for idx = 1:length(vec)
%     % Update the current view
%     viewer.CameraPosition = myPosition(idx,:);
%     % Capture the image using the getframe function
%     I = getframe(hFig);
%     % Write the frame to file
%     writeVideo(v,I.cdata)
% end
% close(v)

%% Clear all
% clear all
% close all
% handles = findall(groot, 'Type', 'figure', 'Tag', 'volshow');
% % close(handles);
end
