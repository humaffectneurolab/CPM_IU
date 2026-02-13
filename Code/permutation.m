clear; clc;

fileList_path = "fileList";
phenoFile_path = "phenoFile";
covarFile_path = "covarFile";
rest_ts_dir = "parcellated_timeseries_rest/";
output_dir = "cpm_outputs/";


m = importdata(fileList_path);
all_mats = zeros(268, 268, length(m));
for i = 1:length(m)
    all_mats(:,:,i) = readmatrix(m{i});
end

all_behav = readmatrix(phenoFile_path);
all_covar = readmatrix(covarFile_path);

[true_r_pos, true_r_neg, true_r_comp] = predict_behavior(all_mats, all_behav, all_covar);

no_iterations = 5000; 
prediction_r = zeros(no_iterations, 3);
prediction_r(1, 1) = true_r_pos;
prediction_r(1, 2) = true_r_neg;
prediction_r(1, 3) = true_r_comp;

no_sub = size(all_mats, 3);

for iter = 2:no_iterations
    new_behav = all_behav(randperm(no_sub));
    [prediction_r(iter, 1), prediction_r(iter, 2), prediction_r(iter, 3)] = predict_behavior(all_mats, new_behav, all_covar);    
end

sorted_r_pos = sort(prediction_r(:,1), 'descend');
position_pos = find(sorted_r_pos == true_r_pos, 1);
pval_pos = position_pos / no_iterations;
fprintf('\n\nP-value (Positive): %f\n', pval_pos);

sorted_r_neg = sort(prediction_r(:,2), 'descend');
position_neg = find(sorted_r_neg == true_r_neg, 1);
pval_neg = position_neg / no_iterations;
fprintf('P-value (Negative): %f\n', pval_neg);

sorted_r_comp = sort(prediction_r(:,3), 'descend');
position_comp = find(sorted_r_comp == true_r_comp, 1);
pval_comp = position_comp / no_iterations;
fprintf('P-value (Composite): %f\n', pval_comp);


function [R_pos, R_neg, R_comp] = predict_behavior(all_mats, all_behav, all_covar)

    thresh = 0.01;
    k = 10;

    no_sub = size(all_mats, 3);
    no_node = size(all_mats, 1);
    
    behav_pred_pos = zeros(no_sub, 1);
    behav_pred_neg = zeros(no_sub, 1);
    behav_pred_comp = zeros(no_sub, 1);
    
    cv = cvpartition(no_sub, 'KFold', k);

    for k = 1:cv.NumTestSets
        train_idx = training(cv, k);
        test_idx = test(cv, k);

        train_mats = all_mats(:, :, train_idx);
        train_behav = all_behav(train_idx);
        train_covar = all_covar(train_idx, :);
        test_mats = all_mats(:, :, test_idx);
        
        train_vcts = reshape(train_mats, [], size(train_mats, 3));

        [r_mat, p_mat] = partialcorr(train_vcts', train_behav, train_covar);
        
        r_mat = reshape(r_mat, no_node, no_node);
        p_mat = reshape(p_mat, no_node, no_node);
        
        pos_mask = (r_mat > 0 & p_mat < thresh);
        neg_mask = (r_mat < 0 & p_mat < thresh);
        
        train_sumpos = zeros(sum(train_idx), 1);
        train_sumneg = zeros(sum(train_idx), 1);
        
        for s = 1:length(train_sumpos)
            train_sumpos(s) = sum(sum(train_mats(:, :, s) .* pos_mask)) / 2;
            train_sumneg(s) = sum(sum(train_mats(:, :, s) .* neg_mask)) / 2;
        end
        
        fit_pos = polyfit(train_sumpos, train_behav, 1);
        fit_neg = polyfit(train_sumneg, train_behav, 1);
        b_comp = regress(train_behav, [train_sumpos, train_sumneg, ones(sum(train_idx),1)]);

        test_subject_indices = find(test_idx);
        for t = 1:length(test_subject_indices)
            current_subject_id = test_subject_indices(t);
            test_mat = test_mats(:, :, t);
            
            test_sumpos = sum(sum(test_mat .* pos_mask)) / 2;
            test_sumneg = sum(sum(test_mat .* neg_mask)) / 2;
            
            behav_pred_pos(current_subject_id) = fit_pos(1) * test_sumpos + fit_pos(2);
            behav_pred_neg(current_subject_id) = fit_neg(1) * test_sumneg + fit_neg(2);
            behav_pred_comp(current_subject_id) = b_comp(1)*test_sumpos + b_comp(2)*test_sumneg + b_comp(3);
        end
    end
    
    [R_pos, ~] = corr(behav_pred_pos, all_behav);
    [R_neg, ~] = corr(behav_pred_neg, all_behav);
    [R_comp, ~] = corr(behav_pred_comp, all_behav);
end

writematrix(prediction_r,'permutation_matrix.csv');