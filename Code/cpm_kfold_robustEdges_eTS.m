clear; clc;

fileList_path = "fileList";
phenoFile_path = "phenoFile";
covarFile_path = "covarFile";
ts_dir = "parcellated_timeseries_rest/";
output_dir = "cpm_outputs/";

thresh = 0.01;
k = 10; 

m = importdata(fileList_path);
task_mats = zeros(268, 268, length(m));
for i = 1:length(m)
    task_mats(:,:,i) = readmatrix(m{i});
end

all_behav = readmatrix(phenoFile_path);
all_covar = readmatrix(covarFile_path);

no_sub = size(task_mats, 3);
no_node = size(task_mats, 1);

behav_pred_pos = [];
behav_pred_neg = [];
behav_pred = [];
test_behav = [];
test_folds = [];
pos_edges = zeros(268,268);
neg_edges = zeros(268,268);

%% K-fold CPM
cv = cvpartition(no_sub, 'KFold', k);

for folds = 1:k
    
    train_idx = cv.training(folds);
    train_mats = task_mats(:, :, train_idx);
    train_behav = all_behav(train_idx);
    train_covar = all_covar(train_idx, :);
    
    train_vcts_corr = reshape(train_mats, no_node*no_node, []);
    [r_mat, p_mat] = partialcorr(train_vcts_corr', train_behav, train_covar);
    p_mat = reshape(p_mat, no_node, no_node);
    r_mat = reshape(r_mat, no_node, no_node);

    pos_mask = (r_mat > 0 & p_mat < thresh);
    neg_mask = (r_mat < 0 & p_mat < thresh);
    
    test_idx = find(cv.test(folds));

    train_vcts_sum = reshape(train_mats, no_node*no_node, []);
    train_sumpos = (pos_mask(:)' * train_vcts_sum) / 2;
    train_sumneg = (neg_mask(:)' * train_vcts_sum) / 2;
    
    fit_pos = polyfit(train_sumpos, train_behav, 1);
    fit_neg = polyfit(train_sumneg, train_behav, 1);
    b = regress(train_behav, [train_sumpos', train_sumneg', ones(cv.TrainSize(folds), 1)]);
    
    test_mat_fold = task_mats(:, :, test_idx);
    test_vcts_fold = reshape(test_mat_fold, no_node*no_node, []);
    test_sumpos = (pos_mask(:)' * test_vcts_fold) / 2;
    test_sumneg = (neg_mask(:)' * test_vcts_fold) / 2;
    
    behav_pred_pos_new = fit_pos(1) * test_sumpos' + fit_pos(2);
    behav_pred_neg_new = fit_neg(1) * test_sumneg' + fit_neg(2);
    behav_pred_new = b(1) * test_sumpos' + b(2) * test_sumneg' + b(3);
    
    behav_pred_pos = [behav_pred_pos; behav_pred_pos_new];
    behav_pred_neg = [behav_pred_neg; behav_pred_neg_new];
    behav_pred = [behav_pred; behav_pred_new];
    test_behav = [test_behav; all_behav(test_idx)];
    test_folds = [test_folds; test_idx];
    pos_edges = pos_edges + pos_mask;
    neg_edges = neg_edges + neg_mask;
end

[~, test_order] = sort(test_folds);
behav_pred_pos = behav_pred_pos(test_order);
behav_pred_neg = behav_pred_neg(test_order);
behav_pred = behav_pred(test_order);
test_behav = test_behav(test_order);

[R_pos, P_pos] = corr(behav_pred_pos, test_behav);
[R_neg, P_neg] = corr(behav_pred_neg, test_behav);
[R_both, P_both] = corr(behav_pred, test_behav);

row_names = {'Positive_Model'; 'Negative_Model'; 'Combined_Model'};
R_values = [R_pos; R_neg; R_both; R_both_sp];
P_values = [P_pos; P_neg; P_both; P_both_sp];

results_table = table(R_values, P_values, 'RowNames', row_names);

writetable(results_table, fullfile(output_dir, "performance_summary.csv"), 'WriteRowNames', true);

writematrix(fullfile(output_dir, "predicted_behavior_combined.csv"), behav_pred);
writematrix(fullfile(output_dir, "predicted_behavior_positive.csv"), behav_pred_pos);
writematrix(fullfile(output_dir, "predicted_behavior_negative.csv"), behav_pred_neg);


%% Robust edge selection

pos_edges_robust = pos_edges==k;
neg_edges_robust = neg_edges==k;

writematrix(pos_edges_robust, fullfile(output_dir, "positive_edge_robust.csv"));
writematrix(neg_edges_robust, fullfile(output_dir, "negative_edge_robust.csv"));

[row_pos, col_pos] = find(pos_edges_robust);
[row_neg, col_neg] = find(neg_edges_robust);


%% Edge timeseries

for i = 1:no_sub

    [~, task_filename, ~] = fileparts(m{i});
    sub_id_parts = split(task_filename, '_');
    sub_id = sub_id_parts{1};
    
    ts_path = fullfile(ts_dir, [sub_id, '_task-rest_confregressed_000_TS.csv']);
    
    if isfile(ts_path)
        rest_ts = readmatrix(ts_path);
        z_rest_ts = zscore(rest_ts, 0, 2);
        
        pos_network_ts = sum(z_rest_ts(row_pos, :) .* z_rest_ts(col_pos, :), 1);
        neg_network_ts = sum(z_rest_ts(row_neg, :) .* z_rest_ts(col_neg, :), 1);
        
        pos_ts_out_path = fullfile(output_dir, [sub_id, '_positive_network_timeseries.csv']);
        neg_ts_out_path = fullfile(output_dir, [sub_id, '_negative_network_timeseries.csv']);
        
        writematrix(pos_network_ts', pos_ts_out_path);
        writematrix(neg_network_ts', neg_ts_out_path);
    else
        fprintf('Skipping %s -- no timeseries data.\n', sub_id);
    end
end
