clear; clc;

fileList_path = "fileList";
phenoFile_path = "phenoFile";
covarFile_path = "covarFile";
ts_dir = "parcellated_timeseries_rest/";
output_dir = "cpm_outputs/";

thresh = 0.01;
k = 10;
n_repeats = 1000;

m = importdata(fileList_path);
task_mats = zeros(268, 268, length(m));
for i = 1:length(m)
    task_mats(:,:,i) = readmatrix(m{i});
end

all_behav = readmatrix(phenoFile_path);
all_covar = readmatrix(covarFile_path);
no_sub = size(task_mats, 3);
no_node = size(task_mats, 1);

distribution_R = zeros(n_repeats, 4);

%% Repeated CPM

for iter = 1:n_repeats
    
    behav_pred_pos = [];
    behav_pred_neg = [];
    behav_pred = [];
    test_behav = [];
    test_folds = [];
    
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
    end
    
    [~, test_order] = sort(test_folds);
    behav_pred_pos = behav_pred_pos(test_order);
    behav_pred_neg = behav_pred_neg(test_order);
    behav_pred = behav_pred(test_order);
    test_behav_ordered = test_behav(test_order);
    
    [R_pos, ~] = corr(behav_pred_pos, test_behav_ordered);
    [R_neg, ~] = corr(behav_pred_neg, test_behav_ordered);
    [R_both, ~] = corr(behav_pred, test_behav_ordered);
    
    distribution_R(iter, :) = [R_pos, R_neg, R_both];
end


dist_table = array2table(distribution_R, 'VariableNames', {'R_Pos', 'R_Neg', 'R_Combined'});
writetable(dist_table, fullfile(output_dir, "performance_distribution.csv"));

mean_R = mean(distribution_R, 1)';
std_R = std(distribution_R, 0, 1)';
min_R = min(distribution_R, [], 1)';
max_R = max(distribution_R, [], 1)';

row_names = {'Positive_Model'; 'Negative_Model'; 'Combined_Model_Pearson'};
results_table = table(mean_R, std_R, min_R, max_R, 'RowNames', row_names, 'VariableNames', {'Mean_R', 'Std_R', 'Min_R', 'Max_R'});
writetable(results_table, fullfile(output_dir, "performance_summary_stats.csv"), 'WriteRowNames', true);

%% Visualization
plot_data = table2array(dist_table(:, 1:3)); 
figure('Name', 'CPM Model Stability', 'Color', 'w', 'Position', [200, 200, 600, 500]);

group_labels = {'Positive', 'Negative', 'Combined'};

boxplot(plot_data, 'Labels', group_labels, 'Widths', 0.5, ...
    'Colors', 'k', 'Symbol', 'k+'); 

set(gca, 'FontSize', 12, 'LineWidth', 1.2);
set(findobj(gca,'type','line'), 'LineWidth', 1.5);

set(gca, 'TickLength', [0 0]);
grid off;
box off;

saveas(gcf, fullfile(output_dir, 'plot_stability_boxplot.png'));