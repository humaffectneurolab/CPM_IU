clear; clc;


fileList_path = "fileList";
phenoFile_path = "phenoFile";
input_dir = "cpm_outputs_final/";
covarFile_path = "covarFile";
output_path = fullfile(input_dir, "dynamic_state_metrics.csv");

m = importdata(fileList_path);
all_behav = readmatrix(phenoFile_path);
all_covar = readmatrix(covarFile_path);
num_subjects = length(m);

% 1.Dominance_Time 2.Pos_DwellTime 3.Pos_PeakAmp 4.Neg_DwellTime 5.Neg_TroughAmp
all_subject_metrics = zeros(num_subjects, 5);
subject_ids = cell(num_subjects, 1);

state_ts_all = zeros(242,num_subjects); % rows are number of timepoints

for i = 1:num_subjects
    [~, task_filename, ~] = fileparts(m{i});
    sub_id_parts = split(task_filename, '_');
    sub_id = sub_id_parts{1};
    subject_ids{i} = sub_id;
    
    pos_ts_path = fullfile(input_dir, [sub_id, '_positive_network_timeseries.csv']);
    neg_ts_path = fullfile(input_dir, [sub_id, '_negative_network_timeseries.csv']);
    
    if ~isfile(pos_ts_path) || ~isfile(neg_ts_path)
        all_subject_metrics(i, :) = NaN;
        continue;
    end
    
    % State timeseries
    pos_ts = readmatrix(pos_ts_path);
    neg_ts = readmatrix(neg_ts_path);
    state_ts = pos_ts - neg_ts;
    
    % Metric 1
    dominance_total_time = sum(state_ts > 0);
    
    % define transitions
    states = sign(state_ts);
    transitions = find(diff(states) ~= 0);
    
    % identify segments of consistent positive or negative state dynamics
    segment_starts = [1; transitions + 1];
    segment_ends = [transitions; length(states)];
    
    pos_dwell_times = [];
    pos_peak_amps = [];
    neg_dwell_times = [];
    neg_trough_amps = [];
    
    for j = 1:length(segment_starts)
        start_idx = segment_starts(j);
        end_idx = segment_ends(j);
        segment = state_ts(start_idx:end_idx);
        
        if states(start_idx) > 0 % Pos state
            pos_dwell_times = [pos_dwell_times; length(segment)]; % Metric 2
            pos_peak_amps = [pos_peak_amps; max(segment)]; % Metric 3
        elseif states(start_idx) < 0 % Neg state
            neg_dwell_times = [neg_dwell_times; length(segment)]; % Metric 4
            neg_trough_amps = [neg_trough_amps; abs(min(segment))]; % Metric 5
        end
    end
    
    all_subject_metrics(i, 1) = dominance_total_time;
    all_subject_metrics(i, 2) = mean(pos_dwell_times, 'omitnan');
    all_subject_metrics(i, 3) = mean(pos_peak_amps, 'omitnan');
    all_subject_metrics(i, 4) = mean(neg_dwell_times, 'omitnan');
    all_subject_metrics(i, 5) = mean(neg_trough_amps, 'omitnan');

    state_ts_all(:,i) = state_ts;
end

metric_labels = {'Dominance_Time','Pos_DwellTime','Pos_PeakAmp','Neg_DwellTime','Neg_TroughAmp'};
results_table = array2table(all_subject_metrics, 'VariableNames', metric_labels, 'RowNames', subject_ids);
writetable(results_table, output_path, 'WriteRowNames', true);

%% Behavior correlation

corr_out = zeros(2,5);

for i = 1:length(metric_labels)

    valid_indices = ~isnan(all_subject_metrics(:, i));
    
    metric_data = all_subject_metrics(valid_indices, i);
    behav_data = all_behav(valid_indices);
    covar_data = all_covar(valid_indices, :);
    
    % regress covariates out
    X = [ones(sum(valid_indices), 1), covar_data];
    
    b_metric = X \ metric_data;
    residuals_metric = metric_data - X * b_metric;
    
    b_behav = X \ behav_data;
    residuals_behav = behav_data - X * b_behav;
    
    [R, P] = corr(residuals_metric, residuals_behav);
    corr_out(1,i) = R; corr_out(2,i) = P;
    writematrix(corr_out, fullfile(input_dir, "behavior_correlation.csv"));
    
    figure();
    hold on;
    
    mdl = fitlm(residuals_behav, residuals_metric);
    
    x_fit = linspace(min(residuals_behav) - 1, max(residuals_behav) + 1, 100)';
    [y_pred, y_ci] = predict(mdl, x_fit, 'Prediction', 'curve');
    fill([x_fit', fliplr(x_fit')], [y_ci(:,1)', fliplr(y_ci(:,2)')], ...
         [1 0.7 0.7], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    
    scatter(residuals_behav, residuals_metric, 100, [0.4 0.65 0.75], 'filled');
    
    plot(x_fit, y_pred, 'r-', 'LineWidth', 2.5);
    
    set(gca, 'TickLength', [0, 0]);
    saveas(gcf, fullfile(input_dir, sprintf("behavior_correlation_plot_%s.png", metric_labels{i})));
    
    hold off;
    close(gcf);
end
