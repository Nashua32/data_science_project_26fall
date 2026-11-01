%% Williams-Otto reactor: N4SID training and two validation datasets
% Requires System Identification Toolbox.
%
% Training:
%   data/n4sid/train.csv
%
% Validation:
%   1. data/n4sid/validation.csv
%   2. Last 20% of data/data_generic.csv
%
% Initial states are estimated separately from each validation dataset.
% The model matrices are fitted using training data only.

clear;
clc;
close all;

%% 1. Settings
projectRoot = ...
    '/Users/nilsbulitta/Documents/GitHub/data_science_project_26fall';

dataDir = fullfile(projectRoot, 'data');
n4sidDir = fullfile(dataDir, 'n4sid');

inputNames  = {'QB', 'Tr'};
outputNames = {'cA', 'cB', 'cC', 'cP', 'cE', 'cG'};

nx = 6;  % Number of model states: starting choice

%% 2. Load training and validation data
trainTable = loadReactorCSV( ...
    fullfile(n4sidDir, 'train.csv'), ...
    inputNames, outputNames);

originalValTable = loadReactorCSV( ...
    fullfile(n4sidDir, 'validation.csv'), ...
    inputNames, outputNames);

genericTable = loadReactorCSV( ...
    fullfile(dataDir, 'data_generic.csv'), ...
    inputNames, outputNames);

% Select the final 20% chronologically.
firstGenericValRow = floor(0.8 * height(genericTable)) + 1;
genericValTable = genericTable(firstGenericValRow:end, :);

assert(height(genericValTable) >= 2, ...
    'The generic validation segment needs at least two samples.');

%% 3. Determine sampling intervals
TsTrain = getSampleTime(trainTable);
TsOriginal = getSampleTime(originalValTable);
TsGeneric = getSampleTime(genericValTable);

fprintf('Training:            %d samples, Ts = %.6g min\n', ...
    height(trainTable), TsTrain);

fprintf('Original validation: %d samples, Ts = %.6g min\n', ...
    height(originalValTable), TsOriginal);

fprintf('Generic last 20%%:    %d samples, Ts = %.6g min\n\n', ...
    height(genericValTable), TsGeneric);

%% 4. Compute preprocessing using TRAINING data only
Utrain = trainTable{:, inputNames};
Ytrain = trainTable{:, outputNames};

uOffset = mean(Utrain, 1);
yOffset = mean(Ytrain, 1);

uScale = std(Utrain, 0, 1);
yScale = std(Ytrain, 0, 1);

assert(all(uScale > 0) && all(yScale > 0), ...
    'Every training input and output must have nonzero variance.');

%% 5. Build the scaled training dataset
trainData = makeIdentificationData( ...
    trainTable, TsTrain, ...
    inputNames, outputNames, ...
    uOffset, yOffset, uScale, yScale);

%% 6. Fit N4SID using ONLY training data
options = n4sidOptions('Focus', 'simulation');

model = n4sid(trainData, nx, ...
    'Feedthrough', false, ...
    'DisturbanceModel', 'none', ...
    options);

model.TimeUnit = 'minutes';

disp(model);

fprintf('Model stability: %d\n\n', isstable(model));

% These matrices operate on the centered and scaled signals.
A = model.A;
B = model.B;
C = model.C;
D = model.D;

%% 7. Validate on the original validation experiment
[originalMetrics, originalPredictions, modelOriginal, x0Original] = ...
    evaluateDataset( ...
        model, originalValTable, TsOriginal, ...
        'Original validation', ...
        inputNames, outputNames, ...
        uOffset, yOffset, uScale, yScale);

%% 8. Validate on the final 20% of data_generic
[genericMetrics, genericPredictions, modelGeneric, x0Generic] = ...
    evaluateDataset( ...
        model, genericValTable, TsGeneric, ...
        'Generic last 20%', ...
        inputNames, outputNames, ...
        uOffset, yOffset, uScale, yScale);

%% 9. Display MAE and RMSE for both datasets
metrics = [originalMetrics; genericMetrics];

fprintf('\nErrors in physical concentration units [mol/dm^3]:\n');
disp(metrics);

%% 10. Save model, preprocessing, metrics and predictions
writetable(metrics, ...
    fullfile(n4sidDir, 'n4sid_validation_metrics.csv'));

save(fullfile(n4sidDir, 'n4sid_model.mat'), ...
    'model', 'A', 'B', 'C', 'D', 'nx', ...
    'TsTrain', 'TsOriginal', 'TsGeneric', ...
    'inputNames', 'outputNames', ...
    'uOffset', 'yOffset', 'uScale', 'yScale', ...
    'modelOriginal', 'modelGeneric', ...
    'x0Original', 'x0Generic', ...
    'metrics', ...
    'originalPredictions', 'genericPredictions');

fprintf('Saved model and metrics to:\n%s\n', n4sidDir);

%% Local functions

function T = loadReactorCSV(filePath, inputNames, outputNames)
    % Load, sort chronologically, and validate the required columns.
    assert(isfile(filePath), 'File not found: %s', filePath);

    T = readtable(filePath);

    requiredNames = [{'time_min'}, inputNames, outputNames];

    assert(all(ismember(requiredNames, T.Properties.VariableNames)), ...
        'Missing required columns in: %s', filePath);

    T = sortrows(T, 'time_min');

    assert(height(T) >= 2, ...
        'At least two samples are required in: %s', filePath);

    values = T{:, requiredNames};

    assert(isnumeric(values) && all(isfinite(values), 'all'), ...
        'Required columns must contain finite numeric values: %s', ...
        filePath);
end


function Ts = getSampleTime(T)
    % Check regular sampling and reject duplicate timestamps.
    intervals = diff(T.time_min);
    Ts = median(intervals);

    tolerance = max(1e-8, abs(Ts) * 1e-6);

    assert(Ts > 0 && all(intervals > 0), ...
        'Timestamps must be strictly increasing.');

    assert(all(abs(intervals - Ts) < tolerance), ...
        'Data must be uniformly sampled.');
end


function data = makeIdentificationData( ...
        T, Ts, inputNames, outputNames, ...
        uOffset, yOffset, uScale, yScale)

    U = T{:, inputNames};
    Y = T{:, outputNames};

    % Always use the TRAINING offsets and scales.
    Uscaled = (U - uOffset) ./ uScale;
    Yscaled = (Y - yOffset) ./ yScale;

    data = iddata(Yscaled, Uscaled, Ts);

    data.TimeUnit = 'minutes';
    data.InputName = inputNames;
    data.OutputName = outputNames;
    data.InterSample = 'zoh';
end


function [metrics, Yhat, evaluationModel, x0] = evaluateDataset( ...
        model, T, Ts, datasetName, ...
        inputNames, outputNames, ...
        uOffset, yOffset, uScale, yScale)

    %% Match the model to this dataset's sampling interval
    tolerance = max(1e-8, abs(model.Ts) * 1e-6);

    if abs(Ts - model.Ts) < tolerance
        evaluationModel = model;
    else
        ratio = Ts / model.Ts;

        % Restrict conversion to integer downsampling.
        % The default data use 10 / 0.1 = 100.
        assert(ratio >= 1 && ...
            abs(ratio - round(ratio)) < 1e-6, ...
            ['Validation sampling must equal the training sampling ' ...
             'or be an integer multiple of it.']);

        evaluationModel = d2d(model, Ts, 'zoh');
        evaluationModel.TimeUnit = 'minutes';
    end

    %% Build validation data using training preprocessing
    data = makeIdentificationData( ...
        T, Ts, inputNames, outputNames, ...
        uOffset, yOffset, uScale, yScale);

    %% Free-running simulation with estimated initial conditions
    compareOpts = compareOptions('InitialCondition', 'e');

    [YhatScaled, ~, x0] = compare( ...
        data, evaluationModel, Inf, compareOpts);

    %% Convert predictions back to mol/dm^3
    Yhat = YhatScaled.OutputData .* yScale + yOffset;
    Ytrue = T{:, outputNames};

    assert(all(isfinite(Yhat), 'all'), ...
        'Simulation produced nonfinite predictions for %s.', ...
        datasetName);

    %% Calculate errors separately for each output
    errors = Ytrue - Yhat;

    mae = mean(abs(errors), 1);
    rmse = sqrt(mean(errors.^2, 1));

    metrics = table( ...
        repmat({datasetName}, numel(outputNames), 1), ...
        outputNames(:), ...
        mae(:), ...
        rmse(:), ...
        'VariableNames', ...
        {'Dataset', 'Output', 'MAE_mol_dm3', 'RMSE_mol_dm3'});

    %% Plot validation results
    figure('Name', datasetName);
    tiledlayout(3, 2);

    % Display time relative to the start of this validation segment.
    time = T.time_min - T.time_min(1);

    for j = 1:numel(outputNames)
        nexttile;

        plot(time, Ytrue(:, j), 'k', 'LineWidth', 1);
        hold on;

        plot(time, Yhat(:, j), '--r', 'LineWidth', 1);

        xlabel('Time since segment start [min]');
        ylabel('Concentration [mol/dm^3]');

        title(sprintf('%s: MAE %.3g | RMSE %.3g', ...
            outputNames{j}, mae(j), rmse(j)));

        legend('Reactor data', 'N4SID', 'Location', 'best');
        grid on;
    end

    sgtitle(datasetName);
end