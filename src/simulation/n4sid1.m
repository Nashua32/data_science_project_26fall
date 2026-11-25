%% Fit N4SID models and export numeric parameters for Python
% Evaluation, plotting and ML comparisons are performed in the notebook.
% Run the notebook's training-export cell first to enable a matched-data fit.
clear;
clc;

% Resolve paths from the current working directory. Search its ancestors so
% running from src/models and running from the project root use the same data.
workingDir = pwd;
projectRoot = workingDir;
candidate = workingDir;
while true
    hasProjectLayout = isfolder(fullfile(candidate, 'src')) && ...
        isfolder(fullfile(candidate, 'data'));
    hasBridgeData = isfile(fullfile(candidate, 'data', 'n4sid', 'train.csv')) || ...
        isfile(fullfile(candidate, 'data', 'n4sid', 'notebook_train.csv'));
    if hasProjectLayout || hasBridgeData
        projectRoot = candidate;
        break;
    end
    parent = fileparts(candidate);
    if isempty(parent) || strcmp(parent, candidate)
        break;
    end
    candidate = parent;
end
fprintf('Project/workspace directory: %s\n', projectRoot);
n4sidDir = fullfile(projectRoot, 'data', 'n4sid');
inputNames = {'QB', 'Tr'};
outputNames = {'cA', 'cB', 'cC', 'cP', 'cE', 'cG'};
nx = 6;  % Retains the order used in the existing project script.

% Keep the original training experiment; also fit a baseline on exactly the
% notebook's training samples for a comparison that controls training data.
trainingSources = {'original', 'notebook'};
trainingFiles = {'train.csv', 'notebook_train.csv'};

for sourceIndex = 1:numel(trainingSources)
    sourceName = trainingSources{sourceIndex};
    inputFile = fullfile(n4sidDir, trainingFiles{sourceIndex});
    if ~isfile(inputFile)
        if strcmp(sourceName, 'notebook')
            fprintf('Skipped notebook fit: run the notebook training-export cell first.\n');
            continue;
        end
        error('Missing training data: %s', inputFile);
    end

    T = readtable(inputFile);
    required = [{'time_min'}, inputNames, outputNames];
    assert(all(ismember(required, T.Properties.VariableNames)), ...
        'Required columns are missing in %s.', inputFile);
    assert(height(T) > nx + 1, 'Insufficient training samples.');
    assert(all(isfinite(T{:, required}), 'all'), ...
        'Training data contains nonfinite values.');
    deltaTime = diff(T.time_min);
    Ts_min = median(deltaTime);
    assert(Ts_min > 0 && all(deltaTime > 0) && ...
        all(abs(deltaTime - Ts_min) < max(1e-8, Ts_min * 1e-6)), ...
        'Training samples must be chronological and uniformly spaced.');

    training_U = T{:, inputNames};
    training_Y = T{:, outputNames};
    uOffset = mean(training_U, 1);
    yOffset = mean(training_Y, 1);
    uScale = std(training_U, 0, 1);
    yScale = std(training_Y, 0, 1);
    assert(all(uScale > 0) && all(yScale > 0), ...
        'All training channels must have nonzero variance.');

    trainData = iddata((training_Y-yOffset)./yScale, ...
        (training_U-uOffset)./uScale, Ts_min);
    trainData.TimeUnit = 'minutes';
    trainData.InputName = inputNames;
    trainData.OutputName = outputNames;
    trainData.InterSample = 'zoh';

    options = n4sidOptions('Focus', 'simulation');
    fitTimer = tic;
    model = n4sid(trainData, nx, 'Feedthrough', false, ...
        'DisturbanceModel', 'none', options);
    training_time_s = toc(fitTimer);
    model.TimeUnit = 'minutes';

    A = model.A;
    B = model.B;
    C = model.C;
    D = model.D;
    training_source = sourceName;
    training_file = inputFile;
    time_unit = 'minutes';
    export_version = 1;

    % Numeric-only MAT v7 can be read directly with scipy.io.loadmat.
    % Training arrays allow the notebook to detect stale/mismatched exports.
    outputFile = fullfile(n4sidDir, ...
        ['n4sid_export_' sourceName '.mat']);
    save(outputFile, 'A', 'B', 'C', 'D', 'Ts_min', ...
        'uOffset', 'yOffset', 'uScale', 'yScale', ...
        'inputNames', 'outputNames', 'nx', 'training_time_s', ...
        'training_source', 'training_file', 'training_U', 'training_Y', ...
        'time_unit', 'export_version', '-v7');
    fprintf('Exported %s model to %s\n', sourceName, outputFile);
end
