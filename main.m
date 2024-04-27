% 导入MNIST数据集
trainImages = loadMNISTImages('train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('train-labels-idx1-ubyte');
testImages = loadMNISTImages('t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');

% 将图像转换为 28x28x1 的数组
trainImages = reshape(trainImages, 28, 28, 1, []);
testImages = reshape(testImages, 28, 28, 1, []);

% 数据预处理：归一化
trainImages = trainImages / 255;
testImages = testImages / 255;

% 将标签转换为分类类型
trainLabels = categorical(trainLabels);
testLabels = categorical(testLabels);

% 构建卷积神经网络模型
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(512)
    reluLayer
    
    dropoutLayer(0.5)
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% 设置训练选项
options = trainingOptions('sgdm', ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 10, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {testImages, testLabels}, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% 训练模型
net = trainNetwork(trainImages, trainLabels, layers, options);

% 使用测试数据集评估模型
YPred = classify(net, testImages);
accuracy = sum(YPred == testLabels) / numel(testLabels);
fprintf('测试集准确率：%.2f%%\n', accuracy * 100);

% 显示一些测试图像和模型预测结果
figure;
for i = 1:16
    subplot(4,4,i);
    idx = randi(numel(testLabels));
    imshow(testImages(:,:,:,idx));
    title(['预测：',char(YPred(idx)),', 实际：',char(testLabels(idx))]);
end
