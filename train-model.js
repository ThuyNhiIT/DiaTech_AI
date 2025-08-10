require('dotenv').config();
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

(async () => {
    console.log("Training model...");

    // Đọc CSV
    const csvPath = path.join(__dirname, 'data', 'pima_diabetes.csv');
    const csvDataset = tf.data.csv(`file://${csvPath}`, { hasHeader: true });
    const points = await csvDataset.toArray();
    let data = points.map(p => Object.values(p).map(Number));

    // Tách X, y
    let X = data.map(d => d.slice(0, 8));
    const y = data.map(d => d[8]);

    // Hàm lấy giá trị median cho các cột có giá trị 0
    function getMedians(arr, cols) {
        const medians = [];
        cols.forEach(colIdx => {
            const nonZeroVals = arr.filter(r => r[colIdx] !== 0).map(r => r[colIdx]);
            nonZeroVals.sort((a, b) => a - b);
            const mid = Math.floor(nonZeroVals.length / 2);
            medians[colIdx] = nonZeroVals.length % 2 !== 0 ?
                nonZeroVals[mid] : (nonZeroVals[mid - 1] + nonZeroVals[mid]) / 2;
        });
        return medians;
    }

    const medianCols = [1, 2, 3, 4, 5];
    // Tính median trước khi xử lý dữ liệu
    const medians = getMedians(X, medianCols);

    // Xử lý giá trị 0 không hợp lệ bằng median của cột
    X.forEach(row => {
        medianCols.forEach(colIdx => {
            if (row[colIdx] === 0) {
                row[colIdx] = medians[colIdx];
            }
        });
    });

    // Hàm chuẩn hóa Min-Max trả về cả mins, maxs
    function minMaxNormalize(arr) {
        const mins = [];
        const maxs = [];
        for (let i = 0; i < arr[0].length; i++) {
            const col = arr.map(r => r[i]);
            mins[i] = Math.min(...col);
            maxs[i] = Math.max(...col);
        }
        const normalized = arr.map(row => row.map((v, i) => (v - mins[i]) / (maxs[i] - mins[i])));
        return { normalized, mins, maxs };
    }

    const { normalized, mins, maxs } = minMaxNormalize(X);
    X = normalized;

    // ... (Phần code tách train-test, xây dựng và huấn luyện model giữ nguyên) ...

    const combined = X.map((x, i) => ({ x, y: y[i] }));
    tf.util.shuffle(combined);

    const trainSize = Math.floor(combined.length * 0.8);
    const trainData = combined.slice(0, trainSize);
    const testData = combined.slice(trainSize);

    const Xtrain = trainData.map(d => d.x);
    const ytrain = trainData.map(d => d.y);
    const Xtest = testData.map(d => d.x);
    const ytest = testData.map(d => d.y);

    const XtrainTensor = tf.tensor2d(Xtrain);
    const ytrainTensor = tf.tensor2d(ytrain, [ytrain.length, 1]);
    const XtestTensor = tf.tensor2d(Xtest);
    const ytestTensor = tf.tensor2d(ytest, [ytest.length, 1]);

    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [8], units: 32, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    await model.fit(XtrainTensor, ytrainTensor, {
        epochs: 100,
        batchSize: 32,
        validationData: [XtestTensor, ytestTensor],
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1} - loss: ${logs.loss.toFixed(4)}, accuracy: ${(logs.acc || logs.accuracy).toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_accuracy: ${(logs.val_acc || logs.val_accuracy).toFixed(4)}`);
            }
        }
    });

    // Lưu model
    const savePath = path.join(__dirname, 'model');
    if (!fs.existsSync(savePath)) fs.mkdirSync(savePath);

    await model.save(tf.io.withSaveHandler(async (modelArtifacts) => {
        fs.writeFileSync(path.join(savePath, 'model.json'), JSON.stringify(modelArtifacts));
        if (modelArtifacts.weightData) {
            fs.writeFileSync(path.join(savePath, 'weights.bin'), Buffer.from(modelArtifacts.weightData));
        }
        return {
            modelArtifactsInfo: {
                dateSaved: new Date(),
                modelTopologyType: 'JSON',
                modelTopologyBytes: modelArtifacts.modelTopology ? JSON.stringify(modelArtifacts.modelTopology).length : 0,
                weightSpecsBytes: modelArtifacts.weightSpecs ? JSON.stringify(modelArtifacts.weightSpecs).length : 0,
                weightDataBytes: modelArtifacts.weightData ? modelArtifacts.weightData.byteLength : 0,
            }
        };
    }));

    // LƯU CẢ MINS, MAXS VÀ MEDIANS
    fs.writeFileSync(path.join(savePath, 'minmax.json'), JSON.stringify({ mins, maxs, medians }));
    console.log('Model saved to', savePath);
    console.log('Min-Max and Median parameters saved to minmax.json');

    // Đánh giá model trên test set (kiểm tra accuracy)
    const preds = model.predict(XtestTensor);
    const predVals = await preds.data();
    let correct = 0;
    for (let i = 0; i < ytest.length; i++) {
        const predLabel = predVals[i] > 0.5 ? 1 : 0;
        if (predLabel === ytest[i]) correct++;
    }
    console.log(`Test set accuracy: ${(correct / ytest.length * 100).toFixed(2)}%`);
})();