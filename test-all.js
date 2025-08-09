const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

(async () => {
    console.log("Loading model...");

    // Load model
    const modelPath = path.join(__dirname, 'model', 'model.json');
    const modelJSON = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
    const weightsPath = path.join(__dirname, 'model', 'weights.bin');
    const weightData = fs.readFileSync(weightsPath);

    const ioHandler = {
        load: async () => ({
            modelTopology: modelJSON.modelTopology,
            weightSpecs: modelJSON.weightSpecs,
            weightData: weightData
        })
    };

    const model = await tf.loadLayersModel(ioHandler);
    console.log("Model loaded");

    // Load minmax (để chuẩn hóa theo min max tập train)
    const minmaxPath = path.join(__dirname, 'model', 'minmax.json');
    if (!fs.existsSync(minmaxPath)) {
        console.error("File minmax.json chưa có. Hãy lưu mins, maxs khi train.");
        process.exit(1);
    }
    const { mins, maxs } = JSON.parse(fs.readFileSync(minmaxPath, 'utf8'));

    // Load raw data
    const csvPath = path.join(__dirname, 'data', 'pima_diabetes.csv');
    const csvDataset = tf.data.csv(`file://${csvPath}`, { hasHeader: true });
    const points = await csvDataset.toArray();
    let data = points.map(p => Object.values(p).map(Number));

    let X = data.map(d => d.slice(0, 8));
    const y = data.map(d => d[8]);

    // Hàm xử lý giá trị 0 không hợp lệ bằng median của cột
    function fixZeros(arr, cols) {
        cols.forEach(colIdx => {
            const nonZeroVals = arr.filter(r => r[colIdx] !== 0).map(r => r[colIdx]);
            nonZeroVals.sort((a, b) => a - b);
            const mid = Math.floor(nonZeroVals.length / 2);
            const median = nonZeroVals.length % 2 !== 0 ?
                nonZeroVals[mid] : (nonZeroVals[mid - 1] + nonZeroVals[mid]) / 2;
            arr.forEach(r => {
                if (r[colIdx] === 0) r[colIdx] = median;
            });
        });
    }
    fixZeros(X, [1, 2, 3, 4, 5]); // giống train

    // Chuẩn hóa Min-Max theo mins, maxs tập train đã lưu
    const Xnorm = X.map(row =>
        row.map((v, i) => (v - mins[i]) / (maxs[i] - mins[i]))
    );

    const Xtensor = tf.tensor2d(Xnorm);

    // Dự đoán
    const predictions = model.predict(Xtensor);
    const predValues = await predictions.data();

    // Tính accuracy với threshold 0.5
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
        const predLabel = predValues[i] > 0.5 ? 1 : 0;
        if (predLabel === y[i]) correct++;
    }
    const accuracy = correct / y.length;

    console.log(`Tổng số mẫu: ${y.length}`);
    console.log(`Số mẫu dự đoán đúng: ${correct}`);
    console.log(`Accuracy trên toàn bộ dữ liệu: ${(accuracy * 100).toFixed(2)}%`);
})();
