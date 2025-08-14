// predict.js
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

(async () => {
    console.log("===== LOAD MODEL =====");

    const modelDir = path.join(__dirname, 'model');

    // 1. Đọc file model.json và weights.bin
    const modelJSON = JSON.parse(fs.readFileSync(path.join(modelDir, 'model.json'), 'utf8'));
    const weightData = fs.readFileSync(path.join(modelDir, 'weights.bin')).buffer;

    // 2. Load model từ memory (chú ý tách các trường)
    const handler = tf.io.fromMemory(
        modelJSON.modelTopology,
        modelJSON.weightSpecs,
        weightData
    );
    const model = await tf.loadLayersModel(handler);
    console.log("✅ Model loaded successfully");

    // 3. Load preprocessing params
    const paramsPath = path.join(modelDir, 'minmax.json');
    const params = JSON.parse(fs.readFileSync(paramsPath, 'utf8'));
    const { medianCols, medians, iqrLower, iqrUpper, mins, maxs, threshold } = params;
    console.log("✅ Preprocessing parameters loaded");

    // ==== Hàm tiền xử lý ====
    function fixZerosInPlace(row) {
        medianCols.forEach(ci => {
            if (row[ci] === 0) row[ci] = medians[ci];
        });
    }
    function winsorizeInPlace(row) {
        for (let j = 0; j < row.length; j++) {
            if (row[j] < iqrLower[j]) row[j] = iqrLower[j];
            if (row[j] > iqrUpper[j]) row[j] = iqrUpper[j];
        }
    }
    function minMaxNormalize(row) {
        return row.map((v, j) => maxs[j] === mins[j] ? 0 : (v - mins[j]) / (maxs[j] - mins[j]));
    }
    function preprocessRow(row) {
        const r = row.slice();
        fixZerosInPlace(r);
        winsorizeInPlace(r);
        return minMaxNormalize(r);
    }

    // 4. Ví dụ input
    let sample = [6, 148, 72, 35, 0, 33.6, 0.627, 50];
    console.log("\nRaw input:", sample);

    const processed = preprocessRow(sample);
    console.log("Processed input:", processed);

    // 5. Predict
    const inputT = tf.tensor2d([processed]);
    const probT = model.predict(inputT);
    const prob = (await probT.data())[0];
    const label = prob >= threshold ? 1 : 0;

    console.log(`\nPrediction probability: ${prob.toFixed(4)}`);
    console.log(`Threshold: ${threshold.toFixed(2)}`);
    console.log(`Predicted class: ${label} (${label === 1 ? 'Có tiểu đường' : 'Không tiểu đường'})`);
})();
