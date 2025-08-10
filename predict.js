const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

(async () => {
    console.log("===== DEBUG PREDICT =====");

    // ==== 1. Load model từ file ====
    const modelPath = path.join(__dirname, 'model', 'model.json');
    const weightsPath = path.join(__dirname, 'model', 'weights.bin');
    const modelJSON = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
    const weightData = fs.readFileSync(weightsPath);

    console.log("Model topology layers:", modelJSON.modelTopology.config.layers.length);
    console.log("Weight specs count:", modelJSON.weightSpecs.length);

    const ioHandler = {
        load: async () => ({
            modelTopology: modelJSON.modelTopology,
            weightSpecs: modelJSON.weightSpecs,
            weightData: weightData
        })
    };
    const model = await tf.loadLayersModel(ioHandler);
    console.log("✅ Model loaded.");

    // ==== 2. Load normalization params ====
    const minmaxPath = path.join(__dirname, 'model', 'minmax.json');
    const { mins, maxs, medians } = JSON.parse(fs.readFileSync(minmaxPath, 'utf8'));
    console.log("mins:", mins);
    console.log("maxs:", maxs);
    console.log("medians:", medians);

    const medianCols = [1, 2, 3, 4, 5];

    function fixZeros(row) {
        medianCols.forEach(colIdx => {
            if (row[colIdx] === 0) {
                row[colIdx] = medians[colIdx];
            }
        });
        return row;
    }

    function minMaxNormalizeSingle(row) {
        return row.map((v, i) => (v - mins[i]) / (maxs[i] - mins[i]));
    }

    // ==== 3. Input ban đầu ====
    let sampleInput = [7, 100, 0, 0, 0, 30, 0.484, 32];
    console.log("\n[STEP 0] Raw input:", sampleInput);

    // ==== 4. Fix zeros ====
    let fixedInput = fixZeros([...sampleInput]);
    console.log("[STEP 1] After fixZeros:", fixedInput);

    // ==== 5. Normalization ====
    let normalizedInput = minMaxNormalizeSingle(fixedInput);
    console.log("[STEP 2] After normalization:", normalizedInput);

    // ==== 6. Tensor conversion ====
    const inputTensor = tf.tensor2d([normalizedInput], [1, 8], 'float32');
    console.log("[STEP 3] Tensor data:", Array.from(inputTensor.dataSync()));

    // ==== 7. Prediction ====
    const prediction = model.predict(inputTensor);
    const predictionValue = (await prediction.data())[0];
    const percent = (predictionValue * 100).toFixed(6);

    console.log("\n===== FINAL RESULT =====");
    console.log("Probability:", predictionValue);
    console.log("Risk %:", percent);
    console.log(predictionValue > 0.5 ? "⚠️ Nguy cơ cao" : "✅ Nguy cơ thấp");
})();
