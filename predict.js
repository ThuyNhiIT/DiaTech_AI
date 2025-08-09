const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

(async () => {
    console.log("📂 Loading model from local files...");

    // Đọc model.json
    const modelPath = path.join(__dirname, 'model', 'model.json');
    const modelJSON = JSON.parse(fs.readFileSync(modelPath, 'utf8'));

    // Đọc weights.bin
    const weightsPath = path.join(__dirname, 'model', 'weights.bin');
    const weightData = fs.readFileSync(weightsPath);

    // Tạo IOHandler thủ công
    const ioHandler = {
        load: async () => ({
            modelTopology: modelJSON.modelTopology,
            weightSpecs: modelJSON.weightSpecs,
            weightData: weightData
        })
    };

    // Load model
    const model = await tf.loadLayersModel(ioHandler);
    console.log("✅ Model loaded successfully!");

    // Dữ liệu mẫu để dự đoán
    const sampleInput = [6, 148, 72, 35, 0, 33.6, 0.627, 50];
    const inputTensor = tf.tensor2d([sampleInput], [1, 8]);

    const prediction = model.predict(inputTensor);
    const predictionValue = (await prediction.data())[0];

    console.log(`🔮 Prediction: ${predictionValue}`);
    console.log(predictionValue > 0.5 ? "🩺 Có khả năng bị tiểu đường" : "✅ Ít khả năng bị tiểu đường");
})();
