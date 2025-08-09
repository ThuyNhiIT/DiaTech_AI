// train-model.js
require('dotenv').config();
const tf = require('@tensorflow/tfjs'); // Chỉ dùng tfjs thuần
const fs = require('fs');
const path = require('path');

(async () => {
    console.log("🚀 Training model...");

    // Đường dẫn tới file CSV
    const csvPath = path.join(__dirname, 'data', 'pima_diabetes.csv');

    // Đọc CSV (có header)
    const csvDataset = tf.data.csv(`file://${csvPath}`, { hasHeader: true });

    // Chuyển sang mảng
    const points = await csvDataset.toArray();
    const data = points.map(p => Object.values(p));

    // X và y
    const X = data.map(d => d.slice(0, 8));
    const y = data.map(d => d[8]);

    // Tensor hóa dữ liệu
    const Xtensor = tf.tensor2d(X);
    const ytensor = tf.tensor2d(y, [y.length, 1]);

    // Tạo model
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [8], units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    // Train model
    await model.fit(Xtensor, ytensor, {
        epochs: 50,
        shuffle: true,
        validationSplit: 0.2
    });

    // Lưu model ra thư mục "model"
    const savePath = path.join(__dirname, 'model');
    if (!fs.existsSync(savePath)) {
        fs.mkdirSync(savePath);
    }

    await model.save(tf.io.withSaveHandler(async (modelArtifacts) => {
        fs.writeFileSync(path.join(savePath, 'model.json'), JSON.stringify(modelArtifacts));

        if (modelArtifacts.weightData) {
            fs.writeFileSync(
                path.join(savePath, 'weights.bin'),
                Buffer.from(modelArtifacts.weightData)
            );
        }

        return {
            modelArtifactsInfo: {
                dateSaved: new Date(),
                modelTopologyType: 'JSON',
                modelTopologyBytes: modelArtifacts.modelTopology ?
                    JSON.stringify(modelArtifacts.modelTopology).length : 0,
                weightSpecsBytes: modelArtifacts.weightSpecs ?
                    JSON.stringify(modelArtifacts.weightSpecs).length : 0,
                weightDataBytes: modelArtifacts.weightData ?
                    modelArtifacts.weightData.byteLength : 0
            }
        };
    }));

    console.log('✅ Model saved to', savePath);
})();
