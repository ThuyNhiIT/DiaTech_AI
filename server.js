require('dotenv').config();
const express = require('express');
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');

const app = express();
app.use(express.json());
app.use(express.static('public'));

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Load model (giống train)
async function loadLocalModel() {
    const modelJsonPath = path.join(__dirname, 'model', 'model.json');
    const weightsPath = path.join(__dirname, 'model', 'weights.bin');
    if (!fs.existsSync(modelJsonPath) || !fs.existsSync(weightsPath)) {
        console.warn('Model files missing in ./model — chạy train-model.js trước.');
        return null;
    }
    const modelJSON = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'));
    const weightData = fs.readFileSync(weightsPath);

    const ioHandler = {
        load: async () => ({
            modelTopology: modelJSON.modelTopology,
            weightSpecs: modelJSON.weightSpecs,
            weightData: weightData
        })
    };
    const model = await tf.loadLayersModel(ioHandler);
    console.log('✅ ML model loaded');
    return model;
}

let model = null;
let vectorIndex = [];

// Biến toàn cục để lưu giá trị min/max cho chuẩn hóa khi predict (bạn nên lưu ra file thực tế)
let featureMins = null;
let featureMaxs = null;

(async () => {
    model = await loadLocalModel();

    // Tính min max từ dữ liệu train để chuẩn hóa input
    // Tốt nhất lưu min max ra file trong train, ở đây tạm tính lại (hoặc bạn copy min max từ train)
    const csvPath = path.join(__dirname, 'data', 'pima_diabetes.csv');
    const csvDataset = tf.data.csv(`file://${csvPath}`, { hasHeader: true });
    const points = await csvDataset.toArray();
    let data = points.map(p => Object.values(p).map(Number));
    let X = data.map(d => d.slice(0, 8));

    // Xử lý 0 giống lúc train
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
    fixZeros(X, [1, 2, 3, 4, 5]);

    // Tính min max cho chuẩn hóa
    featureMins = [];
    featureMaxs = [];
    for (let i = 0; i < X[0].length; i++) {
        const col = X.map(r => r[i]);
        featureMins[i] = Math.min(...col);
        featureMaxs[i] = Math.max(...col);
    }
})();

function minMaxNormalizeSingle(row) {
    return row.map((v, i) => (v - featureMins[i]) / (featureMaxs[i] - featureMins[i]));
}

// API predict
app.post('/predict', async (req, res) => {
    try {
        const { features } = req.body;
        if (!model) return res.status(500).json({ error: 'Model not loaded' });
        if (!Array.isArray(features) || features.length !== 8) {
            return res.status(400).json({ error: 'features phải là mảng 8 số' });
        }

        // Xử lý 0 giống train
        const fixZerosCols = [1, 2, 3, 4, 5];
        let inputFeatures = features.slice();
        fixZerosCols.forEach(idx => {
            if (inputFeatures[idx] === 0) {
                // thay bằng median tương ứng
                // median có thể tính trước rồi lưu ra biến hoặc file
                inputFeatures[idx] = (featureMins[idx] + featureMaxs[idx]) / 2; // tạm dùng trung bình min max thay median
            }
        });

        // Chuẩn hóa
        const normalized = minMaxNormalizeSingle(inputFeatures);

        const inputTensor = tf.tensor2d([normalized], [1, 8]);
        const out = model.predict(inputTensor);
        const prob = (await out.data())[0];

        res.json({ probability: prob });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'server error' });
    }
});

app.listen(3000, () => console.log('🚀 Server running on http://localhost:3000'));
