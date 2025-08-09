const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

(async () => {
    console.log("Loading model and minmax parameters from local files...");

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
    console.log("Model loaded successfully!");

    // Đọc minmax.json (lưu min và max từng cột để chuẩn hóa)
    const minmaxPath = path.join(__dirname, 'model', 'minmax.json');
    const { mins, maxs } = JSON.parse(fs.readFileSync(minmaxPath, 'utf8'));

    // Tính median cho các cột cần fix giá trị 0
    // Ở đây giả sử median được tính trước và lưu hoặc tính trung bình min-max nếu chưa có
    // Bạn có thể thay bằng median thật nếu tính được từ dữ liệu train
    const medianCols = [1, 2, 3, 4, 5]; // các cột Glucose, BloodPressure, SkinThickness, Insulin, BMI

    // Tính median tạm thời = trung bình (min+max)/2
    const medians = mins.map((min, i) => (min + maxs[i]) / 2);

    // Hàm fix zeros: nếu giá trị 0 ở cột đặc biệt thì thay bằng median tương ứng
    function fixZeros(arr, cols, medians) {
        cols.forEach(colIdx => {
            if (arr[colIdx] === 0) {
                arr[colIdx] = medians[colIdx];
            }
        });
    }

    // Hàm chuẩn hóa min-max cho một mẫu đơn
    function minMaxNormalizeSingle(row, mins, maxs) {
        return row.map((v, i) => (v - mins[i]) / (maxs[i] - mins[i]));
    }

    // Dữ liệu mẫu để dự đoán
    let sampleInput = [7, 100, 0, 0, 0, 30, 0.484, 32];

    // Fix zeros theo median
    fixZeros(sampleInput, medianCols, medians);

    // Chuẩn hóa min-max
    const normalizedInput = minMaxNormalizeSingle(sampleInput, mins, maxs);

    // Tạo tensor đầu vào
    const inputTensor = tf.tensor2d([normalizedInput], [1, 8]);

    // Dự đoán
    const prediction = model.predict(inputTensor);
    const predictionValue = (await prediction.data())[0];

    // Tính phần trăm
    const percent = (predictionValue * 100).toFixed(2);

    console.log(`Prediction (xác suất): ${predictionValue}`);
    console.log(`Mức độ tiểu đường (phần trăm): ${percent}%`);
    console.log(predictionValue > 0.5 ? "Có khả năng bị tiểu đường" : "Ít khả năng bị tiểu đường");
})();
