// script.js
let model;
let mins = [];
let maxs = [];
let medians = [];
const medianCols = [1, 2, 3, 4, 5]; // Glucose, BP, Skin, Insulin, BMI

// ===== Hàm fix giá trị 0 =====
function fixZeros(arr, cols, medians) {
    cols.forEach(colIdx => {
        if (arr[colIdx] === 0) {
            arr[colIdx] = medians[colIdx];
        }
    });
}

// ===== Hàm chuẩn hóa Min-Max =====
function minMaxNormalizeSingle(row, mins, maxs) {
    return row.map((v, i) => (v - mins[i]) / (maxs[i] - mins[i]));
}

// ===== Load model và minmax =====
async function loadModelAndParams() {
    console.log("Đang tải model và minmax.json...");
    model = await tf.loadLayersModel('/model/model.json');
    const res = await fetch('/model/minmax.json');
    const minmaxData = await res.json();
    mins = minmaxData.mins;
    maxs = minmaxData.maxs;
    medians = mins.map((min, i) => (min + maxs[i]) / 2);
    console.log("Model và tham số min-max đã sẵn sàng!");
}

// ===== Hàm dự đoán =====
async function predictRisk() {
    if (!model) {
        alert("Model chưa sẵn sàng! Đợi load xong rồi thử lại.");
        return;
    }

    // Lấy dữ liệu từ form
    let sampleInput = [
        parseFloat(document.getElementById('preg').value) || 0,
        parseFloat(document.getElementById('glu').value) || 0,
        parseFloat(document.getElementById('bp').value) || 0,
        parseFloat(document.getElementById('skin').value) || 0,
        parseFloat(document.getElementById('ins').value) || 0,
        parseFloat(document.getElementById('bmi').value) || 0,
        parseFloat(document.getElementById('dpf').value) || 0,
        parseFloat(document.getElementById('age').value) || 0
    ];

    // Fix giá trị 0
    fixZeros(sampleInput, medianCols, medians);

    // Chuẩn hóa
    const normalizedInput = minMaxNormalizeSingle(sampleInput, mins, maxs);

    // Tensor
    const inputTensor = tf.tensor2d([normalizedInput], [1, 8]);

    // Predict
    const prediction = model.predict(inputTensor);
    const predictionValue = (await prediction.data())[0];
    const percent = (predictionValue * 100).toFixed(2);

    // Hiển thị kết quả
    const risk = predictionValue > 0.5 ? '⚠️ Nguy cơ cao' : '✅ Nguy cơ thấp';
    addMessage(`Kết quả dự đoán: ${risk} (Xác suất: ${percent}%)`, 'bot');
}

// ===== Chatbot gửi tin nhắn =====
async function sendMessage() {
    const msg = document.getElementById('message').value;
    if (!msg.trim()) return;

    addMessage(msg, 'user');
    document.getElementById('message').value = '';

    const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg })
    });

    const data = await res.json();
    addMessage(data.reply, 'bot');
}

// ===== Thêm tin nhắn vào chatbox =====
function addMessage(text, sender) {
    const chatBox = document.getElementById('chat-box');
    const msgElem = document.createElement('div');
    msgElem.classList.add('message', sender);
    msgElem.innerText = text;
    chatBox.appendChild(msgElem);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// ===== Reset form =====
function clearMessage() {
    document.getElementById('message').value = '';
}
function resetForm() {
    document.getElementById('preg').value = '';
    document.getElementById('glu').value = '';
    document.getElementById('bp').value = '';
    document.getElementById('skin').value = '';
    document.getElementById('ins').value = '';
    document.getElementById('bmi').value = '';
    document.getElementById('dpf').value = '';
    document.getElementById('age').value = '';
}

// ===== Khởi chạy khi load trang =====
window.onload = async () => {
    await loadModelAndParams();
};
