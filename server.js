require('dotenv').config();
const express = require('express');
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');
const { fixZeros, minMaxNormalizeSingle } = require('./preprocess');

const app = express();
app.use(express.json());

// Serve thư mục model
app.use('/model', express.static(path.join(__dirname, 'model')));
app.use(express.static('public'));

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

let model = null;
let mins = [];
let maxs = [];
let medians = [];
const medianCols = [1, 2, 3, 4, 5];

async function loadModelAndParams() {
    console.log("Đang load model và tham số normalization...");

    // Ép cùng backend CPU để tính giống nhau
    await tf.setBackend('cpu');
    await tf.ready();

    // Load model từ HTTP
    model = await tf.loadLayersModel('http://localhost:3000/model/model.json');

    // Load normalization params
    const minmaxPath = path.join(__dirname, 'model', 'minmax.json');
    const minmaxData = JSON.parse(fs.readFileSync(minmaxPath, 'utf8'));
    mins = minmaxData.mins;
    maxs = minmaxData.maxs;
    medians = minmaxData.medians;

    console.log("✅ Model và tham số đã sẵn sàng!");
}

// API Predict
app.post('/predict', async (req, res) => {
    try {
        const { features } = req.body;
        if (!Array.isArray(features) || features.length !== 8) {
            return res.status(400).json({ error: 'features phải là mảng 8 số' });
        }

        let input = fixZeros(features.map(Number), medianCols, medians);
        input = minMaxNormalizeSingle(input, mins, maxs);

        const inputTensor = tf.tensor2d([input], [1, 8]);
        const prob = (await model.predict(inputTensor).data())[0];
        const label = prob > 0.5 ? '⚠️ Nguy cơ cao' : '✅ Nguy cơ thấp';

        res.json({
            probability: prob,
            percent: (prob * 100).toFixed(2),
            label
        });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Lỗi server khi dự đoán' });
    }
});
let vectorIndex = [];
app.post('/chat', async (req, res) => {
    try {
        const { message, features } = req.body;
        if (!message || typeof message !== 'string') return res.status(400).json({ error: 'message required' });

        // 1) optional ML prediction
        let mlResult = null;
        if (features && Array.isArray(features) && features.length === 8 && model) {
            const input = tf.tensor2d([features], [1, 8]);
            const out = model.predict(input);
            const prob = (await out.data())[0];
            mlResult = { probability: prob, label: prob > 0.5 ? 'high risk' : 'low risk' };
        }

        // 2) embedding query
        const embResp = await client.embeddings.create({
            model: 'text-embedding-3-small',
            input: message
        });
        const qEmb = embResp.data[0].embedding;

        // 3) similarity search in vectorIndex
        const scored = vectorIndex.map(d => ({ ...d, score: cosineSim(qEmb, d.embedding) }));
        scored.sort((a, b) => b.score - a.score);
        const top = scored.slice(0, 3);

        // 4) prepare prompt with retrieved docs
        let prompt = `Bạn là trợ lý y tế. Luôn bắt đầu trả lời bằng một dòng ngắn gọn (tiếng Việt) kèm disclaimer: "Tôi là AI, thông tin chỉ mang tính tham khảo, không thay thế tư vấn bác sĩ".\n\n`;
        if (mlResult) {
            prompt += `Dự đoán ML: khả năng tiểu đường = ${(mlResult.probability || 0).toFixed(3)} (${mlResult.label}).\n\n`;
        }
        prompt += `Tài liệu tham chiếu (ngắn):\n`;
        for (const t of top) {
            prompt += `- ${t.source}: ${t.text}\n`;
        }
        prompt += `\nCâu hỏi người dùng: ${message}\n\nHãy trả lời ngắn gọn (3 phần): 1) đánh giá sơ bộ, 2) bước tiếp theo nên làm (triệu chứng nguy hiểm, xét nghiệm cần chỉ định), 3) nguồn tham khảo và cảnh báo pháp lý.\n`;

        // 5) send to ChatGPT
        const chatResp = await client.chat.completions.create({
            model: 'gpt-3.5-turbo',
            messages: [{ role: 'user', content: prompt }],
            max_tokens: 500
        });

        const reply = chatResp.choices?.[0]?.message?.content || 'Không có phản hồi từ LLM';
        res.json({ reply, ml: mlResult, retrieved: top.map(t => ({ source: t.source, score: t.score })) });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'server error' });
    }
});


(async () => {
    app.listen(3000, async () => {
        console.log('🚀 Server chạy tại http://localhost:3000');
        await loadModelAndParams();
    });
})();
