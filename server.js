// server.js
require('dotenv').config();
const express = require('express');
const tf = require('@tensorflow/tfjs'); // tfjs thuần
const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');

const app = express();
app.use(express.json());
app.use(express.static('public'));

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// --- Load ML model from model/model.json + weights.bin (custom IO handler)
async function loadLocalModel() {
    const modelJsonPath = path.join(__dirname, 'model', 'model.json');
    const weightsPath = path.join(__dirname, 'model', 'weights.bin');

    if (!fs.existsSync(modelJsonPath) || !fs.existsSync(weightsPath)) {
        console.warn('Model files missing in ./model — chạy train-model.js trước.');
        return null;
    }

    const modelJson = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'));
    const weightData = fs.readFileSync(weightsPath);

    const ioHandler = {
        load: async () => ({
            modelTopology: modelJson.modelTopology,
            weightSpecs: modelJson.weightSpecs,
            weightData: weightData
        })
    };

    const model = await tf.loadLayersModel(ioHandler);
    console.log('✅ ML model loaded');
    return model;
}

let model = null;
let vectorIndex = [];
// load model & vector index on startup
(async () => {
    model = await loadLocalModel();
    const idxPath = path.join(__dirname, 'vector-index.json');
    if (fs.existsSync(idxPath)) vectorIndex = JSON.parse(fs.readFileSync(idxPath, 'utf8'));
    console.log('Vector index chunks:', vectorIndex.length);
})();

function cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
}

// API: predict (ML only)
app.post('/predict', async (req, res) => {
    try {
        const { features } = req.body;
        if (!model) return res.status(500).json({ error: 'Model not loaded' });
        if (!Array.isArray(features) || features.length !== 8) {
            return res.status(400).json({ error: 'features phải là mảng 8 số' });
        }
        const input = tf.tensor2d([features], [1, 8]);
        const out = model.predict(input);
        const prob = (await out.data())[0];
        res.json({ probability: prob });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'server error' });
    }
});

// API: chat (RAG + ML optional)
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

app.listen(3000, () => console.log('🚀 Server running on http://localhost:3000'));
