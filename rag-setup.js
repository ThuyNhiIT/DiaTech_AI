require('dotenv').config();
const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function embedText(text) {
    const resp = await client.embeddings.create({
        model: 'text-embedding-3-small',
        input: text
    });
    return resp.data[0].embedding;
}

(async () => {
    const docsDir = path.join(__dirname, 'docs');
    if (!fs.existsSync(docsDir)) {
        console.log('Tạo thư mục ./docs và bỏ các file .txt hoặc .md vào đó rồi chạy lại `node rag-setup.js`');
        return;
    }

    const files = fs.readdirSync(docsDir).filter(f => f.endsWith('.txt') || f.endsWith('.md'));
    if (files.length === 0) {
        console.log('Không tìm thấy file .txt/.md trong ./docs');
        return;
    }

    const index = [];
    for (const file of files) {
        const txt = fs.readFileSync(path.join(docsDir, file), 'utf8');
        // chunk theo paragraph (có thể thay bằng chunk nhỏ hơn nếu cần)
        const chunks = txt.split(/\n\s*\n/).map(s => s.trim()).filter(Boolean);
        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            // bỏ qua đoạn quá ngắn
            if (chunk.length < 20) continue;
            const emb = await embedText(chunk);
            index.push({
                id: `${file}#${i}`,
                source: file,
                text: chunk,
                embedding: emb
            });
            console.log('Ingested', file, 'chunk', i);
        }
    }

    fs.writeFileSync(path.join(__dirname, 'vector-index.json'), JSON.stringify(index, null, 2));
    console.log('Saved vector-index.json with', index.length, 'chunks');
})();
