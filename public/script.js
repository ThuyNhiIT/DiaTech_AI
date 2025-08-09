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

async function predictRisk() {
    const features = [
        parseFloat(document.getElementById('preg').value) || 0,
        parseFloat(document.getElementById('glu').value) || 0,
        parseFloat(document.getElementById('bp').value) || 0,
        parseFloat(document.getElementById('skin').value) || 0,
        parseFloat(document.getElementById('ins').value) || 0,
        parseFloat(document.getElementById('bmi').value) || 0,
        parseFloat(document.getElementById('dpf').value) || 0,
        parseFloat(document.getElementById('age').value) || 0
    ];

    const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
    });

    const data = await res.json();
    const risk = data.probability > 0.5 ? '⚠️ Nguy cơ cao' : '✅ Nguy cơ thấp';
    addMessage(`Kết quả dự đoán: ${risk} (Xác suất: ${(data.probability * 100).toFixed(2)}%)`, 'bot');
}

function addMessage(text, sender) {
    const chatBox = document.getElementById('chat-box');
    const msgElem = document.createElement('div');
    msgElem.classList.add('message', sender);
    msgElem.innerText = text;
    chatBox.appendChild(msgElem);
    chatBox.scrollTop = chatBox.scrollHeight;
}


function clearMessage() {
    // Xóa nội dung trong textarea
    document.getElementById('message').value = '';
}

function resetForm() {
    // Xóa tất cả các ô nhập liệu trong form dự đoán nguy cơ tiểu đường
    document.getElementById('preg').value = '';
    document.getElementById('glu').value = '';
    document.getElementById('bp').value = '';
    document.getElementById('skin').value = '';
    document.getElementById('ins').value = '';
    document.getElementById('bmi').value = '';
    document.getElementById('dpf').value = '';
    document.getElementById('age').value = '';
}