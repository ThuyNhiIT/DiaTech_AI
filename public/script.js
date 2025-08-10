// script.js
async function predictRisk() {
    // Lấy input
    const sampleInput = [
        parseFloat(document.getElementById('preg').value) || 0,
        parseFloat(document.getElementById('glu').value) || 0,
        parseFloat(document.getElementById('bp').value) || 0,
        parseFloat(document.getElementById('skin').value) || 0,
        parseFloat(document.getElementById('ins').value) || 0,
        parseFloat(document.getElementById('bmi').value) || 0,
        parseFloat(document.getElementById('dpf').value) || 0,
        parseFloat(document.getElementById('age').value) || 0
    ];

    // Hiển thị trạng thái loading
    const loadingMsg = addMessage('Đang dự đoán...', 'bot');

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: sampleInput })
        });

        const data = await res.json();
        loadingMsg.remove();

        if (data.error) {
            alert(data.error);
            return;
        }

        addMessage(`Kết quả dự đoán: ${data.label} (Xác suất: ${data.percent}%)`, 'bot');
    } catch (err) {
        loadingMsg.remove();
        addMessage('Lỗi kết nối server', 'bot');
    }
}


function addMessage(text, sender) {
    const chatBox = document.getElementById('chat-box');
    const msgElem = document.createElement('div');
    msgElem.classList.add('message', sender);
    msgElem.innerText = text;
    chatBox.appendChild(msgElem);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msgElem; // trả về phần tử để có thể thao tác sau
}

async function sendMessage() {
    const msg = document.getElementById('message').value;
    if (!msg.trim()) return;

    addMessage(msg, 'user');
    document.getElementById('message').value = '';

    // Hiển thị "Đang xử lý..."
    const loadingMsg = addMessage('Đang xử lý...', 'bot');

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg })
        });
        const data = await res.json();

        // Xóa tin nhắn "Đang xử lý..."
        loadingMsg.remove();

        if (data.error) {
            addMessage('Lỗi: ' + data.error, 'bot');
        } else {
            addMessage(data.reply, 'bot');
        }
    } catch (err) {
        loadingMsg.remove();
        addMessage('Lỗi kết nối server', 'bot');
    }
}


function clearMessage() {
    document.getElementById('message').value = '';
}

function resetForm() {
    document.querySelectorAll('.form-section input').forEach(input => input.value = '');
}
