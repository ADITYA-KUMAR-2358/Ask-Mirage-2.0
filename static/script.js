let uploadedFileName = "";

function uploadFile() {
    let fileInput = document.getElementById("pdfUpload");
    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    fetch("/upload", { method: "POST", body: formData })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            uploadedFileName = data.filename;
            document.getElementById("uploadStatus").innerText = "✅ File uploaded!";
        } else {
            document.getElementById("uploadStatus").innerText = "❌ Upload failed!";
        }
    });
}

function trainModel() {
    if (!uploadedFileName) {
        document.getElementById("trainStatus").innerText = "⚠️ Upload a file first!";
        return;
    }

    fetch("/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: uploadedFileName })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById("trainStatus").innerText = `✅ Model trained on ${data.chunks} chunks!`;
        } else {
            document.getElementById("trainStatus").innerText = "❌ Training failed!";
        }
    });
}

function askQuestion() {
    let query = document.getElementById("queryInput").value;
    
    fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("responseText").innerText = `🤖 ${data.answer}`;
    });
}
