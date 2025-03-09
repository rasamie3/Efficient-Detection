document.getElementById("uploadBtn").addEventListener("click", () => {
     const fileInput = document.getElementById("videoInput");
     const statusDiv = document.getElementById("status");
     const resultDiv = document.getElementById("result");
 
     if (!fileInput || !fileInput.files.length) {
         alert('Please select a video file!');
         return;
     }
 
     const formData = new FormData();
     formData.append("video", fileInput.files[0]);

     statusDiv.textContent = "";
     resultDiv.textContent = "";
 
     statusDiv.textContent = "Uploading...";
 
     fetch("/upload", { method: "POST", body: formData })
         .then(res => res.json())
         .then(data => {
             if (!data.video_path) throw new Error("Upload failed.");
             statusDiv.textContent = "Processing...";
 
             return fetch("/process", {
                 method: "POST",
                 headers: { "Content-Type": "application/json" },
                 body: JSON.stringify({ video_path: data.video_path, model_number: 2 })
             });
         })
         .then(res => res.json())
         .then(data => {
             statusDiv.textContent = "Done!";
             resultDiv.textContent = `Prediction: ${data.prediction || "N/A"} (Confidence: ${(data.score?.toFixed(2) || "N/A")})`;
         })
         .catch(error => {
             console.error(error);
             statusDiv.textContent = "Error occurred.";
             resultDiv.textContent = "";
         });
 });
 