const axios = require("axios");
const multer = require("multer");
const express = require("express");
const FormData = require("form-data");
const fs = require("fs");

const app = express();
const PORT = 2000;
const BACKEND_URL = "http://127.0.0.1:5000";

const upload = multer({ dest: "videos/" });

app.use(express.static("public"));
app.use(express.json());

// Upload video file
app.post("/upload", upload.single("video"), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ "error": "No video uploaded!" });
    }

    try {
        const formData = new FormData();
        formData.append("video", fs.createReadStream(req.file.path), req.file.originalname);

        const flaskRes = await axios.post(`${BACKEND_URL}/upload`, formData, {
            headers: formData.getHeaders(),
        });

        res.json(flaskRes.data);
    } catch (error) {
        console.error("Error sending video to Flask:", error);
        res.status(500).json({ "error": "Error sending video to Flask app (backend)!" });
    }
});

// Process video with Flask backend
app.post("/process", async (req, res) => {
    try {
        console.log('Processing video...');
        const response = await axios.post(`${BACKEND_URL}/process`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error("Error processing video:", error);
        res.status(500).json({ 'error': "Error processing video" });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
