from flask import Flask, request, jsonify
import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

from extract_faces import FaceProcessor
from extract_frames import FrameExtractor

app = Flask(__name__)

UPLOAD_FOLDER = "./videos"
FRAME_FOLDER = "./frames"
MODEL_FOLDER = "./models"
FAKE_THRESHOLD = 0.45

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

MODEL_NUMBER = {
    1: 'EfficientB1_non_sequential_frames',
    2: 'EfficientB3_non_sequential_frames',
    3: 'EfficientB3_sequential_frames',
    4: 'EfficientB1_sequential_frames'
}

MODELS = {
    "EfficientB1_non_sequential_frames": "./models/model_b1_1_non_seq.keras",
    "EfficientB3_non_sequential_frames": "./models/model_b3_1_non_seq.keras",
    "EfficientB3_sequential_frames": "./models/model_b3_1_seq.keras",
    "EfficientB1_sequential_frames": "./models/model_b1_1_seq.keras"
}


def load_model(model_number=1):
    model_key = MODEL_NUMBER.get(model_number)
    if not model_key:
        raise ValueError(f"Invalid model number: {model_number}. Choose from {list(MODEL_NUMBER.keys())}.")
    model_path = MODELS.get(model_key)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path.")
    return tf.keras.models.load_model(model_path)


def process_videos(video_path, face_extractor):
    extractor = FrameExtractor(video_path=video_path, frame_path=FRAME_FOLDER, create_frame_folder=True, auto_start=True)
    return face_extractor.process_frames()

@app.route('/', methods=['GET'])
def home():
    return 'OK'

@app.route("/upload", methods=["POST"])
def upload_video():
    if 'video' not in request.files:
        print('could not get the video')
        return jsonify({"error": "Video file is required"}), 400
    
    print("could get the video")
    
    video = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, secure_filename(video.filename))
    video.save(video_path)
    
    return jsonify({"message": "Video uploaded successfully", "video_path": video_path})


@app.route("/process", methods=["POST"])
def process_and_predict():
    data = request.json
    video_path = data.get('video_path')
    model_number = data.get('model_number', 2)
    count_of_frames = data.get('count_of_frames', -1)
    
    if model_number not in MODEL_NUMBER:
        return jsonify({"error": "Invalid model number"}), 400

    is_sequential = model_number >= 3
    count_of_frames = 15 if is_sequential else count_of_frames # temp

    face_processor = FaceProcessor(
        frames_folder=FRAME_FOLDER,
        count_of_frames=count_of_frames,
        shuffle=False,
        is_sequential_frames=is_sequential
    )
    
    X_data = process_videos(video_path, face_processor)
    model = load_model(model_number)
    
    frame_predictions = model.predict(X_data)
    mean_pred = np.mean(frame_predictions)
    final_prediction = "Fake" if mean_pred >= FAKE_THRESHOLD else "Real"
    
    return jsonify({"success": True, "prediction": final_prediction, "score": float(mean_pred)})


if __name__ == "__main__":
    app.run(port=5000)