## Efficient-Detection: Transfer Learning-Based Deepfake Video Detection

Efficient-Detection is a deepfake video detection system designed to utilize transfer learning with EfficientNet models, making it suitable for deployment on systems with limited computational resources. By leveraging pre-trained EfficientNet models, Efficient-Detection achieves moderate accuracy in detecting manipulated video content while keeping computational requirements low.

Features:
- **Diverse Dataset**: Utilizes a diverse dataset with balanced classes to ensure robust model performance across various deepfake techniques.
- **Random Frame Selection**: Employs a strategy of randomly selecting frames from videos to enhance model generalization and capture inconsistencies in manipulated content.
- **Pre-Trained Models**: Leverages pre-trained EfficientNet models to reduce training time and computational requirements.
- **Efficient Data Types**: Uses memory-saving data types like NumPy arrays for training and preprocessing.

In the next sections, we will explain the process from uploading a video to training and inference.

### Installation
#### Prerequisites
Ensure you have Python installed and set up a virtual environment (recommended):
```sh
python -m venv venv
source venv/bin/activate  
```
#### On Windows (using gitbash) use 
```sh
source venv\Scripts\activate
```
#### Install Required Python Packages
```sh
pip install -r requirements.txt
```

#### Install Required Node.js Packages
Ensure you have Node.js installed, then navigate to the `frontend` directory and install dependencies:
```sh
cd frontend
npm install
```

----

### `FrameExtractor` class in `extract_frames.py`
`FrameExtractor` is a Python class designed to extract frames from video files, which can be useful for deepfake detection using transfer learning. 
This script processes videos from a specified directory, extracts frames at each timestep, and saves them in an organized manner for further analysis.

#### Features
- Supports multiple video formats: `.mp4`, `.mov`, `.mwv`, `.mkv`, `.avi`
- Extracts frames from single or multiple videos in a directory
- Organizes frames into separate folders for each video
- Allows automatic execution upon initialization
- Ensures proper directory structure for frame storage

#### Usage
```python
from frame_extractor import FrameExtractor

# Initialize with default paths
frame_exec = FrameExtractor(video_path='./videos', frame_path='./frames', create_frame_folder=True, auto_start=True)
```

##### Parameters
- `video_path`: Path to the video file or directory containing videos.
- `frame_path`: Path where extracted frames will be saved.
- `create_frame_folder`: If `True`, clears and recreates the frame folder.
- `auto_start`: If `True`, starts extracting frames upon initialization.

##### Folder Structure(videos and frames of the videos)
```
/videos           # Input video files
/frames          # Extracted frames
    /video1      # Frames from video1.mp4
        frame_1.jpg
        frame_2.jpg
    /video2      # Frames from video2.mov
        frame_1.jpg
        frame_2.jpg
```

----

### `FaceProcessor` class in `extract_faces.py`
`FaceProcessor` is a Python-based tool designed to process video frames, detect and extract faces, and optionally apply data augmentation. It utilizes OpenCV and Dlib for face detection and landmark recognition, making it suitable for tasks such as deepfake detection, facial recognition, and machine learning preprocessing.

#### Features
- Extracts frames from videos and detects faces.
- Uses OpenCV's Haar Cascade for face detection.
- Utilizes Dlib's 68-face landmark predictor for precise face localization.
- Supports data augmentation for training deep learning models.
- Allows frame shuffling and sequential ordering.
- Configurable face detection parameters.

#### Usage
```python
from face_processor import FaceProcessor

face_processor = FaceProcessor(frames_folder="./frames", count_of_frames=50, img_size=(299, 299), shuffle=True, is_sequential_frames=False)
```
##### Parameters
- `frames_folder`: Path to frames directory
- `count_of_frames`: Number of frames to process (-1 for all)
- `img_size`: Target size of extracted faces
- `shuffle`: Shuffle frames before processing
- `is_sequential_frames`: Maintain sequential frame order

##### Error Handling
- If no face is detected in a frame, it will be skipped with a warning.
- If the landmark model file is missing, an error will be raised.

----

### `Classifier` class in `train_model.py`
This module classifies faces using an EfficientNet-based model with optional LSTM layers for sequential learning.

#### Usage
```python
from classifier import Classifier

classifier = Classifier(input_shape=(229, 229, 3), selected_model=3)
classifier.train(training_data, validation_data, epochs=5)
loss, accuracy = classifier.evaluate(X_test, y_test)
```
#### Parameters
- `input_shape`: Input image shape.
- `learning_rate`: Learning rate for training.
- `retrain`: Load a pre-trained model if `True`.
- `save_model`: Save trained model if `True`.
- `trained_model`: Path to pre-trained model.
- `selected_model`: Choose EfficientNetB1 (1) or EfficientNetB3 (3).
- `is_sequential_training`: Enable sequential training with LSTM layers.

#### Methods
- `build_model()`: Builds and compiles the model.
- `train(training_data, validation_data, batch_size, epochs)`: Trains the model.
- `evaluate(X_test, y_test)`: Evaluates model performance.
- `save()`: Saves the trained model.

#### Model Training
To train the classifier:
```python
classifier.train(training_data, validation_data, epochs=10)
```
To evaluate:
```python
test_loss, test_acc = classifier.evaluate(X_test, y_test)
```
#### Saving and Loading Models
To save a model after training:
```python
classifier.save()
```
#### To load a pre-trained model:
```python
classifier = Classifier(retrain=True, trained_model="./models/model.keras") # Replace it with a correct model name
```
----

### Notes
- Ensure `shape_predictor_68_face_landmarks.dat` is available in the correct path for Dlib.
- EfficientNet weights are downloaded automatically from TensorFlow.
- Sequential training with LSTM layers is available for temporal modeling.
- Unrar `models.rar` inside a folder named `models`. Download `models.rar` from [here](https://drive.google.com/drive/folders/1SNfEcxFgqMMOSmQk6UzVHxZuM7fYweVb?usp=drive_link)

----

### Running the Full Pipeline
To extract frames, process faces, and train the classifier using command-line arguments:

```sh
python main.py --real_video_path "./videos/real" \
               --fake_video_path "../videos/fake" \
               --frame_path "./frames" \
               --count_of_frames -1 \
               --create_frame_folder 1 \
               --shuffle_frames 0 \
               --is_sequential_frames 0 \
               --implement_augmentation 0 \
               --learning_rate 0.001 \
               --batch_size 1 \
               --epochs 50 \
               --save_model 1 \
               --img_width 229 \
               --img_height 299
```

--- 

### Model Results 
The data used for training comes from the `Celeb DF Dataset`. Since we are performing transfer learning with limited resources, I have used a minimal amount of data for experimentation. Despite this, I achieved moderate results using approximately 8-9K frames extracted from random videos and shuffled.

##### `model_b1_1_non_seq.keras` (EfficientNetB1 - Non-Sequential Frames)
- **Threshold 0.4**: Accuracy: 0.8200, Precision: 0.7759, Recall: 0.9000, F1 Score: 0.8333
- **Threshold 0.45**: Accuracy: 0.8200, Precision: 0.82, Recall: 0.82, F1 Score: 0.82

##### `model_b3_1_non_seq.keras` (EfficientNetB3 - Non-Sequential Frames - Data augmentation used)
- **Threshold 0.45**: Accuracy: 0.8500, Precision: 0.8571, Recall: 0.8400, F1 Score: 0.8485
- **Threshold 0.4**: Accuracy: 0.8500, Precision: 0.8182, Recall: 0.9000, F1 Score: 0.8571

##### `model_b3_1_seq.keras` (EfficientNetB3 - Sequential Frames)
- **Threshold 0.4**: Accuracy: 0.800, Precision: 0.7843, Recall: 0.8000, F1 Score: 0.7921
- **Threshold 0.45**: Accuracy: 0.7800, Precision: 0.7917, Recall: 0.7600, F1 Score: 0.7755

##### `model_b1_1_seq.keras` (EfficientNetB1 - Sequential Frames)
- **Threshold 0.45**: Accuracy: 0.8100, Precision: 0.8039, Recall: 0.8200, F1 Score: 0.8119
- **Threshold 0.4**: Accuracy: 0.8000, Precision: 0.7778, Recall: 0.8400, F1 Score: 0.8077

----

### Web Dev app
I have also created a simple Web Dev app using `Express-NodeJS`, `Flask-Python` and `HTML\CSS\JS` to provide better use of the deepfake detection system

To run the client-side run:
```sh
cd ./frontend
npm run server.js
```
To run the server-side run:
```sh
python -m backend.app
```

----

###### Things to enhance in this project:
- Make the user able to choose number of frames for detection.
- Make the user able to choose different models for detection.
- Make a better UI with a Front-End framework.
- Remove videos from `video` folder after detection to save space
- Remove frames from `frames` folder after detection to save space
- Implement continues learning to enhance model performance and use new datasets with new Deepfake techniques
