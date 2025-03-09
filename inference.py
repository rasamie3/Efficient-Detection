import os
import argparse
import numpy as np
import tensorflow as tf
from extract_faces import FaceProcessor
from extract_frames import FrameExtractor

FAKE_THRESHOLD = 0.45

MODEL_NUMBER = {
    1: 'EfficientB1_non_sequential_frames',
    2: 'EfficientB3_non_sequential_frames',
    3: 'EfficientB3_sequential_frames',
    4: 'EfficientB1_sequential_frames'
}

MODELS = {
    "EfficientB1_non_sequential_frames": "./models/model_b1_1.keras",
    "EfficientB3_non_sequential_frames": "./models/model_b3_1_non_seq.keras",
    "EfficientB3_sequential_frames": "./models/model_b3_1_seq.keras",
    "EfficientB1_sequential_frames": "./models/model_b1_1_seq.keras"
}

def process_videos(video_path, face_extractor, create_frame_folder=True):
    """
    Args:
    video_path (str): Path to the video file.
    face_extractor (FaceProcessor): Instance of FaceProcessor for face extraction.
    create_frame_folder (bool, optional): Whether to create a new folder for extracted frames. Default is True.

    Returns:
        numpy.ndarray: Processed frames containing detected faces.
    """
    extractor = FrameExtractor(video_path=video_path, frame_path="./frames", create_frame_folder=create_frame_folder, auto_start=True)
    return face_extractor.process_frames()

def load_model(model_number=1):
    """
    Args:
    model_number (int, optional): Model selection key. Default is 1.

    Returns:
        tf.keras.Model: Loaded TensorFlow model.

    Raises:
        ValueError: If an invalid model number is provided.
        FileNotFoundError: If the model file does not exist.

    """
    model_key = MODEL_NUMBER.get(model_number)
    
    if not model_key:
        raise ValueError(f"Invalid model number: {model_number}. Choose from {list(MODEL_NUMBER.keys())}.")
    
    model_path = MODELS.get(model_key)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path.")
    
    return tf.keras.models.load_model(model_path)

def main(args):
    """
    Args:
    args (argparse.Namespace): Parsed command-line arguments.

    Workflow:
        1. Validates the model number.
        2. Determines whether sequential frames are needed.
        3. Prompts the user for frame count selection if required.
        4. Validates the video file format.
        5. Extracts and processes frames.
        6. Loads the specified model and performs inference.
        7. Prints the final prediction (Real or Fake).
    """
    print("Processing Video...")

    is_sequential = args.model_number >= 3 


    if args.model_number not in MODEL_NUMBER:
        raise ValueError(f"Invalid model number: {args.model_number}. Choose from {list(MODEL_NUMBER.keys())}.")

    if args.model_number in {1, 2}: 
        count_of_frames = input("Please choose a count of frames from this list [50, 100, -1]. "
                                "Note that -1 means all frames will be processed [default is -1]: ").strip()
        
        if not count_of_frames: 
            count_of_frames = -1
        else:
            try:
                count_of_frames = int(count_of_frames)
                if count_of_frames not in {50, 100, -1}:
                    raise ValueError
            except ValueError:
                raise ValueError(f"Invalid frame count. The value '{count_of_frames}' is not valid. Choose from [50, 100, -1].")

    elif args.model_number in {3, 4}:
        count_of_frames = 15
        print("Models 3 and 4 only accept 15 sequential frames.")        

    video_extensions = {'.mp4', '.avi', '.mov', '.wmv'}

    if not os.path.splitext(args.video_path)[1].lower() in video_extensions:
        print("Invalid video format. Supported formats: .mp4, .avi, .mov, .wmv")
        return

    face_processor = FaceProcessor(
        frames_folder="./frames",
        count_of_frames=count_of_frames,
        shuffle=False,
        is_sequential_frames=is_sequential
    )

    X_data = process_videos(args.video_path, face_processor, create_frame_folder=True)

    # print('X_data shape:', X_data.shape)

    model = load_model(args.model_number)

    print("Detecting the frames...")

    frame_predictions = model.predict(X_data)
    mean_pred = np.mean(frame_predictions)

    last_prediction = "Fake" if mean_pred >= FAKE_THRESHOLD else "Real"


    print('-----------------------------------------------')
    print(args.video_path)
    print(f"Final Prediction: {last_prediction}")
    # print(f'The video is {mean_pred}.')
    # print(f'The video is {frame_predictions}.')
    print('-----------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos to extract faces and train a classifier.")
    
    parser.add_argument('--video_path', type=str, default='../videos', help='Path to the directory containing videos.')
    parser.add_argument('--is_sequential_frames', type=int, default=0, help='Number of frames to extract faces from. Use -1 for all frames.')
    parser.add_argument('--model_number', type=int, default=1, help='Number of model to select.')

    args = parser.parse_args()
    
    main(args)