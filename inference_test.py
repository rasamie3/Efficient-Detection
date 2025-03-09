import os
import pandas as pd
import argparse
import numpy as np
import tensorflow as tf
from extract_faces import FaceProcessor
from extract_frames import FrameExtractor

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

    video_extensions = ['.mp4', '.avi', '.mov', '.wmv']

    video_paths = []
    
    for root, dirs, files in os.walk(args.video_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in video_extensions:
                file_path = os.path.abspath(os.path.join(root, file))
                video_paths.append(file_path)
    
    mean_list = []
    video_list = []

    
    for video in video_paths:
        face_extractor = FaceProcessor(frames_folder='./frames', count_of_frames = args.count_of_frames, shuffle=False, is_sequential_frames=is_sequential)
        
        X_data = process_videos(video,
                                face_extractor,
                                create_frame_folder=True)
        
        X_data = face_extractor.process_frames()

        model = load_model(args.model_number)
        
        print('Detecting the frames...')

        frame_predictions = model.predict(X_data)
        mean_pred = np.mean(frame_predictions.reshape(-1))

        video_list.append(video)
        mean_list.append(mean_pred)
        
        print('-----------------------------------------------')
        print(video)
        print('-----------------------------------------------')

    df = pd.DataFrame()
    
    df['vid'] = video_list
    df['mean_predictions'] = mean_list.round(4)
    df['actual'] = args.videos_actual_flags

    df.to_csv(f'results_{args.videos_actual_flags}.csv', index=False)
    print('Results file saved!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos to extract faces and train a classifier.")
    
    parser.add_argument('--video_path', type=str, default='../videos', help='Path to the directory containing videos.')
    parser.add_argument('--count_of_frames', type=int, default=-1, help='Number of frames to extract faces from. Use -1 for all frames.')
    parser.add_argument('--is_sequential_frames', type=int, default=0, help='Number of frames to extract faces from. Use -1 for all frames.')
    parser.add_argument('--model_number', type=int, default=1, help='Number of model to select.')
    parser.add_argument('--videos_actual_flags', type=int, default=1, help='Actual flag of the videos.')

    args = parser.parse_args()
    
    main(args)