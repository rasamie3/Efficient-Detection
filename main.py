import argparse
import numpy as np

from train_model import Classifier
from extract_faces import FaceProcessor
from extract_frames import FrameExtractor
from sklearn.model_selection import train_test_split


def process_videos(video_path, frame_path, face_extractor, create_frame_folder=False):
    """
    Extracts frames from the given video and processes them to extract faces.
    
    Args:
        video_path (str): Path to the input video.
        frame_path (str): Path where extracted frames will be saved.
        face_extractor (FaceProcessor): Instance of FaceProcessor to process frames.
        create_frame_folder (bool, optional): Whether to create a folder for extracted frames. Defaults to False.
    
    Returns:
        numpy.ndarray: Processed face data extracted from video frames.
    """
    extractor = FrameExtractor(video_path=video_path, frame_path=frame_path, create_frame_folder=create_frame_folder, auto_start=True)
    return face_extractor.process_frames()

def main(args):
    """
    Main function to process videos, extract face data, prepare datasets, and train a classifier.
    
    Args:
        args (Namespace): Parsed command-line arguments containing paths and configuration settings.
    """

    face_extractor = FaceProcessor(frames_folder=args.frame_path, 
                                    count_of_frames=args.count_of_frames,
                                    img_size=(args.img_width, args.img_height),
                                    shuffle=args.shuffle_frames,
                                    is_sequential_frames=args.is_sequential_frames)
    
    X_data_real = process_videos(args.real_video_path, args.frame_path, face_extractor, args.create_frame_folder)
    X_data_fake = process_videos(args.fake_video_path, args.frame_path, face_extractor, args.create_frame_folder)

    if args.is_sequential_frames:
        y_data_real = np.array(np.zeros((X_data_real.shape[0], X_data_real.shape[1])))
        y_data_fake = np.array(np.ones((X_data_fake.shape[0], X_data_fake.shape[1])))
    else:
        y_data_real = np.array(np.zeros(X_data_real.shape[0]))
        y_data_fake = np.array(np.ones(X_data_fake.shape[0]))

    X_data = np.concatenate((X_data_real, X_data_fake))
    y_data = np.concatenate((y_data_real, y_data_fake))

    print(y_data.shape)

    del X_data_real, X_data_fake, y_data_real, y_data_fake

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    test_data = (X_test, y_test) 
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    validation_data = (X_val, y_val)
    
    if args.implement_augmentation and not args.is_sequential_frames: # only works when data isn't sequential (by frame)
        datagen = face_extractor.implement_data_augmentation()
        training_data = datagen.flow(X_train, y_train, batch_size=8, shuffle=False)
        training_data = (training_data)
    else:
        print('not aug')
        training_data = (X_train, y_train)
    
    classifier = Classifier(input_shape=X_data.shape[1:], save_model=args.save_model, is_sequential_training=args.is_sequential_frames)
    classifier.train(training_data, validation_data, batch_size=args.batch_size, epochs=args.epochs)
    classifier.evaluate(*test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos to extract faces and train a classifier.")
    
    parser.add_argument('--real_video_path', type=str, default='../videos/real', help='Path to the directory containing real videos.')
    parser.add_argument('--fake_video_path', type=str, default='../videos/fake', help='Path to the directory containing fake videos.')
    parser.add_argument('--frame_path', type=str, default='../frames', help='Path where extracted frames will be saved.')
    parser.add_argument('--count_of_frames', type=int, default=-1, help='Number of frames to extract faces from. Use -1 for all frames.')
    parser.add_argument('--create_frame_folder', type=int, default=1, help='Whether to create folder for the frames.')
    parser.add_argument('--shuffle_frames', type=int, default=0, help='Whether to shuffle the frames before processing.')
    parser.add_argument('--is_sequential_frames', type=int, default=0, help='Whether to reshape the frames to be sequential before processing.')
    parser.add_argument('--implement_augmentation', type=int, default=0, help='Implement data augmentation.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training.')
    parser.add_argument('--save_model', type=int, default=1, help='Whether to save the model or not.')
    parser.add_argument('--img_width', type=int, default=229,
                        help='Width of the images used for training.')
    parser.add_argument('--img_height', type=int, default=299,
                        help='Height of the images used for training.')

    args = parser.parse_args()
    
    main(args)