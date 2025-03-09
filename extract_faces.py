import os
import cv2
import dlib
import random
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet  import preprocess_input

IMG_EXTENSION = {'.jpg', '.jpeg', '.png'}
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


class FaceProcessor(): 
    LANDMARKS_MODEL_PATH = "./shape_predictor_68_face_landmarks.dat"
    DEFAULT_CONFIG = {
        "scaleFactor": 1.05,
        "minNeighbors": 7,
    }

    def __init__(self, frames_folder='./frames', count_of_frames=-1, img_size=(299, 299), shuffle=True, is_sequential_frames=False):
        """
        Initializes the FaceProcessor class with parameters for processing video frames.

        Args:
            frames_folder (str): Directory containing video frames.
            count_of_frames (int): Number of frames to process (-1 for all available frames).
            img_size (tuple): Target size for extracted face images.
            shuffle (bool): If True, shuffles the frames before processing.
            is_sequential_frames (bool): If True, keeps frames in sequential order.
        """
        self.frames_folder = frames_folder
        self.shuffle = shuffle
        self.count_of_frames = count_of_frames
        self.img_size = img_size
        self.is_sequential_frames = is_sequential_frames
        self.config = self.DEFAULT_CONFIG.copy()

        self.face_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if not os.path.exists(self.LANDMARKS_MODEL_PATH):
            raise FileNotFoundError(f"Landmark model file not found: {self.LANDMARKS_MODEL_PATH}")

        self.landmark_model = dlib.shape_predictor(self.LANDMARKS_MODEL_PATH)

    def set_detection_params(self, scaleFactor=None, minNeighbors=None):
        """
        Updates face detection parameters dynamically.

        Args:
            scaleFactor (float, optional): Scale factor for face detection.
            minNeighbors (int, optional): Minimum number of neighbors required for a face detection.
        """
        if scaleFactor is not None:
            self.config["scaleFactor"] = scaleFactor
        if minNeighbors is not None:
            self.config["minNeighbors"] = minNeighbors


    def get_frames(self):
        """
        Retrieves all frames from the specified folder.

        Returns:
            dict: A dictionary where keys are folder paths and values are lists of frame filenames.
        Raises:
            FileNotFoundError: If the frames folder does not exist.
        """
        if not os.path.exists(self.frames_folder):
            raise FileNotFoundError(f"Frames folder not found: {self.frames_folder}")
        folder_dict = {}
        for dirpath, _, filnames in os.walk(self.frames_folder):

            folder_key = dirpath
            frames = [
                f for f in filnames 
                if os.path.splitext(f)[1].lower().lower() in IMG_EXTENSION]

            if frames:
                if self.shuffle:
                    random.shuffle(frames)
                folder_dict[folder_key] = frames

        self.folder_dict = folder_dict
        return folder_dict
    
    def face_extractor(self, image):
        """
        Detects and extracts the most prominent face from an image.

        Args:
            image (numpy.ndarray): Input image containing a face.

        Returns:
            tuple: Extracted face image and the number of detected landmarks,
                   or None if no face is found.
        """
        if image is None:
            print("No images has been received")
            return None
            
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_model.detectMultiScale(
            gray_frame,
            scaleFactor=self.config['scaleFactor'],
            minNeighbors=self.config['minNeighbors']
        )
        
        if len(faces) == 0:
            return None
            
        best_face = None
        max_landmarks = 0
        
        for (x, y, w, h) in faces:
            try:
                rect = dlib.rectangle(x, y, x+w, y+h)
                landmarks = self.landmark_model(gray_frame, rect)
                num_landmarks = len(landmarks.parts())
                
                if num_landmarks > max_landmarks:
                    max_landmarks = num_landmarks
                    best_face = image[y:y+h, x:x+w]
                    best_face = cv2.resize(best_face, self.img_size)
            except Exception as e:
                print(f'Error detecting the face {e}')
                continue
                
        return (best_face, max_landmarks) if best_face is not None else None 
    

    def process_frames(self): 
        """
        Processes all frames in the folder to extract faces.

        Returns:
            numpy.ndarray: Array of extracted face images reshaped by frames if specified.
        """
        self.get_frames()

        X_data = []
        
        count_of_videos = len(self.folder_dict)
        video_counter = 0

        for folder_path, frames in self.folder_dict.items():
            video_counter += 1
            print(f'Processing video {video_counter} / {count_of_videos}')
            
            count_face_frame = 0
            faces_list = []

            for frame_name in frames:
                
                frame_path = os.path.join(folder_path, frame_name)
                try:
                    frame = cv2.imread(frame_path)
                    face_result = self.face_extractor(frame)
                    
                    if face_result:
                        count_face_frame += 1
                        print(f'processing frame {count_face_frame} / {self.count_of_frames} from video no. {video_counter}')
                        face_image, landmark_count = face_result
                        faces_list.append(face_image)
                    
                        if count_face_frame == self.count_of_frames:
                            break
                    else:
                        print(f"No face found in {frame_path}")
                        
                except Exception as e:
                    print(f"Error processing {frame_path}: {str(e)}")
                    continue

            if (len(faces_list) == self.count_of_frames) or (self.count_of_frames == -1):
                X_data.append(faces_list)

        X_data = np.array(X_data)
        if not self.is_sequential_frames:
            X_data = X_data.reshape(-1, self.img_size[0], self.img_size[1], 3)

        X_data = preprocess_input(X_data)
        return X_data
    
    def implement_data_augmentation(self):
        """
        Applies data augmentation transformations to face images.

        Returns:
            ImageDataGenerator: Configured generator for image augmentation.
        """
        datagen = ImageDataGenerator(
            rotation_range=20, 
            shear_range=0.2,     
            zoom_range=0.2,       
            fill_mode='nearest'     
        )

        return datagen