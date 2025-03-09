import os
import cv2
import shutil


VID_EXTENSIONS = {'.mp4', '.mov', '.mwv', '.mkv', '.avi'}

class FrameExtractor:
    def __init__(self, video_path='./videos', frame_path='./frames', create_frame_folder=True, auto_start=False):
        """
        Initializes the VideoProcessor with paths for videos and frames.

        :param video_path: Path to the video file or directory containing videos.
        :param frame_path: Path where extracted frames will be saved.
        :param auto_start: If True, starts extracting frames immediately.
        
        """

        self.video_path = os.path.normpath(video_path).replace('\\', '/')
        self.frame_path = os.path.normpath(frame_path).replace('\\', '/')
        self.create_frame_folder = create_frame_folder
        self.auto_start = auto_start


        if self.auto_start:
            self.frame_extractor()

    def _path_exists(self, path):
        """Checks if a path exists; creates it if specified."""

        if not os.path.exists(path):
            os.makedirs(path)
            return True

        if self.create_frame_folder:
            shutil.rmtree(path)  
            os.makedirs(path)     
            return True

    
    def list_videos(self):
        """Lists all video files in the specified path."""

        if os.path.isfile(self.video_path) and os.path.splitext(self.video_path)[1].lower() in VID_EXTENSIONS:
            return [self.video_path]

        elif os.path.isdir(self.video_path):
            video_files = []
            for entry in os.listdir(self.video_path):
                full_path = os.path.join(self.video_path, entry)
                if os.path.isfile(full_path) and os.path.splitext(full_path)[1].lower() in VID_EXTENSIONS:
                    video_files.append(os.path.normpath(full_path))
            return video_files

        return []
        

    def frame_extractor(self):
        """Extracts frames from videos and saves them to the specified directory."""
        if not os.path.exists(self.video_path):
            print(f'Please check if the path >> {self.video_path} << exists!')
            return
        
        self._path_exists(self.frame_path)
        videos = self.list_videos()
        
        video_counter = 0
        count_of_videos = len(videos)

        for video in videos:
            video_counter += 1

            video_name, _ = os.path.splitext(os.path.basename(video))
            video_folder = os.path.join(self.frame_path, video_name)
            os.makedirs(video_folder, exist_ok=True)

            print(f"Processing video: {video_name} ({video_counter} / {count_of_videos})")

            vid_obj = cv2.VideoCapture(video)
            frame_counter = 0

            while True:
                frame_exists, frame = vid_obj.read()
                if not frame_exists:
                    break

                frame_counter += 1
                frame_filename = os.path.join(video_folder, f"frame_{frame_counter}.jpg")
                cv2.imwrite(frame_filename, frame)
