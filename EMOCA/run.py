

import cv2
import os
import numpy as np
import torch
import time
import argparse
import threading
import logging
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import queue

# Third-party imports
try:
    from Sim3DR.renderer import render_fvr
except ImportError:
    print("Warning: Sim3DR not found. 3D rendering will be disabled.")
    render_fvr = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("Error: MediaPipe not installed. Please install mediapipe.")
    sys.exit(1)

try:
    from faceversev4 import FaceVerseRecon
except ImportError:
    print("Error: FaceVerse not found. Please install faceversev4.")
    sys.exit(1)


class ProcessingMode(Enum):
    """Enumeration for different processing modes."""
    IMAGE = "image"
    VIDEO = "video"
    WEBCAM = "webcam"
    IMAGE_FOLDER = "image_folder"


@dataclass
class ProcessingConfig:
    """Configuration class for processing parameters."""
    batch_size: int = 4
    enable_smoothing: bool = True
    save_results: bool = True
    save_ply: bool = False
    enable_visualization: bool = True
    max_queue_size: int = 10
    smoothing_window_size: int = 3
    output_quality: int = 95
    enable_profiling: bool = False


@dataclass
class FaceDetectionResult:
    """Data class for face detection results."""
    bounding_box: np.ndarray
    eye_parameters: np.ndarray
    landmarks: np.ndarray
    confidence: float
    timestamp: float


class FaceVerseLogger:
    """Comprehensive logging class for FaceVerse pipeline."""
    
    def __init__(self, log_level=logging.INFO, log_file=None):
        """Initialize logger with specified level and output file."""
        self.logger = logging.getLogger("FaceVersePipeline")
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def performance(self, message: str):
        """Log performance-related message."""
        self.logger.info(f"PERFORMANCE: {message}")


class ConfigurationManager:
    """Manages configuration loading, validation, and saving."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path
        self.default_config = {
            "processing": {
                "batch_size": 4,
                "enable_smoothing": True,
                "smoothing_window_size": 3,
                "max_queue_size": 10
            },
            "output": {
                "save_results": True,
                "save_ply": False,
                "output_quality": 95,
                "create_subdirectories": True
            },
            "rendering": {
                "enable_visualization": True,
                "render_resolution": [512, 512],
                "enable_shading": True
            },
            "performance": {
                "enable_profiling": False,
                "log_performance_metrics": True,
                "gpu_memory_fraction": 0.8
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                return self._merge_configs(self.default_config, loaded_config)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                return self.default_config
        return self.default_config
    
    def save_config(self, config: Dict[str, Any], path: str):
        """Save configuration to file."""
        try:
            with open(path, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving config to {path}: {e}")
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge user config with defaults."""
        merged = default.copy()
        for key, value in user.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged


class PerformanceProfiler:
    """Performance profiling and monitoring class."""
    
    def __init__(self, enabled: bool = False):
        """Initialize profiler."""
        self.enabled = enabled
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timer for an operation."""
        if self.enabled:
            self.start_times[operation] = time.time()
    
    def stop_timer(self, operation: str):
        """Stop timer and record duration."""
        if self.enabled and operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            del self.start_times[operation]
    
    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation."""
        if operation in self.metrics and self.metrics[operation]:
            return sum(self.metrics[operation]) / len(self.metrics[operation])
        return 0.0
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all recorded metrics."""
        summary = {}
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
        return summary
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.start_times.clear()


class FaceDetector:
    """Enhanced face detector using MediaPipe with error handling and logging."""
    
    def __init__(self, model_path: str, logger: FaceVerseLogger):
        """Initialize face detector."""
        self.logger = logger
        self.model_path = model_path
        
        try:
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            
            self.face_detector = vision.FaceLandmarker.create_from_options(options)
            self.logger.info("Face detector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize face detector: {e}")
            raise
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        """
        Detect facial landmarks in the given image.
        
        Args:
            image: Input image in RGB format
            
        Returns:
            FaceDetectionResult or None if no face detected
        """
        try:
            # Convert to MediaPipe image format
            mp_image = mp.Image(mp.ImageFormat.SRGB, image.astype(np.uint8))
            
            # Perform detection
            results = self.face_detector.detect(mp_image)
            
            if not results.face_landmarks or len(results.face_landmarks) == 0:
                return None
            
            # Extract landmarks
            landmarks = results.face_landmarks[0]
            landmarks_array = np.array([(lmk.x, lmk.y, lmk.z) for lmk in landmarks])
            
            # Convert normalized coordinates to pixel coordinates
            landmarks_array[:, 0] = landmarks_array[:, 0] * image.shape[1]
            landmarks_array[:, 1] = landmarks_array[:, 1] * image.shape[0]
            
            # Calculate bounding box
            bbox = self._calculate_bounding_box(landmarks_array)
            
            # Calculate eye parameters
            eye_params = self._calculate_eye_parameters(landmarks_array)
            
            return FaceDetectionResult(
                bounding_box=bbox,
                eye_parameters=eye_params,
                landmarks=landmarks_array,
                confidence=0.7, 
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error during face detection: {e}")
            return None
    
    def _calculate_bounding_box(self, landmarks: np.ndarray) -> np.ndarray:
        """Calculate bounding box from landmarks with padding."""
        min_coords = np.min(landmarks[:, :2], axis=0)
        max_coords = np.max(landmarks[:, :2], axis=0)
        
        # Add padding
        padding = 0.1  # 10% padding
        width = max_coords[0] - min_coords[0]
        height = max_coords[1] - min_coords[1]
        
        min_coords[0] -= width * padding
        min_coords[1] -= height * padding
        max_coords[0] += width * padding
        max_coords[1] += height * padding
        
        # Ensure coordinates are within image bounds
        min_coords = np.maximum(min_coords, 0)
        # Note: max_coords bounds checking would need image dimensions
        
        return np.array([min_coords[0], min_coords[1], max_coords[0], max_coords[1]])
    
    def _calculate_eye_parameters(self, landmarks: np.ndarray) -> np.ndarray:
        """Calculate eye gaze parameters from landmarks."""
        try:
            # Left eye parameters
            left_eye_center = (landmarks[263, :2] + landmarks[362, :2]) / 2
            left_eye_vector = self._normalize(landmarks[362, :2] - landmarks[263, :2])
            
            left_eye_pupil = landmarks[473, :2]
            left_eye_x = np.dot(left_eye_pupil - left_eye_center, left_eye_vector)
            left_eye_y = np.dot(left_eye_pupil - left_eye_center, left_eye_vector[[1, 0]]) * -1.5
            
            # Normalize by eye distance
            left_eye_dist = np.linalg.norm(landmarks[362, :2] - landmarks[263, :2])
            left_eye_x /= left_eye_dist
            left_eye_y /= left_eye_dist
            
            # Right eye parameters
            right_eye_center = (landmarks[33, :2] + landmarks[133, :2]) / 2
            right_eye_vector = self._normalize(landmarks[33, :2] - landmarks[133, :2])
            
            right_eye_pupil = landmarks[468, :2]
            right_eye_x = np.dot(right_eye_pupil - right_eye_center, right_eye_vector)
            right_eye_y = np.dot(right_eye_pupil - right_eye_center, right_eye_vector[[1, 0]]) * -1.5
            
            # Normalize by eye distance
            right_eye_dist = np.linalg.norm(landmarks[33, :2] - landmarks[133, :2])
            right_eye_x /= right_eye_dist
            right_eye_y /= right_eye_dist
            
            return np.array([left_eye_y, left_eye_x, right_eye_y, right_eye_x])
            
        except Exception as e:
            self.logger.warning(f"Error calculating eye parameters: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0])
    
    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize a vector."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


class FrameProcessor:
    """Processes frames for face reconstruction with batching and smoothing."""
    
    def __init__(self, config: ProcessingConfig, logger: FaceVerseLogger):
        """Initialize frame processor."""
        self.config = config
        self.logger = logger
        self.profiler = PerformanceProfiler(config.enable_profiling)
        
        # Initialize smoothing buffers
        self.smoothing_buffer = {
            'boxes': [],
            'frames': [],
            'names': []
        }
    
    def add_frame_for_smoothing(self, frame: np.ndarray, frame_name: str, 
                               detection_result: FaceDetectionResult) -> Tuple[np.ndarray, str, np.ndarray]:
        """Add frame to smoothing buffer and return smoothed result if available."""
        # Store current detection
        current_data = np.concatenate([
            detection_result.bounding_box,
            detection_result.eye_parameters
        ])
        
        # Add to buffer
        self.smoothing_buffer['boxes'].append(current_data)
        self.smoothing_buffer['frames'].append(frame)
        self.smoothing_buffer['names'].append(frame_name)
        
        # Maintain buffer size
        if len(self.smoothing_buffer['boxes']) > self.config.smoothing_window_size:
            self.smoothing_buffer['boxes'].pop(0)
            self.smoothing_buffer['frames'].pop(0)
            self.smoothing_buffer['names'].pop(0)
        
        # Apply smoothing if we have enough frames
        if len(self.smoothing_buffer['boxes']) == self.config.smoothing_window_size:
            smoothed_box = np.mean(self.smoothing_buffer['boxes'], axis=0)
            # Use middle frame for output
            middle_idx = self.config.smoothing_window_size // 2
            output_frame = self.smoothing_buffer['frames'][middle_idx]
            output_name = self.smoothing_buffer['names'][middle_idx]
            
            # Split back into bbox and eye params
            bbox = smoothed_box[:4]
            eye_params = smoothed_box[4:]
            
            return output_frame, output_name, np.stack([bbox, eye_params])
        else:
            # Not enough frames for smoothing, return current
            return frame, frame_name, np.stack([
                detection_result.bounding_box,
                detection_result.eye_parameters
            ])


class OutputManager:
    """Manages output files and formats."""
    
    def __init__(self, output_dir: str, config: ProcessingConfig, logger: FaceVerseLogger):
        """Initialize output manager."""
        self.output_dir = Path(output_dir)
        self.config = config
        self.logger = logger
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories if needed
        if config.save_results:
            self.image_dir = self.output_dir / "images"
            self.image_dir.mkdir(exist_ok=True)
        
        if config.save_ply:
            self.ply_dir = self.output_dir / "ply_models"
            self.ply_dir.mkdir(exist_ok=True)
        
        self.param_dir = self.output_dir / "parameters"
        self.param_dir.mkdir(exist_ok=True)
    
    def save_image_result(self, image: np.ndarray, filename: str):
        """Save processed image result."""
        if self.config.save_results:
            output_path = self.image_dir / filename
            cv2.imwrite(str(output_path), image)
            self.logger.debug(f"Saved image: {output_path}")
    
    def save_parameters(self, parameters: Dict[str, Any], filename: str):
        """Save reconstruction parameters."""
        output_path = self.param_dir / f"{Path(filename).stem}.npy"
        np.save(str(output_path), parameters)
        self.logger.debug(f"Saved parameters: {output_path}")
    
    def save_ply_model(self, vertices: np.ndarray, colors: np.ndarray, 
                      triangles: np.ndarray, filename: str):
        """Save 3D model as PLY file."""
        if self.config.save_ply:
            output_path = self.ply_dir / f"{Path(filename).stem}.ply"
            self._write_ply_file(vertices, colors, triangles, str(output_path))
            self.logger.debug(f"Saved PLY model: {output_path}")
    
    def _write_ply_file(self, vertices: np.ndarray, colors: np.ndarray, 
                       triangles: np.ndarray, output_path: str):
        """Write PLY file with vertex colors."""
        num_vertices = len(vertices)
        num_faces = len(triangles)
        
        header = f"""ply
format ascii 1.0
element vertex {num_vertices}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {num_faces}
property list uchar int vertex_indices
end_header\n"""
        
        try:
            with open(output_path, 'w') as f:
                f.write(header)
                
                # Write vertices with colors
                for i, vertex in enumerate(vertices):
                    color = colors[i] if i < len(colors) else [128, 128, 128]
                    f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} "
                           f"{int(color[0])} {int(color[1])} {int(color[2])}\n")
                
                # Write faces
                for face in triangles:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
                    
        except Exception as e:
            self.logger.error(f"Error writing PLY file {output_path}: {e}")


class EnhancedFrameLoader(threading.Thread):
    """
    Enhanced frame loader with support for multiple input sources
    and better resource management.
    """
    
    def __init__(self, args, config: ProcessingConfig, logger: FaceVerseLogger):
        """Initialize enhanced frame loader."""
        super().__init__()
        self.args = args
        self.config = config
        self.logger = logger
        self.profiler = PerformanceProfiler(config.enable_profiling)
        
        # Determine processing mode
        self.mode = self._determine_processing_mode()
        self.logger.info(f"Processing mode: {self.mode}")
        
        # Initialize face detector
        self.face_detector = FaceDetector('data/face_landmarker.task', logger)
        
        # Initialize queues and state
        self.frames_queue = queue.Queue(maxsize=config.max_queue_size)
        self.frame_info_queue = queue.Queue(maxsize=config.max_queue_size)
        self.stop_event = threading.Event()
        self.done = False
        
        # Video capture (for video and webcam modes)
        self.cap = None
        self.current_frame_index = 0
        
        # Image list (for folder mode)
        self.image_files = []
        self.current_image_index = 0
        
        self._initialize_input_source()
    
    def _determine_processing_mode(self) -> ProcessingMode:
        """Determine the processing mode based on input."""
        input_path = self.args.input.lower()
        
        if input_path == 'webcam':
            return ProcessingMode.WEBCAM
        elif input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return ProcessingMode.VIDEO
        elif input_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            return ProcessingMode.IMAGE
        else:
            # Check if it's a directory
            if os.path.isdir(self.args.input):
                return ProcessingMode.IMAGE_FOLDER
            else:
                raise ValueError(f"Unsupported input format: {self.args.input}")
    
    def _initialize_input_source(self):
        """Initialize the appropriate input source based on mode."""
        try:
            if self.mode in [ProcessingMode.VIDEO, ProcessingMode.WEBCAM]:
                if self.mode == ProcessingMode.WEBCAM:
                    self.cap = cv2.VideoCapture(0)
                    self.logger.info("Initialized webcam capture")
                else:
                    self.cap = cv2.VideoCapture(self.args.input)
                    self.logger.info(f"Initialized video capture: {self.args.input}")
                
                if not self.cap.isOpened():
                    raise RuntimeError(f"Could not open video source: {self.args.input}")
                    
            elif self.mode == ProcessingMode.IMAGE_FOLDER:
                self._load_image_files()
                self.logger.info(f"Loaded {len(self.image_files)} images from folder")
                
            elif self.mode == ProcessingMode.IMAGE:
                self.image_files = [self.args.input]
                self.logger.info(f"Processing single image: {self.args.input}")
                
        except Exception as e:
            self.logger.error(f"Error initializing input source: {e}")
            raise
    
    def _load_image_files(self):
        """Load image files from directory."""
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.image_files = []
        
        for file_path in Path(self.args.input).iterdir():
            if file_path.suffix.lower() in supported_formats:
                self.image_files.append(str(file_path))
        
        self.image_files.sort()
    
    def run(self):
        """Main frame loading loop."""
        self.logger.info("Frame loader started")
        
        try:
            if self.mode in [ProcessingMode.VIDEO, ProcessingMode.WEBCAM]:
                self._process_video_stream()
            else:
                self._process_images()
                
        except Exception as e:
            self.logger.error(f"Error in frame loader: {e}")
        finally:
            self.done = True
            if self.cap:
                self.cap.release()
            self.logger.info("Frame loader finished")
    
    def _process_video_stream(self):
        """Process video stream (file or webcam)."""
        while not self.stop_event.is_set():
            if self.frames_queue.qsize() < self.config.max_queue_size:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.info("End of video stream reached")
                    break
                
                # Convert BGR to RGB
                frame_rgb = frame[:, :, ::-1].copy()
                frame_name = f"frame_{self.current_frame_index:06d}.jpg"
                
                # Detect faces
                detection_result = self.face_detector.detect_landmarks(frame_rgb)
                
                # Add to queue
                try:
                    self.frames_queue.put(frame_rgb, timeout=1.0)
                    self.frame_info_queue.put({
                        'name': frame_name,
                        'detection': detection_result,
                        'index': self.current_frame_index
                    }, timeout=1.0)
                    
                    self.current_frame_index += 1
                    self.logger.debug(f"Loaded frame {frame_name}")
                    
                except queue.Full:
                    self.logger.warning("Frame queue full, skipping frame")
                    continue
                
            else:
                time.sleep(0.01)  # Prevent busy waiting
    
    def _process_images(self):
        """Process image or image folder."""
        for image_path in self.image_files:
            if self.stop_event.is_set():
                break
                
            if self.frames_queue.qsize() < self.config.max_queue_size:
                try:
                    # Load image
                    frame = cv2.imread(image_path)
                    if frame is None:
                        self.logger.warning(f"Could not load image: {image_path}")
                        continue
                    
                    frame_rgb = frame[:, :, ::-1].copy()
                    frame_name = Path(image_path).name
                    
                    # Detect faces
                    detection_result = self.face_detector.detect_landmarks(frame_rgb)
                    
                    # Add to queue
                    self.frames_queue.put(frame_rgb)
                    self.frame_info_queue.put({
                        'name': frame_name,
                        'detection': detection_result,
                        'index': self.current_image_index
                    })
                    
                    self.current_image_index += 1
                    self.logger.debug(f"Loaded image {frame_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing image {image_path}: {e}")
                    continue
            else:
                time.sleep(0.01)
    
    def get_next_frame(self) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Get next frame from queue."""
        try:
            frame = self.frames_queue.get(timeout=1.0)
            frame_info = self.frame_info_queue.get(timeout=1.0)
            return frame, frame_info
        except queue.Empty:
            return None, None
    
    def stop(self):
        """Stop the frame loader."""
        self.stop_event.set()
        self.logger.info("Frame loader stop requested")


class FaceVersePipeline:
    """
    Main FaceVerse 3D face reconstruction pipeline.
    
    This class orchestrates the entire reconstruction process including
    frame loading, face detection, 3D reconstruction, and output generation.
    """
    
    def __init__(self, args, config: ProcessingConfig):
        """Initialize the FaceVerse pipeline."""
        self.args = args
        self.config = config
        self.logger = FaceVerseLogger()
        self.profiler = PerformanceProfiler(config.enable_profiling)
        
        # Initialize components
        self.logger.info("Initializing FaceVerse pipeline...")
        
        # Load FaceVerse model
        self.device = self._initialize_device()
        self.faceverse_model = self._load_faceverse_model()
        
        # Initialize processing components
        self.frame_loader = EnhancedFrameLoader(args, config, self.logger)
        self.frame_processor = FrameProcessor(config, self.logger)
        self.output_manager = OutputManager(args.output, config, self.logger)
        
        # Processing state
        self.frame_count = 0
        self.start_time = None
        self.cache_frame = []
        self.cache_param = []
        
        self.logger.info("FaceVerse pipeline initialized successfully")
    
    def _initialize_device(self) -> torch.device:
        """Initialize and return the appropriate computing device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            
            # Set GPU memory fraction if specified
            if hasattr(self.config, 'gpu_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU")
        
        return device
    
    def _load_faceverse_model(self):
        """Load the FaceVerse reconstruction model."""
        self.logger.info("Loading FaceVerse model...")
        
        try:
            model = FaceVerseRecon(
                "data/faceverse_v4_2.npy",
                "data/faceverse_resnet50.pth",
                self.device
            )
            self.logger.info("FaceVerse model loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load FaceVerse model: {e}")
            raise
    
    def run(self):
        """Run the main reconstruction pipeline."""
        self.logger.info("Starting FaceVerse reconstruction pipeline")
        self.start_time = time.time()
        
        try:
            # Start frame loader
            self.frame_loader.start()
            
            # Main processing loop
            self._process_frames()
            
        except KeyboardInterrupt:
            self.logger.info("Pipeline interrupted by user")
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
        finally:
            # Cleanup
            self._cleanup()
            
            # Log performance summary
            if self.config.enable_profiling:
                self._log_performance_summary()
    
    def _process_frames(self):
        """Process frames from the loader."""
        batch_frames = []
        batch_frame_info = []
        batch_detections = []
        
        while True:
            # Check if we should stop
            if self.frame_loader.stop_event.is_set():
                break
            
            # Get next frame
            frame, frame_info = self.frame_loader.get_next_frame()
            
            if frame is None:
                if self.frame_loader.done:
                    break
                time.sleep(0.01)
                continue
            
            # Process frame
            self.profiler.start_timer("frame_processing")
            
            if frame_info['detection'] is not None:
                # Apply smoothing if enabled
                if self.config.enable_smoothing:
                    processed_frame, processed_name, detection_data = \
                        self.frame_processor.add_frame_for_smoothing(
                            frame, frame_info['name'], frame_info['detection']
                        )
                else:
                    processed_frame = frame
                    processed_name = frame_info['name']
                    detection_data = np.stack([
                        frame_info['detection'].bounding_box,
                        frame_info['detection'].eye_parameters
                    ])
                
                # Add to batch
                batch_frames.append(processed_frame)
                batch_frame_info.append({
                    'name': processed_name,
                    'index': frame_info['index']
                })
                batch_detections.append(detection_data)
                
                # Process batch if full
                if len(batch_frames) >= self.config.batch_size:
                    self._process_batch(batch_frames, batch_frame_info, batch_detections)
                    batch_frames.clear()
                    batch_frame_info.clear()
                    batch_detections.clear()
            else:
                self.logger.warning(f"No face detected in {frame_info['name']}")
                self._handle_no_face_frame(frame, frame_info)
            
            self.profiler.stop_timer("frame_processing")
            self.frame_count += 1
            
            # Log progress periodically
            if self.frame_count % 100 == 0:
                elapsed_time = time.time() - self.start_time
                fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                self.logger.performance(f"Processed {self.frame_count} frames, FPS: {fps:.2f}")
        
        # Process remaining frames in batch
        if batch_frames:
            self._process_batch(batch_frames, batch_frame_info, batch_detections, end=True)
    
    def _process_batch(self, frames: List[np.ndarray], frame_info: List[Dict], 
                      detections: List[np.ndarray], end: bool = False):
        """Process a batch of frames."""
        self.profiler.start_timer("batch_processing")
        
        try:
            # Prepare data for FaceVerse
            boxes = np.stack([det[0] for det in detections])  # Bounding boxes
            eye_params = np.stack([det[1] for det in detections])  # Eye parameters
            frame_batch = np.stack(frames)
            
            # Run FaceVerse reconstruction
            self.profiler.start_timer("faceverse_reconstruction")
            coeffs, bbox_list = self.faceverse_model.process_imgs(frame_batch, boxes[:, np.newaxis, :])
            
            # Use MediaPipe eye parameters
            coeffs[:, -4:] = torch.from_numpy(eye_params).to(coeffs.device)
            self.profiler.stop_timer("faceverse_reconstruction")
            
            # Generate outputs for each frame in batch
            for i, info in enumerate(frame_info):
                self._generate_frame_outputs(
                    coeffs[i:i+1], bbox_list[i:i+1], frames[i], info['name'], end
                )
                
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
        
        self.profiler.stop_timer("batch_processing")
    
    def _generate_frame_outputs(self, coeffs: torch.Tensor, bbox: np.ndarray, 
                               frame: np.ndarray, frame_name: str, end: bool = False):
        """Generate outputs for a single frame."""
        # Apply smoothing in post-processing if enabled
        if self.config.enable_smoothing and not end:
            coeffs_processed, bbox_processed, frame_processed = \
                self._apply_temporal_smoothing(coeffs, bbox, frame, frame_name)
        else:
            coeffs_processed, bbox_processed, frame_processed = coeffs, bbox, frame
        
        if coeffs_processed is not None:
            # Generate 3D reconstruction
            vertices, vertices_proj, normals, colors = self.faceverse_model.from_coeffs(
                coeffs_processed, bbox_processed
            )
            
            # Render if visualization enabled and renderer available
            if self.config.enable_visualization and render_fvr is not None:
                self._render_and_save_outputs(
                    vertices, vertices_proj, normals, colors, 
                    frame_processed, bbox_processed, frame_name
                )
            
            # Save parameters
            head_params = {
                'coeffs': coeffs_processed.cpu().numpy(),
                'bbox_list': bbox_processed
            }
            self.output_manager.save_parameters(head_params, frame_name)
            
            # Save PLY model if requested
            if self.config.save_ply:
                self.output_manager.save_ply_model(
                    vertices[0].cpu().numpy(),
                    (colors[0] * 255).astype(np.uint8),
                    self.faceverse_model.fvd["tri"],
                    frame_name
                )
    
    def _apply_temporal_smoothing(self, coeffs: torch.Tensor, bbox: np.ndarray,
                                frame: np.ndarray, frame_name: str) -> Tuple:
        """Apply temporal smoothing to reconstruction parameters."""
        # Store current frame in cache
        current_param = {'coeffs': coeffs, 'bbox_list': bbox}
        current_frame = {'frame': frame, 'name': frame_name}
        
        self.cache_param.append(current_param)
        self.cache_frame.append(current_frame)
        
        # Maintain cache size
        if len(self.cache_param) > self.config.smoothing_window_size:
            self.cache_param.pop(0)
            self.cache_frame.pop(0)
        
        # Apply smoothing if we have enough frames
        if len(self.cache_param) == self.config.smoothing_window_size:
            middle_idx = self.config.smoothing_window_size // 2
            
            # Simple moving average for coefficients
            smoothed_coeffs = torch.mean(
                torch.cat([p['coeffs'] for p in self.cache_param]), 
                dim=0, keepdim=True
            )
            
            return (smoothed_coeffs, 
                   self.cache_param[middle_idx]['bbox_list'],
                   self.cache_frame[middle_idx]['frame'])
        else:
            # Not enough frames for smoothing
            return coeffs, bbox, frame
    
    def _render_and_save_outputs(self, vertices: torch.Tensor, vertices_proj: torch.Tensor,
                               normals: torch.Tensor, colors: torch.Tensor,
                               frame: np.ndarray, bbox: np.ndarray, frame_name: str):
        """Render and save visualization outputs."""
        try:
            # Render 3D face
            rgb, depth = render_fvr(
                frame, vertices_proj[0], 
                self.faceverse_model.fvd["tri"], 
                normals[0], colors[0]
            )
            
            # Create comparison image
            original_bgr = frame[:, :, ::-1].copy()  # Convert back to BGR for OpenCV
            rendered_bgr = rgb[:, :, ::-1].copy()
            
            # Resize if necessary for concatenation
            if original_bgr.shape != rendered_bgr.shape:
                rendered_bgr = cv2.resize(rendered_bgr, 
                                        (original_bgr.shape[1], original_bgr.shape[0]))
            
            comparison = np.concatenate([original_bgr, rendered_bgr], axis=0)
            
            # Draw bounding box
            bbox_int = bbox[0].astype(np.int32)
            cv2.rectangle(comparison, 
                         (bbox_int[0], bbox_int[1]), 
                         (bbox_int[2], bbox_int[3]), 
                         (0, 255, 0), 2)
            
            # Save result
            self.output_manager.save_image_result(comparison, frame_name)
            
            # Show in window for webcam mode
            if self.frame_loader.mode == ProcessingMode.WEBCAM:
                cv2.imshow('FaceVerse Reconstruction', comparison)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.frame_loader.stop()
                    
        except Exception as e:
            self.logger.error(f"Error rendering frame {frame_name}: {e}")
    
    def _handle_no_face_frame(self, frame: np.ndarray, frame_info: Dict):
        """Handle frames where no face is detected."""
        if self.frame_loader.mode == ProcessingMode.WEBCAM:
            # Display original frame for webcam
            display_frame = np.concatenate([frame[:, :, ::-1], 
                                          np.zeros_like(frame)], axis=0)
            cv2.imshow('FaceVerse Reconstruction', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.frame_loader.stop()
    
    def _log_performance_summary(self):
        """Log performance summary at the end of processing."""
        summary = self.profiler.get_summary()
        total_time = time.time() - self.start_time
        
        self.logger.performance("=" * 50)
        self.logger.performance("PERFORMANCE SUMMARY")
        self.logger.performance("=" * 50)
        self.logger.performance(f"Total frames processed: {self.frame_count}")
        self.logger.performance(f"Total time: {total_time:.2f} seconds")
        self.logger.performance(f"Average FPS: {self.frame_count / total_time:.2f}")
        
        for operation, metrics in summary.items():
            self.logger.performance(
                f"{operation}: {metrics['average']:.4f}s avg "
                f"({metrics['min']:.4f}s min, {metrics['max']:.4f}s max) "
                f"[{metrics['count']} calls]"
            )
    
    def _cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up resources...")
        self.frame_loader.stop()
        
        if self.frame_loader.is_alive():
            self.frame_loader.join(timeout=5.0)
        
        cv2.destroyAllWindows()
        self.logger.info("Cleanup completed")


def main():
    """Main entry point for the FaceVerse pipeline."""
    parser = argparse.ArgumentParser(
        description="FaceVerse 3D Face Reconstruction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output arguments
    parser.add_argument("--input", type=str, default="example/input/test.mp4",
                       help="Input source: video file, image file, image folder, or 'webcam'")
    parser.add_argument("--output", type=str, default="example/output",
                       help="Output directory for results")
    
    # Processing parameters
    parser.add_argument("--batch", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--smooth", action="store_true", default=True,
                       help="Enable temporal smoothing")
    parser.add_argument("--no-smooth", action="store_false", dest="smooth",
                       help="Disable temporal smoothing")
    
    # Output options
    parser.add_argument("--save-results", action="store_true", default=True,
                       help="Save processed images and parameters")
    parser.add_argument("--no-save-results", action="store_false", dest="save_results",
                       help="Don't save results")
    parser.add_argument("--save-ply", action="store_true", default=False,
                       help="Save 3D models as PLY files")
    parser.add_argument("--visual", action="store_true", default=True,
                       help="Enable visualization")
    parser.add_argument("--no-visual", action="store_false", dest="visual",
                       help="Disable visualization")
    
    # Performance options
    parser.add_argument("--profile", action="store_true", default=False,
                       help="Enable performance profiling")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigurationManager(args.config)
    config_dict = config_manager.load_config()
    
    # Create processing config
    processing_config = ProcessingConfig(
        batch_size=args.batch,
        enable_smoothing=args.smooth,
        save_results=args.save_results,
        save_ply=args.save_ply,
        enable_visualization=args.visual,
        enable_profiling=args.profile
    )
    
    # Update from loaded config if available
    if 'processing' in config_dict:
        processing_config.batch_size = config_dict['processing'].get('batch_size', processing_config.batch_size)
        processing_config.enable_smoothing = config_dict['processing'].get('enable_smoothing', processing_config.enable_smoothing)
        processing_config.smoothing_window_size = config_dict['processing'].get('smoothing_window_size', processing_config.smoothing_window_size)
        processing_config.max_queue_size = config_dict['processing'].get('max_queue_size', processing_config.max_queue_size)
    
    if 'output' in config_dict:
        processing_config.save_results = config_dict['output'].get('save_results', processing_config.save_results)
        processing_config.save_ply = config_dict['output'].get('save_ply', processing_config.save_ply)
        processing_config.output_quality = config_dict['output'].get('output_quality', processing_config.output_quality)
    
    if 'performance' in config_dict:
        processing_config.enable_profiling = config_dict['performance'].get('enable_profiling', processing_config.enable_profiling)
    
    # Create and run pipeline
    try:
        pipeline = FaceVersePipeline(args, processing_config)
        pipeline.run()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"Pipeline error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()