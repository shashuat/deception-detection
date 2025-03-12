import torch
import numpy as np
import cv2
import os
from facenet_pytorch import MTCNN
from skimage.transform import SimilarityTransform

class FaceAlignmentTools:
    """
    Modern implementation of face detection and alignment using PyTorch and MTCNN.
    """

    def __init__(
        self,
        min_face_size: int = 40,
        thresholds: list = None,
        device: str = None,
        alignment_style: str = "mtcnn",
    ):
        """
        Initialize the FaceAlignmentTools with MTCNN detector.
        
        Args:
            min_face_size: Minimum face size to detect
            thresholds: Thresholds for the MTCNN detector [p_net, r_net, o_net]
            device: Device to run the model on ('cpu' or 'cuda')
            alignment_style: Style of alignment ("mtcnn", "ms1m", or "lfw")
        """
        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]
            
        # Use CUDA if available, otherwise use CPU
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Initialize MTCNN with more sensitive detection parameters
        self.detector = MTCNN(
            image_size=224,
            margin=0,
            min_face_size=min_face_size,
            thresholds=[0.5, 0.6, 0.6],  # Lower thresholds for better detection
            factor=0.8,                  # Smaller scaling factor for finer scales
            post_process=False,
            keep_all=True,
            device=self.device,
            select_largest=False
        )
        
        # Predefined relative target landmarks for alignment
        self._landmarks = {
            "mtcnn": np.array(
                [
                    [38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.7299, 92.2041],
                ],
                dtype=np.float32,
            )
            / 112,
            "ms1m": np.array(
                [
                    [38.128662, 51.516567],
                    [74.21549, 51.55989],
                    [56.056564, 72.434525],
                    [40.48149, 90.873665],
                    [71.38436, 90.78255],
                ],
                dtype=np.float32,
            )
            / 112,
            "lfw": np.array(
                [
                    [38.411846, 52.59001],
                    [73.68209, 52.300644],
                    [56.092415, 72.949585],
                    [40.763634, 90.94648],
                    [71.64599, 90.62956],
                ],
                dtype=np.float32,
            )
            / 112,
        }
        
        self._alignment_style = alignment_style
    
    def detect_face(self, img, allow_multiface=False):
        """
        Detect faces in an image.
        
        Args:
            img: RGB image as numpy array
            allow_multiface: If True, return all faces, otherwise only the largest face
            
        Returns:
            List of facial landmarks or None if no face is detected
        """
        # Make sure image is RGB and uint8
        assert img.dtype == np.uint8, "Image must be uint8"
        if img.shape[2] != 3:
            raise ValueError("Image must be RGB (3 channels)")
            
        # Detect faces
        try:
            boxes, probs, landmarks = self.detector.detect(img, landmarks=True)
        except Exception as e:
            print(f"Error during detection: {e}")
            return None
            
        # If no faces detected
        if boxes is None or len(boxes) == 0:
            return None
            
        # Convert landmarks to format expected by align_face
        processed_landmarks = []
        for i in range(len(landmarks)):
            # Each landmark is a 5x2 array with x,y coordinates
            landmark = landmarks[i]
            # Reshape to match the expected format
            processed_landmark = landmark.reshape(5, 2).astype(np.float32)
            processed_landmarks.append(processed_landmark)
            
        # If only one face needed, get the largest face
        if not allow_multiface and len(processed_landmarks) > 1:
            # Use the face with highest confidence
            best_idx = np.argmax(probs)
            return np.array([processed_landmarks[best_idx]])
            
        return np.array(processed_landmarks)
    
    def align_face(self, img, src, dst, dsize=(224, 224)):
        """
        Align face using landmarks.
        
        Args:
            img: Input image
            src: Source landmarks
            dst: Target landmarks
            dsize: Output image size
            
        Returns:
            Aligned face image
        """
        tform = SimilarityTransform()
        tform.estimate(src, dst * dsize)
        
        tmatrix = tform.params[0:2, :]
        
        return cv2.warpAffine(img, tmatrix, dsize)
    
    def align(self, img, dsize=(224, 224), allow_multiface=False, central_face=False):
        """
        Detect and align faces in an image.
        
        Args:
            img: Input RGB image
            dsize: Output image size
            allow_multiface: If True, return all aligned faces
            central_face: If True and multiple faces detected, return the most central face
            
        Returns:
            Aligned face image(s) or None if no face detected
        """
        dst_points = self._landmarks[self._alignment_style]
        
        # Detect face landmarks
        src_points = self.detect_face(img, allow_multiface=allow_multiface or central_face)
        
        if src_points is None:
            return None
            
        faces = []
        for points in src_points:
            aligned = self.align_face(img, points, dst_points, dsize)
            faces.append(aligned)
            
        if allow_multiface:
            return faces
        elif central_face and len(faces) > 1:
            return faces[self._determine_center_face_idx(src_points, img.shape[:2])]
        else:
            return faces[0]
    
    def _determine_center_face_idx(self, landmarks, im_size):
        """
        Find the most central face in the image.
        
        Args:
            landmarks: Facial landmarks
            im_size: Image size (height, width)
            
        Returns:
            Index of the most central face
        """
        # Calculate distance of nose tip to image center
        image_center = (im_size[1] / 2, im_size[0] / 2)  # (x, y)
        
        # Use the nose landmark (index 2) to determine centrality
        noses = np.array([landmark[2] for landmark in landmarks])
        distances = np.sum(np.abs(noses - image_center), axis=1)
        
        # Return index of the most central face
        return np.argmin(distances)