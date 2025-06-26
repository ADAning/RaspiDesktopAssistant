"""
For the Raspberry Pi desktop assistant's visual signal acquisition, capturing video input from the Raspberry Pi camera (CSI camera or USB camera).
"""

import cv2
import base64
import numpy as np
import logging
from config_loader import config

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self):
        camera_indexes = self.find_available_cameras()
        if len(camera_indexes) == 0:
            raise RuntimeError(
                "No cameras found. Please connect a camera and try again."
            )
        else:
            logger.info(f"Found {len(camera_indexes)} cameras: {camera_indexes}")
        self.capture = cv2.VideoCapture(camera_indexes[0])
        logger.info(f"Camera initialized with index: {camera_indexes[0]}")

    @staticmethod
    def find_available_cameras(max_tested=5):
        available_cameras = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras

    def capture_frame(self):
        """Capture a single frame from the camera."""
        ret, frame = self.capture.read()
        logger.info(f"Captured frame: {ret}")
        if not ret:
            raise RuntimeError("Failed to capture frame from camera.")
        return frame

    def capture_frame_base64(self):
        """Capture a single frame and return it as a base64 encoded string."""
        frame = self.capture_frame()
        _, buffer = cv2.imencode(".jpg", frame)
        # Convert numpy array to bytes
        bytes_data = buffer.tobytes()
        return base64.b64encode(bytes_data).decode("utf-8")

    def get_camera_info(self):
        """Get current camera parameters and status"""
        info = {
            "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.capture.get(cv2.CAP_PROP_FPS),
            "brightness": self.capture.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": self.capture.get(cv2.CAP_PROP_CONTRAST),
            "saturation": self.capture.get(cv2.CAP_PROP_SATURATION),
            "is_opened": self.capture.isOpened(),
        }
        return info
