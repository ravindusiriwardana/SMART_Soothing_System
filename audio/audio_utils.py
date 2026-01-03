import numpy as np
from collections import deque

class AudioBuffer:
    def __init__(self, max_len):
        # Initialize a thread-safe deque with a fixed maximum length
        self.buffer = deque(maxlen=int(max_len))

    def callback(self, indata, frames, time_info, status):
        """
        Callback function for sounddevice.
        This runs in a separate thread whenever new audio data is available.
        """
        if status:
            print(f"⚠️ Audio Status: {status}")
        
        # indata is shape (frames, channels), we only need channel 0
        self.buffer.extend(indata[:, 0])

    def get_audio_segment(self):
        """
        Returns the current buffer as a numpy array.
        """
        return np.array(self.buffer)