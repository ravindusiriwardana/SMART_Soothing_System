import numpy as np
import librosa
from tensorflow.keras.models import load_model
from config import SAMPLE_RATE, N_MFCC, MAX_LEN

class CryClassifier:
    def __init__(self, model_path, categories):
        self.categories = categories
        try:
            self.model = load_model(model_path)
            print("✅ Cry model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load Cry Model: {e}")
            self.model = None

    def _extract_features(self, audio):
        """Internal helper method to extract MFCC features."""
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
            if mfcc.shape[0] < MAX_LEN:
                pad_width = MAX_LEN - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
            else:
                mfcc = mfcc[:MAX_LEN, :]
            return np.expand_dims(mfcc, axis=0)
        except Exception as e:
            print(f"⚠️ Feature extraction error: {e}")
            return None

    # def _extract_features(self, audio):
    #     """Internal helper method to extract MFCC features for CNN."""
    #     try:
    #         # 1. Extract MFCC
    #         mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
            
    #         # 2. Pad/Truncate to MAX_LEN (157)
    #         if mfcc.shape[1] < MAX_LEN:
    #             pad_width = MAX_LEN - mfcc.shape[1]
    #             mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    #         else:
    #             mfcc = mfcc[:, :MAX_LEN]
            
    #         # 3. Reshape for CNN: (1, N_MFCC, MAX_LEN, 1)
    #         # This '1' at the end is the "channel" dimension required by Conv2D
    #         return mfcc.reshape(1, N_MFCC, MAX_LEN, 1)
            
    #     except Exception as e:
    #         print(f"⚠️ Feature extraction error: {e}")
    #         return None

    def predict(self, audio):
        """Returns (emotion, confidence) or (None, 0.0) on failure."""
        if not self.model:
            return None, 0.0

        features = self._extract_features(audio)
        if features is None:
            return None, 0.0

        try:
            pred = self.model.predict(features, verbose=0)
            idx = np.argmax(pred)
            return self.categories[idx], float(pred[0][idx])
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return None, 0.0