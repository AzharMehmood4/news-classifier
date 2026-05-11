import os
import re
import html
import joblib
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding

# ==============================
# KERA ERROR
# ==============================

original_from_config = Embedding.from_config

def custom_from_config(config):
    # Remove unsupported argument
    config.pop("quantization_config", None)
    return original_from_config(config)

Embedding.from_config = custom_from_config

# ==============================
# AG News Mapping
# ==============================

LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}


class NewsClassifier:
    def __init__(
        self,
        ml_model_name='svm_model.pkl',
        dl_model_name='nn_model.h5',
        tokenizer_name='tokenizer.pkl'
    ):
        """
        Initializes the classifier by loading the saved models.
        """

        # Base models directory
        base_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'models'
        )

        # ==============================
        # LOAD ML MODEL
        # ==============================

        ml_path = os.path.join(base_path, ml_model_name)

        if os.path.exists(ml_path):
            self.ml_pipeline = joblib.load(ml_path)
            print(f" Loaded ML Model: {ml_model_name}")
        else:
            self.ml_pipeline = None
            print(f" ML Model not found at {ml_path}")

        # ==============================
        # LOAD DL MODEL
        # ==============================

        dl_path = os.path.join(base_path, dl_model_name)

        if os.path.exists(dl_path):
            try:
                self.dl_model = tf.keras.models.load_model(
                    dl_path,
                    compile=False
                )

                print(f" Loaded DL Model: {dl_model_name}")

            except Exception as e:
                self.dl_model = None
                print(f" Error loading DL model: {e}")

        else:
            self.dl_model = None
            print(f" DL Model not found at {dl_path}")

        # ==============================
        # LOAD TOKENIZER
        # ==============================

        tok_path = os.path.join(base_path, tokenizer_name)

        if os.path.exists(tok_path):
            with open(tok_path, 'rb') as f:
                self.tokenizer = pickle.load(f)

            print(f" Loaded Tokenizer: {tokenizer_name}")

        else:
            self.tokenizer = None
            print(f" Tokenizer not found at {tok_path}")

    # ==============================
    # CLEAN TEXT
    # ==============================

    def _clean_text(self, text):

        text = html.unescape(text)
        text = text.lower()

        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # ==============================
    # PREDICTION FUNCTION
    # ==============================

    def predict(self, text, model_type='ml'):

        if not text:
            return "No text provided"

        cleaned_text = self._clean_text(text)

        # ==============================
        # ML PREDICTION
        # ==============================

        if model_type == 'ml':

            if self.ml_pipeline is None:
                return "ML model not loaded"

            prediction = self.ml_pipeline.predict(
                [cleaned_text]
            )[0]

            return LABEL_MAP[prediction]

        # ==============================
        # DL PREDICTION
        # ==============================

        elif model_type == 'dl':

            if self.dl_model is None:
                return "DL model not loaded"

            if self.tokenizer is None:
                return "Tokenizer not loaded"

            seq = self.tokenizer.texts_to_sequences(
                [cleaned_text]
            )

            padded = pad_sequences(
                seq,
                maxlen=70,
                padding='post'
            )

            preds = self.dl_model.predict(
                padded,
                verbose=0
            )

            idx = np.argmax(preds[0])

            confidence = float(np.max(preds[0]))

            return f"{LABEL_MAP[idx]} ({confidence*100:.2f}%)"

        else:
            return "Invalid model type"


# ==============================
# QUICK TEST
# ==============================

if __name__ == "__main__":

    print("\n--- Testing Inference Layer ---")

    classifier = NewsClassifier()

    sample_news = (
        "The tech giant announced a new smartphone "
        "with satellite connectivity today."
    )

    print(f"\nInput: {sample_news}")

    print(
        f"ML Prediction: "
        f"{classifier.predict(sample_news, 'ml')}"
    )

    print(
        f"DL Prediction: "
        f"{classifier.predict(sample_news, 'dl')}"
    )