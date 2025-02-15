import numpy as np


class recording():
    def __init__(self):
        self.data = np.empty(dtype=np.int16)
        self.window_bottom = 0
        self.window_top = 0
        self.validated_words = []
        self.words_in_question = []

    def addChunk(self, chunk):
        self.data = np.append(self.data, chunk)

    def transcribe(self):
        pass

pvecording.py
    def validate_words(self):
        pass

