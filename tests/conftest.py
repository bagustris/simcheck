import os
import sys
import types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DIM = 64

class FakeLlama:
    def __init__(self, *args, **kwargs):
        pass

    def embed(self, text: str):
        seed = sum(ord(c) for c in text) % (2**31)
        return np.random.default_rng(seed).random(DIM).astype(np.float32).tolist()

fake_llama_cpp = types.ModuleType("llama_cpp")
fake_llama_cpp.Llama = FakeLlama
sys.modules["llama_cpp"] = fake_llama_cpp
