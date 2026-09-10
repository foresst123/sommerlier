import numpy as np

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculate the cosine similarity between two vectors."""
    if vec_a is None or vec_b is None:
        return -1.0
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def rms_energy(waveform: np.ndarray) -> float:
    """Calculate the Root Mean Square energy of a waveform."""
    if len(waveform) == 0:
        return 0.0
    return float(np.sqrt(np.mean(waveform**2)))
