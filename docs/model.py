# model.py — the neural network, run in your browser by Pyodide (WebAssembly Python).
# Pure NumPy: the SAME forward pass we trained in train.py. No server.

import json
import numpy as np

# Weights are injected from JavaScript via load_weights() once they're fetched.
_W = {}


def load_weights(json_str):
    global _W
    d = json.loads(json_str)
    _W = {
        "W1": np.array(d["W1"], dtype=np.float64),   # (784, 64)
        "b1": np.array(d["b1"], dtype=np.float64),   # (64,)
        "W2": np.array(d["W2"], dtype=np.float64),   # (64, 10)
        "b2": np.array(d["b2"], dtype=np.float64),   # (10,)
    }


def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def center_of_mass(img):
    """Shift the 28x28 image so the digit's center of mass sits dead center —
    the same trick MNIST uses, so it's forgiving about WHERE you draw."""
    total = img.sum()
    if total <= 0:
        return img
    ys, xs = np.indices(img.shape)
    cy = (ys * img).sum() / total
    cx = (xs * img).sum() / total
    return np.roll(np.roll(img, int(round(13.5 - cy)), axis=0),
                   int(round(13.5 - cx)), axis=1)


def predict(pixels):
    """784 grayscale values (0-255, white on black) -> JSON the page can render."""
    try:
        pixels = pixels.to_py()          # JS array -> Python list
    except AttributeError:
        pass

    img = np.array(pixels, dtype=np.float64).reshape(28, 28)
    img = center_of_mass(img)

    x = (img / 255.0).reshape(784)
    h = relu(x @ _W["W1"] + _W["b1"])
    logits = h @ _W["W2"] + _W["b2"]
    probs = softmax(logits)

    return json.dumps({
        "pred": int(np.argmax(probs)),
        "probs": [float(p) for p in probs],
        "seen": [int(v) for v in img.reshape(784)],
    })
