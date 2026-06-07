# main.py — the "brain", running inside your browser via PyScript (Pyodide).
# This is the SAME NumPy math as train.py, just doing a forward pass instead
# of learning. No server, no API calls — the neural network runs on YOUR machine.

import json
import numpy as np
from pyscript import window
from pyscript.ffi import create_proxy

# 1) Load the weights we trained offline (copied into the browser's virtual
#    filesystem by pyscript.toml).
with open("weights.json") as f:
    W = json.load(f)

W1 = np.array(W["W1"], dtype=np.float64)   # (784, 64)
b1 = np.array(W["b1"], dtype=np.float64)   # (64,)
W2 = np.array(W["W2"], dtype=np.float64)   # (64, 10)
b2 = np.array(W["b2"], dtype=np.float64)   # (10,)
ACC = float(W.get("test_accuracy", 0.0))


def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def center_of_mass(img):
    """Shift the 28x28 image so the digit's center of mass sits dead center —
    exactly the trick the original MNIST dataset uses. It makes the model far
    more forgiving about WHERE on the canvas you draw."""
    total = img.sum()
    if total <= 0:
        return img
    ys, xs = np.indices(img.shape)
    cy = (ys * img).sum() / total
    cx = (xs * img).sum() / total
    shift_y = int(round(13.5 - cy))
    shift_x = int(round(13.5 - cx))
    return np.roll(np.roll(img, shift_y, axis=0), shift_x, axis=1)


def predict(pixels):
    """Called from JavaScript with 784 grayscale values (0-255, white on black).
    Returns a JSON string the page can render."""
    # JS arrays arrive as a proxy; turn it into a real Python list.
    try:
        pixels = pixels.to_py()
    except AttributeError:
        pass

    img = np.array(pixels, dtype=np.float64).reshape(28, 28)
    img = center_of_mass(img)

    x = (img / 255.0).reshape(784)        # flatten + scale to 0..1
    h = relu(x @ W1 + b1)                  # hidden layer
    logits = h @ W2 + b2                   # output scores
    probs = softmax(logits)                # turn scores into probabilities

    return json.dumps({
        "pred": int(np.argmax(probs)),
        "probs": [float(p) for p in probs],
        "seen": [int(v) for v in img.reshape(784)],  # the centered image
    })


# 2) Hand the predict() function to JavaScript and announce we're ready.
window.pyPredict = create_proxy(predict)
try:
    window.onPyReady(ACC)
except Exception as err:  # pragma: no cover
    print("onPyReady not found:", err)
