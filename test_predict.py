"""Sanity check: reload the EXPORTED weights.json and reproduce the exact
forward pass model.py runs in the browser, to make sure JSON round-trip +
center-of-mass logic still classifies MNIST correctly."""
import gzip, json, os, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".mnist_cache")


def read_images(p):
    with gzip.open(p, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(n*r*c), np.uint8).astype(np.float64).reshape(n, r*c)


def read_labels(p):
    with gzip.open(p, "rb") as f:
        struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), np.uint8)


X = read_images(os.path.join(CACHE, "t10k-images-idx3-ubyte.gz"))
y = read_labels(os.path.join(CACHE, "t10k-labels-idx1-ubyte.gz"))

W = json.load(open(os.path.join(HERE, "docs", "weights.json")))
W1 = np.array(W["W1"]); b1 = np.array(W["b1"])
W2 = np.array(W["W2"]); b2 = np.array(W["b2"])

relu = lambda z: np.maximum(0, z)
def softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e/e.sum(axis=1, keepdims=True)

def fwd(Xb):
    return softmax(relu(Xb/255.0 @ W1 + b1) @ W2 + b2)

acc = (fwd(X).argmax(1) == y).mean()
print(f"Reloaded weights.json -> test accuracy: {acc*100:.2f}%")
print(f"Declared accuracy in file: {W['test_accuracy']*100:.2f}%")

# Check the bundled example digits classify correctly
ex = json.load(open(os.path.join(HERE, "docs", "examples.json")))["digits"]
Xe = np.array([d["pixels"] for d in ex], dtype=np.float64)
ye = np.array([d["label"] for d in ex])
pe = fwd(Xe).argmax(1)
ok = int((pe == ye).sum())
print(f"Example digits: {ok}/{len(ye)} correct -> {list(zip(ye.tolist(), pe.tolist()))}")
