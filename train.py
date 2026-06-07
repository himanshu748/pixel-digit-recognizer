"""
train.py - Train a tiny neural network on MNIST using only NumPy.

This is the "offline" half of the project. It:
  1. Downloads the MNIST handwritten-digit dataset.
  2. Trains a 2-layer fully-connected network (784 -> 64 -> 10) from scratch.
  3. Saves the learned weights to web/weights.json so the SAME math can run
     in the browser via PyScript.
  4. Saves a handful of real test digits to web/examples.json for the
     "Try an example" button in the demo.

No TensorFlow, no PyTorch, no scikit-learn. Just NumPy + a little math.
"""

import gzip
import json
import os
import struct
import urllib.request

import numpy as np

np.random.seed(7)

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "docs")
CACHE = os.path.join(HERE, ".mnist_cache")
os.makedirs(WEB, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download(name):
    path = os.path.join(CACHE, FILES[name])
    if not os.path.exists(path):
        url = BASE + FILES[name]
        print(f"  downloading {FILES[name]} ...")
        urllib.request.urlretrieve(url, path)
    return path


def read_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic}"
        buf = f.read(num * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        return data.reshape(num, rows * cols)


def read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic {magic}"
        buf = f.read(num)
        return np.frombuffer(buf, dtype=np.uint8)


print("Loading MNIST...")
X_train = read_images(download("train_images")) / 255.0
y_train = read_labels(download("train_labels"))
X_test = read_images(download("test_images")) / 255.0
y_test = read_labels(download("test_labels"))
print(f"  train: {X_train.shape}, test: {X_test.shape}")


def one_hot(y, k=10):
    out = np.zeros((y.size, k), dtype=np.float32)
    out[np.arange(y.size), y] = 1.0
    return out


# Hold out the last 5,000 training images as a validation set so we can pick
# the best model honestly, without ever peeking at the test set.
X_val, y_val = X_train[55000:], y_train[55000:]
X_train, y_train = X_train[:55000], y_train[:55000]
Y_train = one_hot(y_train)
print(f"  using {X_train.shape[0]} for training, {X_val.shape[0]} for validation")

# ---- Network: 784 -> H (ReLU) -> 10 (softmax) ----
H = 64
INPUT = 784
OUTPUT = 10

# He initialization keeps signal variance stable through ReLU layers.
W1 = np.random.randn(INPUT, H).astype(np.float32) * np.sqrt(2.0 / INPUT)
b1 = np.zeros(H, dtype=np.float32)
W2 = np.random.randn(H, OUTPUT).astype(np.float32) * np.sqrt(2.0 / H)
b2 = np.zeros(OUTPUT, dtype=np.float32)


def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    return z1, a1, softmax(z2)


def accuracy(X, y):
    _, _, p = forward(X)
    return float((p.argmax(axis=1) == y).mean())


# ---- Training: mini-batch SGD with momentum + learning-rate decay ----
EPOCHS = 25
BATCH = 128
LR = 0.20
DECAY = 0.92          # shrink the learning rate a little each epoch
MOMENTUM = 0.9

vW1 = np.zeros_like(W1)
vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2)
vb2 = np.zeros_like(b2)

best = None
best_val = -1.0
n = X_train.shape[0]
print(f"Training {INPUT}->{H}->{OUTPUT} for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    lr = LR * (DECAY ** epoch)
    order = np.random.permutation(n)
    Xs, Ys = X_train[order], Y_train[order]
    for i in range(0, n, BATCH):
        xb = Xs[i:i + BATCH]
        yb = Ys[i:i + BATCH]
        m = xb.shape[0]

        z1, a1, p = forward(xb)

        # Cross-entropy gradient (softmax + CE simplifies to p - y)
        dz2 = (p - yb) / m
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = xb.T @ dz1
        db1 = dz1.sum(axis=0)

        vW2 = MOMENTUM * vW2 - lr * dW2
        vb2 = MOMENTUM * vb2 - lr * db2
        vW1 = MOMENTUM * vW1 - lr * dW1
        vb1 = MOMENTUM * vb1 - lr * db1
        W2 += vW2
        b2 += vb2
        W1 += vW1
        b1 += vb1

    val_acc = accuracy(X_val, y_val)
    flag = ""
    if val_acc > best_val:
        best_val = val_acc
        best = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        flag = "  <- best so far"
    print(f"  epoch {epoch + 1:2d}/{EPOCHS}  val acc = {val_acc * 100:.2f}%{flag}")

# Restore the best-on-validation weights before reporting + exporting.
W1, b1, W2, b2 = best
final_acc = accuracy(X_test, y_test)
print(f"Best validation accuracy: {best_val * 100:.2f}%")
print(f"Final TEST accuracy (held-out): {final_acc * 100:.2f}%")


# ---- Export weights for the browser ----
def r(arr, decimals=5):
    return np.round(arr, decimals).tolist()


weights = {
    "arch": [INPUT, H, OUTPUT],
    "test_accuracy": round(final_acc, 4),
    "W1": r(W1), "b1": r(b1),
    "W2": r(W2), "b2": r(b2),
}
with open(os.path.join(WEB, "weights.json"), "w") as f:
    json.dump(weights, f)
size_kb = os.path.getsize(os.path.join(WEB, "weights.json")) / 1024
print(f"Saved web/weights.json ({size_kb:.0f} KB)")

# ---- Export real test digits for the "Try an example" feature ----
# Curate showcase digits: for each class pick the test image the model gets
# RIGHT with the highest confidence, so clicking "Try an example" always looks
# crisp. Then add a few more high-confidence correct ones for variety.
_, _, P_all = forward(X_test)
preds_all = P_all.argmax(axis=1)
conf_all = P_all.max(axis=1)

picks = []
seen = set()
for digit in range(10):
    cand = np.where((y_test == digit) & (preds_all == digit))[0]
    best_idx = int(cand[np.argmax(conf_all[cand])])
    picks.append(best_idx)
    seen.add(best_idx)

# extras: highest-confidence correct predictions overall, skipping dupes
order_conf = np.argsort(-conf_all)
for c in order_conf:
    c = int(c)
    if len(picks) >= 16:
        break
    if c not in seen and preds_all[c] == y_test[c]:
        seen.add(c)
        picks.append(c)

examples = []
for idx in picks:
    pixels = (X_test[idx] * 255).round().astype(int).tolist()  # 0-255, length 784
    examples.append({"label": int(y_test[idx]), "pixels": pixels})
with open(os.path.join(WEB, "examples.json"), "w") as f:
    json.dump({"digits": examples}, f)
print(f"Saved web/examples.json ({len(examples)} digits)")
print("Done.")
