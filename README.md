<!-- HEADER_IMAGE -->

# 🔢 Pixel Digit Recognizer

> Draw a number, and a neural network guesses it — running **100% in your browser** with real **Python (NumPy)** via **Pyodide**. No server, no API calls, no install.

**🎮 [Live demo →](https://himanshu748.github.io/pixel-digit-recognizer/)**

![Python](https://img.shields.io/badge/Python-NumPy_only-3776AB?logo=python&logoColor=white)
![Pyodide](https://img.shields.io/badge/Pyodide-runs_in_the_browser-a78bfa)
![No backend](https://img.shields.io/badge/backend-none-success)
![Accuracy](https://img.shields.io/badge/test_accuracy-97.8%25-ffd166)

---

## What is this?

A handwritten-digit recognizer (the classic MNIST task) with a twist: there's **no backend**. The neural network is trained offline in pure NumPy, and then the *exact same NumPy code* runs **inside your browser** thanks to [Pyodide](https://pyodide.org/) (a full build of Python compiled to WebAssembly).

So the whole thing is just static files you can host for free on GitHub Pages — yet a real Python neural net is doing the recognition on your machine.

## How it works

```
You draw  →  JavaScript shrinks it to 28×28  →  Python (NumPy) runs the
on a canvas    (the way a camera makes pixels)    neural network & predicts
```

- **`train.py`** — downloads MNIST and trains a tiny `784 → 64 → 10` network from scratch in NumPy (mini-batch SGD with momentum + LR decay, ~98% validation / **97.8% test**). Exports the weights to `docs/weights.json`.
- **`docs/model.py`** — the browser brain. Loads those weights and does the forward pass (`ReLU` + `softmax`) plus a center-of-mass trick to match MNIST. Pyodide runs this file in the browser.
- **`docs/app.js`** — the browser hands: smooth canvas drawing, turning the drawing into a 28×28 image, and rendering the confidence bars + a live "what the network sees" preview.

The model is intentionally small (one hidden layer of 64 units) so the weights file stays tiny (~470 KB) and the page loads fast.

## Run it locally

```bash
pip install numpy
python train.py                 # regenerates docs/weights.json + docs/examples.json
python -m http.server -d docs 8000
# open http://localhost:8000
```

Everything in `docs/` is plain static files — drop them on any static host.

## Project structure

```
train.py            # train the network (NumPy only) and export weights
test_predict.py     # sanity-check the exported weights reproduce MNIST accuracy
docs/
  index.html        # the page (loads Pyodide + app.js)
  app.js            # drawing, UI, and booting Python (JavaScript)
  model.py          # the neural network, run in the browser by Pyodide (Python)
  weights.json      # trained weights
  examples.json     # a few real MNIST test digits for "Try an example"
  style.css         # pixel/retro theme
```

## Credits

Built by [Himanshu Kumar](https://github.com/himanshu748) for the [Codédex](https://www.codedex.io/) Monthly Challenge. MNIST dataset by Yann LeCun et al.
