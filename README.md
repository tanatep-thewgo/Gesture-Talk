# GestureTalk — Thai Hand-Sign Recognition

My senior project: a real-time Thai finger-spelling recognizer that runs entirely in the browser.
Point your webcam at your hand and it predicts which letter you're signing, live, with no server
and no installation.

🔗 **Live demo:** https://tanatep-thewgo.github.io/Gesture-Talk/
*(needs a webcam; click "Start" and allow camera access)*

## How it works

The browser app ([`index.html`](index.html)) uses **MediaPipe Tasks Vision** (loaded from a CDN) to
track 21 hand landmarks per frame. Those landmark coordinates are normalized and fed into a small
neural network — trained separately and exported as a JSON weight matrix embedded directly in the
page — which classifies the hand shape into one of 16 classes (15 letters + "not a sign"). Everything
after the initial page load runs client-side; there's no backend and no network calls once the model
and camera are ready.

This repo also includes the Python pipeline I used during development to prototype the same
recognition problem before building the browser version:

| Script | What it does |
|---|---|
| [`python/collect_data.py`](python/collect_data.py) | Opens the webcam, extracts hand landmarks with MediaPipe, and appends labeled samples to a CSV |
| [`python/train_model.py`](python/train_model.py) | Trains a K-Nearest-Neighbors classifier on the collected landmarks and saves it with `joblib` |
| [`python/predict_sign.py`](python/predict_sign.py) | Runs the trained KNN model live against the webcam feed, showing the top-3 predicted letters with confidence bars |

The Python scripts and the browser app are two separate experiments on the same underlying idea
(classifying hand landmarks into Thai finger-spelling letters) — the KNN version was my first
working prototype; the browser version is the polished, deployable result.

## Note on the training data and model file

The raw training data (`hand_sign_data.csv`, ~48MB) and the trained KNN model (`sign_model.pkl`,
~16MB) aren't included in this repo — they're too large for a portfolio project and aren't needed
to run the live demo, since the browser version has its own trained weights built in. To regenerate
them, run `collect_data.py` to gather your own samples, then `train_model.py` to train and save a
model.

## Tech stack

**Browser app:** HTML, CSS, JavaScript, MediaPipe Tasks Vision, a hand-rolled feedforward neural
network for inference.
**Python pipeline:** OpenCV, MediaPipe, scikit-learn (KNeighborsClassifier), pandas.

## Run the Python pipeline locally
```bash
pip install opencv-python mediapipe pandas scikit-learn joblib
python python/collect_data.py     # collect labeled samples (repeat per letter)
python python/train_model.py      # train and save sign_model.pkl
python python/predict_sign.py     # live prediction from your webcam
```

## Run the browser app locally
No build step needed:
```bash
git clone https://github.com/Apinyayaying/gesturetalk.git
cd gesturetalk
```
Then open `index.html` in your browser (camera access requires either `localhost` or HTTPS, which
GitHub Pages provides automatically).
