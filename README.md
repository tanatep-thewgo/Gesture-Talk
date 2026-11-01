# Gesture Talk — Thai Hand-Sign Recognition

My senior project: a real-time Thai sign-language recognizer that runs entirely in the browser
(in-app name: **SignSense**). Point your camera at your hand and it predicts the Thai letter or word
you're signing, live, with no server and no installation.

**Features**
- **Letter mode:** 15 Thai finger-spelling letters, shown in Thai script (ก ด บ ฟ ห ล ม น อ พ ร ส ต ว ย)
- **Word mode (beta):** 4 words — สวัสดี, ขอโทษ, เสียใจ, ขอบคุณ — with a confidence threshold
- **Search:** browse every supported sign with a reference photo of the hand pose (tap to enlarge)
- **History:** log of the predictions made in the current session
- **PWA-friendly, mobile-first UI**; all inference happens on-device

🔗 **Live demo:** [https://tanatep-thewgo.github.io/Gesture-Talk/](https://tanatep-thewgo.github.io/Gesture-Talk/)
*(needs a webcam; open the app, tap the scan button and allow camera access)*

## How it works

The browser app ([`index.html`](index.html)) uses **MediaPipe Tasks Vision** (loaded from a CDN) to
track 21 hand landmarks per frame. Those landmark coordinates are normalized and fed into a small
neural network — trained separately and exported as a JSON weight matrix embedded directly in the
page — which classifies the hand shape into one of 16 letter classes (15 letters + "not a sign"). Word mode
uses a second small network (4 words) on wrist-relative landmark coordinates. Everything after the
initial page load runs client-side; there's no backend, and camera frames never leave the device.
(The page itself loads MediaPipe and web fonts from a CDN.)

This repo also includes the Python pipeline I used during development to prototype the same
recognition problem before building the browser version:

| Script | What it does |
|---|---|
| [`collect_data.py`](collect_data.py) | Opens the webcam, extracts hand landmarks with MediaPipe, and appends labeled samples to a CSV |
| [`train_model.py`](train_model.py) | Trains a K-Nearest-Neighbors classifier on the collected landmarks and saves it with `joblib` |
| [`predict_sign.py`](predict_sign.py) | Runs the trained KNN model live against the webcam feed, showing the top-3 predicted letters with confidence bars |

The Python scripts and the browser app are two separate experiments on the same underlying idea
(classifying hand landmarks into Thai finger-spelling letters) — the KNN version was my first
working prototype; the browser version is the polished, deployable result.

## Note on the training data and model file

The raw training data (`hand_sign_data.csv`, ~48MB) isn't included — it's too large for a portfolio
project and isn't needed to run the live demo, since the browser version has its own trained weights
built in. The KNN prototype model (`sign_model.pkl`, ~16MB) is included for the Python scripts. To
regenerate the data, run `collect_data.py` to gather your own samples, then `train_model.py` to train
and save a model.

## Known limitations

- The ~99% letter accuracy was measured on a random split of the collected frames, so it is optimistic
  (neighbouring frames are highly correlated). A subject-wise split would give a more honest number.
- Word mode covers only 4 words and is a beta; it has no "not a word" class, so a confidence threshold
  is used instead.
- Reference photos in the Search tab are cleaned-up scans of lower-resolution source images.

## Tech stack

**Browser app:** HTML, CSS, JavaScript, MediaPipe Tasks Vision, a hand-rolled feedforward neural
network for inference.
**Python pipeline:** OpenCV, MediaPipe, scikit-learn (KNeighborsClassifier), pandas.
