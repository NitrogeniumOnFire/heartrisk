# Heart Disease Risk Predictor — GitHub Pages App

Files in this folder are ready to be published to GitHub Pages (push to a repo and enable Pages on the main branch).

Included files:
- index.html — main UI
- style.css — styling
- script.js — contains JS forward-pass and UI wiring
- model.json — model weights, feature names, scaler (means/stds)

How it works:
- The model is a small dense neural network trained on the provided heart.csv.
- The JS implements a straightforward forward pass (no external libraries).
- To publish: create a GitHub repo, add these files to the repository root, push, then enable GitHub Pages (Settings → Pages).

Notes:
- Model input features are: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
- The model expects numeric inputs for those features. Use the 'Fill sample' button to populate default values.
- This app performs inference fully in the browser; no server required.
