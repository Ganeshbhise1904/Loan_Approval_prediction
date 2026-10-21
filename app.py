import os
import pickle
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# The app checks these filenames in order. Keep your .pkl file beside app.py.
MODEL_CANDIDATES = [
    BASE_DIR / "model.pkl",
    BASE_DIR / "logistic.pkl",
    BASE_DIR / "logistic(1).pkl",
]

def load_model():
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            # Only load pickle/joblib files that you created or trust.
            return joblib.load(model_path), model_path.name
    raise FileNotFoundError(
        "Model file not found. Put model.pkl, logistic.pkl, or logistic(1).pkl beside app.py."
    )

try:
    model, MODEL_FILENAME = load_model()
    MODEL_ERROR = None
except Exception as exc:
    model, MODEL_FILENAME = None, None
    MODEL_ERROR = str(exc)

def get_feature_names():
    """Use training feature names when the saved estimator has them."""
    if model is None:
        return []
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]
    count = getattr(model, "n_features_in_", None)
    if count is not None:
        return [f"Feature {i + 1}" for i in range(int(count))]
    return []

FEATURE_NAMES = get_feature_names()

PAGE = r"""
<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Studio | ML Prediction</title>
  <meta name="description" content="A modern machine learning prediction dashboard.">
  <style>
    :root { color-scheme: dark; --bg:#0b1020; --panel:#121a2e; --panel2:#17223a; --text:#edf2ff;
      --muted:#9aa8c7; --line:#263451; --accent:#8b7cff; --accent2:#43d9bd; --field:#0d1528;
      --shadow:0 20px 60px rgba(0,0,0,.25); }
    html[data-theme="light"] { color-scheme:light; --bg:#f3f6fc; --panel:#fff; --panel2:#f5f7ff; --text:#172039;
      --muted:#63708c; --line:#e1e7f2; --accent:#6254e7; --accent2:#078e78; --field:#fbfcff;
      --shadow:0 18px 45px rgba(30,50,90,.08); }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(ellipse at 12% 0%,rgba(139,124,255,.14),transparent 34%),var(--bg);
      color:var(--text); font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; min-height:100vh; }
    .shell { max-width:1160px; margin:auto; padding:28px 22px 50px; }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:34px; }
    .brand { display:flex; align-items:center; gap:12px; font-weight:800; letter-spacing:-.4px; font-size:19px; }
    .logo { width:42px;height:42px;border-radius:14px;display:grid;place-items:center;color:white;
      background:linear-gradient(135deg,#8b7cff,#43d9bd);box-shadow:0 8px 25px rgba(139,124,255,.25);font-size:20px; }
    .brand small { display:block;color:var(--muted);font-size:11px;font-weight:600;letter-spacing:1.4px;text-transform:uppercase; }
    button, .button { cursor:pointer;border:1px solid var(--line);border-radius:12px;padding:10px 14px;background:var(--panel);
      color:var(--text);font:inherit;font-weight:700;transition:transform .15s,border-color .15s; }
    button:hover,.button:hover { transform:translateY(-1px);border-color:var(--accent); }
    .hero { display:grid;grid-template-columns:1.3fr .7fr;gap:22px;margin-bottom:22px; }
    .hero-copy { padding:28px 4px; }
    .eyebrow { display:inline-flex;align-items:center;gap:8px;color:var(--accent2);font-weight:800;font-size:12px;
      letter-spacing:1.5px;text-transform:uppercase; }
    .dot { width:7px;height:7px;background:var(--accent2);border-radius:50%;box-shadow:0 0 0 5px rgba(67,217,189,.12); }
    h1 { font-size:clamp(32px,5vw,52px);line-height:1.06;letter-spacing:-2px;margin:16px 0; }
    .gradient { background:linear-gradient(100deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent; }
    .sub { color:var(--muted);max-width:620px;font-size:16px; }
    .status { border:1px solid var(--line);border-radius:20px;background:var(--panel);padding:22px;box-shadow:var(--shadow);align-self:center; }
    .status-top { display:flex;justify-content:space-between;align-items:center;gap:12px; }
    .pill { border-radius:999px;padding:5px 10px;font-size:11px;font-weight:800;letter-spacing:.5px;background:rgba(67,217,189,.12);color:var(--accent2); }
    .status h3 { margin:18px 0 3px;font-size:19px; }
    .muted { color:var(--muted); }
    .status-grid { display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:18px; }
    .mini { background:var(--panel2);border:1px solid var(--line);border-radius:13px;padding:13px; }
    .mini strong { display:block;font-size:17px; }
    .mini span { color:var(--muted);font-size:12px; }
    .layout { display:grid;grid-template-columns:minmax(0,1.1fr) minmax(280px,.9fr);gap:22px;align-items:start; }
    .card { background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:24px;box-shadow:var(--shadow); }
    .card-head { display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:22px; }
    h2 { margin:0;font-size:20px;letter-spacing:-.5px; }
    .card-head p { margin:5px 0 0;color:var(--muted);font-size:13px; }
    .number { width:34px;height:34px;border-radius:11px;display:grid;place-items:center;background:rgba(139,124,255,.14);color:var(--accent);font-weight:900; }
    .fields { display:grid;grid-template-columns:1fr 1fr;gap:16px; }
    label { display:block;font-size:13px;font-weight:700;margin-bottom:8px; }
    input { width:100%;border:1px solid var(--line);background:var(--field);color:var(--text);padding:13px 12px;
      border-radius:11px;outline:none;font:inherit; }
    input:focus { border-color:var(--accent);box-shadow:0 0 0 3px rgba(139,124,255,.14); }
    .hint { display:block;color:var(--muted);font-size:11px;margin-top:6px; }
    .actions { display:flex;gap:10px;flex-wrap:wrap;margin-top:22px; }
    .primary { border:0;color:#fff;background:linear-gradient(100deg,#7666f4,#4aafc5);padding:13px 18px;flex:1; }
    .secondary { background:transparent; }
    .result { margin-top:20px;border:1px solid var(--line);border-radius:16px;padding:18px;background:var(--panel2); }
    .result-label { color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:1.2px;font-weight:800; }
    .prediction { font-size:clamp(25px,4vw,35px);font-weight:850;letter-spacing:-1px;margin:8px 0;overflow-wrap:anywhere; }
    .prob-row { display:flex;justify-content:space-between;gap:10px;font-size:13px;margin:15px 0 8px; }
    .bar { height:8px;background:var(--line);border-radius:99px;overflow:hidden; }
    .bar span { display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:99px; }
    .notice { border-radius:12px;padding:12px 14px;margin-bottom:16px;font-size:13px;background:rgba(255,190,80,.1);border:1px solid rgba(255,190,80,.25); }
    .error { background:rgba(255,100,120,.1);border-color:rgba(255,100,120,.3); }
    .side-list { display:grid;gap:12px; }
    .side-item { display:flex;gap:12px;padding:13px;background:var(--panel2);border:1px solid var(--line);border-radius:13px; }
    .side-icon { flex:0 0 34px;height:34px;display:grid;place-items:center;border-radius:10px;background:rgba(139,124,255,.13); }
    .side-item strong { display:block;font-size:13px; }
    .side-item span { color:var(--muted);font-size:12px;display:block;margin-top:3px; }
    .model-path { word-break:break-all;font-family:ui-monospace,monospace;font-size:12px;color:var(--accent2); }
    footer { text-align:center;color:var(--muted);font-size:12px;padding-top:30px; }
    @media(max-width:850px){.hero,.layout{grid-template-columns:1fr}.hero-copy{padding:10px 0}.status{align-self:stretch}}
    @media(max-width:520px){.shell{padding:18px 13px 35px}.fields{grid-template-columns:1fr}.card{padding:18px}header{margin-bottom:22px}h1{letter-spacing:-1.3px}}
  </style>
</head>
<body>
<div class="shell">
  <header>
    <div class="brand"><div class="logo">✦</div><div>Model Studio<small>Machine Learning Console</small></div></div>
    <button id="themeToggle" type="button" aria-label="Toggle light and dark mode">☀️ Light mode</button>
  </header>

  <section class="hero">
    <div class="hero-copy">
      <div class="eyebrow"><span class="dot"></span> Prediction workspace</div>
      <h1>Turn your data into<br><span class="gradient">decisions.</span></h1>
      <p class="sub">A clean, responsive interface for running predictions with your trained Logistic Regression model. Enter feature values and get a result in seconds.</p>
    </div>
    <div class="status">
      <div class="status-top"><span class="muted">MODEL STATUS</span><span class="pill">{{ 'READY' if model_ready else 'NEEDS SETUP' }}</span></div>
      <h3>{{ 'Logistic Regression' if model_ready else 'Model not loaded' }}</h3>
      <div class="muted" style="font-size:13px">{{ 'Model file: ' ~ model_filename if model_ready else 'Check the model file and dependencies.' }}</div>
      <div class="status-grid">
        <div class="mini"><strong>{{ feature_count if model_ready else '—' }}</strong><span>Input features</span></div>
        <div class="mini"><strong>Flask</strong><span>Web framework</span></div>
      </div>
    </div>
  </section>

  <section class="layout">
    <div class="card">
      <div class="card-head"><div><h2>Prediction inputs</h2><p>Provide a value for every feature expected by your model.</p></div><div class="number">01</div></div>
      {% if not model_ready %}
        <div class="notice error"><strong>Model loading error</strong><br>{{ model_error }}</div>
      {% endif %}
      {% if error %}<div class="notice error">{{ error }}</div>{% endif %}
      <form method="post" action="/">
        <div class="fields">
          {% for feature in features %}
          <div>
            <label for="f{{ loop.index0 }}">{{ feature }}</label>
            <input id="f{{ loop.index0 }}" name="feature_{{ loop.index0 }}" type="number" step="any" required
                   value="{{ values[loop.index0] if values and loop.index0 < values|length else '' }}"
                   placeholder="Enter {{ feature|lower }}" />
            <span class="hint">Numeric input</span>
          </div>
          {% endfor %}
        </div>
        <div class="actions">
          <button class="primary" type="submit" {% if not model_ready %}disabled{% endif %}>✦ Run prediction</button>
          <button class="secondary" type="reset">Reset values</button>
        </div>
      </form>

      {% if prediction is not none %}
      <div class="result" aria-live="polite">
        <div class="result-label">Prediction result</div>
        <div class="prediction">{{ prediction }}</div>
        <div class="muted">Generated by your saved model.</div>
        {% if probabilities %}
          {% for item in probabilities %}
            <div class="prob-row"><span>{{ item.label }}</span><strong>{{ '%.2f'|format(item.value * 100) }}%</strong></div>
            <div class="bar"><span style="width:{{ (item.value * 100)|round(2) }}%"></span></div>
          {% endfor %}
          <span class="hint">Probabilities are model estimates, not guarantees.</span>
        {% endif %}
      </div>
      {% endif %}
    </div>

    <aside class="card">
      <div class="card-head"><div><h2>Workspace details</h2><p>Deployment-ready essentials</p></div><div class="number">02</div></div>
      <div class="side-list">
        <div class="side-item"><div class="side-icon">⚡</div><div><strong>Instant inference</strong><span>Runs prediction on the server using your saved model.</span></div></div>
        <div class="side-item"><div class="side-icon">◐</div><div><strong>Dark / light theme</strong><span>Switch appearance with the button in the top corner.</span></div></div>
        <div class="side-item"><div class="side-icon">▦</div><div><strong>Responsive layout</strong><span>Works on desktop, tablet, and mobile screens.</span></div></div>
        <div class="side-item"><div class="side-icon">⌘</div><div><strong>Model file</strong><span class="model-path">{{ model_filename if model_ready else 'Place your .pkl file beside app.py' }}</span></div></div>
      </div>
      <div style="margin-top:22px">
        <div class="muted" style="font-size:12px;margin-bottom:8px">INPUT FEATURES</div>
        {% if features %}
          {% for feature in features %}<span class="pill" style="display:inline-block;margin:0 5px 7px 0">{{ feature }}</span>{% endfor %}
        {% else %}
          <span class="muted">Feature names will appear when the model loads.</span>
        {% endif %}
      </div>
    </aside>
  </section>
  <footer>Model Studio · Flask inference interface · Verify predictions before using them in real-world decisions.</footer>
</div>
<script>
  const root = document.documentElement;
  const toggle = document.getElementById('themeToggle');
  function setTheme(theme) {
    root.setAttribute('data-theme', theme);
    toggle.textContent = theme === 'dark' ? '☀️ Light mode' : '🌙 Dark mode';
    try { localStorage.setItem('model-studio-theme', theme); } catch (e) {}
  }
  let savedTheme = 'dark';
  try { savedTheme = localStorage.getItem('model-studio-theme') || 'dark'; } catch (e) {}
  setTheme(savedTheme);
  toggle.addEventListener('click', () => setTheme(root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'));
</script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    features = FEATURE_NAMES or []
    values = []
    prediction = None
    probabilities = []
    error = None

    if request.method == "POST" and model is not None:
        try:
            values = [float(request.form[f"feature_{i}"]) for i in range(len(features))]
            if len(values) != int(getattr(model, "n_features_in_", len(values))):
                raise ValueError("The number of inputs does not match the model's expected features.")
            sample = np.asarray(values, dtype=float).reshape(1, -1)
            predicted = model.predict(sample)[0]
            prediction = str(predicted)

            if hasattr(model, "predict_proba"):
                raw_probs = model.predict_proba(sample)[0]
                classes = getattr(model, "classes_", range(len(raw_probs)))
                probabilities = [
                    {"label": str(label), "value": float(prob)}
                    for label, prob in zip(classes, raw_probs)
                ]
                probabilities.sort(key=lambda item: item["value"], reverse=True)
        except KeyError:
            error = "Please enter a value for every feature."
        except (TypeError, ValueError) as exc:
            error = f"Could not run prediction: {exc}"
        except Exception as exc:
            error = f"Prediction failed. Check that your input values match the model: {exc}"

    return render_template_string(
        PAGE,
        model_ready=model is not None,
        model_filename=MODEL_FILENAME,
        model_error=MODEL_ERROR,
        feature_count=len(features),
        features=features,
        values=values,
        prediction=prediction,
        probabilities=probabilities,
        error=error,
    )

if __name__ == "__main__":
    # Use debug=False for safer local testing and deployment.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
