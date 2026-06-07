"""
PhishGuard - AI-Powered Phishing Detection
Flask backend with security best practices.
"""
import os, re, time, logging, hashlib
from functools import wraps
from flask import Flask, request, jsonify, render_template, abort
import joblib, numpy as np, bleach

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', os.urandom(32)),
    MAX_CONTENT_LENGTH=64 * 1024,   # 64 KB max input
    JSON_SORT_KEYS=False,
)

# Security headers
@app.after_request
def security_headers(resp):
    resp.headers['X-Content-Type-Options']    = 'nosniff'
    resp.headers['X-Frame-Options']           = 'DENY'
    resp.headers['X-XSS-Protection']          = '1; mode=block'
    resp.headers['Referrer-Policy']           = 'strict-origin-when-cross-origin'
    resp.headers['Content-Security-Policy']   = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "script-src 'self' 'unsafe-inline';"
    )
    return resp

# ── Rate limiting (simple in-memory) ─────────────────────────────────────────
_rate_store: dict[str, list] = {}

def rate_limit(max_calls=20, window=60):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            ip  = request.remote_addr or 'unknown'
            key = hashlib.sha256(ip.encode()).hexdigest()[:16]
            now = time.time()
            _rate_store.setdefault(key, [])
            _rate_store[key] = [t for t in _rate_store[key] if now - t < window]
            if len(_rate_store[key]) >= max_calls:
                return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
            _rate_store[key].append(now)
            return fn(*args, **kwargs)
        return wrapper
    return decorator

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'phishing_model.pkl')
_model_data = None

def get_model():
    global _model_data
    if _model_data is None:
        _model_data = joblib.load(MODEL_PATH)
        logger.info("Phishing model loaded.")
    return _model_data

# ── Input validation ──────────────────────────────────────────────────────────
URL_REGEX = re.compile(
    r'^(https?://)?'
    r'(([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,})'
    r'(:\d{1,5})?'
    r'(/[^\s]*)?$'
)

def validate_url(url: str) -> str | None:
    url = url.strip()[:500]
    if not url: return None
    if not URL_REGEX.match(url): return None
    return url

def sanitize_text(text: str) -> str:
    text = bleach.clean(text, tags=[], strip=True)
    return text[:5000]

# ── Detection logic ───────────────────────────────────────────────────────────
from utils.features import get_feature_vector, get_risk_factors

def run_detection(url: str = '', email_text: str = '') -> dict:
    data = get_model()
    model, feature_keys = data['model'], data['feature_keys']

    X = get_feature_vector(url, email_text)
    proba = model.predict_proba(X)[0]
    phish_prob = float(proba[1])
    legit_prob = float(proba[0])

    # Verdict thresholds
    if phish_prob >= 0.75:
        verdict, level = 'PHISHING', 'danger'
    elif phish_prob >= 0.45:
        verdict, level = 'SUSPICIOUS', 'warning'
    else:
        verdict, level = 'LIKELY SAFE', 'safe'

    risk_factors = get_risk_factors(url, email_text)
    high_count   = sum(1 for r in risk_factors if r['severity'] == 'high')
    med_count    = sum(1 for r in risk_factors if r['severity'] == 'medium')

    return {
        'verdict':      verdict,
        'level':        level,
        'phish_prob':   round(phish_prob * 100, 1),
        'legit_prob':   round(legit_prob * 100, 1),
        'risk_score':   round(phish_prob * 100),
        'high_risks':   high_count,
        'med_risks':    med_count,
        'risk_factors': risk_factors,
        'analyzed_url': url or None,
        'has_email':    bool(email_text),
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
@rate_limit(max_calls=30, window=60)
def analyze():
    try:
        body = request.get_json(silent=True) or {}
        raw_url   = body.get('url', '').strip()
        raw_email = body.get('email', '').strip()

        if not raw_url and not raw_email:
            return jsonify({'error': 'Provide a URL or email content to analyze.'}), 400

        url = ''
        if raw_url:
            url = validate_url(raw_url)
            if url is None:
                return jsonify({'error': 'Invalid URL format.'}), 400

        email_text = sanitize_text(raw_email) if raw_email else ''

        result = run_detection(url=url, email_text=email_text)
        logger.info(f"Analysis: verdict={result['verdict']} prob={result['phish_prob']}% url={'yes' if url else 'no'} email={'yes' if email_text else 'no'}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({'error': 'Internal analysis error.'}), 500

@app.route('/api/health')
def health():
    try:
        get_model()
        return jsonify({'status':'ok','model':'loaded'})
    except Exception as e:
        return jsonify({'status':'error','message':str(e)}), 500

@app.errorhandler(413)
def too_large(_): return jsonify({'error':'Input too large (max 64KB).'}), 413

@app.errorhandler(429)
def rate_exceeded(_): return jsonify({'error':'Too many requests.'}), 429

if __name__ == '__main__':
    import webbrowser
    import threading
    import time

    get_model()  # pre-load

    def open_browser():
        time.sleep(1)
        webbrowser.open("http://127.0.0.1:5000")

    threading.Thread(target=open_browser).start()

    app.run(debug=False, host='0.0.0.0', port=5000)
