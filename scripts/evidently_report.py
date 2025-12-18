"""
Generate an Evidently HTML report comparing baseline (training) vs production.

Usage:
    python scripts/evidently_report.py

Output:
    monitoring/evidently_report.html
"""
from pathlib import Path
import pandas as pd
import pickle
import sys
import json

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
except Exception:
    Report = None
    DataDriftPreset = None

BASE_DIR = Path(__file__).resolve().parent.parent


def load_model_and_vectorizer():
    model_path = BASE_DIR / 'model_registry' / 'Best_Election_Model' / 'production.pkl'
    vec_path = BASE_DIR / 'model_registry' / 'Best_Election_Model' / 'tfidf_vectorizer.pkl'
    if not model_path.exists() or not vec_path.exists():
        return None, None
    try:
        with open(model_path, 'rb') as mf:
            model = pickle.load(mf)
        with open(vec_path, 'rb') as vf:
            vectorizer = pickle.load(vf)
        return model, vectorizer
    except Exception as e:
        print(f"Warning: could not load model/vectorizer: {e}")
        return None, None


def prepare_datasets(df, model=None, vectorizer=None):
    df = df.copy().reset_index(drop=True)
    if 'prediction_id' not in df.columns:
        import uuid
        df['prediction_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

    split_idx = int(len(df) * 0.8)
    ref = df.iloc[:split_idx].reset_index(drop=True)
    curr = df.iloc[split_idx:].reset_index(drop=True)

    if 'prediction_label' not in ref.columns:
        ref['prediction_label'] = ref.get('target')
        ref['prediction'] = ref['prediction_label']
    else:
        ref['prediction'] = ref['prediction_label']

    # For current, compute predictions if model available
    if model is not None and vectorizer is not None and len(curr) > 0:
        texts = curr.get('comments', curr.get('text', pd.Series([''] * len(curr)))).fillna('').astype(str)
        try:
            X = vectorizer.transform(texts)
            try:
                preds = model.predict(X)
            except Exception:
                preds = model.predict(X.toarray())
        except Exception:
            preds = [None] * len(curr)
        curr['prediction_label'] = preds
        curr['prediction'] = curr['prediction_label']
    else:
        if 'prediction_label' not in curr.columns:
            curr['prediction_label'] = curr.get('target')
        curr['prediction'] = curr['prediction_label']

    return ref, curr


def run_report(reference, current, out_html):
    # Prepare DataFrames with standard column names
    def _prepare_df(df):
        d = df.copy()
        if 'prediction' not in d.columns and 'prediction_label' in d.columns:
            d['prediction'] = d['prediction_label']
        if 'id' not in d.columns and 'prediction_id' in d.columns:
            d['id'] = d['prediction_id']
        return d

    ref2 = _prepare_df(reference)
    curr2 = _prepare_df(current)

    # If Evidently is available, try running a DataDrift report
    if Report is not None and DataDriftPreset is not None:
        report = Report(metrics=[DataDriftPreset()])
        try:
            report.run(reference_data=ref2, current_data=curr2)
        except Exception as e:
            print(f"Warning: Evidently failed to run full report: {e}")
        # Try multiple export methods
        try:
            out_html.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(report, 'save_html'):
                report.save_html(str(out_html))
                print(f"Saved Evidently report to: {out_html} (via save_html)")
                return
            if hasattr(report, 'as_html'):
                html = report.as_html()
                with open(out_html, 'w', encoding='utf-8') as fh:
                    fh.write(html)
                print(f"Saved Evidently report to: {out_html} (via as_html)")
                return
            if hasattr(report, 'as_dict'):
                data = report.as_dict()
                json_path = out_html.with_suffix('.json')
                with open(json_path, 'w', encoding='utf-8') as fh:
                    json.dump(data, fh, indent=2)
                print(f"Saved Evidently report metrics JSON to: {json_path} (via as_dict)")
                return
        except Exception as e:
            print(f"⚠️  Could not export Evidently report: {e}")

    # Fallback: create a minimal HTML summary ourselves
    out_html.parent.mkdir(parents=True, exist_ok=True)

    def _dist_table(s):
        vc = pd.Series(s).dropna().astype(str).value_counts().head(10)
        rows = ''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in vc.items())
        return f"<table border=1><tr><th>Value</th><th>Count</th></tr>{rows}</table>"

    ref_stats = {
        'rows': len(ref2),
        'target_counts': _dist_table(ref2.get('target', pd.Series([]))),
        'prediction_counts': _dist_table(ref2.get('prediction', pd.Series([]))),
        'comment_len_mean': ref2.get('comments', pd.Series([''])).fillna('').astype(str).map(len).mean()
    }
    curr_stats = {
        'rows': len(curr2),
        'target_counts': _dist_table(curr2.get('target', pd.Series([]))),
        'prediction_counts': _dist_table(curr2.get('prediction', pd.Series([]))),
        'comment_len_mean': curr2.get('comments', pd.Series([''])).fillna('').astype(str).map(len).mean()
    }

    html = f"""
    <html><head><meta charset='utf-8'><title>Evidently Fallback Report</title></head><body>
    <h1>Evidently Fallback Report</h1>
    <h2>Reference (baseline)</h2>
    <p>Rows: {ref_stats['rows']}</p>
    <p>Mean comment length: {ref_stats['comment_len_mean']:.2f}</p>
    <h3>Target distribution (top 10)</h3>
    {ref_stats['target_counts']}
    <h3>Prediction distribution (top 10)</h3>
    {ref_stats['prediction_counts']}
    <hr/>
    <h2>Current (production)</h2>
    <p>Rows: {curr_stats['rows']}</p>
    <p>Mean comment length: {curr_stats['comment_len_mean']:.2f}</p>
    <h3>Target distribution (top 10)</h3>
    {curr_stats['target_counts']}
    <h3>Prediction distribution (top 10)</h3>
    {curr_stats['prediction_counts']}
    </body></html>
    """
    with open(out_html, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print(f"Saved fallback Evidently report to: {out_html}")


def main():
    data_path = BASE_DIR / 'data' / 'version1.xlsx'
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        sys.exit(1)

    print(f"Loading data: {data_path}")
    df = pd.read_excel(data_path)

    model, vectorizer = load_model_and_vectorizer()

    ref, curr = prepare_datasets(df, model=model, vectorizer=vectorizer)

    out_html = BASE_DIR / 'monitoring' / 'evidently_report.html'
    run_report(ref, curr, out_html)


if __name__ == '__main__':
    main()
