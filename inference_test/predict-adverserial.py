import requests
import json
import base64
import io
from pathlib import Path
from typing import Optional

def call_api_and_generate_html(
    mitigation_method: str,
    image_path: str, 
    output_html_path: str,    
    ground_truth_path: Optional[str] = None,    
):
    """Call API and generate HTML report with all images."""
    
    url = "http://localhost:8000/predict-adverserial"
    data = {"mitigation_method": mitigation_method}
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        if ground_truth_path:
            with open(ground_truth_path, "rb") as g:
                files["ground_truth"] = g
                response = requests.post(url, data=data, files=files)
        else:
            response = requests.post(url, data=data, files=files)
    
    data = response.json()
    
    # Close files
    for f in files.values():
        f.close()
    
    # Helpers
    def fmt_prob(p):
        return f"{p*100:.2f}%"
    
    def ensure_data_uri(img_data):
        if not img_data:
            return ""
        if img_data.startswith('data:image'):
            return img_data
        return f"data:image/png;base64,{img_data}"
    
    # Build variants rows
    variants_html = []
    for v in data['variants']:
        m = v['mitigation'] or {}
        attack_class = "attack-success" if v['is_attack_successful'] else "attack-failed"
        attack_status = "SUCCESS" if v['is_attack_successful'] else "FAILED"
        restoration = m.get('confidence_restored_pct', 0)
        restoration_class = "mitigation-restored" if restoration > 0 else "mitigation-failed"
        restoration_bar_class = "restoration-positive" if restoration > 0 else "restoration-negative"
        
        mit_image = ensure_data_uri(m.get('heatmap', '')) if m.get('heatmap') else ''
        mit_pred = m.get('prediction', 'N/A')
        mit_prob = fmt_prob(m.get('probability', 0)) if m.get('probability') else 'N/A'
        mit_method = m.get('method', 'N/A')
        
        row = f"""
        <tr>
            <td class="epsilon-cell">{v['epsilon']}</td>
            <td>
                <div class="metric">L2 Dist: <span class="metric-value">{v['l2_distance']:.6f}</span></div>
                <div class="metric">Attack: <span class="{attack_class}">{attack_status}</span></div>
                <div class="metric">Pred: <span class="metric-value">{v['prediction']}</span></div>
                <div class="metric">Prob: <span class="metric-value">{fmt_prob(v['probability'])}</span></div>
            </td>
            <td>
                <img src="{ensure_data_uri(v['image'])}" class="table-image" alt="Adversarial">
                <div class="image-title">{v['prediction']} ({fmt_prob(v['probability'])})</div>
            </td>
            <td>
                <img src="{ensure_data_uri(v['heatmap'])}" class="table-image" alt="Adversarial CAM">
                <div class="image-title">Model Attention</div>
            </td>
            <td>
                <img src="{mit_image}" class="table-image" alt="Mitigated" onerror="this.style.display='none'">
                <div class="image-title">{mit_pred}</div>
            </td>
            <td>
                <div class="metric">Method: <span class="metric-value">{mit_method}</span></div>
                <div class="metric">Result: <span class="metric-value">{mit_pred}</span></div>
                <div class="metric">Prob: <span class="metric-value">{mit_prob}</span></div>
                <div class="metric">Restored: <span class="{restoration_class}">{restoration:.1f}%</span></div>
                <div class="restoration-bar">
                    <div class="restoration-fill {restoration_bar_class}" 
                         style="width: {min(abs(restoration), 100):.1f}%"></div>
                </div>
            </td>
        </tr>
        """
        variants_html.append(row)
    
    # === BUILD GROUND TRUTH HTML CONDITIONALLY ===
    orig = data['original']
    orig_class = orig['prediction'].lower().replace('sicknontb', 'sick')
    
    if orig.get('ground_truth'):
        ground_truth_html = f'''
                <div class="image-card" style="border: 2px solid #27ae60; border-radius: 4px; padding: 10px;">
                    <img src="{ensure_data_uri(orig['ground_truth'])}" alt="Ground Truth">
                    <div class="image-label">Ground Truth (Reference)</div>
                    <span style="color: #27ae60; font-weight: 600;">✓ Reference Standard</span>
                </div>'''
        grid_columns = '1fr 1fr 1fr'
    else:
        ground_truth_html = ''
        grid_columns = '1fr 1fr'
    
    # === BUILD FULL HTML ===
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TB Classifier - Adversarial Analysis Results</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; padding: 20px; line-height: 1.6; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #333; margin-bottom: 10px; font-size: 24px; }}
        .file-info {{ color: #666; margin-bottom: 30px; font-size: 14px; }}
        .original-section {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .original-section h2 {{ color: #2c3e50; margin-bottom: 15px; font-size: 18px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .original-grid {{ display: grid; gap: 20px; }}
        .image-card {{ text-align: center; }}
        .image-card img {{ max-width: 100%; max-height: 400px; height: auto; border-radius: 4px; border: 1px solid #ddd; }}
        .image-label {{ margin-top: 10px; font-weight: 600; color: #555; }}
        .prediction-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 14px; font-weight: 600; margin-top: 5px; }}
        .prediction-healthy {{ background: #d4edda; color: #155724; }}
        .prediction-sick {{ background: #f8d7da; color: #721c24; }}
        .prediction-tb {{ background: #fff3cd; color: #856404; }}
        .variants-section {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .variants-section h2 {{ color: #2c3e50; margin-bottom: 20px; font-size: 18px; border-bottom: 2px solid #e74c3c; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ background: #34495e; color: white; padding: 12px 8px; text-align: center; font-weight: 600; }}
        td {{ padding: 15px 8px; text-align: center; border-bottom: 1px solid #ecf0f1; vertical-align: top; }}
        tr:hover {{ background: #f8f9fa; }}
        .epsilon-cell {{ font-weight: 700; font-size: 16px; color: #2c3e50; }}
        .metric {{ font-family: 'Courier New', monospace; font-size: 12px; color: #666; margin: 2px 0; }}
        .metric-value {{ font-weight: 600; color: #333; }}
        .attack-success {{ color: #e74c3c; font-weight: 700; }}
        .attack-failed {{ color: #27ae60; font-weight: 700; }}
        .mitigation-restored {{ color: #27ae60; font-weight: 600; }}
        .mitigation-failed {{ color: #e74c3c; font-weight: 600; }}
        .table-image {{ max-width: 200px; max-height: 200px; border-radius: 4px; border: 1px solid #ddd; cursor: pointer; transition: transform 0.2s; object-fit: contain; }}
        .table-image:hover {{ transform: scale(1.5); box-shadow: 0 4px 12px rgba(0,0,0,0.3); z-index: 100; position: relative; }}
        .image-title {{ font-size: 11px; color: #888; margin-top: 5px; }}
        .restoration-bar {{ width: 100%; height: 6px; background: #ecf0f1; border-radius: 3px; margin-top: 5px; overflow: hidden; }}
        .restoration-fill {{ height: 100%; border-radius: 3px; }}
        .restoration-positive {{ background: #27ae60; }}
        .restoration-negative {{ background: #e74c3c; }}
        .legend {{ margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 6px; font-size: 12px; }}
        .legend h4 {{ margin-bottom: 10px; color: #555; }}
        .legend-item {{ display: inline-block; margin-right: 20px; margin-bottom: 5px; }}
        .legend-color {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 TB Classifier Adversarial Robustness Analysis</h1>
        <div class="file-info">Analyzing file: <strong>{data['filename']}</strong></div>
        
        <div class="original-section">
            <h2>📋 Original Analysis (Baseline)</h2>
            <div class="original-grid" style="grid-template-columns: {grid_columns};">
                <div class="image-card">
                    <img src="{ensure_data_uri(orig['image'])}" alt="Original X-ray">
                    <div class="image-label">Input Image (to be attacked)</div>
                    <span class="prediction-badge prediction-{orig_class}">{orig['prediction']} ({fmt_prob(orig['probability'])})</span>
                </div>
                <div class="image-card">
                    <img src="{ensure_data_uri(orig['heatmap'])}" alt="Original Grad-CAM">
                    <div class="image-label">Model Attention (Grad-CAM)</div>
                    <span class="prediction-badge prediction-{orig_class}">{orig['prediction']}</span>
                </div>
                {ground_truth_html}
            </div>
        </div>
        
        <div class="variants-section">
            <h2>⚔️ Adversarial Attacks & Mitigations</h2>
            <table>
                <thead>
                    <tr>
                        <th>Epsilon (ε)</th>
                        <th>Metrics</th>
                        <th>Adversarial Image</th>
                        <th>Adversarial Grad-CAM</th>
                        <th>Mitigated Image</th>
                        <th>Mitigation Result</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(variants_html)}
                </tbody>
            </table>
            
            <div class="legend">
                <h4>Legend:</h4>
                <div class="legend-item"><span class="legend-color" style="background: #d4edda;"></span>Healthy</div>
                <div class="legend-item"><span class="legend-color" style="background: #f8d7da;"></span>SickNonTB</div>
                <div class="legend-item"><span class="legend-color" style="background: #fff3cd;"></span>TB</div>
                <div class="legend-item"><span class="legend-color" style="background: #e74c3c;"></span>Attack Successful</div>
                <div class="legend-item"><span class="legend-color" style="background: #27ae60;"></span>Mitigation Restored</div>
            </div>
        </div>
    </div>
</body>
</html>"""
    
    # Write to file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {Path(output_html_path).absolute()}")
    print(f"File size: {Path(output_html_path).stat().st_size / 1024:.1f} KB")
    return output_html_path

# Usage
if __name__ == "__main__":

    mitigation_method="wavelet_spatial_hybrid"

    call_api_and_generate_html(
        mitigation_method=mitigation_method,
        image_path="s0035.png",
        output_html_path=f"report_with_{mitigation_method}_sickNonTb.html"
        # ground_truth_path="tb1117_truth.png"
    )