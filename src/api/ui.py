"""
Interfaz HTML interactiva para la API de prediccion de reingreso hospitalario.
Se sirve desde GET /ui en el mismo servidor FastAPI.
"""

UI_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predictor de Reingreso Hospitalario</title>
<style>
  :root {
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2d3047;
    --accent: #4f8ef7;
    --accent2: #7c5cbf;
    --text: #e2e8f0;
    --muted: #8892a4;
    --low: #22c55e;
    --mid: #f59e0b;
    --high: #ef4444;
    --radius: 10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }
  
  header { background: linear-gradient(135deg, var(--accent2), var(--accent)); padding: 20px 32px; display: flex; align-items: center; gap: 14px; }
  header h1 { font-size: 1.3rem; font-weight: 700; }
  header p { font-size: 0.8rem; opacity: 0.85; margin-top: 2px; }
  .pill { background: rgba(255,255,255,0.2); border-radius: 20px; padding: 3px 10px; font-size: 0.72rem; font-weight: 600; }

  main { max-width: 900px; margin: 0 auto; padding: 28px 20px; display: grid; grid-template-columns: 1fr 340px; gap: 20px; }

  .card { background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px; }
  .card h2 { font-size: 0.85rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 18px; }

  .section-label { font-size: 0.72rem; font-weight: 700; color: var(--accent); text-transform: uppercase; letter-spacing: 0.1em; margin: 18px 0 10px; padding-bottom: 4px; border-bottom: 1px solid var(--border); }
  .section-label:first-child { margin-top: 0; }

  .fields { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 16px; }
  .field { display: flex; flex-direction: column; gap: 4px; }
  .field label { font-size: 0.75rem; color: var(--muted); }
  .field input, .field select {
    background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
    color: var(--text); padding: 7px 10px; font-size: 0.85rem; width: 100%;
    transition: border-color 0.15s;
  }
  .field input:focus, .field select:focus { outline: none; border-color: var(--accent); }
  .field input[type=range] { padding: 4px 0; cursor: pointer; accent-color: var(--accent); }
  .range-row { display: flex; align-items: center; gap: 8px; }
  .range-val { min-width: 28px; text-align: right; font-weight: 600; color: var(--accent); font-size: 0.9rem; }

  .threshold-row { display: flex; align-items: center; gap: 10px; margin-top: 14px; }
  .threshold-row label { font-size: 0.78rem; color: var(--muted); flex: 1; }

  button.predict-btn {
    width: 100%; margin-top: 20px; padding: 12px;
    background: linear-gradient(135deg, var(--accent2), var(--accent));
    border: none; border-radius: var(--radius); color: white;
    font-size: 0.95rem; font-weight: 700; cursor: pointer; transition: opacity 0.15s;
  }
  button.predict-btn:hover { opacity: 0.88; }
  button.predict-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Result panel */
  .result-placeholder { text-align: center; color: var(--muted); font-size: 0.85rem; padding: 40px 0; }
  .result-placeholder svg { opacity: 0.25; margin-bottom: 12px; }

  .risk-badge {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 1.5rem; font-weight: 800; padding: 10px 20px;
    border-radius: 30px; margin: 10px 0 6px;
  }
  .risk-BAJO  { background: rgba(34,197,94,0.15);  color: var(--low);  }
  .risk-MEDIO { background: rgba(245,158,11,0.15); color: var(--mid);  }
  .risk-ALTO  { background: rgba(239,68,68,0.15);  color: var(--high); }

  .prob-bar-wrap { margin: 14px 0; }
  .prob-bar-bg { background: var(--border); border-radius: 20px; height: 10px; overflow: hidden; }
  .prob-bar { height: 10px; border-radius: 20px; transition: width 0.5s ease; }

  .result-meta { font-size: 0.78rem; color: var(--muted); margin-top: 10px; line-height: 1.7; }
  .result-meta strong { color: var(--text); }

  .recommendation {
    margin-top: 16px; padding: 12px 14px;
    background: var(--bg); border-radius: 8px; border-left: 3px solid var(--accent);
    font-size: 0.82rem; line-height: 1.5;
  }
  .recommendation .rec-title { font-size: 0.7rem; font-weight: 700; color: var(--accent); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }

  .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 5px; }
  .dot-ok { background: var(--low); }
  .dot-err { background: var(--high); }
  .dot-warn { background: var(--mid); }

  .health-bar { font-size: 0.78rem; color: var(--muted); margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); display: flex; align-items: center; gap: 6px; }

  .spinner { display: none; width: 18px; height: 18px; border: 2px solid rgba(79,142,247,0.3); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.7s linear infinite; margin: 0 auto; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .error-box { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 8px; padding: 12px; font-size: 0.82rem; color: #fca5a5; margin-top: 10px; }

  @media (max-width: 680px) {
    main { grid-template-columns: 1fr; }
    .fields { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>🏥 Predictor de Reingreso Hospitalario</h1>
    <p>Diabetes 130-US Hospitals · XGBoost + MLflow · <span class="pill">v1.1.0</span></p>
  </div>
</header>

<main>
  <!-- LEFT: Form -->
  <div class="card">
    <h2>Datos del paciente</h2>

    <div class="section-label">Demografía</div>
    <div class="fields">
      <div class="field">
        <label>Edad</label>
        <div class="range-row">
          <input type="range" id="age" min="0" max="100" value="65" oninput="document.getElementById('age_val').textContent=this.value">
          <span class="range-val" id="age_val">65</span>
        </div>
      </div>
      <div class="field">
        <label>Género</label>
        <select id="gender">
          <option value="Female">Femenino</option>
          <option value="Male">Masculino</option>
          <option value="Unknown">Desconocido</option>
        </select>
      </div>
    </div>

    <div class="section-label">Datos de Admisión</div>
    <div class="fields">
      <div class="field">
        <label>Tipo de admisión</label>
        <select id="admission_type_id">
          <option value="1">1 · Emergencia</option>
          <option value="2">2 · Urgente</option>
          <option value="3">3 · Electiva</option>
          <option value="4">4 · Recién nacido</option>
          <option value="5">5 · Sin info</option>
          <option value="6">6 · N/A</option>
          <option value="7">7 · Trauma</option>
          <option value="8">8 · Desconocido</option>
        </select>
      </div>
      <div class="field">
        <label>Fuente de admisión</label>
        <select id="admission_source_id">
          <option value="7">7 · Emergencias</option>
          <option value="1">1 · Referido médico</option>
          <option value="2">2 · Referido clínica</option>
          <option value="4">4 · Transfer hospital</option>
          <option value="9">9 · Sin info</option>
        </select>
      </div>
      <div class="field">
        <label>Tipo de alta</label>
        <select id="discharge_disposition_id">
          <option value="1">1 · Alta a domicilio</option>
          <option value="2">2 · Alta con asistencia</option>
          <option value="3">3 · Transfer SNF</option>
          <option value="6">6 · Alta a casa con salud</option>
          <option value="11">11 · Fallecido</option>
          <option value="18">18 · Sin info</option>
        </select>
      </div>
      <div class="field">
        <label>Días hospitalizado</label>
        <div class="range-row">
          <input type="range" id="time_in_hospital" min="1" max="14" value="5" oninput="document.getElementById('tih_val').textContent=this.value">
          <span class="range-val" id="tih_val">5</span>
        </div>
      </div>
    </div>

    <div class="section-label">Procedimientos y Medicamentos</div>
    <div class="fields">
      <div class="field">
        <label>Procedimientos de laboratorio</label>
        <div class="range-row">
          <input type="range" id="num_lab_procedures" min="0" max="132" value="44" oninput="document.getElementById('nlp_val').textContent=this.value">
          <span class="range-val" id="nlp_val">44</span>
        </div>
      </div>
      <div class="field">
        <label>Otros procedimientos</label>
        <div class="range-row">
          <input type="range" id="num_procedures" min="0" max="6" value="1" oninput="document.getElementById('np_val').textContent=this.value">
          <span class="range-val" id="np_val">1</span>
        </div>
      </div>
      <div class="field">
        <label>Medicamentos distintos</label>
        <div class="range-row">
          <input type="range" id="num_medications" min="0" max="81" value="14" oninput="document.getElementById('nm_val').textContent=this.value">
          <span class="range-val" id="nm_val">14</span>
        </div>
      </div>
      <div class="field">
        <label>Diagnósticos ingresados</label>
        <div class="range-row">
          <input type="range" id="number_diagnoses" min="1" max="16" value="9" oninput="document.getElementById('nd_val').textContent=this.value">
          <span class="range-val" id="nd_val">9</span>
        </div>
      </div>
    </div>

    <div class="section-label">Visitas Previas (último año)</div>
    <div class="fields">
      <div class="field">
        <label>Visitas ambulatorias</label>
        <div class="range-row">
          <input type="range" id="number_outpatient" min="0" max="42" value="0" oninput="document.getElementById('no_val').textContent=this.value">
          <span class="range-val" id="no_val">0</span>
        </div>
      </div>
      <div class="field">
        <label>Visitas a urgencias</label>
        <div class="range-row">
          <input type="range" id="number_emergency" min="0" max="76" value="1" oninput="document.getElementById('ne_val').textContent=this.value">
          <span class="range-val" id="ne_val">1</span>
        </div>
      </div>
      <div class="field">
        <label>Hospitalizaciones previas</label>
        <div class="range-row">
          <input type="range" id="number_inpatient" min="0" max="21" value="1" oninput="document.getElementById('ni_val').textContent=this.value">
          <span class="range-val" id="ni_val">1</span>
        </div>
      </div>
    </div>

    <div class="section-label">Control Glucémico</div>
    <div class="fields">
      <div class="field">
        <label>Resultado HbA1c</label>
        <select id="A1Cresult">
          <option value="None">Sin medición</option>
          <option value="Norm">Normal</option>
          <option value=">7">&gt;7</option>
          <option value=">8" selected>&gt;8</option>
        </select>
      </div>
      <div class="field">
        <label>Insulina</label>
        <select id="insulin">
          <option value="No">Sin cambio</option>
          <option value="Steady">Estable</option>
          <option value="Up" selected>Aumentada</option>
          <option value="Down">Reducida</option>
        </select>
      </div>
      <div class="field">
        <label>Medicamento diabetes</label>
        <select id="diabetesMed">
          <option value="Yes" selected>Sí</option>
          <option value="No">No</option>
        </select>
      </div>
      <div class="field">
        <label>Cambio en medicación</label>
        <select id="change">
          <option value="Ch" selected>Sí hubo cambio</option>
          <option value="No">Sin cambio</option>
        </select>
      </div>
    </div>

    <div class="threshold-row">
      <label>Umbral de decisión de riesgo alto</label>
      <div class="range-row" style="flex:0 0 auto">
        <input type="range" id="threshold" min="0.1" max="0.9" step="0.05" value="0.3" style="width:100px" oninput="document.getElementById('thr_val').textContent=parseFloat(this.value).toFixed(2)">
        <span class="range-val" id="thr_val">0.30</span>
      </div>
    </div>

    <button class="predict-btn" id="predictBtn" onclick="predict()">⚡ Predecir Riesgo</button>
  </div>

  <!-- RIGHT: Result -->
  <div>
    <div class="card" id="resultCard">
      <h2>Resultado</h2>
      <div id="resultContent">
        <div class="result-placeholder">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M9 12h6M12 9v6M3 12a9 9 0 1 0 18 0 9 9 0 0 0-18 0z"/>
          </svg>
          <p>Completa los datos del paciente<br>y presiona <strong>Predecir</strong>.</p>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <h2>Estado del Servicio</h2>
      <div id="healthContent" style="font-size:0.82rem;color:var(--muted)">Cargando...</div>
    </div>
  </div>
</main>

<script>
const API = '';  // same origin

async function checkHealth() {
  try {
    const r = await fetch(API + '/health');
    const d = await r.json();
    const dot = d.model_loaded ? 'dot-ok' : 'dot-warn';
    const label = d.model_loaded ? 'Modelo cargado' : 'Modelo no disponible';
    document.getElementById('healthContent').innerHTML = `
      <span class="status-dot ${dot}"></span><strong>${label}</strong><br>
      <span style="margin-left:13px">v${d.version} · status: ${d.status}</span>
    `;
  } catch(e) {
    document.getElementById('healthContent').innerHTML = `<span class="status-dot dot-err"></span>API no disponible`;
  }
}

async function predict() {
  const btn = document.getElementById('predictBtn');
  btn.disabled = true;
  btn.textContent = 'Calculando...';

  const threshold = parseFloat(document.getElementById('threshold').value);

  const payload = {
    age: parseInt(document.getElementById('age').value),
    gender: document.getElementById('gender').value,
    admission_type_id: parseInt(document.getElementById('admission_type_id').value),
    discharge_disposition_id: parseInt(document.getElementById('discharge_disposition_id').value),
    admission_source_id: parseInt(document.getElementById('admission_source_id').value),
    time_in_hospital: parseInt(document.getElementById('time_in_hospital').value),
    num_lab_procedures: parseInt(document.getElementById('num_lab_procedures').value),
    num_procedures: parseInt(document.getElementById('num_procedures').value),
    num_medications: parseInt(document.getElementById('num_medications').value),
    number_outpatient: parseInt(document.getElementById('number_outpatient').value),
    number_emergency: parseInt(document.getElementById('number_emergency').value),
    number_inpatient: parseInt(document.getElementById('number_inpatient').value),
    number_diagnoses: parseInt(document.getElementById('number_diagnoses').value),
    A1Cresult: document.getElementById('A1Cresult').value,
    insulin: document.getElementById('insulin').value,
    diabetesMed: document.getElementById('diabetesMed').value,
    change: document.getElementById('change').value,
  };

  try {
    const r = await fetch(API + '/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!r.ok) {
      const err = await r.json();
      showError(err.detail || 'Error en la prediccion');
      return;
    }

    const d = await r.json();
    // Override high_risk with local threshold
    const isHigh = d.readmission_probability >= threshold;
    let risk = d.readmission_probability < 0.2 ? 'BAJO' : d.readmission_probability < threshold ? 'MEDIO' : 'ALTO';
    // Adjust if threshold changes the boundary
    if (threshold <= 0.2) risk = d.readmission_probability >= threshold ? 'ALTO' : 'BAJO';

    const pct = Math.round(d.readmission_probability * 100);
    const barColor = risk === 'BAJO' ? 'var(--low)' : risk === 'MEDIO' ? 'var(--mid)' : 'var(--high)';
    const emoji = risk === 'BAJO' ? '✅' : risk === 'MEDIO' ? '⚠️' : '🚨';

    document.getElementById('resultContent').innerHTML = `
      <div style="text-align:center;padding:10px 0">
        <div class="risk-badge risk-${risk}">${emoji} RIESGO ${risk}</div>
        <div style="font-size:2.4rem;font-weight:800;color:${barColor}">${pct}%</div>
        <div style="font-size:0.78rem;color:var(--muted)">probabilidad de reingreso &lt;30 días</div>
      </div>

      <div class="prob-bar-wrap">
        <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:var(--muted);margin-bottom:4px">
          <span>0%</span><span>Umbral: ${Math.round(threshold*100)}%</span><span>100%</span>
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar" style="width:${pct}%;background:${barColor}"></div>
        </div>
        <div style="position:relative;height:16px">
          <div style="position:absolute;left:${Math.round(threshold*100)}%;top:0;transform:translateX(-50%);width:2px;height:12px;background:var(--muted);border-radius:2px"></div>
        </div>
      </div>

      <div class="recommendation">
        <div class="rec-title">📋 Recomendación clínica</div>
        ${d.recommendation}
      </div>

      <div class="result-meta" style="margin-top:14px">
        <strong>Modelo:</strong> ${d.model_name} [${d.model_stage}]<br>
        <strong>Umbral aplicado:</strong> ${threshold.toFixed(2)} (local) / ${d.threshold_used} (servidor)<br>
      </div>
    `;

  } catch(e) {
    showError('No se pudo contactar la API: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '⚡ Predecir Riesgo';
  }
}

function showError(msg) {
  document.getElementById('resultContent').innerHTML = `
    <div class="error-box">❌ ${msg}</div>
    <div style="font-size:0.78rem;color:var(--muted);margin-top:10px">
      Asegúrate de que el modelo esté cargado (ver Estado del Servicio).<br>
      También puedes revisar la <a href="/docs" style="color:var(--accent)">documentación Swagger</a>.
    </div>
  `;
}

// Init
checkHealth();
setInterval(checkHealth, 15000);
</script>
</body>
</html>"""
