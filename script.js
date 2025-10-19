
// Fixed script.js: use model.weights, map activations, friendly dropdown labels
let modelMeta = null;

async function loadModel() {
  try {
    const res = await fetch('model.json');
    if(!res.ok) throw new Error('Failed to load model.json ('+res.status+')');
    modelMeta = await res.json();
    initUI();
    console.log('Model loaded, features:', modelMeta.feature_names);
  } catch(e) { console.error('loadModel error', e); alert('Failed to load model.json: ' + e.message); }
}

function relu_vec(vec){ return vec.map(v=> Math.max(0, v)); }
function sigmoid_scalar(x){ return 1/(1+Math.exp(-x)); }

// compute z = b + W^T x  where W is [in_dim][out_dim], x is length in_dim
function dense_forward(W, b, x){
  if(!W || W.length === 0) throw new Error('Weight matrix is empty');
  const out_dim = W[0].length;
  const z = new Array(out_dim).fill(0.0);
  for(let k=0;k<out_dim;k++){ z[k] = (b && b[k]) ? b[k] : 0.0; }
  for(let i=0;i<x.length;i++){ const xi = x[i]; const Wi = W[i]; if(!Wi) throw new Error('Weight row missing for input index '+i); for(let k=0;k<out_dim;k++){ z[k] += Wi[k]*xi; } }
  return z;
}

// Build layers array by combining weights + activations (activations stored in modelMeta.layers[].activation)
function getLayers(){
  const layers = [];
  const meta_layers = modelMeta.layers || [];
  if(modelMeta.weights){
    for(let i=0;i<modelMeta.weights.length;i++){
      const W = modelMeta.weights[i].W;
      const b = modelMeta.weights[i].b;
      const act = (meta_layers[i] && meta_layers[i].activation) ? meta_layers[i].activation : (i<modelMeta.weights.length-1 ? 'relu' : 'sigmoid');
      layers.push({W: W, b: b, activation: act});
    }
  } else if(meta_layers.length){
    for(let i=0;i<meta_layers.length;i++){ layers.push(meta_layers[i]); }
  }
  return layers;
}

function predictFromModel(inputObj){
  if(!modelMeta) throw new Error('Model not loaded');
  const featureNames = modelMeta.feature_names;
  const means = modelMeta.means;
  const stds = modelMeta.stds;
  const layers = getLayers();
  if(layers.length === 0) throw new Error('No layers found in model');
  // build normalized input vector (original scale)
  const x = featureNames.map((f,i)=>{ const v = parseFloat(inputObj[f]); return isNaN(v)?0.0: (v - means[i]) / (stds[i] || 1.0); });

  let curr = x;
  for(let li=0; li<layers.length; li++){
    const W = layers[li].W;
    const b = layers[li].b;
    const z = dense_forward(W, b, curr);
    const act = layers[li].activation || 'linear';
    if(li === layers.length-1){
      const out = z.map(v=>sigmoid_scalar(v));
      return out[0];
    } else { curr = relu_vec(z); }
  }
  return 0.0;
}

// Standard ranges (server-side mapping)
const STANDARD = {"age": {"min": 20, "max": 90, "step": 1, "desc": "Age in years"}, "sex": {"min": 0, "max": 1, "step": 1, "desc": "Sex"}, "cp": {"min": 0, "max": 3, "step": 1, "desc": "Chest pain type"}, "trestbps": {"min": 90, "max": 200, "step": 1, "desc": "Resting blood pressure (mm Hg)"}, "chol": {"min": 100, "max": 400, "step": 1, "desc": "Serum cholesterol (mg/dl)"}, "fbs": {"min": 0, "max": 1, "step": 1, "desc": "Fasting blood sugar > 120 mg/dl"}, "restecg": {"min": 0, "max": 2, "step": 1, "desc": "Resting ECG results"}, "thalach": {"min": 60, "max": 240, "step": 1, "desc": "Maximum heart rate achieved"}, "exang": {"min": 0, "max": 1, "step": 1, "desc": "Exercise-induced angina"}, "oldpeak": {"min": 0.0, "max": 6.0, "step": 0.1, "desc": "ST depression induced by exercise relative to rest"}, "slope": {"min": 0, "max": 2, "step": 1, "desc": "Slope of the peak exercise ST segment"}, "ca": {"min": 0, "max": 4, "step": 1, "desc": "Number of major vessels (0\u20134) colored by fluoroscopy"}, "thal": {"min": 1, "max": 3, "step": 1, "desc": "Thalassemia"}};

// Friendly labels for selects
const FRIENDLY = {"sex": {"0": "Female", "1": "Male"}, "cp": {"0": "Typical angina", "1": "Atypical angina", "2": "Non-anginal pain", "3": "Asymptomatic"}, "fbs": {"0": "False (<120)", "1": "True (>=120)"}, "restecg": {"0": "Normal", "1": "ST-T wave abnormality", "2": "Left ventricular hypertrophy"}, "exang": {"0": "No", "1": "Yes"}, "slope": {"0": "Upsloping", "1": "Flat", "2": "Downsloping"}, "ca": {"0": "0 vessels", "1": "1 vessel", "2": "2 vessels", "3": "3 vessels", "4": "4 vessels"}, "thal": {"1": "Normal", "2": "Fixed defect", "3": "Reversible defect"}};

function clamp(v,min,max){ return Math.min(max, Math.max(min, v)); }

// Render UI: sliders for continuous, selects for small integer categorical (range size <= 5)
function initUI(){
  const inputsGrid = document.getElementById('inputs-grid');
  inputsGrid.innerHTML = '';
  const featureNames = modelMeta.feature_names;
  featureNames.forEach(f => {
    const meta = STANDARD[f] || {min:0,max:10,step:1,desc:f};
    const wrapper = document.createElement('div'); wrapper.className='input';
    const label = document.createElement('label');
    const nameSpan = document.createElement('span'); nameSpan.textContent = f.replace(/_/g,' ');
    const valSpan = document.createElement('span'); valSpan.className='val'; valSpan.id = 'val__' + f;
    label.appendChild(nameSpan); label.appendChild(valSpan);

    const rangeSize = meta.max - meta.min;
    // decide control: if integer categorical small range -> select, else slider+number
    if(meta.step === 1 && rangeSize <= 5){
      const select = document.createElement('select');
      select.id = 'sel__' + f;
      const friendly = FRIENDLY[f] || null;
      for(let v = meta.min; v <= meta.max; v++){
        const opt = document.createElement('option'); opt.value = v; opt.textContent = friendly && friendly[v] ? friendly[v] + ' ('+v+')' : v; select.appendChild(opt);
      }
      select.value = Math.round((meta.min + meta.max)/2);
      select.addEventListener('change', ()=>{ document.getElementById('val__'+f).textContent = select.value; });
      wrapper.appendChild(label); wrapper.appendChild(select);
      const desc = document.createElement('div'); desc.className='desc'; desc.textContent = meta.desc + ' (options: ' + meta.min + '–' + meta.max + ')';
      wrapper.appendChild(desc);
      inputsGrid.appendChild(wrapper);
      document.getElementById('val__'+f).textContent = select.value;
    } else {
      const range = document.createElement('input'); range.type = 'range'; range.id = 'f__' + f; range.min = meta.min; range.max = meta.max; range.step = meta.step; range.value = Math.round((meta.min + meta.max)/2);
      const number = document.createElement('input'); number.type = 'number'; number.id = 'num__' + f; number.min = meta.min; number.max = meta.max; number.step = meta.step; number.value = range.value;
      range.addEventListener('input', ()=>{ number.value = range.value; document.getElementById('val__'+f).textContent = range.value; });
      number.addEventListener('change', ()=>{ let v=parseFloat(number.value); if(isNaN(v)) v=meta.min; v=clamp(v,meta.min,meta.max); number.value=v; range.value=v; document.getElementById('val__'+f).textContent=v; });
      const desc = document.createElement('div'); desc.className='desc'; desc.textContent = meta.desc + ' (range: ' + meta.min + '–' + meta.max + ')';
      wrapper.appendChild(label); wrapper.appendChild(range); wrapper.appendChild(number); wrapper.appendChild(desc);
      inputsGrid.appendChild(wrapper);
      document.getElementById('val__'+f).textContent = number.value;
    }
  });

  // ensure buttons exist then wire listeners
  const predictBtn = document.getElementById('predict-btn');
  const fillBtn = document.getElementById('fill-sample');
  const resetBtn = document.getElementById('reset-btn');
  if(predictBtn){ predictBtn.removeEventListener('click', onPredict); predictBtn.addEventListener('click', onPredict); }
  if(fillBtn){ fillBtn.removeEventListener('click', fillSample); fillBtn.addEventListener('click', fillSample); }
  if(resetBtn){ resetBtn.removeEventListener('click', resetForm); resetBtn.addEventListener('click', resetForm); }

  // make sure form doesn't submit
  const form = document.getElementById('input-form');
  if(form) form.addEventListener('submit', (e)=> e.preventDefault());
}

function getCurrentInputs(){ const featureNames = modelMeta.feature_names; const inputs = {}; featureNames.forEach(f=>{ const sel = document.getElementById('sel__'+f); if(sel) inputs[f] = sel.value; else inputs[f] = document.getElementById('num__'+f).value || document.getElementById('f__'+f).value; }); return inputs; }

function fillSample(){ const featureNames = modelMeta.feature_names; featureNames.forEach(f=>{ const meta = STANDARD[f] || {min:0,max:10}; const mid = Math.round((meta.min + meta.max)/2); const sel = document.getElementById('sel__'+f); if(sel){ sel.value = mid; document.getElementById('val__'+f).textContent = sel.value; } else { document.getElementById('num__'+f).value = mid; document.getElementById('f__'+f).value = mid; document.getElementById('val__'+f).textContent = mid; } }); document.getElementById('result').classList.add('hidden'); document.getElementById('explain').classList.add('hidden'); }

function resetForm(){ const featureNames = modelMeta.feature_names; featureNames.forEach(f=>{ const meta = STANDARD[f] || {min:0,max:10}; const sel = document.getElementById('sel__'+f); const val = meta.min; if(sel){ sel.value = val; document.getElementById('val__'+f).textContent = sel.value; } else { document.getElementById('num__'+f).value = val; document.getElementById('f__'+f).value = val; document.getElementById('val__'+f).textContent = val; } }); document.getElementById('result').classList.add('hidden'); document.getElementById('explain').classList.add('hidden'); }

function onPredict(){ try { const inputs = getCurrentInputs(); for(const k in inputs){ if(inputs[k] === '' || inputs[k] === null){ alert('Please provide all inputs.'); return; } if(isNaN(parseFloat(inputs[k]))){ alert('Invalid number for ' + k); return; } } const prob = predictFromModel(inputs); const pct = Math.round(prob*100); const resDiv = document.getElementById('result'); resDiv.classList.remove('hidden'); resDiv.innerHTML = `<strong>Estimated risk:</strong> ${pct}% <br/><small>Probability: ${prob.toFixed(3)}</small>`; computeLocalSensitivity(inputs); } catch(e) { console.error('onPredict error', e); alert('Prediction failed: ' + e.message); } }

function computeLocalSensitivity(inputs){ const featureNames = modelMeta.feature_names; const sensitivities = []; featureNames.forEach(f=>{ const meta = STANDARD[f] || {min:0,max:10}; const range = meta.max - meta.min || 1; const delta = Math.max(range*0.05, meta.step || 1); const orig = parseFloat(inputs[f]); const plus = Object.assign({}, inputs); plus[f] = orig + delta; const minus = Object.assign({}, inputs); minus[f] = orig - delta; const pPlus = predictFromModel(plus); const pMinus = predictFromModel(minus); const grad = (pPlus - pMinus) / (2*delta); const effect = grad * orig; sensitivities.push({feature:f, grad:grad, effect:effect, absEffect:Math.abs(effect)}); }); sensitivities.sort((a,b)=>b.absEffect - a.absEffect); renderExplain(sensitivities.slice(0,6)); }

function renderExplain(items){ const ex = document.getElementById('explain'); const list = document.getElementById('explain-list'); list.innerHTML = ''; if(items.length === 0) { ex.classList.add('hidden'); return; } ex.classList.remove('hidden'); const maxAbs = Math.max(...items.map(it=>Math.abs(it.effect))) || 1; items.forEach(it=>{ const wrap = document.createElement('div'); wrap.className = 'explain-item'; const name = document.createElement('div'); name.className='feature-name'; name.textContent = it.feature.replace(/_/g,' '); const bar = document.createElement('div'); bar.className='bar'; const span = document.createElement('span'); const frac = Math.abs(it.effect)/maxAbs; span.style.width = Math.max(6, Math.round(frac*100)) + '%'; span.className = it.effect >= 0 ? 'effect-pos' : 'effect-neg'; bar.appendChild(span); const txt = document.createElement('div'); txt.style.minWidth='90px'; txt.innerHTML = (it.effect>=0?'+':'') + (it.effect).toFixed(4); wrap.appendChild(name); wrap.appendChild(bar); wrap.appendChild(txt); list.appendChild(wrap); }); }

document.addEventListener('DOMContentLoaded', ()=>{ loadModel(); });
