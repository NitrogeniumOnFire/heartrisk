
let modelMeta = null;

async function loadModel() {
  const res = await fetch('model.json');
  if(!res.ok) throw new Error('Failed to load model.json: ' + res.status);
  modelMeta = await res.json();
  initUI();
}

function relu(x){ return x>0?x:0; }
function sigmoid(x){ return 1/(1+Math.exp(-x)); }

function matVecMul(W, x){
  const out_dim = W[0].length;
  const out = new Array(out_dim).fill(0.0);
  for(let i=0;i<x.length;i++){ const xi = x[i]; const row = W[i]; for(let j=0;j<out_dim;j++){ out[j] += row[j]*xi; } }
  return out;
}

function addBias(vec, b){ return vec.map((v,i)=>v + b[i]); }

function applyActivation(vec, name){
  if(name === 'relu') return vec.map(v=>relu(v));
  if(name === 'sigmoid') return vec.map(v=>sigmoid(v));
  return vec;
}

function predictFromModel(inputObj){
  const featureNames = modelMeta.feature_names;
  const means = modelMeta.means;
  const stds = modelMeta.stds;
  const weights = modelMeta.weights;
  const activations = modelMeta.layers.map(l => l.activation || 'linear');
  const x = featureNames.map((f,i)=>{
    const v = parseFloat(inputObj[f]);
    return isNaN(v)?0.0: (v - means[i]) / stds[i];
  });
  let out = x;
  for(let li=0; li<weights.length; li++){ const W = weights[li].W; const b = weights[li].b; out = matVecMul(W, out); out = addBias(out, b); out = applyActivation(out, activations[li]); }
  let prob = out[0];
  if(typeof prob !== 'number') prob = parseFloat(prob);
  prob = Math.min(1, Math.max(0, prob));
  return prob;
}

// Standard ranges (from server-side mapping)
const STANDARD = 
{"age": {"min": 20, "max": 90, "step": 1, "desc": "Age in years"}, "sex": {"min": 0, "max": 1, "step": 1, "desc": "Sex (0 = female, 1 = male)"}, "cp": {"min": 0, "max": 3, "step": 1, "desc": "Chest pain type (0\u20133)"}, "trestbps": {"min": 90, "max": 200, "step": 1, "desc": "Resting blood pressure (mm Hg)"}, "chol": {"min": 100, "max": 400, "step": 1, "desc": "Serum cholesterol (mg/dl)"}, "fbs": {"min": 0, "max": 1, "step": 1, "desc": "Fasting blood sugar > 120 mg/dl (1 = true)"}, "restecg": {"min": 0, "max": 2, "step": 1, "desc": "Resting electrocardiographic results (0\u20132)"}, "thalach": {"min": 60, "max": 240, "step": 1, "desc": "Maximum heart rate achieved"}, "exang": {"min": 0, "max": 1, "step": 1, "desc": "Exercise-induced angina (1 = yes)"}, "oldpeak": {"min": 0.0, "max": 6.0, "step": 0.1, "desc": "ST depression induced by exercise relative to rest"}, "slope": {"min": 0, "max": 2, "step": 1, "desc": "Slope of the peak exercise ST segment (0\u20132)"}, "ca": {"min": 0, "max": 4, "step": 1, "desc": "Number of major vessels (0\u20134) colored by fluoroscopy"}, "thal": {"min": 1, "max": 3, "step": 1, "desc": "Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)"}};
function clamp(v,min,max){ return Math.min(max, Math.max(min, v)); }

function initUI(){
  const inputsGrid = document.getElementById('inputs-grid');
  inputsGrid.innerHTML = '';
  const featureNames = modelMeta.feature_names;
  featureNames.forEach(f => {
    const meta = STANDARD[f] || {min:0,max:10,step:1,desc:f};
    const wrapper = document.createElement('div');
    wrapper.className = 'input';
    const label = document.createElement('label');
    const nameSpan = document.createElement('span');
    nameSpan.textContent = f.replace(/_/g,' ');
    const valSpan = document.createElement('span');
    valSpan.className = 'val';
    valSpan.id = 'val__' + f;
    label.appendChild(nameSpan); label.appendChild(valSpan);
    const range = document.createElement('input');
    range.type = 'range';
    range.id = 'f__' + f;
    range.min = meta.min;
    range.max = meta.max;
    range.step = meta.step;
    range.value = Math.round((meta.min + meta.max)/2 * (1/meta.step)) / (1/meta.step);
    range.addEventListener('input', ()=>{ document.getElementById('val__'+f).textContent = range.value; });
    const number = document.createElement('input');
    number.type = 'number';
    number.id = 'num__' + f;
    number.min = meta.min; number.max = meta.max; number.step = meta.step;
    number.value = range.value;
    // sync
    range.addEventListener('input', ()=>{ number.value = range.value; document.getElementById('val__'+f).textContent = range.value; });
    number.addEventListener('change', ()=>{ let v = parseFloat(number.value); if(isNaN(v)) v = meta.min; v = clamp(v, meta.min, meta.max); number.value = v; range.value = v; document.getElementById('val__'+f).textContent = v; });
    // description
    const desc = document.createElement('div'); desc.className='desc'; desc.textContent = meta.desc + ' (range: ' + meta.min + '–' + meta.max + ')';
    // append
    wrapper.appendChild(label); wrapper.appendChild(range); wrapper.appendChild(number); wrapper.appendChild(desc);
    inputsGrid.appendChild(wrapper);
    document.getElementById('val__'+f).textContent = number.value;
  });

  document.getElementById('predict-btn').addEventListener('click', onPredict);
  document.getElementById('fill-sample').addEventListener('click', fillSample);
  document.getElementById('reset-btn').addEventListener('click', resetForm);
}

function getCurrentInputs(){
  const featureNames = modelMeta.feature_names;
  const inputs = {};
  featureNames.forEach(f=>{ const v = document.getElementById('num__'+f).value; inputs[f] = v; });
  return inputs;
}

function fillSample(){
  // fill with midpoint values
  const featureNames = modelMeta.feature_names;
  featureNames.forEach(f=>{ const meta = STANDARD[f] || {min:0,max:10}; const val = Math.round((meta.min + meta.max)/2); document.getElementById('num__'+f).value = val; document.getElementById('f__'+f).value = val; document.getElementById('val__'+f).textContent = val; });
}

function resetForm(){
  const featureNames = modelMeta.feature_names;
  featureNames.forEach(f=>{ const meta = STANDARD[f] || {min:0,max:10}; const val = meta.min; document.getElementById('num__'+f).value = val; document.getElementById('f__'+f).value = val; document.getElementById('val__'+f).textContent = val; });
  document.getElementById('result').classList.add('hidden');
  document.getElementById('explain').classList.add('hidden');
}

async function onPredict(){
  const inputs = getCurrentInputs();
  // basic validation: ensure numeric
  for(const k in inputs){
    if(inputs[k] === '' || inputs[k] === null){ alert('Please provide all inputs.'); return; }
    if(isNaN(parseFloat(inputs[k]))){ alert('Invalid number for ' + k); return; }
  }
  const prob = predictFromModel(inputs);
  const pct = Math.round(prob*100);
  const resDiv = document.getElementById('result');
  resDiv.classList.remove('hidden');
  resDiv.innerHTML = `<strong>Estimated risk:</strong> ${pct}% <br/><small>Probability: ${prob.toFixed(3)}</small>`;
  // Compute explainability via sensitivity
  computeLocalSensitivity(inputs, prob);
}

function computeLocalSensitivity(inputs, baseProb){
  const featureNames = modelMeta.feature_names;
  const sensitivities = [];
  featureNames.forEach(f=>{
    const meta = STANDARD[f] || {min:0,max:10};
    const range = meta.max - meta.min || 1;
    const delta = Math.max(range*0.05, meta.step || 1);
    const orig = parseFloat(inputs[f]);
    const plus = Object.assign({}, inputs); plus[f] = orig + delta;
    const minus = Object.assign({}, inputs); minus[f] = orig - delta;
    const pPlus = predictFromModel(plus);
    const pMinus = predictFromModel(minus);
    const grad = (pPlus - pMinus) / (2*delta);
    const effect = grad * orig;
    sensitivities.push({feature:f, grad:grad, effect:effect, absEffect:Math.abs(effect)});
  });
  sensitivities.sort((a,b)=>b.absEffect - a.absEffect);
  const top = sensitivities.slice(0,6);
  renderExplain(top);
}

function renderExplain(items){
  const ex = document.getElementById('explain');
  const list = document.getElementById('explain-list');
  list.innerHTML = '';
  if(items.length === 0){ ex.classList.add('hidden'); return; }
  ex.classList.remove('hidden');
  const maxAbs = Math.max(...items.map(it=>Math.abs(it.effect))) || 1;
  items.forEach(it=>{
    const wrap = document.createElement('div'); wrap.className = 'explain-item';
    const name = document.createElement('div'); name.className='feature-name'; name.textContent = it.feature.replace(/_/g,' ');
    const bar = document.createElement('div'); bar.className='bar';
    const span = document.createElement('span');
    const frac = Math.abs(it.effect)/maxAbs;
    span.style.width = Math.max(6, Math.round(frac*100)) + '%';
    span.className = it.effect >= 0 ? 'effect-pos' : 'effect-neg';
    bar.appendChild(span);
    const txt = document.createElement('div'); txt.style.minWidth='90px'; txt.innerHTML = (it.effect>=0?'+':'') + (it.effect).toFixed(4);
    wrap.appendChild(name); wrap.appendChild(bar); wrap.appendChild(txt);
    list.appendChild(wrap);
  });
}

loadModel().catch(e=>{ console.error(e); alert('Failed to load model.json: ' + e.message); });
