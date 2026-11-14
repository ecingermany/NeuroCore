"""Freeze the original JS N128 engine on public synthetic raw samples."""
import json
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
from synthetic import signals

raw = signals(seconds=183).tolist()
source = Path(r'Z:\Work\BQNF\NHS data collection tool\fft-live')
script = '''const fs=require('fs');
const input=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));
const D=require(process.argv[2]+'/dynamic-core.js');
const I=require(process.argv[2]+'/dynamic-input.js');
const e=new D.Engine();
const results=input.map((pair,i)=>{
 const f=I.adaptFrame({a:pair[0],b:pair[1]}, {sourceType:0x82000000,rawCount:256,fftN:128,rawRange:256});
 return e.process(f.a,f.b,i<61?1:i<122?4:2,D.rawGate(f.gateA,f.gateB).pass);
});
process.stdout.write(JSON.stringify(results));'''
input_path = ROOT/'local/synthetic-monitor-input.json'
input_path.write_text(json.dumps(raw), encoding='utf8')
data = subprocess.check_output(['node', '-e', script, str(input_path), str(source)], text=True)
(ROOT/'tests/golden/monitor-js.json').write_text(json.dumps(dict(seed=20261114, results=json.loads(data))), encoding='utf8')
(ROOT/'docs/monitor-reference.json').write_text(json.dumps(dict(
    profile='monitorr-fa398d9a-n128-hann-reconstruction-v1',
    source='NHS local working tree, existing mode5 extension present before extraction',
    compared_modes=[1,4,2], frames=183,
    sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
            for name in ('dynamic-core.js','dynamic-input.js')}
), indent=2)+'\n', encoding='utf8')
print('Original JS engine: 183-frame public synthetic baseline saved')
