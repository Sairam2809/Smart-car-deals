"""Measure server-side app execution; excludes browser, network and dependency setup."""
from pathlib import Path
import json
import sys
import time
from streamlit.testing.v1 import AppTest
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
start=time.perf_counter()
app=AppTest.from_file(str(root/'app/app.py'),default_timeout=90).run()
cold=time.perf_counter()-start
assert not app.exception,[e.message for e in app.exception]
heavy={n:n in sys.modules for n in ['sklearn','matplotlib.pyplot','plotly.express','src.classification','src.clustering']}
warm=[]
for _ in range(3):
    start=time.perf_counter();app.run();warm.append(time.perf_counter()-start)
result={'project':root.name,'first_app_execution_seconds':cold,'warm_reruns_seconds':warm,'loaded_after_landing':heavy}
print(json.dumps(result,indent=2))
