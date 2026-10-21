"""Execute all notebooks sequentially, preserving actual outputs and failing on errors."""
from pathlib import Path
import os
import sys
import time
import subprocess
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
names=sys.argv[1:]
paths=[ROOT/'notebooks'/n for n in names] if names else sorted((ROOT/'notebooks').glob('*.ipynb'))
for path in paths:
    print('START',path.name,flush=True);start=time.time()
    nb=nbformat.read(path,as_version=4)
    try:
        NotebookClient(nb,timeout=3600,kernel_name='python3',resources={'metadata':{'path':str(ROOT/'notebooks')}}).execute()
    finally:
        nbformat.write(nb,path)
    print('PASS',path.name,round(time.time()-start,1),'seconds',flush=True)
if not names:
    subprocess.run([sys.executable,str(ROOT/'scripts/write_summary.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/export_gui_assets.py')],check=True)
