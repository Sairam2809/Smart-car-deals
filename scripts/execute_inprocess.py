"""Execute real notebook cells without a socket-based kernel (restricted environments)."""
from pathlib import Path
import os
import sys
import time
import traceback
import io
import subprocess
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Image

def show_figures(*args, **kwargs):
    for number in plt.get_fignums():
        buffer=io.BytesIO()
        plt.figure(number).savefig(buffer,format='png',dpi=110,bbox_inches='tight')
        display(Image(data=buffer.getvalue()))
    plt.close('all')
plt.show=show_figures

ROOT=Path(__file__).resolve().parents[1]
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.chdir(ROOT/'notebooks')
paths=[ROOT/'notebooks'/n for n in sys.argv[1:]] if len(sys.argv)>1 else sorted((ROOT/'notebooks').glob('*.ipynb'))
for path in paths:
    print('START',path.name,flush=True);start=time.time()
    InteractiveShell.clear_instance()
    shell=InteractiveShell.instance()
    nb=nbformat.read(path,as_version=4)
    count=0
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        count+=1
        with capture_output() as captured:
            source=cell.source.replace('%matplotlib inline', '# In-process runner captures figures through plt.show().')
            result=shell.run_cell(source,store_history=True)
            show_figures()
        outputs=[]
        if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for rich in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
        cell.outputs=outputs;cell.execution_count=count
        error=result.error_before_exec or result.error_in_exec
        if error:
            cell.outputs.append(nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),traceback=traceback.format_exception(error)))
            nbformat.write(nb,path)
            raise RuntimeError(f'{path.name} cell {count}: {error}') from error
        nbformat.write(nb,path)
        print('CELL',path.name,count,'done',flush=True)
    print('PASS',path.name,round(time.time()-start,1),'seconds',flush=True)
if len(sys.argv)==1:
    subprocess.run([sys.executable,str(ROOT/'scripts/write_summary.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/export_gui_assets.py')],check=True)
