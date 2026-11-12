"""Publish the validated full-turn comparison into the sibling design docs."""
from pathlib import Path
import runpy
publisher=Path(__file__).resolve().parents[2]/'spark2_design_docs/review/full_rotation/publish.py'
runpy.run_path(str(publisher),run_name='__main__')
