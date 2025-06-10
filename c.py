import nbconvert
import nbformat

notebook_filename = "Classical_ML.ipynb"
with open(notebook_filename, "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

script_exporter = nbconvert.PythonExporter()
script, _ = script_exporter.from_notebook_node(notebook)

with open("Classical_ML.py", "w", encoding="utf-8") as f:
    f.write(script)
