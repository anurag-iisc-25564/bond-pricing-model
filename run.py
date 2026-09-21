import subprocess
import sys
import venv
from pathlib import Path

BASE_DIR =Path(__file__).resolve().parent
VENV_DIR =(BASE_DIR/".venv")

if not VENV_DIR.exists():
    print("Creating virtual environment.....")
    venv.create(VENV_DIR,with_pip=True)

if sys.platform =="win32":
    python = VENV_DIR/"Scripts"/"Python.exe"
else:
    python = VENV_DIR/"bin"/"python"

print("Installing required libraries......")

subprocess.check_call([
    str(python),
    "-m",
    "pip",
    "install",
    "-r",
    str(BASE_DIR/"requirements.txt")
])

print("Running the program.......")

subprocess.check_call([
    str(python),
    str(BASE_DIR/"code_file.py")
])

