from config import INTERPRETER
import subprocess

for i in range(10):
    subprocess.call([INTERPRETER, "main.py"])
