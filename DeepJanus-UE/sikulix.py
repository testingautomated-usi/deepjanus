import subprocess
import requests
import os

from properties import SIKULIX_API_PATH, SIKULIX_REST, SIKULIX_SCRIPTS_HOME

def start_sikulix_server():
    try:
        subprocess.run(["java", "-jar", SIKULIX_API_PATH, "-s"])#, check=True, capture_output=True
        print("Done")
    except subprocess.CalledProcessError as e:
        print(e)


def set_sikulix_scripts_home():
    request = SIKULIX_REST + "scripts/" + SIKULIX_SCRIPTS_HOME
    try:
        r = requests.get(request)
        print(r)
    except requests.exceptions.RequestException as e:
        print(e)
        # raise SystemExit(e)


def run_sikulix(script_name):
    res = run_sikulix_script(script_name)

    if res == 1:
        try:
            os.system("TASKKILL /F /IM unityeyes.exe")
        except:
            pass
        run_sikulix(script_name)


def run_sikulix_script(script_name):
    request = SIKULIX_REST + "run/" + script_name
    try:
        r = requests.get(request)
        status = str(r._content)
        status = status.replace('PASS 200 runScript: returned: ', '')
        status = status[2]
        return int(status)
    except requests.exceptions.RequestException as e:
        print(e)
        return 1