from os.path import exists
from properties import RESULTS_PATH, INTERPRETER
import shutil
import time
from os import makedirs
import subprocess
import re
import datetime
from sikulix import start_sikulix_server, set_sikulix_scripts_home

start = datetime.datetime.now()
print(start)

src = RESULTS_PATH

print("Starting")
#start_sikulix_server()
print("Sikulix server started")
set_sikulix_scripts_home()
print("Sikulix scripts home set")


for i in range(0,10):
#for i in range(10):
    with open("properties.py", "r") as file_obj:
        #file_obj.write(pop_string)
        data = file_obj.read()

    with open("properties.py", "w") as file_obj:
        data = re.sub(r'DATASET = \"population_\d\"', 'DATASET = \"population_'+str(i)+'\"', data)
        file_obj.write(data)

    try:
        subprocess.call([INTERPRETER, "main.py"])
    except Exception as e:
        print(e)
        raise e

    dst = src+str()
    #timestr = str(time.strftime("%Y%m%d-%H%M%S"))
    dst = src+str(i)

    if not exists(dst):
        makedirs(dst)

    shutil.move(src, dst)

end = datetime.datetime.now()
print(end)

diff = end - start
print(diff)

with open("run_time.txt", "w") as text_file:
    text_file.write("Start: %s \n" % str(start))
    text_file.write("END: %s \n" % str(end))
    text_file.write("DIFF: %s \n" % str(diff))