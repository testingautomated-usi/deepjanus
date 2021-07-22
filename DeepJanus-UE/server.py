import subprocess

try:
    print("asdada")
    subprocess.run(["java", "-jar", "D:\\Nargiz\\Sikuli\\sikulixapi-2.0.4.jar", "-s"])


    print("Done")
except subprocess.CalledProcessError as e:
    print(e)