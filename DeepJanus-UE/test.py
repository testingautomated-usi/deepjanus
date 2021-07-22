import subprocess

try:
    print("server running")
    subprocess.run(["java", "-jar", "Sikuli-jars//sikulixapi-2.0.4.jar", "-s"])
    print("Done")
except subprocess.CalledProcessError as e:
    print(e)
