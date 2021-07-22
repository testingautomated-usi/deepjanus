# Getting the angles from the file
from os.path import join

BASE = "C://"

SIKULIX_SCRIPT_FOLDER = "sikulix_scripts"

SIKULIX_SCRIPT_NAME_W_FMT = "unityeyes.sikuli"

SIKULIX_ANGLES_FILE_NAME = "angles.txt"

dst = join(BASE, SIKULIX_SCRIPT_FOLDER, SIKULIX_SCRIPT_NAME_W_FMT, SIKULIX_ANGLES_FILE_NAME)
with open(dst) as f:
    content = [i.strip() for i in f.readlines()]
    cam_angles = content[0]
    cam_angles = cam_angles.split(',')
    eye_angles = content[1]
    eye_angles = eye_angles.split(',')
    c_pitch = cam_angles[0]
    c_yaw = cam_angles[1]
    e_pitch = eye_angles[0]
    e_yaw = eye_angles[1]

#exit()
# Starting Unity eyes
doubleClick("eye.png")
wait(3)

# Proceed
click("play.png")
wait(6)

# Input camera angles
cam_input = str(c_pitch)+","+str(c_yaw)+",0,0"
click("first.png")
type(cam_input)

# Input eye agnles
eye_input = str(e_pitch)+","+str(e_yaw)+",0,0"
click("second.png")
type(eye_input)


# Start generating images
click("start.png")
wait(5)

# Stop generating images
click("x.png")
#wait(0.5)
