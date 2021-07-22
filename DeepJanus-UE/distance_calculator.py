import json
import random
import math
import numpy as np

from glob import glob
from datetime import datetime
import similarity_charts as sch

# chromosome = {
#     # eye & skin details
#     "pupil_size": "0.01930688",
#     "iris_size": "0.9619961",
#     "iris_texture": "eyeball_green",
#     "primary_skin_texture": "m09_color",
#     # skybox details
#     "skybox_texture": "bergen_2k",
#     "skybox_exposure": "1.191213",
#     "skybox_rotation": "231",
#     "ambient_intensity": "1.026327",
#     # light details
#     "light_rotation_1": "0.8",
#     "light_rotation_2": "189.0",
#     "light_intensity": "0.8476323"
# }


# # datetime object containing current date and time
# now = datetime.now()
# dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
# print("date and time =", dt_string)


def extract_chromosome_params(data):

    json_params = {}

    # camera and eye angles
    # TODO: N int or float?
    json_params["camera_angle_1"] = int(float(data['camera_angle_1']))
    json_params["camera_angle_2"] = int(float(data['camera_angle_2']))
    json_params["eye_angle_1"] = int(float(data['eye_angle_1']))
    json_params["eye_angle_2"] = int(float(data['eye_angle_2']))
    # eye & skin details
    json_params["pupil_size"] = data['eye_details']["pupil_size"]
    json_params["iris_size"] = data['eye_details']["iris_size"]
    json_params["iris_texture"] = data['eye_details']["iris_texture"]
    json_params["primary_skin_texture"] = data['eye_region_details']['primary_skin_texture']
    # skybox details
    json_params["skybox_texture"] = data['lighting_details']['skybox_texture']
    json_params["skybox_exposure"] = data['lighting_details']['skybox_exposure']
    json_params["skybox_rotation"] = data['lighting_details']['skybox_rotation']
    json_params["ambient_intensity"] = data['lighting_details']['ambient_intensity']
    # light details
    if not isinstance(data['lighting_details']['light_rotation'], list):
        light_rot = eval(data['lighting_details']['light_rotation'])
    else:
        light_rot = data['lighting_details']['light_rotation']
    json_params["light_rotation_angle_1"] = light_rot[0]
    json_params["light_rotation_angle_2"] = light_rot[1]
    json_params["light_intensity"] = data['lighting_details']['light_intensity']

    return json_params


def normalise(dist):
    return dist/(dist + 1)


def calc_norm_cont_distance(value1, value2):
    return normalise(abs(float(value1) - float(value2)))


def calc_iris_txtr_distance(value1, value2):
    iris_txtr_distance = 1

    if value1 == value2:
        iris_txtr_distance = 0
    else:
        for dist in sch.iris_txtr_dist:
            if value1 in dist and value2 in dist:
                iris_txtr_distance = dist[2]

    return iris_txtr_distance


def calc_skin_txtr_distance(value1, value2):
    if value1 == value2:
        skin_txtr_distance = 0
    else:
        skin_txtr_distance = 1

    return skin_txtr_distance


def calc_skybox_txtr_distance(value1, value2):
    if value1 == value2:
        skybox_txtr_distance = 0
    else:
        skybox_txtr_distance = 1

    return skybox_txtr_distance


def calc_angle_distance(angles1, angles2):
    x_p = math.sin(angles1[0]) * math.cos(angles1[1])
    y_p = math.sin(angles1[0]) * math.sin(angles1[1])
    z_p = math.cos(angles1[0])

    x_t = math.sin(angles2[0]) * math.cos(angles2[1])
    y_t = math.sin(angles2[0]) * math.sin(angles2[1])
    z_t = math.cos(angles2[0])

    norm_p = math.sqrt(x_p * x_p + y_p * y_p + z_p * z_p)
    norm_t = math.sqrt(x_t * x_t + y_t * y_t + z_t * z_t)

    dot_pt = x_p * x_t + y_p * y_t + z_p * z_t

    angle_value = dot_pt/(norm_p * norm_t)

    angle_value = np.clip(angle_value, a_min=-1, a_max=1)

    loss_val = (math.acos(angle_value))

    return loss_val


def calc_distance_total(chromosome, pop_chromosome, property=None, old_value = None):
    pop_chromosome = extract_chromosome_params(pop_chromosome)
    chromosome = extract_chromosome_params(chromosome)

    # TODO: Normalise or not? Ask PT
    # angles
    camera_angles_chr = (np.radians(chromosome["camera_angle_1"]), np.radians(chromosome["camera_angle_2"]))
    camera_angles_pop_chr = (np.radians(pop_chromosome["camera_angle_1"]), np.radians(pop_chromosome["camera_angle_2"]))
    head_angles_dist = calc_angle_distance(camera_angles_chr, camera_angles_pop_chr)

    eye_angles_chr = (np.radians(chromosome["eye_angle_1"]), np.radians(chromosome["eye_angle_2"]))
    eye_angles_pop_chr = (np.radians(pop_chromosome["eye_angle_1"]), np.radians(pop_chromosome["eye_angle_2"]))
    eye_angles_dist = calc_angle_distance(eye_angles_chr, eye_angles_pop_chr)

    # eye & skin details
    pupil_size_dist = calc_norm_cont_distance(chromosome["pupil_size"], pop_chromosome["pupil_size"])
    iris_texture_dist = calc_iris_txtr_distance(chromosome["iris_texture"], pop_chromosome["iris_texture"])
    primary_skin_texture_dist = calc_skin_txtr_distance(chromosome["primary_skin_texture"], pop_chromosome["primary_skin_texture"])

    # skybox details
    skybox_texture_dist = calc_skybox_txtr_distance(chromosome["skybox_texture"], pop_chromosome["skybox_texture"])
    skybox_exposure_dist = calc_norm_cont_distance(chromosome["skybox_exposure"], pop_chromosome["skybox_exposure"])
    skybox_rotation_dist = calc_norm_cont_distance(chromosome["skybox_rotation"], pop_chromosome["skybox_rotation"])
    ambient_intensity_dist = calc_norm_cont_distance(chromosome["ambient_intensity"], pop_chromosome["ambient_intensity"])

    # light details
    light_rotation_1_dist = calc_norm_cont_distance(chromosome["light_rotation_angle_1"], pop_chromosome["light_rotation_angle_1"])
    light_rotation_2_dist = calc_norm_cont_distance(chromosome["light_rotation_angle_2"], pop_chromosome["light_rotation_angle_2"])
    light_intensity_dist = calc_norm_cont_distance(chromosome["light_intensity"], pop_chromosome["light_intensity"])

    total_dist = 5 * head_angles_dist + 10 * eye_angles_dist + pupil_size_dist + iris_texture_dist + primary_skin_texture_dist + \
                 skybox_texture_dist + 0.5 * skybox_exposure_dist + 0.5 * skybox_rotation_dist + 0.5 * ambient_intensity_dist + \
                 ambient_intensity_dist + light_intensity_dist + light_rotation_1_dist + light_rotation_2_dist

    total_dist = normalise(total_dist)

    # print("Tot before" + str(total_dist))

    if old_value is not None and str(old_value) == str(pop_chromosome[property]):
        #print(property + ": " + str(chromosome[property]))
        total_dist = None
    # print("Tot after" + str(total_dist))

    return total_dist


def find_closest_indv(folder, chromosome, property = None, old_value = None, excluded = set()):

    min_found_distance = np.inf
    closest_indv = []

    path = folder + "\*.json"
    json_files = glob(path)
    #TODO: check
    assert (len(json_files) > 0)

    json_files = set(json_files) - excluded
    assert(len(json_files)>0)

    for json_file in json_files:
        data_file = open(json_file)
        data = json.load(data_file)

        distance = calc_distance_total(chromosome, data, property, old_value)

        if distance is not None:
            if distance < min_found_distance:
                closest_indv.clear()
                min_found_distance = distance
                closest_indv.append(json_file)
            elif distance == min_found_distance:
                closest_indv.append(json_file)
        #else:
        #    print("distance is None")

    # TODO: check
    assert (min_found_distance < np.inf)
    return closest_indv


if __name__ == "__main__":
    from glob import glob
    from os.path import splitext
    DATA = 'eye_dataset/'
    from eye_input import Eye

    sample_list = glob(DATA + '/*.jpg')

    image_path = sample_list[1]
    path = splitext(image_path)
    json_path = path[0] + ".json"

    sample1:Eye = Eye(json_path, image_path)

    image_path = sample_list[2]
    path = splitext(image_path)
    json_path = path[0] + ".json"

    sample2: Eye = Eye(json_path, image_path)

    dist = calc_distance_total(sample1.model_params, sample1.model_params)

    #print("distance from self: "+ str(dist))
    #print("distance > 0 "+str(dist > 0.0))

    dist = calc_distance_total(sample1.model_params, sample2.model_params)

    #print("distance from other: " + str(dist))
    #print("distance > 0 " + str(dist > 0.0))

    closest = find_closest_indv(DATA, sample1.model_params)
    print("closest to "+sample1.model)
    print(closest)

    closest = find_closest_indv(DATA, sample2.model_params)
    print("closest to " + sample2.model)
    print(closest)
