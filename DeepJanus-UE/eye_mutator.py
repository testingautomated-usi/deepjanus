import glob
from os.path import exists, join, splitext, dirname, abspath
from eye_input import Eye
from properties import MUTOPPROB, DATASET, POPSIZE
from utils import *
from chromosome_properties import chromosome_properties_list
from properties import SIKULIX_SCRIPT_PATH, SIKULIX_SCRIPT_FOLDER, \
                        SIKULIX_SCRIPT_NAME, SIKULIX_ANGLES_FILE_NAME, \
                        SIKULIX_SCRIPT_NAME_W_FMT, UNITY_GENERATED_IMGS_PATH
from sikulix import run_sikulix
from distance_calculator import find_closest_indv, calc_distance_total
from copy import deepcopy


class EyeMutator:
    EXCLUDED = set()

    def __init__(self, sample):
        self.eye_sample = sample

    def mutate(self, reference=None):
        old_model_params = deepcopy(self.eye_sample.model_params)
        # Select property to mutate
        property = random.choice(chromosome_properties_list)
        old_value = self.eye_sample.__dict__[property]

        condition = True
        while condition:
            if property in ["camera_angle_1", "camera_angle_2", "eye_angle_1", "eye_angle_2", "light_rotation_angle_2"]:
                new_value = mutate_integer_parameter(property, old_value)
            elif property in ["iris_texture", "skybox_texture", "primary_skin_texture"]:
                new_value = mutate_categorical_parameter(property, old_value)
            elif property in ["skybox_rotation", "light_rotation_angle_1"]:
                new_value = mutate_circular_parameter(property, old_value)
            elif property in ["pupil_size", "iris_size", "skybox_exposure", "ambient_intensity", "light_intensity"]:
                new_value = mutate_continuous_parameter(property, old_value)

            # Overwrite the value of the attribute
            self.eye_sample.__dict__[property] = new_value
            # Create new json description
            self.eye_sample.create_desc()

            # check if the mutation produces a different input than the other input in the pair
            if reference is not None:
                distance_inputs = calc_distance_total(reference.model_params,
                                                      self.eye_sample.model_params)
                if distance_inputs != 0:
                    condition = False
            else:
                condition = False

        condition = True
        while condition:
            imgs_folder = ''.join((UNITY_GENERATED_IMGS_PATH,
                               str(self.eye_sample.__dict__["camera_angle_1"]), "_",
                               str(self.eye_sample.__dict__["camera_angle_2"]), "_",
                               str(self.eye_sample.__dict__["eye_angle_1"]), "_",
                               str(self.eye_sample.__dict__["eye_angle_2"])))

            # TODO: check if folder - excluded is empty(discuss with Nargiz)
            avl_samples = glob(imgs_folder + '/*.json')
            avl_samples = set(avl_samples) - EyeMutator.EXCLUDED

            if ((not exists(imgs_folder)) or (len(avl_samples) < POPSIZE)):
                #print("img folder does not exist: "+str(not exists(imgs_folder)))
                #print("avl samples number: "+str(len(avl_samples)))
                dst = join(SIKULIX_SCRIPT_FOLDER, SIKULIX_SCRIPT_NAME_W_FMT, SIKULIX_ANGLES_FILE_NAME)
                # TODO: Nargiz check the correct position of angles.txt
                file = open(dst, 'w+')
                file.write(str(self.eye_sample.camera_angle_1)+"," +str(self.eye_sample.camera_angle_2)+",0,0\n")
                file.write(str(self.eye_sample.eye_angle_1) + "," + str(self.eye_sample.eye_angle_2) + ",0,0")
                file.close()

                #Generate new images with Sikulix + UnityEyes
                run_sikulix(SIKULIX_SCRIPT_NAME)

                rename_generated_imgs_folder(imgs_folder, self.eye_sample.camera_angle_1, self.eye_sample.camera_angle_2,
                                                    self.eye_sample.eye_angle_1, self.eye_sample.eye_angle_2)

            # Select the closest individual
            #TODO: check
            closest_indv = find_closest_indv(imgs_folder, self.eye_sample.model_params,
                                             property, old_value, excluded = EyeMutator.EXCLUDED)

            assert len(closest_indv) > 0

            closest_indv = random.choice(closest_indv)
            EyeMutator.EXCLUDED.add(closest_indv)

            if reference is not None:
                data = self.eye_sample.get_json_data(closest_indv)
                distance_inputs = calc_distance_total(reference.model_params,
                                                      data)
                if distance_inputs != 0:
                    condition = False
            else:
                condition = False

        self.eye_sample.update_individual(closest_indv)

        self.eye_sample.predicted_label = None
        self.eye_sample.diff = None
        self.eye_sample.correctly_classified = None
        self.eye_sample.is_misb = None


if __name__ == "__main__":
    DATA = 'eye_dataset/'
    sample_list = glob(DATA + '/*.jpg')
    print(len(set(sample_list))>0)

    image_path = sample_list[0]
    path = splitext(image_path)
    json_path = path[0] + ".json"

    image_path2 = sample_list[1]
    path = splitext(image_path2)
    json_path2 = path[0] + ".json"

    sample:Eye = Eye(json_path, image_path)
    reference: Eye = Eye(json_path2, image_path2)
    EyeMutator(sample).mutate(reference=reference)

    exit()

    print(sample.ambient_intensity)
    sample.__dict__["ambient_intensity"] = None
    print(sample.ambient_intensity)
    print(sample.model_params['lighting_details']['ambient_intensity'])
    sample.create_desc()
    print(sample.model_params['lighting_details']['ambient_intensity'])
    for property in chromosome_properties_list:
        sample.__dict__[property] = None
        sample.create_desc()
    print(sample.model_params)


