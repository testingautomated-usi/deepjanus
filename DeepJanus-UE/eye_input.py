import json
import cv2
import numpy as np

from os.path import splitext, split

from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import img_to_array
from distance_calculator import calc_distance_total


class Eye:
    def __init__(self, json_path, image_path):
        # model is a json path
        self.model = json_path
        self.image_path = image_path

        self.camera_angle_1 = None
        self.camera_angle_2 = None
        self.eye_angle_1 = None
        self.eye_angle_2 = None
        self.pupil_size = None
        self.iris_size = None
        self.iris_texture = None
        self.skybox_texture = None
        self.skybox_exposure = None
        self.skybox_rotation = None
        self.ambient_intensity = None
        self.light_rotation_angle_1 = None
        self.light_rotation_angle_2 = None
        self.light_intensity = None
        self.primary_skin_texture = None

        self.h_angles_rad_np = None
        self.eye_angles_rad = None

        # reads json data
        self.model_params = self.get_desc(json_path)
        # assign values to model params
        self.extract_params(self.model_params)
        self.is_misb = None

        # IMG, NP
        self.representation = self.get_representation(image_path)
        # TODO: Nargiz check preprocess and reshape
        self.img_np = self.preprocess_img(image_path)

        self.diff = None
        self.predicted_label = None
        self.correctly_classified = None

    def clone(self):
        clone_sample = Eye(self.model, self.image_path)
        return clone_sample

################################################################################
    #TODO: how to read image
    # What do we need it for here? how do we want to store it?
    @staticmethod
    def get_representation(image_path):
        #image = cv2.imread(image_path)
        with open(image_path, 'rb') as f:
            image = f.read()
        return image

    def get_representation_tf_rgb(self, image_path):
        image = load_img(image_path, color_mode = "rgb")
        image = img_to_array(image)
        return image

    def get_representation_tf_gray(self, image_path):
        image = load_img(image_path, color_mode = "grayscale")
        image = img_to_array(image)
        return image

    ################################################################################

    def read_img_cv2_gray(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return img

    def reshape_image(self, img):

        ldmks_caruncle = self.process_json_list(self.model_params['caruncle_2d'], img)

        x = int(ldmks_caruncle[6][0] - 50)
        y = int(ldmks_caruncle[6][1] - 130)
        h = 216
        w = 360

        crop_img = img[y:y + h, x:x + w]

        dim = (60, 36)
        res_img = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)

        return res_img

    def normalise_image(self, img):
        img = img.astype('float32')
        img = img / 255.0
        return img

    def preprocess_img(self, image_path):
        # will be a bit different.
        img = self.read_img_cv2_gray(image_path)

        img = self.reshape_image(img)
        img = self.normalise_image(img)

        img = img.reshape((img.shape[0], img.shape[1], 1))
        img_arr = np.array([img])  # Convert single image to a batch.

        return img_arr

    ################################################################################

    def process_json_list(self, json_list, img):
        ldmks = [eval(s) for s in json_list]
        return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])

    def get_json_data(self, json_file):
        with open(json_file) as data_file:
            data = json.load(data_file)
        return data

    def get_desc(self, json_path):
        data = self.get_json_data(json_path)
        return data

    def extract_params(self, data):
        self.pupil_size = float(data['eye_details']["pupil_size"])
        self.iris_size = float(data['eye_details']["iris_size"])
        self.iris_texture = data['eye_details']["iris_texture"]

        self.skybox_texture = data['lighting_details']['skybox_texture']
        self.skybox_exposure = float(data['lighting_details']['skybox_exposure'])
        self.skybox_rotation = float(data['lighting_details']['skybox_rotation'])

        self.ambient_intensity = float(data['lighting_details']['ambient_intensity'])
        if not isinstance(data['lighting_details']['light_rotation'], list):
            light_rot = eval(data['lighting_details']['light_rotation'])
        else:
            light_rot = data['lighting_details']['light_rotation']
        self.light_rotation_angle_1 = int(light_rot[0])
        self.light_rotation_angle_2 = int(light_rot[1])
        self.light_intensity = float(data['lighting_details']['light_intensity'])

        self.primary_skin_texture = data['eye_region_details']['primary_skin_texture']

        # TODO: Nargiz check this change, before it was crashing
        self.camera_angle_1 = int(float(data['camera_angle_1']))
        self.camera_angle_2 = int(float(data['camera_angle_2']))
        self.eye_angle_1 = int(float(data['eye_angle_1']))
        self.eye_angle_2 = int(float(data['eye_angle_2']))

        h_a_1 = np.radians(self.camera_angle_1)
        h_a_2 = np.radians(self.camera_angle_2)

        self.h_angles_rad_np = np.array([h_a_1, h_a_2])
        self.h_angles_rad_np = np.array([self.h_angles_rad_np])

        e_a_1 = np.radians(self.eye_angle_1)
        e_a_2 = np.radians(self.eye_angle_2)

        self.eye_angles_rad = [e_a_1, e_a_2]
        # self.gaze_direction = data['gaze_direction']

    def create_desc(self):
        self.model_params['eye_details']["pupil_size"] = self.pupil_size
        self.model_params['eye_details']["iris_size"] = self.iris_size
        self.model_params['eye_details']["iris_texture"] = self.iris_texture
        self.model_params['lighting_details']['skybox_texture'] = self.skybox_texture
        self.model_params['lighting_details']['skybox_exposure'] = self.skybox_exposure
        self.model_params['lighting_details']['skybox_rotation'] = self.skybox_rotation
        self.model_params['lighting_details']['ambient_intensity'] = self.ambient_intensity
        self.model_params['lighting_details']['light_rotation'] = [self.light_rotation_angle_1,
                                                                   self.light_rotation_angle_2]
        self.model_params['lighting_details']['light_intensity'] = self.light_intensity
        self.model_params['eye_region_details']['primary_skin_texture'] = self.primary_skin_texture
        self.model_params['camera_angle_1'] = str(self.camera_angle_1)
        self.model_params['camera_angle_2'] = str(self.camera_angle_2)
        self.model_params['eye_angle_1'] = str(self.eye_angle_1)
        self.model_params['eye_angle_2'] = str(self.eye_angle_2)


    # TODO: put the function that compares the label and prediction here (N) / it may be not necessary
    #       It is in distance calculator, do we really need it here?
    # def calculate_loss(self, pred, true):
    #     return calc_angle_distance(pred, true)

    def update_individual(self, json_path):
        self.model = json_path
        mut_img_path = splitext(json_path)[0] + ".jpg"
        self.image_path = mut_img_path
        self.representation = self.get_representation(mut_img_path)
        self.model_params = self.get_desc(json_path)
        self.extract_params(self.model_params)

    def __lt__(self, other):
        return self.image_path < other.image_path

    def __eq__(self, other):
        dist = calc_distance_total(self.model_params, other.model_params)
        return dist == 0.0


if __name__ == "__main__":
    import glob
    from os import makedirs
    from os.path import splitext, exists, join
    from shutil import copyfile
    DATA = 'eye_dataset/'
    OUTPUT_DATASET = 'population01'

    sample_list = glob.glob(DATA + '/*.jpg')

    image_path = sample_list[1]
    path = splitext(image_path)
    json_path = path[0] + ".json"

    sample1:Eye = Eye(json_path, image_path)

    image_path = sample_list[2]
    path = splitext(image_path)
    json_path = path[0] + ".json"

    sample2: Eye = Eye(json_path, image_path)

    list1 = []
    list2 = []
    list1.append(sample1)
    list1.append(sample2)
    list2.append(sample1)

    print(list1)
    print(list2)

    result = np.setdiff1d(list1, list2)
    result2 = np.intersect1d(list1, list2)

    print(result)
    print(result2)

    if not exists(OUTPUT_DATASET):
        makedirs(OUTPUT_DATASET)
    for element in list1:
        element_path = split(element.model)[-1]
        dst = join(OUTPUT_DATASET, element_path)
        copyfile(element.model, dst)
        element_path = split(element.image_path)[-1]
        dst = join(OUTPUT_DATASET, element_path)
        copyfile(element.image_path, dst)