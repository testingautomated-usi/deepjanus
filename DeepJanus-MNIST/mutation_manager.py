import random
import xml.etree.ElementTree as ET
import re
import numpy as np
from random import randint, uniform
from config import MUTLOWERBOUND, MUTUPPERBOUND, \
    MUTOFPROB, NUMBER_OF_POINTS, SQUARE_SIZE

from attention_maps import AM_get_attetion_svg_points_images_mth1, \
    AM_get_attetion_svg_points_images_mth2, \
    AM_get_attetion_svg_points_images_mth5

from predictor import Predictor
from attention_manager import AttentionManager

NAMESPACE = '{http://www.w3.org/2000/svg}'


def apply_displacement_to_mutant(value, extent):
    displ = uniform(MUTLOWERBOUND, MUTUPPERBOUND) * extent
    if random.uniform(0, 1) >= MUTOFPROB:
        result = float(value) + displ
    else:
        result = float(value) - displ
    return repr(result)


def apply_mutoperator1(input_img, svg_path, extent):

    while(True):
        # find all the vertexes
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        segments = pattern.findall(svg_path)

        svg_iter = re.finditer(pattern, svg_path)
        # chose a random vertex
        num_matches = len(segments) * 2

        random_coordinate_index = randint(0, num_matches - 1)
        # print(random_coordinate_index)

        vertex = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index / 2)))
        group_index = (random_coordinate_index % 2) + 1

        value = apply_displacement_to_mutant(vertex.group(group_index), extent)

        if 0 <= float(value) <= 28:
            break

    path = svg_path[:vertex.start(group_index)] + value + svg_path[vertex.end(group_index):]
    return path


def apply_mutoperator2(input_img, svg_path, extent):
    # find all the vertexes
    pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
    segments = pattern.findall(svg_path)

    # chose a random control point
    num_matches = len(segments) * 4
    path = svg_path
    if num_matches > 0:
        random_coordinate_index = randint(0, num_matches - 1)
        svg_iter = re.finditer(pattern, svg_path)
        control_point = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index/4)))
        group_index = (random_coordinate_index % 4) + 1
        value = apply_displacement_to_mutant(control_point.group(group_index), extent)
        path = svg_path[:control_point.start(group_index)] + value + svg_path[control_point.end(group_index):]
    else:
        print("ERROR")
        print(svg_path)
    return path


def get_attetion_region_prob(xai_image, svg_path_list, sqr_size):
    x_dim = xai_image.shape[0]
    y_dim = xai_image.shape[1]

    if sqr_size == 3:
        y_border_up = -1
        y_border_bottom = 1
        x_border_right = 1
        x_border_left = -1
    elif sqr_size == 5:
        y_border_up = -2
        y_border_bottom = 2
        x_border_right = 2
        x_border_left = -2
    else:
        print("Choose a valid value for square_size (sqr_size): 3 or 5")
        return 0

    xai_list = []
    for pos in svg_path_list:
        x_sqr_pos = int(pos[0])
        y_sqr_pos = int(pos[1])
        sum_xai = 0
        for y_in_sqr in range(y_border_up, y_border_bottom + 1):
            y_pixel_pos = y_sqr_pos + y_in_sqr
            if 0 <= y_pixel_pos <= y_dim - 1:
                for x_in_sqr in range(x_border_left, x_border_right + 1):
                    x_pixel_pos = x_sqr_pos + x_in_sqr
                    if 0 <= x_pixel_pos <= x_dim - 1:
                        sum_xai += xai_image[y_pixel_pos][x_pixel_pos]
        xai_list.append(sum_xai)

    sum_xai_list = sum(xai_list)
    list_of_weights = []
    list_of_probabilities = []

    for sum_value, pos in zip(xai_list, svg_path_list):
        list_of_weights.append(np.exp((sum_value / sum_xai_list) * 100))

    sum_weights_list = sum(list_of_weights)
    for weight in list_of_weights:
        list_of_probabilities.append(weight / sum_weights_list)

    return list_of_weights, list_of_probabilities


def AM_get_attetion_svg_points_images_prob(images, svg_path, xai):
    """
    AM_get_attetion_svg_points_images_mth2 Calculate the attetion score around each SVG path point and return a list of points (tuples) and the respective non-uniform distribution weights for all the SVG path points

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param sqr_size: X and Y size of the square region
    :param model: The model object that will predict the value of the digit in the image
    :return: A a list of points (tuples) and the respective non-uniform distribution weights for all the SVG path points. A well detailed explanation about the structure of the list returned is described at the end of this function.
    """
    import copy
    #copy_images = copy.deepcopy(images)
    #xai = AttentionManager.compute_attention_maps(copy_images)

    #print(xai[0].shape)
    #print(attmap.shape)
    #exit()

    pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
    ControlPoints = pattern.findall(svg_path)
    controlPoints = [(float(i[0]), float(i[1])) for i in ControlPoints]
    if len(ControlPoints) != 0:
        #weight_list, prob_list = get_attetion_region_prob(xai[0], controlPoints, SQUARE_SIZE)
        weight_list, prob_list = get_attetion_region_prob(xai, controlPoints, SQUARE_SIZE)
    else:
        return None, "NA", xai, controlPoints

    return weight_list, controlPoints


def apply_mutoperator_roulette_attention(input_img, svg_path, extent, attmap):
    list_of_weights, original_svg_points = AM_get_attetion_svg_points_images_prob(input_img, svg_path, attmap)
    # TODO: check if the image is valid
    if list_of_weights is not None:
        original_point = random.choices(population=original_svg_points, weights=list_of_weights, k=1)[0]
        original_coordinate = random.choice(original_point)
        mutated_coordinate = apply_displacement_to_mutant(original_coordinate, extent)
        path = svg_path.replace(str(original_coordinate), str(mutated_coordinate))
        return path
    else:
        # TODO: this else branch would start an infinite loop
        return svg_path


def mutate(input_img, svg_desc, attmap, operator_name, mutation_extent):
    root = ET.fromstring(svg_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    mutant_vector = svg_path

    if operator_name == 1:
        mutant_vector = apply_mutoperator1(input_img, svg_path, mutation_extent)
    elif operator_name == 2:
        mutant_vector = apply_mutoperator2(input_img, svg_path, mutation_extent)
    elif operator_name == 3:
        #mutant_img = input_img.reshape(1, 28, 28)
        #mutant_vector = apply_mutoperator_attention_2(mutant_img, svg_path, mutation_extent)
        #mutant_vector = apply_mutoperator_roulette_attention(input_img, svg_path, mutation_extent)
        mutant_vector = apply_mutoperator_roulette_attention(input_img, svg_path, mutation_extent, attmap)
    return mutant_vector


def generate(svg_desc, operator_name):
    root = ET.fromstring(svg_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    if operator_name == 1:
        vector1, vector2 = apply_operator1(svg_path)
    elif operator_name == 2:
        vector1, vector2 = apply_operator2(svg_path)
    return vector1, vector2


def apply_displacement(value):
    displ = uniform(MUTLOWERBOUND, MUTUPPERBOUND)
    result = float(value) + displ
    difference = float(value) - displ
    return repr(result), repr(difference)


def apply_operator1(svg_path):
    while(True):
        # find all the vertexes
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        segments = pattern.findall(svg_path)
        svg_iter = re.finditer(pattern, svg_path)
        # chose a random vertex
        num_matches = len(segments) * 2

        random_coordinate_index = randint(0, num_matches - 1)

        vertex = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index / 2)))
        group_index = (random_coordinate_index % 2) + 1

        value1, value2 = apply_displacement(vertex.group(group_index))

        if 0 <= float(value1) <= 28 and 0 <= float(value2) <= 28:
            break

    path1 = svg_path[:vertex.start(group_index)] + value1 + svg_path[vertex.end(group_index):]
    path2 = svg_path[:vertex.start(group_index)] + value2 + svg_path[vertex.end(group_index):]
    return path1, path2


def apply_operator2(svg_path):
    # find all the vertexes
    pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
    segments = pattern.findall(svg_path)

    # chose a random control point
    num_matches = len(segments) * 4
    path1 = svg_path
    path2 = svg_path
    if num_matches > 0:
        random_coordinate_index = randint(0, num_matches - 1)
        svg_iter = re.finditer(pattern, svg_path)
        control_point = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index/4)))
        group_index = (random_coordinate_index % 4) + 1
        value1, value2 = apply_displacement(control_point.group(group_index))
        path1 = svg_path[:control_point.start(group_index)] + value1 + svg_path[control_point.end(group_index):]
        path2 = svg_path[:control_point.start(group_index)] + value2 + svg_path[control_point.end(group_index):]
    else:
        print("ERROR")
        print(svg_path)
    return path1, path2
