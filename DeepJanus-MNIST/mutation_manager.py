import random
import xml.etree.ElementTree as ET
import re
from random import randint, uniform
from config import MUTLOWERBOUND, MUTUPPERBOUND, MUTOFPROB

from attention_maps import AM_get_attetion_svg_points_images_mth1, \
    AM_get_attetion_svg_points_images_mth2, \
    AM_get_attetion_svg_points_images_mth5

from predictor import Predictor

NAMESPACE = '{http://www.w3.org/2000/svg}'

def apply_displacement_to_mutant_2(list_of_points, extent):
    displ = uniform(MUTLOWERBOUND, MUTUPPERBOUND) * extent
    x_or_y = random.choice((0,1))
    y_or_x = (x_or_y-1) * -1
    list_of_mutated_coordinates_string = []
    coordinate_matutated = [0,0]
    for point in list_of_points:
        coordinate_matutated[y_or_x] = point[y_or_x]
        value = point[x_or_y]        
        if random.uniform(0, 1) >= MUTOFPROB:
            result = float(value) + displ
            coordinate_matutated[x_or_y] = result
            list_of_mutated_coordinates_string.append(str(coordinate_matutated[0])+","+str(coordinate_matutated[1]))
        else:
            result = float(value) - displ
            coordinate_matutated[x_or_y] = result
            list_of_mutated_coordinates_string.append(str(coordinate_matutated[0])+","+str(coordinate_matutated[1]))
    return list_of_mutated_coordinates_string

def apply_mutoperator_attention_2(input_img, svg_path, extent):
    # start_time = time.time()
    attention_mth = 5
    
    if attention_mth != 2:
        if attention_mth == 1:
            list_of_points_inside_square_attention_patch, elapsed_time = AM_get_attetion_svg_points_images_mth1(input_img, 3, 3, svg_path)            
        elif attention_mth == 5:
            list_of_points_inside_square_attention_patch, elapsed_time = AM_get_attetion_svg_points_images_mth5(input_img, 4, svg_path)
        else:
            print("Choose a valid attention_mth option in mutation_manager.py")
            
        list_of_mutated_coordinates_string = apply_displacement_to_mutant_2(list_of_points_inside_square_attention_patch[0], extent)
            
        # print("LIMCS", list_of_mutated_coordinates_string)  
        
        originalPath = svg_path
        list_of_points = list_of_points_inside_square_attention_patch[0]                                                                                                
        for original_coordinate_tuple, mutated_coordinate_tuple in zip(list_of_points, list_of_mutated_coordinates_string):
            original_coordinate = str(original_coordinate_tuple[0]) + "," + str(original_coordinate_tuple[1])
            # print("original coordinate", original_coordinate)
            # print("mutated coordinate", mutated_coordinate_tuple)
            mutatedPath = originalPath.replace(original_coordinate, mutated_coordinate_tuple)

    elif attention_mth == 2:
        original_point, elapsed_time = AM_get_attetion_svg_points_images_mth2(input_img, 3, svg_path)
        original_coordinate = random.choice(original_point)
        # print(original_point)
        # print(original_coordinate)

        mutated_coordinate = apply_displacement_to_mutant(original_coordinate, extent)

        mutatedPath = svg_path.replace(str(original_coordinate), str(mutated_coordinate))
    else:
        print("Choose a valid attention_mth option in mutation_manager.py")

    # end_time = time.time()
    # print("apply_mutoperator_attention_2 mth5 time: ", (end_time - start_time))
    return mutatedPath

def apply_displacement_to_mutant(value, extent):
    displ = uniform(MUTLOWERBOUND, MUTUPPERBOUND) * extent
    if random.uniform(0, 1) >= MUTOFPROB:
        result = float(value) + displ
    else:
        result = float(value) - displ
    return repr(result)


def apply_mutoperator_attention(input_img, svg_path, extent):
    list_of_points_inside_square_attention_patch, elapsed_time = AM_get_attetion_svg_points_images_mth1(input_img, 3, 3,
                                                                                                        Predictor.model)
    original_point = random.choice(list_of_points_inside_square_attention_patch[0])
    original_coordinate = random.choice(original_point)

    mutated_coordinate = apply_displacement_to_mutant(original_coordinate, extent)


    path = svg_path.replace(str(original_coordinate), str(mutated_coordinate))

    # TODO: it seems that the points inside the square attention patch do not precisely match the point coordinates in the svg, to be tested
    return path


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


def mutate(input_img, svg_desc, operator_name, mutation_extent):
    root = ET.fromstring(svg_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    mutant_vector = svg_path

    operator_name = 3

    if operator_name == 1:
        mutant_vector = apply_mutoperator1(input_img, svg_path, mutation_extent)
    elif operator_name == 2:
        mutant_vector = apply_mutoperator2(input_img, svg_path, mutation_extent)
    elif operator_name == 3:
        mutant_img = input_img.reshape(1, 28, 28)
        mutant_img = mutant_img * 255
        mutant_vector = apply_mutoperator_attention_2(mutant_img, svg_path, mutation_extent)
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
