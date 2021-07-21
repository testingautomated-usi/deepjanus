import random
import xml.etree.ElementTree as ET
import re
from random import randint, uniform
from config import MUTLOWERBOUND, MUTUPPERBOUND, MUTOFPROB

NAMESPACE = '{http://www.w3.org/2000/svg}'


def apply_displacement_to_mutant(value, extent):
    displ = uniform(MUTLOWERBOUND, MUTUPPERBOUND) * extent
    if random.uniform(0, 1) >= MUTOFPROB:
        result = float(value) + displ
    else:
        result = float(value) - displ
    return repr(result)


def apply_mutoperator1(svg_path, extent):

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


def apply_mutoperator2(svg_path, extent):
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


def mutate(svg_desc, operator_name, mutation_extent):
    root = ET.fromstring(svg_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    mutant_vector = svg_path
    if operator_name == 1:
        mutant_vector = apply_mutoperator1(svg_path, mutation_extent)
    elif operator_name == 2:
        mutant_vector = apply_mutoperator2(svg_path, mutation_extent)
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
