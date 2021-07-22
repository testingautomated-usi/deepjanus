chromosome_properties_list = ["camera_angle_1", "camera_angle_2", "eye_angle_1",
                         "eye_angle_2", "pupil_size", "iris_size", "iris_texture",
                         "skybox_texture", "skybox_exposure", "skybox_rotation",
                         "ambient_intensity", "light_rotation_angle_1",
                         "light_rotation_angle_2", "light_intensity", "primary_skin_texture"]

# camera_angle_1 (theta)
camera_angle_1_low_bound = - 20
camera_angle_1_up_bound = 20
camera_angle_1_mut_mgnt = 3

# camera_angle_2 (ph)\
camera_angle_2_low_bound = -40
camera_angle_2_up_bound = 40
camera_angle_2_mut_mgnt = 3

# eye_angle_1 (theta)
eye_angle_1_low_bound = -30
eye_angle_1_up_bound = 30
eye_angle_1_mut_mgnt = 3

# eye_angle_2 (phi)
eye_angle_2_low_bound = -30
eye_angle_2_up_bound = 30
eye_angle_2_mut_mgnt = 3

#JSON PARAMS

# Pupil Size
pupil_size_low_bound = -0.9287103
pupil_size_up_bound = 0.9613182
pupil_size_mut_mgnt = 0.19

# Iris Size
iris_size_low_bound = 0.9
iris_size_up_bound = 1.0 #0.9999997
iris_size_mut_mgnt = 0.01

    # Iris Textures
iris_texture = ['eyeball_grey', 'eyeball_amber', 'eyeball_grey_1', 'eyeball_brown', 'eyeball_green']


# Skybox Textures
skybox_texture = ['DF360_003_Ref', 'Protospace_A_Ref', 'DF360_002c_Ref',
                   'parking_lot_2k', 'st_lucia_interior_2k', 'delta_2k',
                   'DF360_008b_Ref', 'st_lucia_beach_2k', 'northcliff_2k',
                   'courtyard_2k', 'oribi_2k', 'DF360_001a_Ref', 'garage_2k',
                   'bergen_2k', 'lapa_2k', 'cabin_2k', 'bathroom_2k',
                   'golden_gate_2k']

# Skybox Exposure
skybox_exposure_low_bound = 1 #1.000008
skybox_exposure_up_bound = 1.2 #1.199997
skybox_exposure_mut_mgnt = 0.02

# Skybox light_rotation
skybox_rotation_low_bound = 0.0
skybox_rotation_up_bound = 359.0
skybox_rotation_mut_mgnt = 5

# Ambient Intensity 0.9 -> 0.94
ambient_intensity_low_bound = 0.8 #0.8000012
ambient_intensity_up_bound = 1.2
ambient_intensity_mut_mgnt = 0.04

# Light Rotation
light_rotation_angle_1_low_bound = 0.0
light_rotation_angle_1_up_bound = 360.0
light_rotation_angle_1_mut_mgnt = 5

light_rotation_angle_2_low_bound = 0.0
light_rotation_angle_2_up_bound = 270.0
light_rotation_angle_2_mut_mgnt = 5

# Light Intensity
light_intensity_low_bound = 0.6 # 0.6000005
light_intensity_up_bound = 1.2 # 1.199999
light_intensity_mut_mgnt = 0.06


# Skin Textures
primary_skin_texture = ['m01_color', 'm02_color', 'm03_color', 'm04_color', 'm05_color',
                                                      'm06_color', 'm07_color', 'm08_color', 'm09_color', 'm10_color',
                                                      'm11_color', 'm12_color', 'm13_color', 'm14_color', 'm15_color',
                                                      'f01_color', 'f02_color', 'f03_color', 'f04_color', 'f05_color']
