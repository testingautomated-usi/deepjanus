import rasterization_tools


class Digit:
    def __init__(self, desc, label):
        self.xml_desc = desc
        self.expected_label = label
        self.predicted_label = None
        self.purified = rasterization_tools.rasterize_in_memory(self.xml_desc)
        self.P_class = None
        self.P_notclass = None

    def clone(self):
        clone_digit = Digit(self.xml_desc, self.expected_label)
        return clone_digit
