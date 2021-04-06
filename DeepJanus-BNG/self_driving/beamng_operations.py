

class Operations:
    def __init__(self):
        raise NotImplemented()

    def add_obstacle(self, coordinates):
        raise NotImplemented()

    def change_illumination(self, time_of_day):
        raise NotImplemented()

    def change_slope(self):
        raise NotImplemented()

    def change_weather(self):
        raise NotImplemented()

global operations
operations = Operations()