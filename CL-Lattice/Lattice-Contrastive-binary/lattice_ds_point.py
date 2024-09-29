class DSpoint:

    def __init__(self, latency, area, configuration):
        self.area = area
        self.latency = latency
        self.configuration = configuration
        self.radius = 1

    def set_radius(self, radius):
        self.radius = radius
