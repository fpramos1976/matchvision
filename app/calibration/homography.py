class Homography:

    def __init__(self, image_points, configuration):
        self.image_points = image_points
        self.configuration = configuration
        self.matrix = None

        print("Homography criada!")

    def compute(self):
        print("Calculando homografia...")
        self.matrix = None

    def transform(self, frame):
        return frame