class DataRepo:

    _category_mapping = {
        0: 'World',
        1: 'Sports',
        2: 'Business',
        3: 'Sci/Tech'
    }

    def __init__(self):

        self.test_data = [
            "Possible Source of Cosmic Rays Found (SPACE.com) SPACE.com - Astronomers have produced the first truly "
            "useful image ever of something in space using gamma rays. It's a picture only an astronomer could love, "
            "but it appears to help solve a century-long mystery.",

            "No. 10 W.Virginia Tops E.Carolina, 56-23 (AP) AP - Kay-Jay Harris rushed for a school-record 337 yards "
            "and four touchdowns to lead No. 10 West Virginia to a 56-23 victory over East Carolina on Saturday night.",

            "The Dow closed almost a precentage point up for the day. Nasdaq closed 1.5 % up, S&P was up a bit over 1%",

            "Zimbabwe court Friday convicted a British man accused of leading a coup plot against the government of "
            "oil-rich Equatorial Guinea on weapons charges, but acquitted most of the 69 other men held with him."
        ]

        self.test_labels = [4, 2, 3, 1]

    def get_smoke_test_data(self):
        return self.test_data

    def get_smoke_test_labels(self):
        return self.test_labels

    @staticmethod
    def int_labels_to_words(int_labels, index_start=0):
        return [DataRepo._category_mapping[index - index_start] for index in int_labels]