from cmn.author import Author

class Inventor(Author):
    def __init__(self, id, name, gender):
        super().__init__(id, name, None)
        self.gender = gender

        # override Member.set() to list() to keep track of inventor-location-patent association
        self.teams = []
        self.locations = []  # this is in association with self.teams (what were the location of self in each of the teams/patent)
