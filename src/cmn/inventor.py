from member import Member


class Inventor(Member):
    def __init__(self, id, name, location_id):
        super().__init__(id, name)
        self.location_id = location_id
        self.teams = []
        self.roles = []
