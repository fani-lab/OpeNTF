class Member(object):
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.teams = set()
        self.skills = set()
