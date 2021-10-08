class member(object):
    def __init__(self, id, member):
        self.id = id
        self.members = member

    def get_id(self):
        return self.id

    def get_inventor_id(self):
        return self.members
