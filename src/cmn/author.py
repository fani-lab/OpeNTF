from cmn.member import Member

class Author(Member):
    def __init__(self, id, name, org):
        super().__init__(id, name)
        self.org = org