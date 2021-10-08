from cmn.member import member


class author(member):
    def __init__(self, id,  inventor_id, loc_id):
        super().__init__(id, inventor_id)
        self.loc_id = loc_id

    def get_loc(self):
        return self.loc_id
