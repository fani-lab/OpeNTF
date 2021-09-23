class Member(object):
    count = 0
    def __init__(self, id, name):
        Member.count += 1
        self.id = id
        self.name = name
        self.n_papers = 1
    
    def get_id(self):
        return self.id
    
    def get_name(self):
        return self.name

    def get_n_papers(self):
        return self.n_papers

    def increase_n(self):
        self.n_papers += 1