class Member(object):
    count = 0
    def __init__(self, id, name):
        Member.count += 1
        self.id = id
        self.name = name
    
    def get_id(self):
        return self.id
    
    def get_name(self):
        return self.name