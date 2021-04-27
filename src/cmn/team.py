class Team(object):
    count = 0
    def __init__(self, id, members):
        Team.count += 1
        self.id = id
        self.members = members
        self.uid = self.set_uid()
        
    # Set a unique id for each team by adding ids of each team member    
    def set_uid(self):
        sum_of_ids = 0
        for mem in self.members:
            sum_of_ids += mem.get_id()
        return sum_of_ids

    # Return the unique id of the team
    def get_uid(self):
        return self.uid    
    
    # Return a list of members' names    
    def get_members_names(self):
        members_names = []
        for member in self.members:
            members_names.append(member.get_name())
        return members_names
    
    # Return a list of members' ids
    def get_members_ids(self):
        members_ids = []
        for member in self.members:
            members_ids.append(member.get_id())
        return members_ids