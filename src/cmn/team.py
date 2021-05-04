from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
from datetime import datetime

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

    def get_skills(self):
        pass

    @staticmethod
    def build_index_members(all_members):
        idx = 0
        member_to_index = {}
        index_to_member = {}

        for member in all_members.values():
            index_to_member[idx] = member.get_id()
            member_to_index[member.get_id()] = idx
            idx += 1

        return index_to_member, member_to_index

    @staticmethod
    def build_index_skills(teams):
        idx = 0
        skill_to_index = {}
        index_to_skill = {}

        for team in teams.values():
            for skill in team.get_skills():
                if skill not in skill_to_index.keys():
                    skill_to_index[skill] = idx
                    index_to_skill[idx] = skill
                    idx += 1

        return index_to_skill, skill_to_index

    @staticmethod
    def read_data(data_path, topn=None):
        #should be override by the children classes, customize their loading data
        pass

    @staticmethod
    def build_dataset(teams, skill_to_index, member_to_index):

        training_size = len(teams)
        BUCKET_SIZE = 100
        SKILL_SIZE = len(skill_to_index)
        AUTHOR_SIZE = len(member_to_index)

        # Sparse Matrix and bucketing
        data = lil_matrix((training_size, SKILL_SIZE + AUTHOR_SIZE + 1))
        data_ = np.zeros((BUCKET_SIZE, SKILL_SIZE + AUTHOR_SIZE + 1))
        j = -1
        for i, team in enumerate(teams.values()):
            if i >= training_size: break

            # Generating one hot encoded vector for input
            X = np.zeros((1, SKILL_SIZE))
            input_fields = team.get_skills()
            for field in input_fields:
                X[0, skill_to_index[field]] = 1

            # This does not work since the number of authors are different for each sample, therefore we need to build the output as a one hot encoding
            # y_index = []
            # for id in output_ids:
            #     y_index.append(member_to_index[id])
            # y_index.append(len(output_ids))
            # y = np.asarray([y_index])

            # Generating one hot encoded vector for output
            y = np.zeros((1, AUTHOR_SIZE + 1))
            output_ids = team.get_members_ids()
            for id in output_ids:
                y[0, member_to_index[id]] = 1
            y[0, -1] = len(output_ids)

            # Building a training instance
            X_y = np.hstack([X, y])

            # Bucketing
            try:
                j += 1
                data_[j] = X_y
            except:
                s = int(((i / BUCKET_SIZE) - 1) * BUCKET_SIZE)
                e = int(s + BUCKET_SIZE)
                data[s: e] = data_
                j = 0
                data_[j] = X_y
            if (i % BUCKET_SIZE == 0):
                print(f'Loading {i}/{len(teams)} instances!{datetime.now()}')
        if j > -1:
            data[-j - 1:] = data_[0:j + 1]

        input_matrix = data[:, :SKILL_SIZE]
        output_matrix = data[:, -1 - AUTHOR_SIZE:]
        print(input_matrix.shape)
        print(output_matrix.shape)
        # input_matrix = torch.rand(100,100)
        # output_matrix = torch.rand(100,100)

        return input_matrix, output_matrix