from cmn.member import Member

class CastnCrew(Member):
    def __init__(self, id, name, birth, death, p_prof, knownfor):
        super().__init__(id, name)
        self.birth = birth
        self.death =death
        self.p_prof = p_prof
        self.knownfor = knownfor

        #override Member.set() to list() to keep track of movie-role association
        self.teams = []
        self.roles = []#this is in association with self.teams (what were the roles of self in each of the teams/movies)
