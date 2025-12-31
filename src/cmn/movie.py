import logging
log = logging.getLogger(__name__)

import pkgmgr as opentf
from .team import Team
from .castncrew import CastnCrew

class Movie(Team):

    def __init__(self, id, castncrew, primary_title, original_title, release, end, runtime, genres, members_roles):
        super().__init__(id=id, members=castncrew, skills=set(), datetime=release)
        self.primary_title = primary_title #primaryTitle
        self.original_title = original_title #originalTitle
        self.release = release #the release year of a title. In the case of TV Series, it is the series start year
        self.end = end #endYear – TV Series end year. '\N' for all other title types
        self.runtime = runtime
        self.genres = genres
        self.members_roles = members_roles

        self.skills = {g.replace(' ', '_').lower() for g in genres.split(',')} #TODO: ordered skills based on order of genres
        #this part won't have effect since the members are empty the way we create movie objects
        for i, member in enumerate(self.members):
            member.teams.append(self.id)
            member.skills.update(set(self.skills))
            member.role.append(self.members_roles[i])
        self.members_locations = [(None, None, None)] * len(self.members) #TODO: finding the birth location of castncrew from wiki, thought most of them are from holywood, US.??
        #self.location #TODO: finding the country of movie from wiki, thought most of movies are from holywood, US. ??

    @staticmethod
    def read_data(datapath, output, cfg, indexes_only=False):
        pd = opentf.install_import('pandas')# should be here as pickle uses references to existing modules when serialize the objects!
        tqdm = opentf.install_import('tqdm', from_module='tqdm')
        try: return super(Movie, Movie).load_data(output, indexes_only)
        except (FileNotFoundError, EOFError) as e:
            log.info(f'Pickles not found! Reading raw data from {datapath} ...')
            # in imdb, title.* represent movies and name.* represent crew members
            strid2int = lambda x : int(x[2:])
            text = lambda x : x.lower().replace(' ', '_')
            log.info('Reading movie data ...')
            title_basics = pd.read_csv(datapath, sep='\t', header=0, na_values='\\N', converters={'tconst': strid2int, 'primaryTitle': text, 'originalTitle': text}, dtype={'startYear': 'UInt16', 'endYear': 'UInt16'}, low_memory=False)  # title.basics.tsv
            title_basics = title_basics[title_basics['titleType'].isin(['movie', ''])]
            #due to this buggy line, first keep movies only, then change dtype
            #buggy line >> no closing double quotation >> tt10233364	tvEpisode	"Rolling in the Deep Dish	"Rolling in the Deep Dish	0	2019	\N	\N	Reality-TV
            title_basics['runtimeMinutes'] = title_basics['runtimeMinutes'].astype('UInt16')
            log.info('Reading castncrew data ...')
            title_principals = pd.read_csv(datapath.replace('title.basics', 'title.principals'), sep='\t', header=0, na_values='\\N', converters={'tconst': strid2int, 'nconst': strid2int}, low_memory=False)  # movie-crew association for top-10 cast
            name_basics = pd.read_csv(datapath.replace('title.basics', 'name.basics'), sep='\t', header=0, na_values='\\N', converters={'nconst': strid2int,'primaryName': text},dtype={'birthYear': 'UInt16', 'deathYear': 'UInt16'}, low_memory=False)  # name.basics.tsv

            log.info('Joining movie-crew data ...')
            movies_crewids = pd.merge(title_basics, title_principals, on='tconst', how='inner', copy=False)
            movies_crewids_crew = pd.merge(movies_crewids, name_basics, on='nconst', how='inner', copy=False)

            movies_crewids_crew.dropna(subset=['genres'], inplace=True)
            movies_crewids_crew.sort_values(by=['tconst'], inplace=True)
            movies_crewids_crew.loc[len(movies_crewids_crew)] = pd.NA  #last empty row to break the following loop
            movies_crewids_crew = movies_crewids_crew.astype({'tconst':'UInt64', 'nconst':'UInt64'}) # pd.NA or None upcast int to float to make it nullable!

            log.info('Reading data to objects ...')
            teams = {}; candidates = {}; n_row = 0
            current = None
            # for index, movie_crew in tqdm(movies_crewids_crew.iterrows(), total=movies_crewids_crew.shape[0]):#54%|█████▍    | 2036802/3776643 [04:20<03:37, 7989.97it/s]
            # for index in tqdm(range(0, movies_crewids_crew.shape[0], 1)):#50%|█████     | 1888948/3776643 [06:06<06:12, 5074.40it/s]
            #     movie_crew = movies_crewids_crew.loc[index]
            for movie_crew in tqdm(movies_crewids_crew.itertuples(), total=movies_crewids_crew.shape[0]):#100%|███████████|3776642it [01:05, 57568.62it/s]
                try:
                    if pd.isnull(new := movie_crew.tconst): break
                    if current != new:
                        team = Movie(movie_crew.tconst,
                                     [],#empty members!
                                     movie_crew.primaryTitle,
                                     movie_crew.originalTitle,
                                     movie_crew.startYear if pd.notna(movie_crew.startYear) else None,
                                     movie_crew.endYear,
                                     movie_crew.runtimeMinutes,
                                     movie_crew.genres,
                                     [])
                        current = new
                        teams[team.id] = team

                    member_id = movie_crew.nconst
                    member_name = movie_crew.primaryName
                    if (idname := f'{member_id}_{member_name}') not in candidates:
                        candidates[idname] = CastnCrew(movie_crew.nconst,
                                                       movie_crew.primaryName,
                                                       movie_crew.birthYear,
                                                       movie_crew.deathYear,
                                                       movie_crew.primaryProfession,
                                                       movie_crew.knownForTitles)
                    team.members.append(candidates[idname])
                    team.members_details.append((movie_crew.category, movie_crew.job, movie_crew.characters))
                    team.members_locations.append((None, None, None))

                    candidates[idname].skills.update(set(team.skills))
                    candidates[idname].teams.append(team.id)
                    candidates[idname].roles.append(team.members_details[-1])
                    team.members_locations = [(None, None, None)] * len(team.members)

                except Exception as e: raise e
            return super(Movie, Movie).read_data(teams, output, cfg)
