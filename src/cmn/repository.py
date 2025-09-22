import logging, numpy as np
log = logging.getLogger(__name__)

import pkgmgr as opentf
from .team import Team
from .developer import Developer

class Repository(Team):
    # we assume the skill set is fixed and from the top-100 prog. langs. in github.
    # https://madnight.github.io/githut/#/pull_requests/2024/1
    top_100_langs = {
        'python', 'java', 'go', 'javascript', 'c++', 'typescript', 'php', 'ruby', 'c', 'c#',
        'nix', 'shell', 'html', 'rust', 'scala', 'kotlin', 'swift', 'dart', 'jupyter notebook',
        'makefile', 'groovy', 'perl', 'lua', 'hcl', 'dm', 'systemverilog', 'css', 'objective-c',
        'elixir', 'scss', 'starlark', 'roff', 'json', 'codeql', 'ocaml', 'dockerfile', 'haskell',
        'vue', 'powershell', 'erlang', 'smarty', 'emacs lisp', 'jinja', 'julia', 'clojure', 'r',
        'coffeescript', 'f#', 'verilog', 'webassembly', 'mlir', 'adblock filter list', 'bicep',
        'tex', 'xslt', 'fortran', 'markdown', 'cython', 'gap', 'matlab', 'puppet', 'plpgsql',
        'sass', 'jetbrains mps', 'smalltalk', 'yaml', 'bitbake', 'vala', 'haxe', 'pascal',
        'elm', 'vim script', 'nunjucks', 'assembly', 'coq', 'haml', 'antlr', 'gherkin',
        'reason', 'common lisp', 'jsonnet', 'standard ml', 'mathematica', 'cuda', 'd',
        'cmake', 'glsl', 'hoon', 'scheme', 'coldfusion', 'swig', 'llvm', 'twig', 'nim',
        'less', 'saltstack', 'sqf', 'qml', 'zap', 'tcl'
    }

    def __init__(self, idx: int, contributors: list, name: str, languages_lines: list, nforks: int,
                 nstargazers: int, created_at: str, year: int, pushed_at: str, ncontributions: list, releases: list):

        super().__init__(id=idx, members=contributors, skills=set(), datetime=year)
        self.name = name
        self.nforks = nforks
        self.nstargazers = nstargazers
        self.created_at = created_at
        self.pushed_at = pushed_at
        self.ncontributions = ncontributions
        self.releases = releases
        self.languages_lines = languages_lines

        self.skills = {l.replace(' ', '_').lower() for (l, _) in self.languages_lines.items()} #[(lang, line#)] TODO: ordered skills based on line#
        for dev in self.members:
            dev.teams.add(self.id)
            dev.skills.update(set(self.skills))
        self.members_locations = [(None, None, None)] * len(self.members)

    @staticmethod
    def read_data(datapath, output, cfg, indexes_only=False):
        pd = opentf.install_import('pandas')# should be here as pickle uses references to existing modules when serialize the objects!
        tqdm = opentf.install_import('tqdm', from_module='tqdm')
        try: return super(Repository, Repository).load_data(output, indexes_only)
        except (FileNotFoundError, EOFError) as e:
            log.info(f'Pickles not found! Reading raw data from {datapath} ...')
            ds = pd.read_csv(datapath, converters={'collabs': eval, 'langs': lambda x: {k.lower(): v for k, v in eval(x).items()}, 'rels': eval}, encoding='latin-1')
            ds = ds[ds['collabs'].map(type) != dict] # remove repos with error in contributors like "{'message': 'The history or contributor list ....
            # memory demand but fast
            ds = ds[ds.explode('langs').assign(keep=lambda d: d['langs'].isin(Repository.top_100_langs)).groupby(level=0)['keep'].any()]
            # slow but no memory demand
            # ds = ds[~ds['langs'].apply(lambda lst: set(lst).isdisjoint(Repository.top_100_langs))]
            teams = dict(); repos = dict(); candidates = dict()
            ds['created_at'] = pd.to_datetime(ds['created_at'])
            ds['year'] = ds['created_at'].dt.year
            try:
                for idx, row in tqdm(ds.iterrows(), total=len(ds)):
                    contributors = row['collabs']
                    developers = list(); contributions = list()

                    for contributor in contributors:
                        if isinstance(contributor, str): continue
                        if (idname := f"{contributor['id']}_{contributor['login']}") not in candidates:
                            candidates[idname] = Developer(name=contributor['login'], id=contributor['id'], url=contributor['url'])
                        developers.append(candidates[idname])
                        contributions.append(contributor['contributions'])

                    repo_name = row['repo']
                    languages_lines = row['langs']# a list of (lang, nlines)
                    nstargazers = row['stargazers_count']
                    nforks = row['forks_count']
                    created_at = row['created_at']
                    year = row['year']
                    pushed_at = row['pushed_at']
                    ncontributions = row['collabs']
                    releases = row['rels']

                    if repo_name not in repos:
                        teams[idx] = Repository(idx=idx, contributors=developers, name=repo_name, releases=releases,
                                                  languages_lines=languages_lines, nstargazers=nstargazers,
                                                  nforks=nforks, created_at=created_at, year=year, pushed_at=pushed_at, ncontributions=ncontributions)
                        repos[repo_name] = teams[idx]
                    else: pass

            except Exception as e: raise e
            return super(Repository, Repository).read_data(teams, output, cfg)
