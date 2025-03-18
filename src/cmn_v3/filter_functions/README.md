# Filter Functions

This directory contains modular filter functions for preprocessing team data. Each filter function follows a consistent pattern and can be applied independently or in combination with other filters.

## Available Files

- `__init__.py`: Module initialization with imports of all filter functions
- `remove_dup_teams.py`: Removes duplicate teams based on member and skill composition
- `remove_empty_skills_teams.py`: Removes teams with no skills
- `remove_empty_experts_teams.py`: Removes teams with no experts/members
- `filter_min_team_size.py`: Filters out teams with fewer than specified number of members
- `filter_max_team_size.py`: Filters out teams with more than specified number of members
- `filter_min_skills_team.py`: Filters out teams with fewer than specified number of skills
- `filter_max_skills_team.py`: Filters out teams with more than specified number of skills
- `filter_min_teams_per_expert.py`: Filters out experts who are in fewer than specified number of teams
- `filter_max_teams_per_member.py`: Filters out experts who are in more than specified number of teams

## Overview

The filtering process is designed to preprocess team data by removing teams, members, or skills that don't meet certain criteria. The filtered data is saved separately from the original data, allowing for easy comparison and analysis.

## Filter Process

1. The `generate_sparse_vectors_v3` method in `team.py` now checks for existing filtered data first.
2. If filtered data exists, it loads and returns it directly.
3. If filtered data doesn't exist but unfiltered data does, it applies filters, rebuilds indexes, and creates a filtered teams list.
4. If no data exists, it generates new data, applies filters, rebuilds indexes, and creates a filtered teams list.
5. Filtered data is saved as:
   - `teamsvecs.pkl`: Contains the filtered sparse vectors
   - `indexes.pkl`: Contains the updated indexes for the filtered data
   - `teams.pkl`: Contains the filtered Team objects

## Data Consistency

The filtering process ensures that all three data files remain consistent:

1. `teamsvecs.pkl`: Contains only the teams that passed the filters
2. `indexes.pkl`: Contains updated indexes that only include members and skills present in the filtered teams
3. `teams.pkl`: Contains only the Team objects that correspond to the filtered teamsvecs

This consistency is maintained by:

- Using the same team IDs across all files
- Rebuilding indexes based on the filtered data
- Creating a filtered teams list based on the filtered IDs

## Available Filters

### Team Removal Filters

- `remove_dup_teams`: Removes duplicate teams based on member composition
- `remove_empty_skills_teams`: Removes teams with no skills
- `remove_empty_experts_teams`: Removes teams with no members

### Team Size Filters

- `filter_min_team_size`: Filters teams with fewer than a specified number of members
- `filter_max_team_size`: Filters teams with more than a specified number of members

### Skill Count Filters

- `filter_min_skills_team`: Filters teams with fewer than a specified number of skills
- `filter_max_skills_team`: Filters teams with more than a specified number of skills

### Member Participation Filters

- `filter_min_teams_per_expert`: Filters experts who participate in fewer than a specified number of teams
- `filter_max_teams_per_member`: Filters members who participate in more than a specified number of teams

## Usage

Filters are applied using the `apply_filters.py` script in the helper_functions directory, which can be used as follows:

```python
from cmn_v3.helper_functions.apply_filters import apply_filters

# Apply filters to teamsvecs
filtered_teamsvecs = apply_filters(teamsvecs, indexes, domain_params)

# Rebuild indexes based on filtered data
from cmn_v3.team import Team
filtered_indexes = Team.rebuild_indexes_from_filtered_vecs(filtered_teamsvecs, indexes)

# Create filtered teams list
filtered_teams = Team.create_filtered_teams(teams, filtered_teamsvecs["id"])
```

You can also run the script directly:

```bash
python -m cmn_v3.helper_functions.apply_filters -i /path/to/data -o /path/to/output -d gith
```

## Configuration

Filters are configured in the `param.py` file. The configuration includes:

```python
"filters": {
    "common": {
        "remove_dup_teams": True,
        "remove_empty_skills_teams": True,
        "remove_empty_experts_teams": True,
        "min_team_size": None,
        "max_team_size": None,
        "min_skills": None,
        "max_skills": None,
        "min_teams_per_expert": None,
        "max_teams_per_expert": None,
    }
}
```

## Filter Order

Filters are applied in a specific order for maximum efficiency:

1. First, filters that remove teams with empty/invalid data
2. Then, team size filters
3. Next, skill count filters
4. Then, member participation filters
5. Finally, duplicate team filter (most expensive)

## Adding New Filters

To add a new filter:

1. Create a new Python file in the `filter_functions` directory
2. Implement the filter function with the following signature:
   ```python
   def filter_name(teamsvecs, indexes=None, parameter=None, verbose=True):
       # Filter implementation
       return filtered_teamsvecs
   ```
3. Add the filter to the imports in `__init__.py`
4. Add the filter to the `apply_filters` function in `apply_filters.py`
5. Add the filter parameter to the configuration in `param.py`

## Index Rebuilding

After filtering, the indexes are rebuilt to ensure consistency between the filtered data and the indexes. This is done using the `rebuild_indexes_from_filtered_vecs` method in the `Team` class, which:

1. Identifies active members and skills in the filtered data
2. Creates new mappings for these active elements
3. Updates the member and skill matrices to use the new indices
4. Returns the updated indexes

This ensures that the filtered data and indexes are consistent and can be used together for further processing.
