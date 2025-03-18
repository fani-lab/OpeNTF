# Helper Functions

This directory contains utility functions that can be reused across different domain code in the OpeNTF-nmt2 project.

## Available Files

### Core Processing Functions

- `apply_filters.py`: Applies various filters to team data based on configuration settings
- `generate_reports.py`: Generates detailed markdown reports and visualizations for dataset statistics
- `run_gpu_processing.py`: Handles GPU-accelerated processing for large datasets

### Data Logging Functions

- `log_entries_processed.py`: Logs information about processed team entries to entries_processed.log
- `log_skills.py`: Logs unique skills in the dataset, optionally with frequency information

### GPU and Threading Utilities

- `get_default_threads.py`: Determines the default number of threads to use for processing
- `get_gpu_device.py`: Selects appropriate GPU devices for processing
- `get_nthreads.py`: Calculates optimal thread count based on system resources
- `import_gpu_libs.py`: Handles conditional importing of GPU-related libraries

### Initialization

- `__init__.py`: Provides module initialization and exports functions

## Core Functions Documentation

### apply_filters

Applies a series of filters to team data based on configuration settings.

```python
from cmn_v3.helper_functions import apply_filters

filtered_teamsvecs = apply_filters(teamsvecs, indexes, domain_params)
```

### generate_reports

Generates detailed markdown reports and visualizations for dataset statistics.

```python
from cmn_v3.helper_functions import generate_markdown_report

md_file_path = generate_markdown_report(stats, report_params, out_file=md_file_path)
```

### log_entries_processed

Logs information about processed team entries.

```python
from cmn_v3.helper_functions import log_entries_processed

entry_text = log_entries_processed(
    output_path,       # Path to store the logs
    team_id,           # Team/Repository ID
    skills,            # List of skills
    members,           # List of team members
    entry_idx=None     # Optional index for the entry
)
```

For batch processing of multiple entries:

```python
from cmn_v3.helper_functions import log_entries_processed_batch

log_entries_processed_batch(
    output_path,       # Path to store the logs
    entries            # List of formatted entry strings
)
```

### log_skills

Logs all unique skills in the dataset, one per line.

```python
from cmn_v3.helper_functions import log_skills

log_skills(
    output_path,       # Path to store the logs
    teams,             # List of team objects
    skill_index=None   # Optional dictionary mapping skill indices to names
)
```

For skills with frequency information:

```python
from cmn_v3.helper_functions import log_skills_with_frequencies

log_skills_with_frequencies(
    output_path,        # Path to store the logs
    teams,              # List of team objects
    skill_index=None,   # Optional dictionary mapping skill indices to names
    member_skill_matrix=None  # Optional matrix of skill usage
)
```

### get_nthreads

Calculates optimal number of threads based on available system resources.

```python
from cmn_v3.helper_functions import get_nthreads

n_threads = get_nthreads(thread_fraction=0.75)  # Use 75% of available threads
```

### get_default_threads

Determines the default number of threads to use based on whether GPU is available.

```python
from cmn_v3.helper_functions import get_default_threads

n_threads = get_default_threads(mode_str="cpu")  # For CPU mode
n_threads = get_default_threads(mode_str="gpu")  # For GPU mode
```

## Usage Example

```python
from cmn_v3.helper_functions import (
    is_duplicate_team,
    log_entries_processed,
    log_skills
)

# Initialize seen_teams dictionary for duplicate detection
seen_teams = {}

# Process teams
for idx, team in enumerate(teams):
    # Check if this team is a duplicate
    is_duplicate, dup_index = is_duplicate_team(
        team.id, team.skills, team.members, seen_teams
    )

    # Log processed entry
    log_entries_processed(
        output_path,
        team.id,
        team.skills,
        team.members,
        idx
    )

    # Handle duplicates as needed
    if is_duplicate:
        # Skip or handle duplicate team
        pass

# Log unique skills
log_skills(output_path, teams)
```
