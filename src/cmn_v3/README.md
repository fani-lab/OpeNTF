# CMN V3 - Common Module for Neural Team Formation

This directory contains the common modules for neural team formation, including domain-specific data processing, filtering, and reporting.

## Directory Structure

- `filter_functions/`: Contains filter functions for preprocessing team data
- `helper_functions/`: Contains helper functions for various tasks
- Domain-specific implementation files:
  - `dblp.py`: Implementation for DBLP dataset (academic papers and authors)
  - `gith.py`: Implementation for GitHub dataset (repositories and contributors)
- Domain-specific model files:
  - `dblp_author.py`: Author model for DBLP dataset
  - `dblp_venue.py`: Venue model for DBLP dataset
  - `gith_contributor.py`: Contributor model for GitHub dataset
- `team.py`: Base class for team-based datasets
- `__init__.py`: Module initialization

## Preprocessing Pipeline

The preprocessing pipeline follows these steps:

1. **Raw Data Processing**: Domain-specific classes process raw data and generate sparse vectors (teamsvecs)
2. **Filtering**: Apply filters to the teamsvecs data to remove unwanted teams/experts
3. **Report Generation**: Generate reports and visualizations for the filtered data

### Raw Data Processing

Each domain has its own processing class (e.g., `dblp.py`, `gith.py`) that handles the specifics of parsing and processing raw data. These classes generate sparse vectors in the teamsvecs format.

### Filtering

The filtering step is handled by the `apply_filters.py` script, which applies a series of filters to the teamsvecs data based on the configuration specified in the settings file or command-line arguments.

Filters are organized into two categories:

1. **Common Filters**: Applicable to all datasets (e.g., removing duplicate teams)
2. **Domain-Specific Filters**: Specific to each domain (e.g., filtering by year for DBLP)

See the `filter_functions/README.md` file for more details on available filters and how to use them.

### Report Generation

After filtering, reports and visualizations are generated to provide insights into the dataset. This is handled by the `generate_reports.py` script in the `helper_functions` directory.

## Configuration

Configuration is managed through parameter files:

- `param.py`: Contains common settings for all domains

The filter configuration follows this structure:

```python
"filters": {
    "common": {
        "remove_dup_teams": True,           # Remove duplicate teams
        "remove_empty_skills_teams": True,  # Remove teams with no skills
        "remove_empty_experts_teams": True, # Remove teams with no experts
        "min_team_size": 2,                 # Minimum number of experts per team
        "min_teams_per_expert": 2           # Minimum number of teams an expert must be in
    },
    "domain": {
        # Domain-specific filters
    }
}
```

## Usage

The preprocessing pipeline is typically run through the main.py script, which handles the entire process from raw data to filtered teamsvecs and reports.

To run the pipeline for a specific domain:

```bash
python src/main.py -i path/to/raw/data -d domain_name -o output_folder_name
```

To run individual steps:

1. **Apply Filters Only**:

   ```bash
   python src/cmn_v3/helper_functions/apply_filters.py -i path/to/teamsvecs.pkl -o path/to/output/dir -c path/to/config.json -d domain_name
   ```

2. **Generate Reports Only**:
   ```bash
   python src/cmn_v3/helper_functions/generate_reports.py -i path/to/teamsvecs.pkl -o path/to/output/dir
   ```
