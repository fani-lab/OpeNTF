#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import ast  # Import ast module for literal_eval
from time import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import sys
from pathlib import Path
import traceback

# Add the project root to the Python path if it's not already there
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .team import Team
from .gith_contributor import GithContributor
from .gith_params import GITH_PARAMS
from utils.tprint import tprint

# Error logging function
def log_error(datapath, error_type, repo_id, message, data=None):
    """
    Log an error to gith_errors.log
    
    Args:
        datapath: Path to the data file (used to determine where to save the log)
        error_type: Type of error (e.g. 'collabs', 'skills', 'contributor')
        repo_id: Repository ID where the error occurred
        message: Error message
        data: The problematic data object that caused the error
    """
    log_path = os.path.join(os.path.dirname(datapath), "gith_errors.log")
    with open(log_path, 'a', encoding='utf-8') as log_file:
        # Format timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write error header
        log_file.write(f"[{timestamp}] {error_type.upper()} ERROR for {repo_id}: {message}\n")
        
        # Write the problematic data
        if data is not None:
            log_file.write("DATA DUMP:\n")
            try:
                # Try to format data nicely if possible
                if isinstance(data, (dict, list)):
                    import pprint
                    formatted_data = pprint.pformat(data, indent=2)
                    log_file.write(formatted_data)
                else:
                    log_file.write(f"Type: {type(data)}\n")
                    log_file.write(f"Value: {str(data)}")
            except Exception as e:
                log_file.write(f"Error dumping data: {str(e)}")
        
        # Add double newline to separate entries
        log_file.write("\n\n")


class Repository(Team):
    """
    Class representing a Gith repository team
    
    This class extends the base Team class for Gith-specific functionality.
    """
    
    # Add domain_params as a class attribute
    domain_params = GITH_PARAMS
    
    def __init__(self, id, contributors, skills, year, location=None, created_at=None, pushed_at=None):
        """
        Initialize a Gith repository team
        
        Args:
            id: Repository identifier (typically 'owner/repo')
            contributors: List of contributors (GithContributor objects)
            skills: List of programming languages used in the repository
            year: Year the repository was created
            location: Repository location (optional)
            created_at: Repository creation timestamp (optional)
            pushed_at: Last push timestamp (optional)
        """
        # Safety validation of inputs
        if id is None:
            id = "unknown_repo"
            
        # Ensure contributors is a valid list with at least one member
        if contributors is None or len(contributors) == 0:
            # Create a default contributor based on the repository name
            repo_parts = id.split('/')
            if len(repo_parts) >= 2:
                owner = repo_parts[0]
                contributors = [GithContributor(id=owner, login=owner, contributions=1)]
            else:
                contributors = [GithContributor(id="unknown", login="unknown", contributions=1)]
        
        # Validate each contributor has a valid ID
        valid_contributors = []
        for contrib in contributors:
            if contrib is not None and hasattr(contrib, 'id') and contrib.id is not None:
                valid_contributors.append(contrib)
        
        # If no valid contributors remain, add a default one
        if not valid_contributors:
            repo_parts = id.split('/')
            if len(repo_parts) >= 2:
                owner = repo_parts[0]
                valid_contributors = [GithContributor(id=owner, login=owner, contributions=1)]
            else:
                valid_contributors = [GithContributor(id="unknown", login="unknown", contributions=1)]
        
        # Ensure skills is not None
        if skills is None or len(skills) == 0:
            skills = ["unknown"]
            
        # Handle year as None or invalid
        if year is None:
            year = datetime.now().year
            
        # Convert year to datetime for chronological ordering
        try:
            creation_datetime = datetime.strptime(f"{year}-01-01", "%Y-%m-%d") if isinstance(year, int) else None
        except (ValueError, TypeError):
            creation_datetime = datetime.now()
            
        if created_at:
            try:
                creation_datetime = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, TypeError):
                # Keep existing creation_datetime if parsing fails
                pass
                
        # Base class initialization with the datetime
        super().__init__(id, valid_contributors, skills, creation_datetime, location)
        
        # Gith specific attributes
        self.created_at = created_at
        self.pushed_at = pushed_at
        self.year = year
    
    def __str__(self):
        """
        String representation of the repository
        
        Returns:
            String with repository information
        """
        return f"Repository: {self.id} ({self.year}) - {len(self.members)} contributors, {len(self.skills)} languages"
    
    @classmethod
    def read_and_filter_data_v3(cls, datapath):
        """
        Read and filter Gith repository data
        
        Args:
            datapath: Path to the Gith dataset file (CSV)
            
        Returns:
            List of Repository objects
        """
        start_time = time()
        
        # Clear the error log before starting
        error_log_path = os.path.join(os.path.dirname(datapath), "gith_errors.log")
        try:
            with open(error_log_path, 'w') as f:
                f.write(f"GitHub Data Processing Errors Log - Started: {datetime.now()}\n\n")
        except Exception as e:
            tprint(f"Warning: Could not initialize error log: {str(e)}")
            
        # Initialize invalid repositories log
        invalids_log_path = os.path.join(os.path.dirname(datapath), "gith_invalids.log")
        try:
            with open(invalids_log_path, 'w') as f:
                f.write(f"GitHub Invalid Repositories Log - Started: {datetime.now()}\n")
                f.write(f"This file contains repositories that didn't pass filters and why.\n\n")
        except Exception as e:
            tprint(f"Warning: Could not initialize invalids log: {str(e)}")
            
        # Initialize repository output log
        output_log_path = os.path.join(os.path.dirname(datapath), "gith_output.log")
        try:
            with open(output_log_path, 'w') as f:
                f.write(f"GitHub Repository Data - Started: {datetime.now()}\n\n")
        except Exception as e:
            tprint(f"Warning: Could not initialize output log: {str(e)}")
            
        # Function to determine which log file to use based on team index
        def get_output_log_path(idx, datapath):
            """
            Determine which log file to use based on the team index
            
            The pattern is:
            - First 1K entries (1-1000): gith_output_1K01.log
            - Next 100K entries (1001-101000): gith_output_100K01.log
            - Next 1K entries (101001-102000): gith_output_1K02.log
            - Next 100K entries (102001-202000): gith_output_100K02.log
            
            And so on, alternating between 1K and 100K files
            """
            base_dir = os.path.dirname(datapath)
            
            # Offset by 1 to make 1-indexed (idx is 0-based, but we want files to start at 1)
            team_num = idx + 1
            
            # Calculate the segment and position within the pattern
            segment = 1
            position = team_num
            
            # Each pattern is 1K + 100K = 101K entries
            pattern_size = 101000
            
            # Determine which segment we're in
            if team_num > pattern_size:
                segment = (team_num - 1) // pattern_size + 1
                position = (team_num - 1) % pattern_size + 1
            
            # If position is within the first 1000, use the 1K file
            if position <= 1000:
                log_file = f"gith_output_1K{segment:02d}.log"
            else:
                # Otherwise use the 100K file
                log_file = f"gith_output_100K{segment:02d}.log"
            
            return os.path.join(base_dir, log_file)

        # Dictionary to keep track of initialized log files
        initialized_log_files = set()
            
        tprint(f"Reading and filtering Gith data from {datapath}")

        # Track statistics
        repositories = {}  # Use a dictionary to track by ID
        repo_ids = set()  # Use a set for faster lookups
        all_members = set()  # For statistics only
        all_skills = set()   # For statistics only

        # Get filtering parameters
        # pass1_filters
        remove_duplicates = GITH_PARAMS['pass1_filters']['remove_duplicates']
        remove_empty_skills = GITH_PARAMS['pass1_filters']['remove_empty_skills']
        remove_empty_members = GITH_PARAMS['pass1_filters']['remove_empty_members']
        
        min_team_size = GITH_PARAMS['pass1_filters']['min_team_size']
        max_team_size = GITH_PARAMS['pass1_filters']['max_team_size']
        min_skills = GITH_PARAMS['pass1_filters']['min_skills']
        max_skills = GITH_PARAMS['pass1_filters']['max_skills']
        min_year = GITH_PARAMS['pass1_filters']['min_year']
        max_year = GITH_PARAMS['pass1_filters']['max_year']
        languages = GITH_PARAMS['pass1_filters']['languages']

        # passn_filters
        min_nteam = GITH_PARAMS['passn_filters']['min_nteam']

        # processing
        debug_logs = GITH_PARAMS['processing']['debug_logs']

        progress_message = tprint("Reading Gith data")

        # Read the CSV file with appropriate encoding
        repositories = {}
        all_members = set()  # Initialize set to track all unique members
        all_skills = set()   # Initialize set to track all unique skills
        repo_ids = set()     # Initialize set to track all unique repository IDs
        try:
            # Try multiple encodings if necessary
            encodings_to_try = ['latin1', 'utf-8', 'cp1252']
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(datapath, encoding=encoding)
                    tprint(f"Successfully loaded data using {encoding} encoding")
                    
                    # Only log essential column information
                    important_columns = ['repo', 'langs', 'collabs', 'created_at']
                    missing_columns = [col for col in important_columns if col not in df.columns]
                    if missing_columns:
                        tprint(f"WARNING: The following important columns are missing: {missing_columns}")
                    
                    # Debug: Check for collabs column
                    if 'collabs' in df.columns:
                        collabs_count = df['collabs'].notna().sum()
                        tprint(f"Found 'collabs' column with {collabs_count} non-NA values out of {len(df)} rows")
                        
                        # Skip verbose logging of sample values
                        # non_empty_collabs = df[df['collabs'].notna()]['collabs'].iloc[:5]
                        # if len(non_empty_collabs) > 0:
                        #     tprint(f"Sample collabs values: {non_empty_collabs.values}")
                    else:
                        tprint(f"WARNING: 'collabs' column not found in the data!")
                    
                    break
                except UnicodeDecodeError:
                    tprint(f"Failed to load with {encoding} encoding, trying another...")
                    continue
            
            # Process each repository
            progress_update_interval = 5000  # Print status every 5000 repos
            with tqdm(total=len(df), desc=progress_message, unit="repos", unit_scale=True) as pbar:
                for idx, (_, row) in enumerate(df.iterrows()):
                    try:
                        # Print occasional progress updates
                        if idx % progress_update_interval == 0 and idx > 0:
                            tprint(f"Processed {idx} repositories, found {len(repositories)} valid repos, {len(all_members)} contributors, {len(all_skills)} skills")
                        
                        # Extract repository details
                        repo_id = str(row['repo']) if 'repo' in row and pd.notna(row['repo']) else "unknown_repo"
                        
                        # Initialize variables for logging
                        langs_value = "None"
                        skills = []
                        contributors = []
                        
                        # Extract and parse created_at to get the year
                        created_at = row['created_at'] if 'created_at' in row and pd.notna(row['created_at']) else None
                        year = None
                        
                        try:
                            if created_at and isinstance(created_at, str):
                                created_dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
                                year = created_dt.year
                        except (ValueError, TypeError) as e:
                            if debug_logs:
                                tprint(f"Error parsing date for {repo_id}: {str(e)}")
                            # Try to extract year from string if possible
                            if isinstance(created_at, str) and len(created_at) >= 4:
                                try:
                                    year = int(created_at[:4])
                                except ValueError:
                                    pass
                        
                        # Extract languages (skills)
                        repo_skills = set()
                        language_error = None

                        if 'langs' in row and pd.notna(row['langs']):
                            langs = row['langs']
                            langs_value = str(langs)  # Store original value for logging
                            try:
                                # Store the type for debugging
                                with open(error_log_path, 'a', encoding='utf-8') as log_file:
                                    log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LANGS INFO for {repo_id}:\n")
                                    log_file.write(f"Type: {type(langs)}\n")
                                    log_file.write(f"Value: {langs}\n\n")
                                
                                # Initialize langs_dict
                                langs_dict = {}
                                
                                # Handle different types of langs value
                                if isinstance(langs, dict):
                                    # Already a dict, use as is
                                    langs_dict = langs
                                elif isinstance(langs, str):
                                    # Parse from string representation
                                    try:
                                        # First try ast.literal_eval for more flexible parsing
                                        langs_dict = ast.literal_eval(langs)
                                    except (SyntaxError, ValueError) as e:
                                        # Log the attempt
                                        if debug_logs:
                                            tprint(f"ast.literal_eval failed for {repo_id}: {str(e)}")
                                            
                                        # Try with replaced single quotes
                                        try:
                                            fixed_langs = langs.replace("'", "\"")
                                            langs_dict = json.loads(fixed_langs)
                                            if debug_logs:
                                                tprint(f"Successfully parsed after quote replacement for {repo_id}")
                                        except json.JSONDecodeError as je:
                                            # Try regular JSON parsing
                                            if debug_logs:
                                                tprint(f"Quote replacement failed for {repo_id}: {str(je)}")
                                                
                                            # Try special handling for Ren'Py and similar cases
                                            if "'Py" in langs or "'py" in langs:
                                                if debug_logs:
                                                    tprint(f"Detected likely 'Py language issue, attempting fix for {repo_id}")
                                                # Try to handle the special case with embedded single quotes
                                                try:
                                                    # Replace problematic patterns
                                                    fixed_langs = langs.replace("'Py", "_Py").replace("'py", "_py")
                                                    langs_dict = json.loads(fixed_langs)
                                                    # Restore original keys in the result
                                                    original_langs_dict = {}
                                                    for k, v in langs_dict.items():
                                                        original_key = k.replace("_Py", "'Py").replace("_py", "'py")
                                                        original_langs_dict[original_key] = v
                                                    langs_dict = original_langs_dict
                                                except Exception:
                                                    # If all else fails, create a basic dict from what we can extract
                                                    langs_dict = {}
                                                    for part in langs.strip("{}").split(","):
                                                        if ":" in part:
                                                            try:
                                                                key, value = part.split(":", 1)
                                                                key = key.strip().strip('"\'')
                                                                langs_dict[key] = 1
                                                            except Exception:
                                                                pass
                                            else:
                                                # Last resort - standard JSON parsing
                                                langs_dict = json.loads(langs)
                                
                                # Extract skills from langs_dict keys
                                skills = [key.lower() for key in langs_dict.keys()]
                                
                                # Track all unique skills
                                for skill in skills:
                                    # Ensure skill is lowercase and valid
                                    if skill and isinstance(skill, str):
                                        # Convert to lowercase and strip whitespace
                                        clean_skill = skill.lower().strip()
                                        if clean_skill:
                                            all_skills.add(clean_skill)
                                            repo_skills.add(clean_skill)
                                    
                                    if debug_logs and not skills:
                                        tprint(f"No languages found for {repo_id}, langs data: {langs}")
                            except Exception as e:
                                if debug_logs:
                                    tprint(f"Error processing languages for {repo_id}: {str(e)}")
                                
                                # Store the error for logging
                                language_error = str(e)
                                
                                # Log the language processing error with the actual value
                                log_error(datapath, "skills", repo_id, 
                                         f"Error processing languages: {str(e)}", 
                                         langs)
                        
                        # Extract contributors (team members)
                        contributor_error = None
                        if 'collabs' in row and pd.notna(row['collabs']):
                            collabs = row['collabs']
                            collabs_value = str(collabs)  # Store original value for logging
                            try:
                                # Remove verbose debug output
                                # if debug_logs:
                                #     tprint(f"Processing collabs for {repo_id}, type: {type(collabs)}, value: {collabs}")
                                
                                if isinstance(collabs, str):
                                    # Parse JSON string
                                    # Check if the string looks like JSON or is empty
                                    if collabs.strip() in ['[]', '{}', '']:
                                        if debug_logs:
                                            tprint(f"Empty collabs string for {repo_id}: '{collabs}'")
                                        collabs_list = []
                                    else:
                                        try:
                                            # First, try safe literal evaluation (for Python dict notation)
                                            try:
                                                collabs_list = ast.literal_eval(collabs)
                                            except (SyntaxError, ValueError):
                                                # Fall back to JSON parsing if literal_eval fails
                                                try:
                                                    collabs_list = json.loads(collabs)
                                                except json.JSONDecodeError:
                                                    # Try with replaced single quotes
                                                    try:
                                                        collabs_list = json.loads(collabs.replace("'", "\""))
                                                    except json.JSONDecodeError:
                                                        # If all parsing fails, use a simple extraction approach
                                                        if debug_logs:
                                                            tprint(f"Cannot parse collabs for {repo_id} - creating default")
                                                            
                                                        # Create a default contributor based on repo owner
                                                        repo_parts = repo_id.split('/')
                                                        if len(repo_parts) >= 2:
                                                            owner = repo_parts[0]
                                                            collabs_list = [{'login': owner, 'id': owner, 'contributions': 1}]
                                                        else:
                                                            collabs_list = []
                                        except Exception as e:
                                            if debug_logs:
                                                tprint(f"Error processing collabs for {repo_id}: {str(e)}")
                                else:
                                    # Already a list
                                    collabs_list = collabs
                                
                                # Safety check - ensure collabs_list is actually a list
                                if not isinstance(collabs_list, list):
                                    if debug_logs:
                                        tprint(f"collabs_list is not a list for {repo_id} - type: {type(collabs_list)}")
                                    
                                    # Log this error to the error log
                                    log_error(datapath, "collabs", repo_id, 
                                             f"collabs_list is not a list - type: {type(collabs_list)}", 
                                             collabs_list)
                                    
                                    # Mark this specific error for the team
                                    if contributor_error is None:
                                        contributor_error = f"collabs_list is not a list - type: {type(collabs_list)}"
                                    
                                    # Try to convert to list if it's a dict
                                    if isinstance(collabs_list, dict):
                                        collabs_list = [collabs_list]
                                    else:
                                        # Create a default contributor
                                        repo_parts = repo_id.split('/')
                                        if len(repo_parts) >= 2:
                                            owner = repo_parts[0]
                                            collabs_list = [{'login': owner, 'id': owner, 'contributions': 1}]
                                        else:
                                            collabs_list = []
                                
                                contributor_count = 0
                                for collab in collabs_list:
                                    if not isinstance(collab, dict):
                                        if debug_logs:
                                            tprint(f"Non-dict collab for {repo_id}: {collab}, type: {type(collab)}")
                                        continue
                                    
                                    # Extract only the specific fields we need
                                    extracted_fields = {}
                                    
                                    # Try to get login field
                                    if 'login' in collab:
                                        extracted_fields['login'] = collab['login']
                                    
                                    # Try to get id field
                                    if 'id' in collab:
                                        extracted_fields['id'] = collab['id']
                                        
                                    # Try to get contributions field
                                    if 'contributions' in collab:
                                        extracted_fields['contributions'] = collab['contributions']
                                    else:
                                        extracted_fields['contributions'] = 1  # Default value
                                    
                                    # Get ID and login, ensuring both have values
                                    collab_id = extracted_fields.get('id')
                                    login = extracted_fields.get('login')
                                    
                                    # If both are missing, skip this contributor
                                    if collab_id is None and login is None:
                                        if debug_logs:
                                            tprint(f"Skipping contributor with no ID or login for {repo_id}")
                                        
                                        # Log this error
                                        log_error(datapath, "contributor", repo_id, 
                                                 "Contributor has no ID or login", 
                                                 collab)
                                        continue
                                    
                                    # If ID is missing but login exists, use login as ID
                                    if collab_id is None:
                                        collab_id = login
                                    # If login is missing but ID exists, use ID as login
                                    elif login is None:
                                        login = str(collab_id)
                                    
                                    # Ensure collab_id is not None and is hashable
                                    try:
                                        hash(collab_id)
                                    except (TypeError, ValueError):
                                        collab_id = str(collab_id) if collab_id is not None else login
                                        
                                    try:
                                        contributor = GithContributor(
                                            id=collab_id,
                                            login=login,
                                            contributions=extracted_fields['contributions']
                                        )
                                        contributors.append(contributor)
                                        all_members.add(collab_id)  # Use the possibly fixed collab_id
                                        contributor_count += 1
                                    except Exception as e:
                                        if debug_logs:
                                            tprint(f"Error creating contributor for {repo_id}: {str(e)}")
                                            tprint(f"  ID: {collab_id}, Login: {login}")
                                
                                # Log contributor count for this repo
                                if debug_logs:
                                    # Don't log for every repo
                                    # tprint(f"Added {contributor_count} contributors for {repo_id}")
                                    
                                    # Only log when debug_logs is true AND no contributors were found
                                    if not contributors:
                                        tprint(f"No contributors found for {repo_id}")
                            except Exception as e:
                                if debug_logs:
                                    tprint(f"Error processing contributors for {repo_id}: {str(e)}")
                                contributor_error = str(e)
                        
                        # Log repository details to output log BEFORE applying filters
                        try:
                            # Track errors for this team
                            team_errors = []
                            
                            # Check for common error conditions
                            if repo_id == "unknown_repo" or not repo_id:
                                team_errors.append('no_repo_id')
                                
                            if not contributors:
                                team_errors.append('no_members')
                                
                            if not skills:
                                team_errors.append('no_skills')
                                
                            if contributor_error and 'collabs_list is not a list' in contributor_error:
                                team_errors.append('collabs_list_not_list')
                            
                            # Get the appropriate log file path
                            current_log_path = get_output_log_path(idx, datapath)
                            
                            # Initialize log file if it's the first time using it
                            if current_log_path not in initialized_log_files:
                                try:
                                    with open(current_log_path, 'w', encoding='utf-8') as f:
                                        f.write(f"GitHub Repository Data - Started: {datetime.now()}\n")
                                        f.write(f"File segment: {os.path.basename(current_log_path)}\n\n")
                                    initialized_log_files.add(current_log_path)
                                except Exception as e:
                                    if debug_logs:
                                        tprint(f"Error initializing log file {current_log_path}: {str(e)}")
                            
                            with open(current_log_path, 'a', encoding='utf-8') as output_file:
                                # Add team number (using idx + 1)
                                output_file.write(f"team#: {idx + 1}\n")
                                
                                # Include errors section
                                output_file.write(f"errors: {team_errors}\n")
                                
                                # Handle missing repo_id
                                if 'no_repo_id' in team_errors:
                                    output_file.write(f"repo_id: ''\n")
                                else:
                                    output_file.write(f"repo_id: {repo_id}\n")
                                
                                output_file.write(f"year: {year}\n")
                                
                                # Log contributors
                                if contributor_error:
                                    output_file.write(f"contributors: [] # ERROR - {contributor_error}\n")
                                else:
                                    output_file.write("contributors: [")
                                    if contributors:
                                        output_file.write("\n")
                                        for contributor in contributors:
                                            output_file.write(f"  {contributor.id} (login: {contributor.login}, contributions: {contributor.contributions}),\n")
                                    output_file.write("]\n")
                                
                                # Format langs data in a cleaner way
                                if 'langs' in row and pd.notna(row['langs']):
                                    if language_error:
                                        output_file.write(f"langs: {langs_value} # ERROR - {language_error}\n")
                                    else:
                                        try:
                                            # Try to format as a proper Python dict
                                            if isinstance(langs_value, str):
                                                formatted_langs = langs_value
                                            else:
                                                # If it's a dict-like object, format it prettily
                                                formatted_langs = json.dumps(langs_dict)
                                            output_file.write(f"langs: {formatted_langs}\n")
                                        except Exception:
                                            # Fallback if formatting fails
                                            output_file.write(f"langs: {langs_value}\n")
                                else:
                                    output_file.write("langs: {}\n")
                                
                                # Add horizontal line separator
                                output_file.write("\n" + "-" * 80 + "\n\n")
                        except Exception as e:
                            if debug_logs:
                                tprint(f"Error writing to output log for {repo_id}: {str(e)}")
                                tprint(traceback.format_exc())
                        
                        # NOW apply filters
                        # Track if this repo is valid and why if it's not
                        is_valid = True
                        invalid_reason = ""
                        
                        # Check year filter
                        if min_year > 0 and (year is None or year < min_year):
                            is_valid = False
                            invalid_reason = f"Year {year} is below minimum year {min_year}"
                        elif max_year > 0 and (year is None or year > max_year):
                            is_valid = False
                            invalid_reason = f"Year {year} is above maximum year {max_year}"
                        # Check duplicate filter
                        elif repo_id in repo_ids and remove_duplicates:
                            is_valid = False
                            invalid_reason = f"Duplicate repository ID"
                        # Check empty skills filter
                        elif not skills and remove_empty_skills:
                            is_valid = False
                            invalid_reason = f"No programming languages found"
                        # Check min/max skills filter
                        elif min_skills > 0 and len(skills) < min_skills:
                            is_valid = False
                            invalid_reason = f"Has {len(skills)} skills, below minimum of {min_skills}"
                        elif max_skills > 0 and len(skills) > max_skills:
                            is_valid = False
                            invalid_reason = f"Has {len(skills)} skills, above maximum of {max_skills}"
                        elif languages and not any(lang in skills for lang in languages):
                            is_valid = False
                            invalid_reason = f"Does not contain any required languages ({', '.join(languages)})"
                        # Check empty members filter
                        elif not contributors and remove_empty_members:
                            is_valid = False
                            invalid_reason = f"No contributors found"
                        # Check team size filters
                        elif min_team_size > 0 and len(contributors) < min_team_size:
                            is_valid = False
                            invalid_reason = f"Team size {len(contributors)} is below minimum of {min_team_size}"
                        elif max_team_size > 0 and len(contributors) > max_team_size:
                            is_valid = False
                            invalid_reason = f"Team size {len(contributors)} is above maximum of {max_team_size}"
                        
                        # Add repository ID to set if it's not a duplicate
                        if (repo_id not in repo_ids or not remove_duplicates):
                            repo_ids.add(repo_id)
                        
                        # If valid, add to repositories dict, otherwise log invalid reason
                        if is_valid:
                            # Create repository object
                            repository = Repository(
                                id=repo_id,
                                contributors=contributors,
                                skills=skills,
                                year=year,
                                created_at=created_at,
                                pushed_at=row['pushed_at'] if 'pushed_at' in row and pd.notna(row['pushed_at']) else None
                            )
                            
                            # Add to repositories dict
                            repositories[repo_id] = repository
                        else:
                            # Log invalid repository with reason
                            try:
                                with open(invalids_log_path, 'a', encoding='utf-8') as invalid_file:
                                    invalid_file.write(f"repo_id: {repo_id}, reason: {invalid_reason}\n")
                            except Exception as e:
                                if debug_logs:
                                    tprint(f"Error writing to invalids log for {repo_id}: {str(e)}")
                        
                        # Update progress bar
                        pbar.update(1)
                    except Exception as e:
                        if debug_logs:
                            tprint(f"Error processing repository {repo_id if 'repo_id' in locals() else 'unknown'}: {str(e)}")
                            tprint(traceback.format_exc())
                        pbar.update(1)
        
        except Exception as e:
            tprint(f"Error reading Gith data: {str(e)}")
            tprint(traceback.format_exc())
            return []
        
        processing_time = time() - start_time
        tprint(f"Processed {len(repositories)} valid repositories in {processing_time:.2f} seconds")
        tprint(f"Found {len(all_members)} unique contributors and {len(all_skills)} unique skills")
        
        # Write all unique skills to a log file, sorted alphabetically
        skills_log_path = os.path.join(os.path.dirname(datapath), "gith_skills.log")
        try:
            with open(skills_log_path, 'w', encoding='utf-8') as skills_file:
                # Sort skills alphabetically
                sorted_skills = sorted(all_skills)
                
                # Filter out some common non-programming language entries
                suspicious_entries = 0
                excluded_entries = []
                
                for skill in sorted_skills:
                    # Skip skills that are too long or contain suspicious characters
                    if len(skill) > 50 or any(c in skill for c in ['<', '>', '=', '{', '}', ';', '(', ')']):
                        suspicious_entries += 1
                        excluded_entries.append(skill)
                        continue
                        
                    # Skip skills that are just numbers
                    if skill.replace('.', '').isdigit():
                        suspicious_entries += 1
                        excluded_entries.append(skill)
                        continue
                        
                    # Write valid skills to the log file
                    skills_file.write(f"{skill}\n")
                
                # Write a summary at the end
                if suspicious_entries > 0:
                    skills_file.write(f"\n# {suspicious_entries} suspicious entries excluded:\n")
                    for entry in excluded_entries[:20]:  # Show first 20 excluded entries
                        skills_file.write(f"# - {entry}\n")
                    if len(excluded_entries) > 20:
                        skills_file.write(f"# ... and {len(excluded_entries) - 20} more\n")
                        
            tprint(f"Wrote {len(all_skills) - suspicious_entries} unique skills to {skills_log_path} ({suspicious_entries} suspicious entries excluded)")
        except Exception as e:
            tprint(f"Error writing skills log: {str(e)}")
        
        # Add summary to error log
        try:
            with open(error_log_path, 'a', encoding='utf-8') as error_file:
                error_file.write(f"\n\n--- PROCESSING SUMMARY ---\n")
                error_file.write(f"Completed: {datetime.now()}\n")
                error_file.write(f"Total repositories processed: {len(df)}\n")
                error_file.write(f"Valid repositories: {len(repositories)}\n")
                error_file.write(f"Unique contributors: {len(all_members)}\n")
                error_file.write(f"Unique skills: {len(all_skills)}\n")
                error_file.write(f"Processing time: {time() - start_time:.2f} seconds\n")
        except Exception as e:
            tprint(f"Error writing error log summary: {str(e)}")
        
        # Add summary to output log
        try:
            # Add summary to each log file
            for log_path in initialized_log_files:
                with open(log_path, 'a', encoding='utf-8') as output_file:
                    output_file.write(f"\n\n{'=' * 80}\n")
                    output_file.write(f"DATASET SUMMARY\n")
                    output_file.write(f"{'=' * 80}\n\n")
                    output_file.write(f"Completed: {datetime.now()}\n")
                    output_file.write(f"Total repositories processed: {len(df)}\n")
                    output_file.write(f"Valid repositories logged: {len(repositories)}\n")
                    output_file.write(f"Unique contributors: {len(all_members)}\n")
                    output_file.write(f"Unique skills: {len(all_skills)}\n")
                    output_file.write(f"Processing time: {time() - start_time:.2f} seconds\n")
                    output_file.write(f"Log files created: {len(initialized_log_files)}\n")
                    output_file.write(f"Log files: {', '.join([os.path.basename(p) for p in initialized_log_files])}\n")
        except Exception as e:
            tprint(f"Error writing output log summary: {str(e)}")
            
        return list(repositories.values()) 