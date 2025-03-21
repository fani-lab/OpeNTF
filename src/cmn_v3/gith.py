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
import pickle
import re

# Add the project root to the Python path if it's not already there
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .team import Team
from .gith_contributor import GithContributor
from utils.tprint import tprint
from .helper_functions import get_settings

# Get settings from the appropriate param.py
SETTINGS = get_settings()


class Repository(Team):
    """
    Class representing a Gith repository team

    This class extends the base Team class for Gith-specific functionality.
    """

    # Add domain_params as a class attribute, using the consolidated settings
    domain_params = SETTINGS["data"]

    def __init__(
        self,
        id,
        contributors,
        skills,
        year,
        location=None,
        created_at=None,
        pushed_at=None,
    ):
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
            repo_parts = id.split("/")
            if len(repo_parts) >= 2:
                owner = repo_parts[0]
                contributors = [GithContributor(id=owner, login=owner, contributions=1)]
            else:
                contributors = [
                    GithContributor(id="unknown", login="unknown", contributions=1)
                ]

        # Validate each contributor has a valid ID
        valid_contributors = []
        for contrib in contributors:
            if (
                contrib is not None
                and hasattr(contrib, "id")
                and contrib.id is not None
            ):
                valid_contributors.append(contrib)

        # If no valid contributors remain, add a default one
        if not valid_contributors:
            repo_parts = id.split("/")
            if len(repo_parts) >= 2:
                owner = repo_parts[0]
                valid_contributors = [
                    GithContributor(id=owner, login=owner, contributions=1)
                ]
            else:
                valid_contributors = [
                    GithContributor(id="unknown", login="unknown", contributions=1)
                ]

        # Ensure skills is not None
        if skills is None or len(skills) == 0:
            skills = []

        # Handle year as None or invalid
        if year is None:
            year = datetime.now().year

        # Convert year to datetime for chronological ordering
        try:
            creation_datetime = (
                datetime.strptime(f"{year}-01-01", "%Y-%m-%d")
                if isinstance(year, int)
                else None
            )
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
    def read_and_filter_data_v3(cls, datapath, output_path=None):
        """
        Read and filter Gith repository data

        Args:
            datapath: Path to the Gith dataset file (CSV)
            output_path: Path to store preprocessed results and logs (optional)

        Returns:
            List of Repository objects
        """
        start_time = time()

        # Determine the output path - use the output_path parameter if provided,
        # otherwise default to the same directory as the input file
        output_dir = output_path if output_path else os.path.dirname(datapath)

        # Check if teams data exists
        if output_path:
            teams_path = os.path.join(output_path, "teams.pkl")

            if os.path.exists(teams_path):
                tprint(f"Found teams data at {teams_path}. Using existing teams.")
                try:
                    with open(teams_path, "rb") as f:
                        teams = pickle.load(f)
                    tprint(f"Successfully loaded {len(teams)} repositories")
                    return teams
                except Exception as e:
                    tprint(f"Error loading teams: {str(e)}")
                    tprint("Proceeding with vectors conversion...")

            # If teams don't exist but vectors do, convert them
            teamsvecs_path = os.path.join(output_path, "teamsvecs.pkl")
            indexes_path = os.path.join(output_path, "indexes.pkl")

            if os.path.exists(teamsvecs_path) and os.path.exists(indexes_path):
                tprint(
                    f"Found vectors at {teamsvecs_path}. Converting to repositories."
                )
                try:
                    with open(teamsvecs_path, "rb") as f:
                        vecs = pickle.load(f)

                    with open(indexes_path, "rb") as f:
                        indexes = pickle.load(f)

                    # Convert data back to Repository objects
                    repositories = cls.convert_vecs_to_repositories(
                        vecs, indexes, output_path
                    )

                    # Save the converted repositories for future use
                    with open(teams_path, "wb") as f:
                        pickle.dump(repositories, f)

                    tprint(
                        f"Successfully converted and saved {len(repositories)} repositories"
                    )
                    return repositories
                except Exception as e:
                    tprint(f"Error converting vectors: {str(e)}")
                    tprint("Proceeding with full data processing...")

        tprint(f"Reading Gith data from {datapath}")

        # Track repositories
        repositories = []

        # Read the CSV file
        try:
            # Try multiple encodings if necessary
            encodings_to_try = ["latin1", "utf-8", "cp1252"]

            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(datapath, encoding=encoding)
                    tprint(f"Successfully loaded data using {encoding} encoding")

                    # Check for required columns
                    required_columns = ["repo", "langs", "collabs", "created_at"]
                    missing_columns = [
                        col for col in required_columns if col not in df.columns
                    ]
                    if missing_columns:
                        tprint(
                            f"WARNING: The following required columns are missing: {missing_columns}"
                        )

                    break
                except UnicodeDecodeError:
                    tprint(
                        f"Failed to load with {encoding} encoding, trying another..."
                    )
                    continue

            # Process each repository
            with tqdm(
                total=len(df),
                desc="Processing repositories",
                unit="repos",
                unit_scale=True,
                miniters=5000,  # Update progress bar less frequently
            ) as pbar:
                # Use a list with pre-allocated capacity for better performance
                repositories = []

                # Pre-compile regex patterns for common operations
                single_quote_pattern = re.compile(r"'")

                # Prepare a set of common languages for fast lookup
                common_languages = {
                    "javascript",
                    "python",
                    "java",
                    "c++",
                    "c#",
                    "php",
                    "ruby",
                    "typescript",
                    "go",
                    "swift",
                    "kotlin",
                    "rust",
                    "c",
                    "html",
                    "css",
                    "shell",
                    "perl",
                    "scala",
                }

                # Store log entries to show processed data
                log_entries = []

                # Process in batches for better performance
                batch_size = 10000
                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i : i + batch_size]
                    batch_repositories = []

                    # Process each team in the batch
                    for _, row in batch_df.iterrows():
                        try:
                            # Extract repository ID first (fastest operation)
                            repo_id = (
                                str(row["repo"])
                                if "repo" in row and pd.notna(row["repo"])
                                else f"unknown_repo_{len(repositories) + len(batch_repositories)}"
                            )

                            # Extract created_at and year in a more efficient way
                            created_at = None
                            year = None

                            if "created_at" in row and pd.notna(row["created_at"]):
                                created_at = row["created_at"]
                                if isinstance(created_at, str):
                                    # Fast year extraction without datetime parsing
                                    if (
                                        len(created_at) >= 4
                                        and created_at[0:4].isdigit()
                                    ):
                                        year = int(created_at[0:4])
                                    else:
                                        try:
                                            created_dt = datetime.strptime(
                                                created_at, "%Y-%m-%dT%H:%M:%SZ"
                                            )
                                            year = created_dt.year
                                        except (ValueError, TypeError):
                                            pass

                            # FILTER BY YEAR - only if explicitly enabled
                            min_year = SETTINGS["data"]["filters"]["gith"].get(
                                "min_year"
                            )
                            max_year = SETTINGS["data"]["filters"]["gith"].get(
                                "max_year"
                            )
                            if (
                                year is not None
                                and min_year is not None
                                and max_year is not None
                            ):
                                if year < min_year or year > max_year:
                                    # Skip this repository as it's outside the year range
                                    pbar.update(1)
                                    continue

                            # Process languages (skills) more efficiently
                            skills = []
                            if "langs" in row and pd.notna(row["langs"]):
                                langs = row["langs"]
                                # Handle most common case first - empty dict-like strings
                                if isinstance(langs, str):
                                    if langs.strip() in ["[]", "{}", ""]:
                                        skills = []
                                    else:
                                        # Try faster methods first
                                        clean_langs = langs.lower()

                                        # First try proper parsing before relying on common language detection
                                        try:
                                            # Try ast.literal_eval first as it's safer and often faster
                                            langs_dict = ast.literal_eval(langs)
                                            if isinstance(langs_dict, dict):
                                                skills = [
                                                    key.lower()
                                                    for key in langs_dict.keys()
                                                ]
                                            elif isinstance(langs_dict, list):
                                                skills = [
                                                    str(item).lower()
                                                    for item in langs_dict
                                                    if item
                                                ]
                                            else:
                                                skills = []
                                        except (SyntaxError, ValueError):
                                            # Try JSON parsing with quotes fixed
                                            try:
                                                fixed_langs = single_quote_pattern.sub(
                                                    '"', langs
                                                )
                                                langs_dict = json.loads(fixed_langs)
                                                if isinstance(langs_dict, dict):
                                                    skills = [
                                                        key.lower()
                                                        for key in langs_dict.keys()
                                                    ]
                                                elif isinstance(langs_dict, list):
                                                    skills = [
                                                        str(item).lower()
                                                        for item in langs_dict
                                                        if item
                                                    ]
                                                else:
                                                    skills = []
                                            except Exception:
                                                # If JSON parsing failed, try more basic approaches
                                                # Check for common languages with more precise matching
                                                found_langs = []
                                                for lang in common_languages:
                                                    if (
                                                        f'"{lang}"' in clean_langs
                                                        or f"'{lang}'" in clean_langs
                                                        or f" {lang} "
                                                        in f" {clean_langs} "
                                                        or f",{lang},"
                                                        in f",{clean_langs},"
                                                    ):
                                                        found_langs.append(lang)

                                                if found_langs:
                                                    skills = found_langs
                                                else:
                                                    # Last resort - split by commas and other separators
                                                    if "," in langs:
                                                        skills = [
                                                            l.strip().lower()
                                                            for l in langs.split(",")
                                                            if l.strip()
                                                        ]
                                                    elif ";" in langs:
                                                        skills = [
                                                            l.strip().lower()
                                                            for l in langs.split(";")
                                                            if l.strip()
                                                        ]
                                                    else:
                                                        # Just use the whole string if it's not too long
                                                        skills = (
                                                            [langs.lower()]
                                                            if len(langs) < 30
                                                            else []
                                                        )
                                elif isinstance(langs, dict):
                                    # Direct dict - fastest case
                                    skills = [key.lower() for key in langs.keys()]
                                elif isinstance(langs, list):
                                    # Direct list
                                    skills = [
                                        str(item).lower() for item in langs if item
                                    ]
                                else:
                                    skills = []
                            else:
                                skills = []

                            # Filter out non-language items
                            if skills:
                                skills = [
                                    s
                                    for s in skills
                                    if len(s) < 30
                                    and not (s.startswith("http") or s.startswith("{"))
                                ]

                            # Process contributors more efficiently
                            contributors = []
                            if "collabs" in row and pd.notna(row["collabs"]):
                                collabs = row["collabs"]

                                # Handle more complex cases
                                try:
                                    collabs_list = None

                                    # Direct list/dict case
                                    if isinstance(collabs, list):
                                        collabs_list = collabs
                                    elif isinstance(collabs, dict):
                                        # Check for API error responses or "too many contributors" indicator
                                        if any(
                                            key in collabs
                                            for key in [
                                                "message",
                                                "error",
                                                "documentation_url",
                                            ]
                                        ):
                                            # This is an error response, treat as empty list
                                            collabs_list = []
                                        elif any(
                                            key in str(collabs).lower()
                                            for key in [
                                                "too many",
                                                "max",
                                                "limit exceed",
                                                "truncated",
                                            ]
                                        ):
                                            # This appears to be a "too many contributors" response
                                            collabs_list = []
                                        else:
                                            # Treat as a single contributor
                                            collabs_list = [collabs]
                                    else:
                                        # String parsing - more expensive
                                        try:
                                            # Use ast.literal_eval for speed
                                            collabs_list = ast.literal_eval(collabs)
                                        except (SyntaxError, ValueError, TypeError):
                                            try:
                                                # Fallback to JSON with quote fixes
                                                fixed_collabs = (
                                                    single_quote_pattern.sub(
                                                        '"', collabs
                                                    )
                                                )
                                                collabs_list = json.loads(fixed_collabs)
                                            except Exception:
                                                # Default to owner from repo ID
                                                repo_parts = repo_id.split("/")
                                                if len(repo_parts) >= 2:
                                                    owner = repo_parts[0]
                                                    collabs_list = [
                                                        {
                                                            "login": owner,
                                                            "id": owner,
                                                            "contributions": 1,
                                                        }
                                                    ]
                                                else:
                                                    collabs_list = []

                                    # Process contributors from the list
                                    if isinstance(collabs_list, list):
                                        for collab in collabs_list:
                                            if isinstance(collab, dict):
                                                collab_id = collab.get("id")
                                                login = collab.get("login")
                                                contributions = collab.get(
                                                    "contributions", 1
                                                )

                                                # Ensure ID and login are valid
                                                if collab_id is None and login is None:
                                                    collab_id = f"unknown_{len(contributors)}_{repo_id}"
                                                    login = collab_id
                                                elif collab_id is None:
                                                    collab_id = login
                                                elif login is None:
                                                    login = str(collab_id)

                                                # Create contributor
                                                try:
                                                    contributors.append(
                                                        GithContributor(
                                                            id=str(collab_id),
                                                            login=str(login),
                                                            contributions=int(
                                                                contributions
                                                            ),
                                                        )
                                                    )
                                                except:
                                                    pass
                                            elif collab is not None:
                                                # Use string value as ID and login
                                                try:
                                                    collab_str = str(collab)
                                                    contributors.append(
                                                        GithContributor(
                                                            id=collab_str,
                                                            login=collab_str,
                                                            contributions=1,
                                                        )
                                                    )
                                                except:
                                                    pass
                                    elif isinstance(collabs_list, dict):
                                        # Single contributor as dict
                                        try:
                                            collab_id = collabs_list.get("id")
                                            login = collabs_list.get("login")
                                            contributions = collabs_list.get(
                                                "contributions", 1
                                            )

                                            if collab_id is None and login is None:
                                                collab_id = f"unknown_{repo_id}"
                                                login = collab_id
                                            elif collab_id is None:
                                                collab_id = login
                                            elif login is None:
                                                login = str(collab_id)

                                            contributors.append(
                                                GithContributor(
                                                    id=str(collab_id),
                                                    login=str(login),
                                                    contributions=int(contributions),
                                                )
                                            )
                                        except:
                                            pass
                                except Exception:
                                    # Default to owner from repo ID
                                    repo_parts = repo_id.split("/")
                                    if len(repo_parts) >= 2:
                                        owner = repo_parts[0]
                                        contributors = [
                                            GithContributor(
                                                id=owner, login=owner, contributions=1
                                            )
                                        ]
                                    else:
                                        contributors = [
                                            GithContributor(
                                                id=f"default_{repo_id}",
                                                login=f"default_{repo_id}",
                                                contributions=1,
                                            )
                                        ]

                            # Ensure contributors is not empty
                            if not contributors:
                                contributors = [
                                    GithContributor(
                                        id=f"default_{repo_id}",
                                        login=f"default_{repo_id}",
                                        contributions=1,
                                    )
                                ]

                            # Create repository object - more efficient with minimal validation
                            repository = Repository(
                                id=repo_id,
                                contributors=contributors,
                                skills=skills,
                                year=year,
                                created_at=created_at,
                            )

                            # Log every entry in memory to reduce I/O
                            if output_path:
                                # Use contributor logins instead of IDs in members array
                                log_members = [c.login for c in contributors]

                                log_entries.append(
                                    f"entry_{len(repositories) + len(batch_repositories) + 1}\n"
                                    f"repo_id: {repo_id}\n"
                                    f"skills: {skills}\n"
                                    f"members: {log_members}\n"
                                    f"---\n"
                                )

                            # Add to batch repositories
                            batch_repositories.append(repository)

                        except Exception as e:
                            # Create minimal repository with default values
                            try:
                                repo_id = f"error_repo_{len(repositories) + len(batch_repositories)}"
                                if "repo" in row and pd.notna(row["repo"]):
                                    repo_id = str(row["repo"])

                                repository = Repository(
                                    id=repo_id,
                                    contributors=[
                                        GithContributor(
                                            id=f"default_{repo_id}",
                                            login=f"default_{repo_id}",
                                            contributions=1,
                                        )
                                    ],
                                    skills=[],
                                    year=None,
                                )

                                batch_repositories.append(repository)
                            except:
                                pass

                        # Update progress bar
                        pbar.update(1)

                    # Write log entries in batch to reduce I/O operations
                    if output_path and log_entries:
                        logs_dir = os.path.join(output_dir, "logs")
                        os.makedirs(logs_dir, exist_ok=True)

                        entries_log_path = os.path.join(
                            logs_dir, "entries_processed.log"
                        )
                        with open(entries_log_path, "a", encoding="utf-8") as log_file:
                            log_file.writelines(log_entries)

                        # Clear log entries after writing to file
                        log_entries = []

                    # Add batch repositories to main list
                    repositories.extend(batch_repositories)

            # After processing all batches, collect all unique skills across the entire dataset
            if output_path:
                all_unique_skills = set()
                for repo in repositories:
                    all_unique_skills.update(repo.skills)

                # Write all unique skills to skills.log (sorted for readability)
                skills_log_path = os.path.join(
                    os.path.join(output_dir, "logs"), "skills.log"
                )
                with open(skills_log_path, "w", encoding="utf-8") as skills_file:
                    for skill in sorted(all_unique_skills):
                        skills_file.write(f"{skill}\n")

        except Exception as e:
            tprint(f"Error reading Gith data: {str(e)}")
            tprint(traceback.format_exc())
            return []

        processing_time = time() - start_time
        tprint(
            f"Processed {len(repositories)} repositories in {processing_time:.2f} seconds"
        )

        return repositories

    @classmethod
    def convert_vecs_to_repositories(cls, vecs, indexes, output_path=None):
        """
        Convert filtered vectors back to Repository objects

        Args:
            vecs: Dictionary containing sparse vectors for teams
            indexes: Dictionary containing indexes for skills and members
            output_path: Path to store logs (optional)

        Returns:
            List of Repository objects
        """
        tprint("Converting filtered vectors back to Repository objects...")

        # Extract necessary indexes
        i2s = indexes["i2s"]  # index to skill
        i2c = indexes["i2c"]  # index to contributor

        # Get matrices
        id_vector = vecs["id"]
        skill_matrix = vecs["skill"]
        member_matrix = vecs["member"]

        # Convert to arrays if they are sparse matrices
        if hasattr(skill_matrix, "toarray"):
            skill_array = skill_matrix.toarray()
        else:
            skill_array = skill_matrix

        if hasattr(member_matrix, "toarray"):
            member_array = member_matrix.toarray()
        else:
            member_array = member_matrix

        # Create Repository objects
        repositories = []

        with tqdm(
            total=len(id_vector), desc="Converting repositories", unit="repos"
        ) as pbar:
            for i in range(len(id_vector)):
                try:
                    # Get repository ID
                    repo_id = id_vector[i]

                    # Get skills for this repository
                    repo_skills = []
                    for j in range(skill_array.shape[1]):
                        if skill_array[i, j] > 0:
                            skill = i2s.get(j)
                            if skill:
                                repo_skills.append(skill)

                    # Get contributors for this repository
                    repo_contributors = []
                    for j in range(member_array.shape[1]):
                        if member_array[i, j] > 0:
                            contributor_id = i2c.get(j)
                            if contributor_id:
                                # Create a GithContributor object
                                # Since we don't have all the original data, we'll create a simplified version
                                contributor = GithContributor(
                                    id=contributor_id,
                                    login=contributor_id,  # Use ID as login
                                    contributions=1,  # Default value
                                )
                                repo_contributors.append(contributor)

                    # Extract year from repo_id if possible (format: owner/repo_year)
                    year = None
                    try:
                        # Try to extract year from the end of the repo name
                        if isinstance(repo_id, str) and "_" in repo_id:
                            year_part = repo_id.split("_")[-1]
                            if year_part.isdigit() and len(year_part) == 4:
                                year = int(year_part)
                    except:
                        year = None

                    # Create Repository object
                    repository = Repository(
                        id=repo_id,
                        contributors=repo_contributors,
                        skills=repo_skills,
                        year=year,
                    )

                    # Log entry for this repository
                    if output_path:
                        logs_dir = os.path.join(output_path, "logs")
                        os.makedirs(logs_dir, exist_ok=True)

                        # Use contributor logins instead of IDs in members array
                        log_members = [c.login for c in repo_contributors]

                        entries_log_path = os.path.join(
                            logs_dir, "entries_processed.log"
                        )
                        with open(entries_log_path, "a", encoding="utf-8") as log_file:
                            log_file.write(f"entry_{i + 1}\n")
                            log_file.write(f"repo_id: {repo_id}\n")
                            log_file.write(f"skills: {repo_skills}\n")
                            log_file.write(f"members: {log_members}\n")
                            log_file.write("---\n")

                    repositories.append(repository)

                except Exception as e:
                    tprint(f"Error converting repository {i}: {str(e)}")

                pbar.update(1)

        # After processing all repositories, collect and write unique skills
        if output_path:
            all_unique_skills = set()
            for repo in repositories:
                all_unique_skills.update(repo.skills)

            # Write all unique skills to skills.log (sorted for readability)
            logs_dir = os.path.join(output_path, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            skills_log_path = os.path.join(logs_dir, "skills.log")
            with open(skills_log_path, "w", encoding="utf-8") as skills_file:
                for skill in sorted(all_unique_skills):
                    skills_file.write(f"{skill}\n")

        tprint(f"Successfully converted {len(repositories)} repositories")
        return repositories
