#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
from time import time
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import traceback
import sys
import re
import ast

# Add the project root to the Python path if it's not already there
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use relative imports for modules in the same package
from .team import Team
from .dblp_author import DblpAuthor
from utils.tprint import tprint
from .helper_functions import get_settings

# Get settings from the appropriate param.py
SETTINGS = get_settings()


class Publication(Team):
    """
    Class representing a publication team from DBLP dataset

    This class extends the base Team class for DBLP-specific functionality.
    """

    # Add domain_params as a class attribute with domain_name set to "dblp"
    domain_params = SETTINGS["data"]
    domain_params["domain_name"] = "dblp"

    def __init__(
        self,
        id,
        title,
        authors,
        skills,
        year,
        venue_raw=None,
        published_at=None,
        doc_type=None,
    ):
        """
        Initialize a Publication object

        Args:
            id: Unique identifier for the publication
            title: Title of the publication
            authors: List of Author objects representing the publication authors
            skills: List of skills associated with the publication
            year: Year of publication
            venue_raw: Venue name as a string (e.g., journal, conference)
            published_at: Publication timestamp (optional)
            doc_type: Document type (e.g., 'article', 'inproceedings', 'book') (optional)
        """
        # Safety validation of inputs
        if id is None:
            id = "unknown_paper"

        # Ensure authors is a valid list with at least one member
        if authors is None or len(authors) == 0:
            # Create a default author
            authors = [DblpAuthor(id="unknown", name="unknown", org="")]

        # Validate each author has a valid ID and ensure uniqueness
        valid_authors = []
        authors_dict = {}
        for author in authors:
            if author is not None and hasattr(author, "id") and author.id is not None:
                # Only add each author once (by ID)
                if author.id not in authors_dict:
                    authors_dict[author.id] = author

        # Convert dictionary to list
        valid_authors = list(authors_dict.values())

        # If no valid authors remain, add a default one
        if not valid_authors:
            valid_authors = [DblpAuthor(id="unknown", name="unknown", org="")]

        # Ensure skills is not None and deduplicate while preserving order
        if skills is None:
            skills = []
        else:
            # Deduplicate skills while preserving order
            seen_skills = set()
            unique_skills = []
            for skill in skills:
                if skill not in seen_skills:
                    seen_skills.add(skill)
                    unique_skills.append(skill)
            skills = unique_skills

        # Handle year as None or invalid
        if year is None:
            year = datetime.now().year

        # Convert year to datetime for chronological ordering
        try:
            publication_datetime = (
                datetime.strptime(f"{year}-01-01", "%Y-%m-%d")
                if isinstance(year, int)
                else None
            )
        except (ValueError, TypeError):
            publication_datetime = datetime.now()

        if published_at:
            try:
                publication_datetime = datetime.strptime(
                    published_at, "%Y-%m-%dT%H:%M:%SZ"
                )
            except (ValueError, TypeError):
                # Keep existing publication_datetime if parsing fails
                pass

        # Initialize the base Team class
        super().__init__(
            id, valid_authors, skills, publication_datetime, location=venue_raw
        )

        # DBLP-specific attributes
        self.id = id
        self.title = title
        self.authors = valid_authors
        self.skills = skills
        self.year = year
        self.venue_raw = venue_raw
        self.published_at = published_at
        self.doc_type = doc_type

    def __str__(self):
        """String representation of the publication"""
        authors_str = ", ".join(a.name for a in self.members[:3])
        if len(self.members) > 3:
            authors_str += f" +{len(self.members) - 3} more"
        return f"Publication({self.id}): {self.title} by {authors_str} ({self.year})"

    @staticmethod
    def detect_json_format(filepath):
        """
        Detect if the JSON file is in array format or lines format.

        Args:
            filepath: Path to the JSON file

        Returns:
            str: 'array' or 'lines' or 'unknown'
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                # Read the first 10 lines or up to 1000 characters
                first_chars = f.read(1000)

            # Check if it starts with '[' - Array format
            if first_chars.strip().startswith("["):
                return "array"

            # Check if each line is a valid JSON object - Lines format
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line and first_line[0] == "{" and json.loads(first_line):
                        return "lines"
            except json.JSONDecodeError:
                pass

            return "unknown"
        except Exception as e:
            tprint(f"Error detecting JSON format: {str(e)}")
            return "unknown"

    @classmethod
    def read_and_filter_data_v3(cls, datapath, output_path=None):
        """
        Read DBLP data from a JSON file, apply filters during reading, and convert to Publication objects

        Args:
            datapath: Path to the DBLP JSON file
            output_path: Path to store preprocessed results and logs (optional)

        Returns:
            List of Publication objects
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
                    tprint(f"Successfully loaded {len(teams)} publications")
                    return teams
                except Exception as e:
                    tprint(f"Error loading teams: {str(e)}")
                    tprint("Proceeding with full data processing...")

        tprint(f"Reading and filtering DBLP data from {datapath}")

        # Track statistics
        publications = {}  # Use a dictionary to track by ID
        paper_ids = set()  # Use a set for faster lookups
        all_members = set()  # For statistics only
        all_skills = set()  # For statistics only

        # Debugging counters
        total_lines = 0
        json_parse_failures = 0
        missing_id = 0
        exceptions = 0

        # We're not doing inline filtering anymore, so we don't need these counters
        # Filter parameters are handled by the filter_functions modules

        # Get processing parameters
        debug_logs = SETTINGS["data"]["processing"].get("debug_logs", False)

        # Store log entries to show processed data
        log_entries = []

        # Detect the JSON format
        json_format = cls.detect_json_format(datapath)
        tprint(f"Detected JSON format: {json_format}")

        # For array format, try to convert to lines format first
        if json_format == "array":
            tprint(
                "Converting JSON array format to JSON Lines format for faster processing..."
            )
            # Create a temporary file with .jsonl extension
            jsonl_path = os.path.splitext(datapath)[0] + ".jsonl"

            # Check if the jsonl file already exists
            if os.path.exists(jsonl_path) and os.path.getsize(jsonl_path) > 0:
                tprint(
                    f"Found existing JSON Lines file at {jsonl_path}. Using it instead."
                )
                datapath = jsonl_path
            else:
                # If jsonl doesn't exist or is empty, run the conversion utility
                tprint(
                    "Need to convert JSON array to JSON Lines. This might take some time."
                )
                tprint("Please run this command first:")
                tprint(f"python utils/convert_dblp_json_to_jsonl.py {datapath}")
                tprint(f"Then rerun the data processing with the output .jsonl file")
                return []

        # Read and process the JSON file
        with tqdm(
            desc="Reading DBLP data",
            unit="B",
            unit_scale=True,
            total=os.path.getsize(datapath),
        ) as pbar:
            with open(datapath, "r", encoding="utf-8") as jf:
                for line_num, line in enumerate(jf):
                    pbar.update(len(line))
                    total_lines += 1

                    if not line.strip():
                        continue

                    try:
                        # Parse JSON line
                        line = line.strip()

                        # Try to parse JSON
                        try:
                            jsonline = json.loads(line.lower())
                        except json.JSONDecodeError:
                            json_parse_failures += 1
                            if json_parse_failures < 10:
                                tprint(
                                    f"JSON parse error on line {line_num+1}: {line[:100]}..."
                                )
                            continue

                        # Extract basic fields
                        paper_id = jsonline.get("id", None)
                        if paper_id is None:
                            missing_id += 1
                            continue

                        title = jsonline.get("title", "")
                        year = int(jsonline.get("year", 0))
                        doc_type = jsonline.get("doc_type", "")

                        # Add to paper_ids set for tracking
                        paper_ids.add(paper_id)

                        # Create a dictionary to track unique authors by ID
                        paper_members_dict = {}

                        # Process authors
                        authors = jsonline.get("authors", [])

                        # Process authors without filtering - we'll use filter_functions instead
                        for author in authors:
                            author_id = author.get("id", None)
                            if author_id is None:
                                continue

                            name = author.get("name", "").replace(" ", "_")
                            org = (
                                author.get("org", "").replace(" ", "_")
                                if "org" in author
                                else ""
                            )

                            # Create author and add to paper's members dictionary (overwriting if duplicate ID)
                            author_obj = DblpAuthor(author_id, name, org)
                            paper_members_dict[author_id] = author_obj
                            all_members.add(author_id)  # For statistics

                        # Convert dictionary values to list
                        paper_members = list(paper_members_dict.values())

                        # Process skills without filtering - we'll use filter_functions instead
                        fos = jsonline.get("fos", None)
                        keywords = jsonline.get("keywords", [])

                        # Create a set to collect all skills first (for deduplication)
                        temp_skills_set = set()

                        # Process field of study - add to skills first
                        if fos:
                            if isinstance(fos, dict) and "name" in fos:
                                skill = fos["name"].lower()
                                temp_skills_set.add(skill)
                                all_skills.add(skill)
                            elif isinstance(fos, list):
                                for field in fos:
                                    if isinstance(field, dict) and "name" in field:
                                        skill = field["name"].lower()
                                        temp_skills_set.add(skill)
                                        all_skills.add(skill)

                        # Now add keywords to skills
                        if keywords:
                            for kw in keywords:
                                if kw:
                                    skill = kw.lower()
                                    temp_skills_set.add(skill)
                                    all_skills.add(skill)

                        # Create the final skills list, maintaining correct order (fos first, then keywords)
                        paper_skills = []

                        # Add fos skills first
                        if fos:
                            if isinstance(fos, dict) and "name" in fos:
                                skill = fos["name"].lower()
                                if skill in temp_skills_set:
                                    paper_skills.append(skill)
                                    temp_skills_set.remove(skill)
                            elif isinstance(fos, list):
                                for field in fos:
                                    if isinstance(field, dict) and "name" in field:
                                        skill = field["name"].lower()
                                        if skill in temp_skills_set:
                                            paper_skills.append(skill)
                                            temp_skills_set.remove(skill)

                        # Then add remaining skills (keywords)
                        paper_skills.extend(sorted(temp_skills_set))

                        # Get venue as string
                        venue_raw = ""
                        venue = jsonline.get("venue", None)
                        if venue and isinstance(venue, dict):
                            venue_raw = venue.get("raw", "")

                        # Format published_at timestamp (if available)
                        published_at = None
                        if year:
                            # Create a basic timestamp from the year
                            published_at = f"{year}-01-01T00:00:00Z"

                        # Create Publication object
                        publication = cls(
                            id=paper_id,
                            title=title,
                            authors=paper_members,
                            skills=paper_skills,
                            year=year,
                            venue_raw=venue_raw,
                            published_at=published_at,
                            doc_type=doc_type,
                        )

                        # Log entry for this publication if output path is specified
                        if output_path:
                            # Use author names for log
                            log_members = [a.name for a in paper_members]

                            log_entries.append(
                                f"entry_{len(publications) + 1}\n"
                                f"paper_id: {paper_id}\n"
                                f"skills: {paper_skills}\n"
                                f"members: {log_members}\n"
                                f"---\n"
                            )

                            # Batch write logs periodically
                            if len(log_entries) >= 1000:
                                logs_dir = os.path.join(output_dir, "logs")
                                os.makedirs(logs_dir, exist_ok=True)

                                entries_log_path = os.path.join(
                                    logs_dir, "entries_processed.log"
                                )
                                with open(
                                    entries_log_path, "a", encoding="utf-8"
                                ) as log_file:
                                    log_file.writelines(log_entries)

                                # Clear log entries after writing
                                log_entries = []

                        publications[paper_id] = publication

                        # Print progress periodically
                        if len(publications) % 10000 == 0 and len(publications) > 0:
                            tprint(
                                f"Processed {len(publications)} valid publications so far..."
                            )

                    except json.JSONDecodeError:
                        # Badly formatted JSON, skip this line
                        json_parse_failures += 1
                        continue
                    except Exception as e:
                        exceptions += 1
                        if debug_logs:
                            tprint(f"Error processing line {line_num+1}: {str(e)}")
                            if debug_logs == "verbose":
                                tprint(traceback.format_exc())
                        continue

        # Print detailed stats about filtering
        tprint("\nProcessing statistics:")
        tprint(f"  Total lines processed: {total_lines}")
        tprint(f"  JSON parse failures: {json_parse_failures}")
        tprint(f"  Missing paper IDs: {missing_id}")
        tprint(f"  Other exceptions: {exceptions}")
        tprint(f"  Valid publications: {len(publications)}")

        # Write remaining log entries
        if output_path and log_entries:
            logs_dir = os.path.join(output_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)

            entries_log_path = os.path.join(logs_dir, "entries_processed.log")
            with open(entries_log_path, "a", encoding="utf-8") as log_file:
                log_file.writelines(log_entries)

        # After processing all publications, collect all unique skills
        if output_path:
            all_unique_skills = set()
            for pub_id, pub in publications.items():
                all_unique_skills.update(pub.skills)

            # Write all unique skills to skills.log (sorted for readability)
            logs_dir = os.path.join(output_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)

            skills_log_path = os.path.join(logs_dir, "skills.log")
            with open(skills_log_path, "w", encoding="utf-8") as skills_file:
                for skill in sorted(all_unique_skills):
                    skills_file.write(f"{skill}\n")

        # Log statistics
        publications_list = list(publications.values())
        elapsed = time() - start_time
        tprint(
            f"Processed {len(publications_list)} valid publications in {elapsed:.2f} seconds"
        )
        tprint(
            f"Found {len(all_members)} unique authors and {len(all_skills)} unique skills"
        )

        # Save the processed data if output path is specified
        if output_path and publications_list:
            teams_path = os.path.join(output_path, "teams.pkl")
            with open(teams_path, "wb") as f:
                pickle.dump(publications_list, f)
            tprint(f"Saved processed publications to {teams_path}")
        elif not publications_list:
            tprint("WARNING: No valid publications found, nothing saved.")

        return publications_list

    @classmethod
    def convert_vecs_to_publications(cls, vecs, indexes, output_path=None):
        """
        Convert filtered vectors back to Publication objects

        Args:
            vecs: Dictionary containing sparse vectors for teams
            indexes: Dictionary containing indexes for skills and members
            output_path: Path to store logs (optional)

        Returns:
            List of Publication objects
        """
        tprint("Converting filtered vectors back to Publication objects...")

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

        # Create Publication objects
        publications = []

        with tqdm(
            total=len(id_vector), desc="Converting publications", unit="pubs"
        ) as pbar:
            for i in range(len(id_vector)):
                try:
                    # Get publication ID
                    pub_id = id_vector[i]

                    # Get skills for this publication - maintain uniqueness
                    pub_skills = []
                    skill_set = set()
                    for j in range(skill_array.shape[1]):
                        if skill_array[i, j] > 0:
                            skill = i2s.get(j)
                            if skill and skill not in skill_set:
                                pub_skills.append(skill)
                                skill_set.add(skill)

                    # Get authors for this publication - use dictionary to ensure uniqueness
                    pub_authors_dict = {}
                    for j in range(member_array.shape[1]):
                        if member_array[i, j] > 0:
                            author_id = i2c.get(j)
                            if author_id and author_id not in pub_authors_dict:
                                # Create a DblpAuthor object
                                # Since we don't have all the original data, we'll create a simplified version
                                author = DblpAuthor(
                                    id=author_id,
                                    name=author_id,  # Use ID as name
                                    org="",  # Default value
                                )
                                pub_authors_dict[author_id] = author

                    # Convert the dictionary to a list
                    pub_authors = list(pub_authors_dict.values())

                    # Extract year from pub_id if possible
                    year = None
                    try:
                        # Try to extract year from the end of the paper ID
                        if isinstance(pub_id, str) and "_" in pub_id:
                            year_part = pub_id.split("_")[-1]
                            if year_part.isdigit() and len(year_part) == 4:
                                year = int(year_part)
                    except:
                        year = None

                    if not year:
                        year = datetime.now().year

                    # Create Publication object
                    publication = Publication(
                        id=pub_id,
                        title=f"Publication {pub_id}",  # Default title
                        authors=pub_authors,
                        skills=pub_skills,
                        year=year,
                        venue_raw="",
                        doc_type=None,
                    )

                    publications.append(publication)

                except Exception as e:
                    tprint(f"Error converting publication {i}: {str(e)}")

                pbar.update(1)

        tprint(f"Successfully converted {len(publications)} publications")
        return publications

    @staticmethod
    def build_indexes(teams, include_locations=True):
        """
        Build various indexes from teams data for DBLP publications

        This is the default implementation from the parent Team class.
        We're not adding document type indexes at this time.

        Args:
            teams: List of Publication objects
            include_locations: Whether to include location indexes

        Returns:
            Dictionary of indexes
        """
        # Use the parent class's build_indexes method directly
        from .team import Team

        return Team.build_indexes(teams, include_locations)
