#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
from time import time
from pyparsing import List
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import sys

# Add the project root to the Python path if it's not already there
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use relative imports for modules in the same package
from .team import Team
from .dblp_author import DblpAuthor
from .dblp_venue import DblpVenue
from param import settings as SETTINGS
from utils.tprint import tprint


class Publication(Team):
    """
    Class representing a publication team from DBLP dataset

    This class extends the base Team class for DBLP-specific functionality.
    """

    # Add domain_params as a class attribute
    domain_params = SETTINGS["data"]

    def __init__(self, id, title, authors, skills, year, venue):
        """
        Initialize a Publication object

        Args:
            id: Unique identifier for the publication
            title: Title of the publication
            authors: List of Author objects representing the publication authors
            skills: List of skills associated with the publication
            year: Year of publication
            venue: Venue of publication (e.g., journal, conference)
        """

        # Additional domain-specific attributes
        self.id = id
        self.title = title
        self.authors = authors
        self.skills = skills
        self.year = year
        self.venue = venue

        # Initialize the base Team class
        super().__init__(id, authors, skills, year, location=venue)

    def __str__(self):
        """String representation of the publication"""
        authors_str = ", ".join(a.name for a in self.members[:3])
        if len(self.members) > 3:
            authors_str += f" +{len(self.members) - 3} more"
        return (
            f"Publication({self.id}): {self.title} by {authors_str} ({self.datetime})"
        )

    @classmethod
    def read_and_filter_data_v3(cls, datapath):
        """
        Read DBLP data from a JSON file, apply filters during reading, and convert to Publication objects

        Args:
            datapath: Path to the DBLP JSON file

        Returns:
            List of Publication objects
        """
        start_time = time()
        tprint(f"Reading and filtering DBLP data from {datapath}")

        # Track statistics
        publications = {}  # Use a dictionary to track by ID
        paper_ids = set()  # Use a set for faster lookups
        all_members = set()  # For statistics only
        all_skills = set()  # For statistics only

        # Get filter parameters from settings
        min_team_size = SETTINGS["data"]["filters"]["common"].get("min_team_size")
        max_team_size = SETTINGS["data"]["filters"]["common"].get("max_team_size")
        min_skills = SETTINGS["data"]["filters"]["common"].get("min_skills")
        max_skills = SETTINGS["data"]["filters"]["common"].get("max_skills")
        remove_dup_teams = SETTINGS["data"]["filters"]["common"].get("remove_dup_teams")
        remove_empty_skills = SETTINGS["data"]["filters"]["common"].get(
            "remove_empty_skills_teams"
        )
        remove_empty_members = SETTINGS["data"]["filters"]["common"].get(
            "remove_empty_experts_teams"
        )
        min_year = SETTINGS["data"]["filters"]["dblp"].get("min_year")
        max_year = SETTINGS["data"]["filters"]["dblp"].get("max_year")
        doc_types = SETTINGS["data"]["filters"]["dblp"].get(
            "doc_types", ["article", "inproceedings"]
        )

        # Get other parameters
        min_nteam = SETTINGS["data"]["filters"]["common"].get("min_teams_per_expert")

        # Get processing parameters
        debug_logs = SETTINGS["data"]["processing"].get("debug_logs", False)

        progress_message = tprint("Reading DBLP data")

        # Read and process the JSON file
        with tqdm(
            desc=progress_message,
            unit="B",
            unit_scale=True,
            total=os.path.getsize(datapath),
        ) as pbar:
            with open(datapath, "r", encoding="utf-8") as jf:
                for line in jf:
                    pbar.update(len(line))

                    if not line.strip():
                        continue

                    try:
                        # Parse JSON line (remove leading comma if present)
                        line = line.strip()
                        if line.startswith(","):
                            line = line[1:]
                        if not line:
                            continue

                        jsonline = json.loads(line.lower())

                        # Extract basic fields
                        paper_id = jsonline.get("id", None)
                        if paper_id is None:
                            continue

                        title = jsonline.get("title", "")
                        year = int(jsonline.get("year", 0))
                        doc_type = jsonline.get("doc_type", "")

                        # FILTER: doc_type
                        if doc_types and (doc_type not in doc_types):
                            continue

                        # FILTER: min and max years
                        if min_year > 0 and max_year > 0:  # Only apply if both are set
                            if year < min_year or year > max_year:
                                continue

                        # FILTER: duplicate
                        if paper_id in paper_ids:
                            if remove_dup_teams:
                                continue
                        paper_ids.add(paper_id)

                        # Create paper-specific sets for members and skills
                        paper_members = []
                        paper_skills = set()

                        authors = jsonline.get("authors", [])

                        # FILTER: empty authors
                        if not authors and remove_empty_members:
                            continue

                        # Process authors
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

                            # Create author and add to paper's members
                            author_obj = DblpAuthor(author_id, name, org)
                            paper_members.append(author_obj)
                            all_members.add(author_id)  # For statistics

                        # FILTER: min and max members
                        if len(paper_members) < min_team_size or (
                            max_team_size > 0 and len(paper_members) > max_team_size
                        ):
                            continue

                        # Process skills
                        fos = jsonline.get("fos", None)
                        keywords = jsonline.get("keywords", [])

                        # Add skills
                        if fos:
                            if isinstance(fos, dict) and "name" in fos:
                                skill = fos["name"].lower()
                                paper_skills.add(skill)
                                all_skills.add(skill)
                            elif isinstance(fos, list):
                                for field in fos:
                                    if isinstance(field, dict) and "name" in field:
                                        skill = field["name"].lower()
                                        paper_skills.add(skill)
                                        all_skills.add(skill)

                        if keywords:
                            for kw in keywords:
                                if kw:
                                    skill = kw.lower()
                                    paper_skills.add(skill)
                                    all_skills.add(skill)

                        # FILTER: empty skills
                        if not paper_skills and remove_empty_skills:
                            continue

                        # FILTER: min and max skills
                        if len(paper_skills) < min_skills or (
                            max_skills > 0 and len(paper_skills) > max_skills
                        ):
                            continue

                        # Add venue
                        venue = jsonline.get("venue", None)
                        venue_obj = None

                        if venue:
                            venue_id = venue.get("id", None)
                            venue_raw = venue.get("raw", "")
                            if venue_id:
                                venue_obj = DblpVenue(venue_id, venue_raw)

                        # Create Publication object
                        publication = cls(
                            id=paper_id,
                            title=title,
                            authors=paper_members,
                            skills=list(paper_skills),
                            year=year,
                            venue=venue_obj,
                        )

                        publications[paper_id] = publication

                    except json.JSONDecodeError:
                        # Badly formatted JSON, skip this line
                        continue
                    except Exception as e:
                        if debug_logs:
                            tprint(f"Error processing line: {str(e)}")
                        continue

        # Log statistics
        publications_list = list(publications.values())
        elapsed = time() - start_time
        tprint(
            f"Processed {len(publications_list)} valid publications in {elapsed:.2f} seconds"
        )
        tprint(
            f"Found {len(all_members)} unique authors and {len(all_skills)} unique skills"
        )

        # De-duplication
        if remove_dup_teams:
            orig_len = len(publications_list)
            publications_list = cls.remove_duplicate_papers(publications_list)
            dup_count = orig_len - len(publications_list)
            tprint(f"Removed {dup_count} duplicate papers")

        return publications_list
