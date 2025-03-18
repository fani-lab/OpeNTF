#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from typing import List, Any, Optional


def log_entries_processed(
    output_path: str,
    team_id: str,
    skills: List[str],
    members: List[Any],
    entry_idx: Optional[int] = None,
) -> None:
    """
    Log processed team entries to entries_processed.log

    Args:
        output_path: Path to store the logs
        team_id: Unique identifier for the team
        skills: List of team skills/attributes
        members: List of team members/contributors
        entry_idx: Optional index/counter for the entry
    """
    if not output_path:
        return

    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(output_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Define log file path
    entries_log_path = os.path.join(logs_dir, "entries_processed.log")

    # Initialize the log file if it doesn't exist
    if not os.path.exists(entries_log_path):
        with open(entries_log_path, "w", encoding="utf-8") as log_file:
            start_time = datetime.now()
            log_file.write(f"Processed Entries Log - Started: {start_time}\n\n")
            log_file.write(f"Each entry represents a team that was processed\n")
            log_file.write(
                f"Format: entry_X team_id: ID, skills: [skills], members: [members]\n"
            )
            log_file.write(f"{'='*80}\n\n")

    # Format members for logging
    formatted_members = []
    for member in members:
        if hasattr(member, "login"):
            formatted_members.append(member.login)
        elif hasattr(member, "id"):
            formatted_members.append(member.id)
        else:
            formatted_members.append(str(member))

    # Create log entry
    entry_text = (
        f"entry_{entry_idx if entry_idx is not None else 'X'}\n"
        f"team_id: {team_id}\n"
        f"skills: {skills}\n"
        f"members: {formatted_members}\n"
        f"---\n"
    )

    # Write to log file
    with open(entries_log_path, "a", encoding="utf-8") as log_file:
        log_file.write(entry_text)

    return entry_text


def log_entries_processed_batch(output_path: str, entries: List[str]) -> None:
    """
    Log multiple processed team entries at once to entries_processed.log

    Args:
        output_path: Path to store the logs
        entries: List of formatted entry strings
    """
    if not output_path or not entries:
        return

    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(output_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Define log file path
    entries_log_path = os.path.join(logs_dir, "entries_processed.log")

    # Initialize the log file if it doesn't exist
    if not os.path.exists(entries_log_path):
        with open(entries_log_path, "w", encoding="utf-8") as log_file:
            start_time = datetime.now()
            log_file.write(f"Processed Entries Log - Started: {start_time}\n\n")
            log_file.write(f"Each entry represents a team that was processed\n")
            log_file.write(
                f"Format: entry_X team_id: ID, skills: [skills], members: [members]\n"
            )
            log_file.write(f"{'='*80}\n\n")

    # Write all entries at once
    with open(entries_log_path, "a", encoding="utf-8") as log_file:
        log_file.writelines(entries)
