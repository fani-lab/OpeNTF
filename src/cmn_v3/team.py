#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import multiprocessing
from time import time
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from functools import partial
from scipy.sparse import lil_matrix, csr_matrix
import sys
import gc
import torch
from datetime import datetime
from cmn_v3.helper_functions.get_nthreads import get_nthreads

# Add the project root to the Python path if it's not already there
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.tprint import tprint


class Team:
    """
    Base Team class for preprocessing different datasets

    This class provides common functionality for team-based datasets, including:
    - Loading and saving data
    - Building indexes
    - Generating sparse vectors
    - Applying filters

    Each domain (DBLP, USPT, IMDB, GITH) will extend this class with specific implementations
    """

    def __init__(self, id, members, skills, datetime, location=None):
        """
        Initialize a Team object

        Args:
            id: Unique identifier for the team
            members: List of team members
            skills: List of skills/attributes of the team
            datetime: Timestamp or date of the team formation
            location: Optional location of the team
        """
        self.id = id
        self.members = members
        self.skills = skills
        self.datetime = datetime
        self.location = location

    def get_sparse_vector(self, skill_index, member_index):
        """
        Generate sparse vectors for this team

        Args:
            skill_index: Dictionary mapping skills to indices
            member_index: Dictionary mapping members to indices

        Returns:
            Dictionary of sparse vectors for different features
        """
        result = {}

        # ID vector (just the team ID)
        result["id"] = self.id

        # Skills vector
        skill_vec = np.zeros(len(skill_index))
        for skill in self.skills:
            if skill in skill_index:
                skill_vec[skill_index[skill]] = 1
        result["skill"] = skill_vec

        # Members vector
        member_vec = np.zeros(len(member_index))
        for member in self.members:
            if member.id in member_index:
                member_vec[member_index[member.id]] = 1
        result["member"] = member_vec

        return result

    @staticmethod
    def build_indexes(teams, include_locations=True):
        """
        Build various indexes from teams data

        Args:
            teams: List of Team objects
            include_locations: Whether to include location indexes

        Returns:
            Dictionary of indexes
        """
        tprint(f"Building indexes from {len(teams)} teams...")

        # Initialize indexes
        indexes = {
            "i2s": {},  # Index to skill
            "s2i": {},  # Skill to index
            "i2c": {},  # Index to candidate/member
            "c2i": {},  # Candidate/member to index
            "i2t": {},  # Index to team
            "t2i": {},  # Team to index
        }

        # Add location indexes if requested
        if include_locations:
            indexes["i2l"] = {}  # Index to location
            indexes["l2i"] = {}  # Location to index

        # Add year index for temporal data
        indexes["i2y"] = []  # Index to year (list of tuples (year, index))

        # Build skill index
        skill_idx = 0
        for team in teams:
            for skill in team.skills:
                if skill not in indexes["s2i"]:
                    indexes["s2i"][skill] = skill_idx
                    indexes["i2s"][skill_idx] = skill
                    skill_idx += 1

        # Build member index
        member_idx = 0
        for team in teams:
            for member in team.members:
                if member.id not in indexes["c2i"]:
                    indexes["c2i"][member.id] = member_idx
                    indexes["i2c"][member_idx] = member.id
                    member_idx += 1

        # Build location index if requested
        if include_locations:
            loc_idx = 0
            for team in teams:
                if team.location and team.location not in indexes["l2i"]:
                    indexes["l2i"][team.location] = loc_idx
                    indexes["i2l"][loc_idx] = team.location
                    loc_idx += 1

        # Build team index and year index
        team_idx = 0
        years = defaultdict(list)

        # First sort teams by year for temporal consistency
        sorted_teams = sorted(teams, key=lambda t: t.datetime)

        for team in sorted_teams:
            indexes["t2i"][team.id] = team_idx
            indexes["i2t"][team_idx] = team.id

            # Add to year index
            # Handle different datetime formats
            if isinstance(team.datetime, int):
                year = team.datetime
            elif hasattr(team.datetime, "year"):  # datetime object
                year = team.datetime.year
            elif isinstance(team.datetime, str):
                year = int(team.datetime)
            else:
                # Default to current year if datetime is None or in an unknown format
                year = datetime.now().year

            years[year].append(team_idx)
            team_idx += 1

        # Convert year index to list of (year, index) tuples
        for year in sorted(years.keys()):
            for idx in years[year]:
                indexes["i2y"].append((year, idx))

        tprint(
            f"Built indexes: {len(indexes['s2i'])} skills, {len(indexes['c2i'])} members, {len(indexes['i2t'])} teams"
        )
        if include_locations:
            tprint(f"Location index has {len(indexes['l2i'])} locations")

        return indexes

    @classmethod
    def save_data(cls, teams, indexes, output_dir):
        """
        Save teams and indexes to pickle files

        Args:
            teams: List of Team objects
            indexes: Dictionary of indexes
            output_dir: Directory to save the files
        """
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Save teams
        teams_path = output_dir / "teams.pkl"
        tprint(f"Saving {len(teams)} teams to {teams_path}")
        with open(teams_path, "wb") as f:
            pickle.dump(teams, f)

        # Save indexes
        indexes_path = output_dir / "indexes.pkl"
        tprint(f"Saving indexes to {indexes_path}")
        with open(indexes_path, "wb") as f:
            pickle.dump(indexes, f)

    @classmethod
    def load_data(cls, output_dir):
        """
        Load teams and indexes from pickle files

        Args:
            output_dir: Directory containing the files

        Returns:
            Tuple of (teams, indexes)
        """
        output_dir = Path(output_dir)

        # Load teams
        teams_path = output_dir / "teams.pkl"
        tprint(f"Loading teams from {teams_path}")
        with open(teams_path, "rb") as f:
            teams = pickle.load(f)

        # Load indexes
        indexes_path = output_dir / "indexes.pkl"
        tprint(f"Loading indexes from {indexes_path}")
        with open(indexes_path, "rb") as f:
            indexes = pickle.load(f)

        return teams, indexes

    @staticmethod
    def process_team_batch(team_batch, skill_index, member_index):
        """
        Process a batch of teams to generate sparse vectors

        Args:
            team_batch: List of Team objects to process
            skill_index: Dictionary mapping skills to indices
            member_index: Dictionary mapping members to indices

        Returns:
            Dictionary of sparse matrices for this batch
        """
        batch_size = len(team_batch)

        # Initialize result vectors with more memory-efficient format
        id_vector = np.zeros(
            batch_size, dtype=np.int32
        )  # int32 is usually sufficient for IDs

        # Initialize with lists of (row, col) coordinates for sparse matrices
        # This is more memory efficient than directly populating a lil_matrix
        skill_coords = []
        member_coords = []

        # For faster lookups
        skill_idx_cache = {}
        member_idx_cache = {}

        # Process each team
        for i, team in enumerate(team_batch):
            # Set team ID
            id_vector[i] = team.id

            # Process skills
            for skill in team.skills:
                if skill in skill_idx_cache:
                    col_idx = skill_idx_cache[skill]
                    skill_coords.append((i, col_idx))
                elif skill in skill_index:
                    col_idx = skill_index[skill]
                    skill_idx_cache[skill] = col_idx
                    skill_coords.append((i, col_idx))

            # Process members
            for member in team.members:
                if member.id in member_idx_cache:
                    col_idx = member_idx_cache[member.id]
                    member_coords.append((i, col_idx))
                elif member.id in member_index:
                    col_idx = member_index[member.id]
                    member_idx_cache[member.id] = col_idx
                    member_coords.append((i, col_idx))

        # Create sparse matrices from coordinate lists
        skill_matrix = lil_matrix((batch_size, len(skill_index)), dtype=np.int8)
        member_matrix = lil_matrix((batch_size, len(member_index)), dtype=np.int8)

        # Set values all at once is faster than one by one
        if skill_coords:
            rows, cols = zip(*skill_coords)
            skill_matrix[rows, cols] = 1

        if member_coords:
            rows, cols = zip(*member_coords)
            member_matrix[rows, cols] = 1

        # Convert to CSR for more efficient memory usage during transfer
        return {
            "id": id_vector,
            "skill": skill_matrix.tocsr() if batch_size > 0 else skill_matrix,
            "member": member_matrix.tocsr() if batch_size > 0 else member_matrix,
        }

    @classmethod
    def generate_sparse_vectors_v3(cls, datapath, output_path, gpus=None):
        """
        Generate sparse vectors for teams

        Args:
            datapath: Path to the raw data
            output_path: Directory to save results
            gpus: GPU identifiers to use (comma-separated string like "0" or "0,1")

        Returns:
            Tuple of (sparse vectors, indexes)
        """
        start_time = time()
        tprint(f"Starting preprocessing for {cls.__name__} with data from {datapath}")
        output_dir = Path(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Setup domain-specific settings
        processing = cls.domain_params["processing"]
        cpu_batch_size = processing.get("cpu_batch_size")
        gpu_batch_size = processing.get("gpu_batch_size")

        # Use get_nthreads() function to determine optimal thread count
        nthreads = get_nthreads()

        # Try loading existing data first
        teamsvecs_path = output_dir / "teamsvecs.pkl"
        teams_path = output_dir / "teams.pkl"
        indexes_path = output_dir / "indexes.pkl"

        # First check if teams data exists - it's either already filtered or will need filtering
        try:
            tprint(f"Attempting to load existing teams data from {teams_path}")
            start_load = time()
            with open(teams_path, "rb") as infile:
                teams = pickle.load(infile)

            tprint(
                f"Successfully loaded {len(teams)} teams in {time() - start_load:.2f} seconds"
            )
            return None, None, teams
        except (FileNotFoundError, EOFError):
            tprint("No existing teams data found, checking for vector data...")

        # Then check for vector data
        try:
            tprint(f"Attempting to load existing sparse matrices from {teamsvecs_path}")
            start_load = time()
            with open(teamsvecs_path, "rb") as infile:
                vecs = pickle.load(infile)

            with open(indexes_path, "rb") as infile:
                indexes = pickle.load(infile)

            tprint(
                f"Successfully loaded existing data in {time() - start_load:.2f} seconds"
            )

            # Data is either already filtered or will be used as is
            return vecs, indexes, None

        except (FileNotFoundError, EOFError) as e:
            tprint(f"Existing files not found or incomplete: {str(e)}")
            tprint("Generating new sparse matrices...")

        # Read domain-specific raw data with filtering applied during reading
        tprint(f"Reading and filtering raw data from {datapath}...")
        try:
            teams = cls.read_and_filter_data_v3(datapath, output_dir)
        except Exception as e:
            tprint(f"Error reading data: {str(e)}")
            raise

        if not teams:
            raise ValueError(f"No teams were loaded from {datapath}")

        # Build indexes
        indexes = cls.build_indexes(teams)

        # Save teams and indexes
        cls.save_data(teams, indexes, output_dir)

        use_gpu_avail = torch.cuda.is_available()

        batch_size = gpu_batch_size if use_gpu_avail else cpu_batch_size
        batch_type = "GPU" if use_gpu_avail else "CPU"

        # Generate sparse vectors
        tprint(
            f"Generating sparse vectors for {len(teams)} teams with {batch_type} batch size {batch_size}..."
        )
        start_sparse = time()

        # Configure GPU usage if specified
        if gpus is not None:
            # Set CUDA_VISIBLE_DEVICES if gpus is specified
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus

        # Check if GPU is available after setting environment variable
        if use_gpu_avail:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )  # Convert to GB
            tprint(
                f"CUDA GPU detected: {gpu_name} with {gpu_memory:.1f}GB memory. Using GPU acceleration."
            )
            tprint(f"Using GPU indices: {gpus if gpus else 'all available'}")
            vecs = cls.generate_sparse_vectors_gpu(teams, indexes, batch_size)
        else:
            if gpus is not None:
                tprint(
                    "GPU acceleration requested but no CUDA GPU detected. Falling back to CPU-based processing."
                )
            else:
                tprint("Using CPU-based processing.")
            # Features to include
            skill_index = indexes["s2i"]
            member_index = indexes["c2i"]

            # Create result vectors
            vecs = {
                "id": np.zeros(len(teams)),
                "skill": lil_matrix((len(teams), len(skill_index))),
                "member": lil_matrix((len(teams), len(member_index))),
            }

            # Log the threading and batch size information
            tprint(
                f"Using {nthreads} threads for parallel processing with batch size of {batch_size}"
            )

            # Split teams into batches - use the batch size from domain_params
            team_batches = []
            # Convert teams from list to numpy array for faster slicing if possible
            teams_array = np.array(teams, dtype=object)

            for i in range(0, len(teams), batch_size):
                end = min(i + batch_size, len(teams))
                team_batches.append(teams_array[i:end])

            # Create a function for processing each batch
            process_func = partial(
                cls.process_team_batch,
                skill_index=skill_index,
                member_index=member_index,
            )

            # Process batches in parallel
            with multiprocessing.Pool(nthreads) as pool:
                # Create a progress bar
                with tqdm(total=len(team_batches), desc="Processing batches") as pbar:
                    # Use a smaller chunksize to get more frequent updates
                    # This tells imap to send smaller chunks to each worker
                    chunksize = max(1, min(5, len(team_batches) // nthreads))

                    # Process each batch with chunksize optimization
                    for i, batch_result in enumerate(
                        pool.imap(process_func, team_batches, chunksize=chunksize)
                    ):
                        # Calculate the start and end indices for this batch
                        start_idx = i * batch_size
                        end_idx = min(start_idx + batch_size, len(teams))

                        # Update the result vectors
                        vecs["id"][start_idx:end_idx] = batch_result["id"]
                        vecs["skill"][start_idx:end_idx] = batch_result["skill"]
                        vecs["member"][start_idx:end_idx] = batch_result["member"]

                        # Update progress bar
                        pbar.update(1)

                        # Force garbage collection periodically to prevent memory buildup
                        if i % 10 == 0:
                            batch_result = None
                            gc.collect()

        # Save sparse vectors
        sparse_time = time() - start_sparse

        # Create output
        tprint(f"Saving unfiltered sparse matrices to {teamsvecs_path}...")
        with open(teamsvecs_path, "wb") as outfile:
            pickle.dump(vecs, outfile)

        # Apply filters
        tprint("Applying filters to teamsvecs...")
        from .helper_functions.apply_filters import apply_filters

        filtered_vecs = apply_filters(vecs, indexes, cls.domain_params)

        # Rebuild indexes based on filtered data
        tprint("Rebuilding indexes for filtered data...")
        filtered_indexes = cls.rebuild_indexes_from_filtered_vecs(
            filtered_vecs, indexes
        )

        # Create filtered teams list
        tprint("Creating filtered teams list...")
        filtered_teams = cls.create_filtered_teams(teams, filtered_vecs["id"])

        # Save filtered data and indexes
        tprint(f"Saving data to {teamsvecs_path}...")
        with open(teamsvecs_path, "wb") as outfile:
            pickle.dump(filtered_vecs, outfile)

        tprint(f"Saving indexes to {indexes_path}...")
        with open(indexes_path, "wb") as outfile:
            pickle.dump(filtered_indexes, outfile)

        tprint(f"Saving teams to {teams_path}...")
        with open(teams_path, "wb") as outfile:
            pickle.dump(filtered_teams, outfile)

        # Display stats
        mem_usage = 0
        tprint("Sparse matrix shapes:")
        for key, matrix in filtered_vecs.items():
            if hasattr(matrix, "shape"):
                shape = matrix.shape
                tprint(f"  {key}: {shape}")
                # Check if it's a sparse matrix (has both data and indptr attributes)
                if hasattr(matrix, "data") and hasattr(matrix, "indptr"):
                    mem_usage += (
                        matrix.data.nbytes
                        + matrix.indptr.nbytes
                        + matrix.indices.nbytes
                    )
                else:
                    mem_usage += matrix.nbytes

        tprint(f"Approximate memory usage: {mem_usage / (1024*1024):.2f} MB")
        tprint(f"Sparse vector generation took {sparse_time:.2f} seconds")
        total_time = time() - start_time
        tprint(f"Total processing time: {total_time:.2f} seconds")

        return filtered_vecs, filtered_indexes

    @classmethod
    def generate_sparse_vectors_gpu(cls, teams, indexes, gpu_batch_size=None):
        """
        Generate sparse vectors using GPU acceleration

        Args:
            teams: List of teams
            indexes: Dictionary mapping skills and members to indices
            batch_size: Batch size for GPU processing

        Returns:
            Dictionary of sparse vectors
        """
        # Extract indexes
        skill_index = indexes["s2i"]
        member_index = indexes["c2i"]
        n_teams = len(teams)
        n_skills = len(skill_index)
        n_members = len(member_index)

        # Initialize result vectors on CPU
        id_vector = np.array(
            [team.id for team in teams], dtype=object
        )  # Use object dtype to store string IDs

        # Create COO format sparse matrices (more efficient for construction)
        skill_rows = []
        skill_cols = []
        member_rows = []
        member_cols = []

        # Check for multi-GPU setup
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            tprint(f"Multi-GPU setup detected with {num_gpus} GPUs!")
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if visible_devices:
                tprint(f"Using GPUs: {visible_devices}")
                num_gpus = len(visible_devices.split(","))

        # Get the primary GPU device
        device = torch.device("cuda:0")

        # Adjust batch size based on GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
            1024**3
        )  # Convert to GB

        # Scale batch size based on available GPU memory
        # H100 with 80GB can handle much larger batches
        if gpu_batch_size is None:
            if gpu_memory >= 70:  # Likely an H100 or A100 or similar high-end GPU
                gpu_batch_size = (
                    500_000  # Conservative estimate for extremely large datasets
                )
            elif gpu_memory >= 30:  # Likely a A10/A40/A6000 or similar
                gpu_batch_size = 250_000
            elif gpu_memory >= 15:  # Likely a RTX 3090/4090 or similar consumer GPU
                gpu_batch_size = 100_000
            else:  # Smaller GPUs
                gpu_batch_size = 50_000

        tprint(
            f"Processing {n_teams} teams with {num_gpus} GPU(s), using batch size {gpu_batch_size}"
        )

        # Create skill and member indices lookup dictionaries
        skill_to_idx = {skill: idx for skill, idx in skill_index.items()}
        member_to_idx = {member_id: idx for member_id, idx in member_index.items()}

        # Process in batches
        with tqdm(
            total=(n_teams + gpu_batch_size - 1) // gpu_batch_size,
            desc="GPU processing",
        ) as pbar:
            for start_idx in range(0, n_teams, gpu_batch_size):
                end_idx = min(start_idx + gpu_batch_size, n_teams)
                batch_teams = teams[start_idx:end_idx]
                current_batch_size = end_idx - start_idx

                # First pass: count number of non-zeros for pre-allocation
                skill_nnz = 0
                member_nnz = 0

                for team in batch_teams:
                    for skill in team.skills:
                        if skill in skill_to_idx:
                            skill_nnz += 1

                    for member in team.members:
                        if member.id in member_to_idx:
                            member_nnz += 1

                # Pre-allocate arrays for this batch
                batch_skill_rows = np.zeros(skill_nnz, dtype=np.int32)
                batch_skill_cols = np.zeros(skill_nnz, dtype=np.int32)
                batch_member_rows = np.zeros(member_nnz, dtype=np.int32)
                batch_member_cols = np.zeros(member_nnz, dtype=np.int32)

                # Second pass: fill arrays
                skill_idx = 0
                member_idx = 0

                for i, team in enumerate(batch_teams):
                    # Global index in the full dataset
                    global_idx = start_idx + i

                    # Set team ID
                    id_vector[global_idx] = team.id

                    # Process skills for this team
                    for skill in team.skills:
                        if skill in skill_to_idx:
                            batch_skill_rows[skill_idx] = (
                                i  # Local row index within batch
                            )
                            batch_skill_cols[skill_idx] = skill_to_idx[skill]
                            skill_idx += 1

                    # Process members for this team
                    for member in team.members:
                        if member.id in member_to_idx:
                            batch_member_rows[member_idx] = (
                                i  # Local row index within batch
                            )
                            batch_member_cols[member_idx] = member_to_idx[member.id]
                            member_idx += 1

                # Create sparse tensors on GPU
                if skill_nnz > 0:
                    skill_values = torch.ones(
                        skill_nnz, dtype=torch.int8, device=device
                    )
                    skill_indices = torch.stack(
                        [
                            torch.tensor(
                                batch_skill_rows, dtype=torch.int64, device=device
                            ),
                            torch.tensor(
                                batch_skill_cols, dtype=torch.int64, device=device
                            ),
                        ]
                    )

                    # Create sparse tensor
                    sparse_skill = torch.sparse_coo_tensor(
                        skill_indices,
                        skill_values,
                        (current_batch_size, n_skills),
                        device=device,
                    )

                    # Add directly from sparse indices
                    for i in range(skill_indices.shape[1]):
                        row_idx = skill_indices[0, i].item()
                        col_idx = skill_indices[1, i].item()
                        skill_rows.append(start_idx + row_idx)
                        skill_cols.append(col_idx)

                if member_nnz > 0:
                    member_values = torch.ones(
                        member_nnz, dtype=torch.int8, device=device
                    )
                    member_indices = torch.stack(
                        [
                            torch.tensor(
                                batch_member_rows, dtype=torch.int64, device=device
                            ),
                            torch.tensor(
                                batch_member_cols, dtype=torch.int64, device=device
                            ),
                        ]
                    )

                    # Create sparse tensor
                    sparse_member = torch.sparse_coo_tensor(
                        member_indices,
                        member_values,
                        (current_batch_size, n_members),
                        device=device,
                    )

                    # Add directly from sparse indices
                    for i in range(member_indices.shape[1]):
                        row_idx = member_indices[0, i].item()
                        col_idx = member_indices[1, i].item()
                        member_rows.append(start_idx + row_idx)
                        member_cols.append(col_idx)

                # Update progress bar
                pbar.update(1)

                # Free GPU memory explicitly, including tensors we created
                if "skill_values" in locals():
                    del skill_values, skill_indices, sparse_skill
                if "member_values" in locals():
                    del member_values, member_indices, sparse_member
                torch.cuda.empty_cache()

                # Periodic garbage collection to free memory
                if start_idx % (gpu_batch_size * 5) == 0 and start_idx > 0:
                    gc.collect()

        # Pre-allocate arrays for better memory efficiency
        tprint("Pre-allocating arrays for sparse matrix construction...")
        combined_nnz = len(skill_rows)
        skill_data = np.ones(combined_nnz, dtype=np.int8)

        tprint("Converting to sparse matrices...")
        # Create sparse matrices from coordinate lists - use efficient construction
        skill_matrix = csr_matrix(
            (skill_data, (skill_rows, skill_cols)), shape=(n_teams, n_skills)
        )

        # Clear temporary variables to free memory
        del skill_data, skill_rows, skill_cols
        gc.collect()

        combined_nnz = len(member_rows)
        member_data = np.ones(combined_nnz, dtype=np.int8)

        member_matrix = csr_matrix(
            (member_data, (member_rows, member_cols)), shape=(n_teams, n_members)
        )

        # Clear temporary variables to free memory
        del member_data, member_rows, member_cols
        gc.collect()

        tprint(
            f"Created sparse matrices with {combined_nnz} skill entries and {len(member_matrix.data)} member entries"
        )

        # Return results
        return {"id": id_vector, "skill": skill_matrix, "member": member_matrix}

    @classmethod
    def process_team_batches(cls, teams, indexes, batch_size, n_threads=None):
        """
        Process teams in batches using CPU multi-threading

        Args:
            teams: List of teams
            indexes: Dictionary mapping skills and members to indices
            batch_size: Batch size for processing
            n_threads: Number of threads to use (defaults to get_nthreads() if None)

        Returns:
            Dictionary of sparse vectors
        """
        from scipy.sparse import lil_matrix, vstack
        from functools import partial
        import multiprocessing
        import gc

        # Use get_nthreads() if n_threads is not specified
        if n_threads is None:
            n_threads = get_nthreads()

        # Extract indexes
        skill_index = indexes["s2i"]
        member_index = indexes["c2i"]
        n_teams = len(teams)
        n_skills = len(skill_index)
        n_members = len(member_index)

        # Initialize result vectors
        id_vector = np.zeros(n_teams, dtype=np.int32)

        # Create empty matrices for the results
        skill_matrix = lil_matrix((n_teams, n_skills), dtype=np.int8)
        member_matrix = lil_matrix((n_teams, n_members), dtype=np.int8)

        # Split teams into batches
        team_batches = []
        for i in range(0, n_teams, batch_size):
            end = min(i + batch_size, n_teams)
            team_batches.append((i, teams[i:end]))

        # Create partial function for processing batches
        process_func = partial(
            cls.process_team_batch_with_offset,
            skill_index=skill_index,
            member_index=member_index,
        )

        # Process batches in parallel
        with multiprocessing.Pool(n_threads) as pool:
            # Use a smaller chunksize for more frequent updates
            chunksize = max(1, min(5, len(team_batches) // n_threads))

            # Process with progress tracking
            with tqdm(total=len(team_batches), desc="Processing batches") as pbar:
                for batch_offset, batch_result in pool.imap(
                    process_func, team_batches, chunksize=chunksize
                ):
                    # Update results
                    batch_size = len(batch_result["id"])
                    batch_end = batch_offset + batch_size

                    # Update ID vector
                    id_vector[batch_offset:batch_end] = batch_result["id"]

                    # Update sparse matrices
                    for i in range(batch_size):
                        row = batch_offset + i

                        # Update skill matrix
                        for j in batch_result["skill_indices"][i]:
                            skill_matrix[row, j] = 1

                        # Update member matrix
                        for j in batch_result["member_indices"][i]:
                            member_matrix[row, j] = 1

                    # Update progress
                    pbar.update(1)

                    # Force garbage collection periodically
                    if batch_offset % (10 * batch_size) == 0:
                        gc.collect()

        # Convert to CSR format for efficiency
        tprint("Converting to CSR format...")
        skill_matrix = skill_matrix.tocsr()
        member_matrix = member_matrix.tocsr()

        return {"id": id_vector, "skill": skill_matrix, "member": member_matrix}

    @classmethod
    def process_team_batch_with_offset(cls, batch_tuple, skill_index, member_index):
        """
        Process a batch of teams with offset information

        Args:
            batch_tuple: Tuple of (offset, batch)
            skill_index: Mapping of skills to indices
            member_index: Mapping of members to indices

        Returns:
            Dictionary with batch results and offset
        """
        offset, batch = batch_tuple

        # Create skill and member indices dictionaries for faster lookup
        skill_to_idx = {skill: idx for skill, idx in skill_index.items()}
        member_to_idx = {member_id: idx for member_id, idx in member_index.items()}

        # Initialize results
        batch_size = len(batch)
        id_vector = np.zeros(batch_size, dtype=np.int32)
        skill_indices = [[] for _ in range(batch_size)]
        member_indices = [[] for _ in range(batch_size)]

        # Process each team
        for i, team in enumerate(batch):
            # Set team ID
            id_vector[i] = team.id

            # Process skills
            for skill in team.skills:
                if skill in skill_to_idx:
                    skill_indices[i].append(skill_to_idx[skill])

            # Process members
            for member in team.members:
                if member.id in member_to_idx:
                    member_indices[i].append(member_to_idx[member.id])

        # Return results with offset
        return {
            "id": id_vector,
            "skill_indices": skill_indices,
            "member_indices": member_indices,
        }

    @classmethod
    def read_and_filter_data_v3(cls, datapath):
        """
        Read raw data and convert to Team objects
        This is a placeholder that should be overridden by subclasses

        Args:
            datapath: Path to the raw data

        Returns:
            List of Team objects
        """
        pass

    @classmethod
    def rebuild_indexes_from_filtered_vecs(cls, filtered_teamsvecs, original_indexes):
        """
        Rebuild the indexes based on the filtered teamsvecs - with GPU acceleration.

        Args:
            filtered_teamsvecs: Dictionary containing filtered sparse vectors for teams
            original_indexes: Original dictionary containing indexes

        Returns:
            Updated indexes dictionary
        """
        import torch
        import numpy as np

        tprint("Rebuilding indexes based on filtered data...")

        # Check if GPU is available
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )  # Convert to GB
            tprint(
                f"Using GPU: {gpu_name} with {gpu_memory:.1f}GB memory for index rebuilding"
            )
        else:
            tprint(
                "WARNING: CUDA GPU not available. Falling back to CPU for index rebuilding."
            )

        # Create a copy of the original indexes structure
        indexes = {
            "s2i": {},  # skill to index
            "i2s": {},  # index to skill
            "c2i": {},  # collaborator to index
            "i2c": {},  # index to collaborator
        }

        # Get the skill and member matrices
        skill_matrix = filtered_teamsvecs["skill"]
        member_matrix = filtered_teamsvecs["member"]

        # Process skill matrix to find used skills
        tprint("Finding used skills...")
        if use_gpu:
            try:
                # Convert to GPU tensor if it's not already one
                if hasattr(skill_matrix, "toarray"):
                    # For sparse matrices, convert to dense numpy array, then to torch tensor
                    skill_tensor = torch.tensor(
                        skill_matrix.toarray(), dtype=torch.float32, device=device
                    )
                elif isinstance(skill_matrix, torch.Tensor):
                    # If already a tensor, ensure it's on GPU
                    skill_tensor = (
                        skill_matrix.to(device)
                        if skill_matrix.device != device
                        else skill_matrix
                    )
                else:
                    # For numpy arrays or other array-like objects
                    skill_tensor = torch.tensor(
                        skill_matrix, dtype=torch.float32, device=device
                    )

                # Find skills that are used (column indices where any value > 0)
                skill_counts = torch.sum(skill_tensor > 0, dim=0)
                used_skill_indices = torch.nonzero(skill_counts).squeeze().cpu().numpy()

                # Free GPU memory
                del skill_tensor
                del skill_counts
                torch.cuda.empty_cache()

                # Create mappings only for skills that are actually used
                for idx in used_skill_indices:
                    skill = original_indexes["i2s"].get(int(idx))
                    if skill:
                        old_idx = original_indexes["s2i"].get(skill)
                        if old_idx is not None:
                            indexes["s2i"][skill] = old_idx
                            indexes["i2s"][old_idx] = skill

            except torch.cuda.OutOfMemoryError:
                tprint(
                    "GPU out of memory during skill index rebuilding. Falling back to CPU."
                )
                use_gpu = False
                torch.cuda.empty_cache()

                # Fall back to CPU implementation
                if hasattr(skill_matrix, "toarray"):
                    # Efficiently process sparse matrix on CPU
                    # Get row and column indices of non-zero elements
                    rows, cols = skill_matrix.nonzero()
                    used_skill_indices = np.unique(cols)
                else:
                    # For dense arrays
                    used_skill_indices = np.where(np.sum(skill_matrix > 0, axis=0) > 0)[
                        0
                    ]

                # Create mappings only for skills that are actually used
                for idx in used_skill_indices:
                    skill = original_indexes["i2s"].get(int(idx))
                    if skill:
                        old_idx = original_indexes["s2i"].get(skill)
                        if old_idx is not None:
                            indexes["s2i"][skill] = old_idx
                            indexes["i2s"][old_idx] = skill

        else:
            # CPU implementation
            if hasattr(skill_matrix, "toarray"):
                # Efficiently process sparse matrix on CPU
                # Get row and column indices of non-zero elements
                rows, cols = skill_matrix.nonzero()
                used_skill_indices = np.unique(cols)
            else:
                # For dense arrays
                used_skill_indices = np.where(np.sum(skill_matrix > 0, axis=0) > 0)[0]

            # Create mappings only for skills that are actually used
            for idx in used_skill_indices:
                skill = original_indexes["i2s"].get(int(idx))
                if skill:
                    old_idx = original_indexes["s2i"].get(skill)
                    if old_idx is not None:
                        indexes["s2i"][skill] = old_idx
                        indexes["i2s"][old_idx] = skill

        # Process member matrix to find used members - use batched processing to avoid OOM
        tprint("Finding used members...")

        if use_gpu:
            try:
                # Process member matrix in batches to avoid OOM
                batch_size = 10000  # Adjust based on available GPU memory

                # Initialize a set to collect all used member indices
                used_member_indices = set()

                # Convert member matrix to array format for batched processing
                if hasattr(member_matrix, "toarray"):
                    member_array = member_matrix
                else:
                    member_array = member_matrix

                # Process in batches
                for start_idx in range(0, member_array.shape[1], batch_size):
                    end_idx = min(start_idx + batch_size, member_array.shape[1])

                    if hasattr(member_array, "toarray"):
                        # Get subset of columns for this batch (sparse format)
                        batch_cols = member_array.tocsc()[:, start_idx:end_idx]

                        # Get column indices with non-zero values
                        rows, cols = batch_cols.nonzero()
                        batch_used_indices = np.unique(cols) + start_idx
                        used_member_indices.update(batch_used_indices)
                    else:
                        # Dense format - convert to GPU tensor
                        batch_tensor = torch.tensor(
                            member_array[:, start_idx:end_idx],
                            dtype=torch.float32,
                            device=device,
                        )

                        # Find columns with any non-zero values
                        member_counts = torch.sum(batch_tensor > 0, dim=0)
                        batch_used_indices = (
                            torch.nonzero(member_counts).squeeze().cpu().numpy()
                        )

                        # Adjust indices to account for batch offset
                        batch_used_indices = batch_used_indices + start_idx
                        used_member_indices.update(batch_used_indices)

                        # Free GPU memory
                        del batch_tensor
                        del member_counts
                        torch.cuda.empty_cache()

                # Convert set to list for easier iteration
                used_member_indices = list(used_member_indices)

            except torch.cuda.OutOfMemoryError:
                tprint(
                    "GPU out of memory during member index rebuilding. Falling back to CPU."
                )
                use_gpu = False
                torch.cuda.empty_cache()

                # Fall back to CPU implementation - process efficiently
                if hasattr(member_matrix, "toarray"):
                    # Get row and column indices of non-zero elements
                    rows, cols = member_matrix.nonzero()
                    used_member_indices = np.unique(cols)
                else:
                    # For dense arrays
                    used_member_indices = np.where(
                        np.sum(member_matrix > 0, axis=0) > 0
                    )[0]

        else:
            # CPU implementation
            if hasattr(member_matrix, "toarray"):
                # Get row and column indices of non-zero elements - efficient for sparse matrices
                rows, cols = member_matrix.nonzero()
                used_member_indices = np.unique(cols)
            else:
                # For dense arrays
                used_member_indices = np.where(np.sum(member_matrix > 0, axis=0) > 0)[0]

        # Create mappings only for members that are actually used
        tprint(
            f"Creating member mappings for {len(used_member_indices)} used members..."
        )
        for idx in used_member_indices:
            member = original_indexes["i2c"].get(int(idx))
            if member:
                old_idx = original_indexes["c2i"].get(member)
                if old_idx is not None:
                    indexes["c2i"][member] = old_idx
                    indexes["i2c"][old_idx] = member

        tprint(
            f"Rebuilt indexes: {len(indexes['s2i'])} skills, {len(indexes['c2i'])} members"
        )
        return indexes

    @classmethod
    def create_filtered_teams(cls, teams, filtered_ids):
        """
        Create a filtered list of teams based on the filtered IDs

        Args:
            teams: List of Team objects
            filtered_ids: Array of team IDs that passed the filters

        Returns:
            List of filtered Team objects
        """
        # Create a set of filtered IDs for faster lookup
        filtered_id_set = set(filtered_ids)

        # Create a dictionary mapping team IDs to team objects for faster lookup
        team_dict = {team.id: team for team in teams}

        # Create the filtered teams list
        filtered_teams = []
        for team_id in filtered_ids:
            if team_id in team_dict:
                filtered_teams.append(team_dict[team_id])

        tprint(
            f"Created filtered teams list with {len(filtered_teams)} teams (from original {len(teams)} teams)"
        )
        return filtered_teams

    @classmethod
    def save_data_v3(cls, teams, output_dir=None, save_teams=True):
        """
        Save teams data using the v3 format.

        Args:
            teams: List of Team objects
            output_dir: Directory to save the data (optional)
            save_teams: Whether to save the teams as a pickle file

        Returns:
            Tuple of (teamsvecs, indexes)
        """
        start_time = time()

        if output_dir is None:
            output_dir = Path("./data")
        else:
            output_dir = Path(output_dir)

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        tprint(f"Converting {len(teams)} teams to sparse vectors...")

        # Convert teams to sparse vectors
        teamsvecs, indexes = cls.generate_sparse_vectors_v3(teams)

        # Save the raw sparse vectors
        teamsvecs_path = output_dir / "teamsvecs.pkl"
        teams_path = output_dir / "teams.pkl"

        tprint(f"Saving teamsvecs to {teamsvecs_path}...")
        with open(teamsvecs_path, "wb") as outfile:
            pickle.dump(teamsvecs, outfile)

        if save_teams:
            tprint(f"Saving teams to {teams_path}...")
            with open(teams_path, "wb") as outfile:
                pickle.dump(teams, outfile)

        # Save the indexes
        indexes_path = output_dir / "indexes.pkl"
        tprint(f"Saving indexes to {indexes_path}...")
        with open(indexes_path, "wb") as outfile:
            pickle.dump(indexes, outfile)

        processing_time = time() - start_time
        tprint(
            f"Data saved in {processing_time:.2f}s. Vectors shape: {teamsvecs['skill'].shape}, {teamsvecs['member'].shape}"
        )

        return teamsvecs, indexes

    @classmethod
    def apply_filters_and_save_v3(
        cls, teamsvecs, indexes, output_dir, teams=None, domain_params=None
    ):
        """
        Apply filters to the teamsvecs based on parameters and save the results.

        Args:
            teamsvecs: Dictionary containing sparse vectors for teams
            indexes: Dictionary containing indexes for skills and members
            output_dir: Directory to save the filtered data
            teams: List of Team objects (optional)
            domain_params: Domain-specific parameters (optional)

        Returns:
            Tuple of (filtered_teamsvecs, filtered_indexes)
        """
        from .helper_functions.apply_filters import apply_filters

        start_time = time()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Apply filters
        tprint("Applying filters to teamsvecs...")
        filtered_teamsvecs = apply_filters(teamsvecs, indexes, domain_params)

        # Save the filtered sparse vectors
        tprint(f"Saving filtered teamsvecs to {output_dir / 'teamsvecs.pkl'}...")
        with open(output_dir / "teamsvecs.pkl", "wb") as outfile:
            pickle.dump(filtered_teamsvecs, outfile)

        # Rebuild the indexes based on the filtered data
        filtered_indexes = cls.rebuild_indexes_from_filtered_vecs(
            filtered_teamsvecs, indexes
        )

        # Save the filtered indexes
        tprint(f"Saving filtered indexes to {output_dir / 'indexes.pkl'}...")
        with open(output_dir / "indexes.pkl", "wb") as outfile:
            pickle.dump(filtered_indexes, outfile)

        # If teams are provided, save the filtered teams too
        if teams is not None:
            # Create a filtered teams list
            filtered_team_ids = set(filtered_teamsvecs["id"])
            filtered_teams = [team for team in teams if team.id in filtered_team_ids]

            # Save the filtered teams
            tprint(f"Saving filtered teams to {output_dir / 'teams.pkl'}...")
            with open(output_dir / "teams.pkl", "wb") as outfile:
                pickle.dump(filtered_teams, outfile)

            tprint(
                f"Original teams: {len(teams)}, Filtered teams: {len(filtered_teams)}"
            )

        processing_time = time() - start_time
        tprint(
            f"Filtering complete in {processing_time:.2f}s. "
            f"Original: {len(teamsvecs['id'])} teams, Filtered: {len(filtered_teamsvecs['id'])} teams."
        )

        return filtered_teamsvecs, filtered_indexes
