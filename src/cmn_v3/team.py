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
        result['id'] = self.id
        
        # Skills vector
        skill_vec = np.zeros(len(skill_index))
        for skill in self.skills:
            if skill in skill_index:
                skill_vec[skill_index[skill]] = 1
        result['skill'] = skill_vec
        
        # Members vector
        member_vec = np.zeros(len(member_index))
        for member in self.members:
            if member.id in member_index:
                member_vec[member_index[member.id]] = 1
        result['member'] = member_vec
            
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
            'i2s': {},  # Index to skill
            's2i': {},  # Skill to index
            'i2c': {},  # Index to candidate/member
            'c2i': {},  # Candidate/member to index
            'i2t': {},  # Index to team
            't2i': {},  # Team to index
        }
        
        # Add location indexes if requested
        if include_locations:
            indexes['i2l'] = {}  # Index to location
            indexes['l2i'] = {}  # Location to index
            
        # Add year index for temporal data
        indexes['i2y'] = []  # Index to year (list of tuples (year, index))
        
        # Build skill index
        skill_idx = 0
        for team in teams:
            for skill in team.skills:
                if skill not in indexes['s2i']:
                    indexes['s2i'][skill] = skill_idx
                    indexes['i2s'][skill_idx] = skill
                    skill_idx += 1
        
        # Build member index
        member_idx = 0
        for team in teams:
            for member in team.members:
                if member.id not in indexes['c2i']:
                    indexes['c2i'][member.id] = member_idx
                    indexes['i2c'][member_idx] = member.id
                    member_idx += 1
        
        # Build location index if requested
        if include_locations:
            loc_idx = 0
            for team in teams:
                if team.location and team.location not in indexes['l2i']:
                    indexes['l2i'][team.location] = loc_idx
                    indexes['i2l'][loc_idx] = team.location
                    loc_idx += 1
        
        # Build team index and year index
        team_idx = 0
        years = defaultdict(list)
        
        # First sort teams by year for temporal consistency
        sorted_teams = sorted(teams, key=lambda t: t.datetime)
        
        for team in sorted_teams:
            indexes['t2i'][team.id] = team_idx
            indexes['i2t'][team_idx] = team.id
            
            # Add to year index
            # Handle different datetime formats
            if isinstance(team.datetime, int):
                year = team.datetime
            elif hasattr(team.datetime, 'year'):  # datetime object
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
                indexes['i2y'].append((year, idx))
        
        tprint(f"Built indexes: {len(indexes['s2i'])} skills, {len(indexes['c2i'])} members, {len(indexes['i2t'])} teams")
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
        with open(teams_path, 'wb') as f:
            pickle.dump(teams, f)
        
        # Save indexes
        indexes_path = output_dir / "indexes.pkl"
        tprint(f"Saving indexes to {indexes_path}")
        with open(indexes_path, 'wb') as f:
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
        with open(teams_path, 'rb') as f:
            teams = pickle.load(f)
        
        # Load indexes
        indexes_path = output_dir / "indexes.pkl"
        tprint(f"Loading indexes from {indexes_path}")
        with open(indexes_path, 'rb') as f:
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
        id_vector = np.zeros(batch_size, dtype=np.int32)  # int32 is usually sufficient for IDs
        
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
            'id': id_vector,
            'skill': skill_matrix.tocsr() if batch_size > 0 else skill_matrix,
            'member': member_matrix.tocsr() if batch_size > 0 else member_matrix
        }
    
    @classmethod
    def generate_sparse_vectors_v3(cls, datapath, output_dir, gpus=None):
        """
        Generate sparse vectors for teams
        
        Args:
            datapath: Path to the raw data
            output_dir: Directory to save results
            gpus: GPU identifiers to use (comma-separated string like "0" or "0,1")
            
        Returns:
            Tuple of (sparse vectors, indexes)
        """
        start_time = time()
        tprint(f"Starting preprocessing for {cls.__name__} with data from {datapath}")
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup domain-specific settings
        processing = cls.domain_params['processing']
        cpu_batch_size = processing.get('cpu_batch_size')
        gpu_batch_size = processing.get('gpu_batch_size')
        
        # Always enable parallel processing
        nthreads = processing.get('nthreads', 0)
        if nthreads <= 0:
            nthreads = max(1, int(multiprocessing.cpu_count() * 0.75))  # Use 75% of available threads by default
            
        
        # Try loading existing data first
        teamsvecs_path = output_dir / "teamsvecs.pkl"
        try:
            tprint(f"Attempting to load existing sparse matrices from {teamsvecs_path}")
            start_load = time()
            with open(teamsvecs_path, 'rb') as infile:
                vecs = pickle.load(infile)
            
            indexes_path = output_dir / "indexes.pkl" 
            with open(indexes_path, 'rb') as infile:
                indexes = pickle.load(infile)
                
            tprint(f"Successfully loaded existing data in {time() - start_load:.2f} seconds")
            return vecs, indexes
            
        except (FileNotFoundError, EOFError) as e:
            tprint(f"Existing files not found or incomplete: {str(e)}")
            tprint("Generating new sparse matrices...")
        
        # Read domain-specific raw data with filtering applied during reading
        tprint(f"Reading and filtering raw data from {datapath}...")
        try:
            teams = cls.read_and_filter_data_v3(datapath)
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
        batch_type = 'GPU' if use_gpu_avail else 'CPU'
        
        # Generate sparse vectors
        tprint(f"Generating sparse vectors for {len(teams)} teams with {batch_type}batch size {batch_size}...")
        start_sparse = time()
        
        # Configure GPU usage if specified
        if gpus is not None:
            # Set CUDA_VISIBLE_DEVICES if gpus is specified
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
            
        # Check if GPU is available after setting environment variable
        if use_gpu_avail:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            tprint(f"CUDA GPU detected: {gpu_name} with {gpu_memory:.1f}GB memory. Using GPU acceleration.")
            tprint(f"Using GPU(s): {gpus if gpus else 'all available'}")
            vecs = cls.generate_sparse_vectors_gpu(teams, indexes, batch_size)
        else:
            if gpus is not None:
                tprint("GPU acceleration requested but no CUDA GPU detected. Falling back to CPU-based processing.")
            else:
                tprint("Using CPU-based processing.")
            # Features to include
            skill_index = indexes['s2i']
            member_index = indexes['c2i']
            
            # Create result vectors
            vecs = {
                'id': np.zeros(len(teams)),
                'skill': lil_matrix((len(teams), len(skill_index))),
                'member': lil_matrix((len(teams), len(member_index)))
            }

            # Log the threading and batch size information
            tprint(f"Using {nthreads} threads for parallel processing with batch size of {batch_size}")
            
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
                    for i, batch_result in enumerate(pool.imap(process_func, team_batches, chunksize=chunksize)):
                        # Calculate the start and end indices for this batch
                        start_idx = i * batch_size
                        end_idx = min(start_idx + batch_size, len(teams))
                        
                        # Update the result vectors
                        vecs['id'][start_idx:end_idx] = batch_result['id']
                        vecs['skill'][start_idx:end_idx] = batch_result['skill']
                        vecs['member'][start_idx:end_idx] = batch_result['member']
                        
                        # Update progress bar
                        pbar.update(1)
                        
                        # Force garbage collection periodically to prevent memory buildup
                        if i % 10 == 0:
                            batch_result = None
                            gc.collect()
        
        # Save sparse vectors
        sparse_time = time() - start_sparse
        total_time = time() - start_time
        
        # Create output
        sparse_vectors_path = output_dir / "teamsvecs.pkl"
        tprint(f"Saving sparse matrices to {sparse_vectors_path}...")
        with open(sparse_vectors_path, 'wb') as outfile:
            pickle.dump(vecs, outfile)
            
        # Display stats    
        mem_usage = 0
        tprint("Sparse matrix shapes:")
        for key, matrix in vecs.items():
            if hasattr(matrix, 'shape'):
                shape = matrix.shape
                tprint(f"  {key}: {shape}")
                # Check if it's a sparse matrix (has both data and indptr attributes)
                if hasattr(matrix, 'data') and hasattr(matrix, 'indptr'):
                    mem_usage += matrix.data.nbytes + matrix.indptr.nbytes + matrix.indices.nbytes
                else:
                    mem_usage += matrix.nbytes
        
        tprint(f"Approximate memory usage: {mem_usage / (1024*1024):.2f} MB")
        tprint(f"Sparse vector generation took {sparse_time:.2f} seconds")
        tprint(f"Total processing time: {total_time:.2f} seconds")
        
        return vecs, indexes
    
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
        skill_index = indexes['s2i']
        member_index = indexes['c2i']
        n_teams = len(teams)
        n_skills = len(skill_index)
        n_members = len(member_index)
        
        # Initialize result vectors on CPU
        id_vector = np.array([team.id for team in teams], dtype=object)  # Use object dtype to store string IDs
        
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
                num_gpus = len(visible_devices.split(','))
        
        # Get the primary GPU device
        device = torch.device("cuda:0")
        
        # Adjust batch size based on GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        
        # Scale batch size based on available GPU memory
        # H100 with 80GB can handle much larger batches
        if gpu_batch_size is None:
            if gpu_memory >= 70:  # Likely an H100 or A100 or similar high-end GPU
                gpu_batch_size = 500_000  # Conservative estimate for extremely large datasets
            elif gpu_memory >= 30:  # Likely a A10/A40/A6000 or similar
                gpu_batch_size = 250_000
            elif gpu_memory >= 15:  # Likely a RTX 3090/4090 or similar consumer GPU
                gpu_batch_size = 100_000
            else:  # Smaller GPUs
                gpu_batch_size = 50_000
            
        tprint(f"Processing {n_teams} teams with {num_gpus} GPU(s), using batch size {gpu_batch_size}")
        
        # Create skill and member indices lookup dictionaries
        skill_to_idx = {skill: idx for skill, idx in skill_index.items()}
        member_to_idx = {member_id: idx for member_id, idx in member_index.items()}
        
        # Process in batches
        with tqdm(total=(n_teams + gpu_batch_size - 1) // gpu_batch_size, desc="GPU processing") as pbar:
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
                            batch_skill_rows[skill_idx] = i  # Local row index within batch
                            batch_skill_cols[skill_idx] = skill_to_idx[skill]
                            skill_idx += 1
                    
                    # Process members for this team
                    for member in team.members:
                        if member.id in member_to_idx:
                            batch_member_rows[member_idx] = i  # Local row index within batch
                            batch_member_cols[member_idx] = member_to_idx[member.id]
                            member_idx += 1
                
                # Create sparse tensors on GPU
                if skill_nnz > 0:
                    skill_values = torch.ones(skill_nnz, dtype=torch.int8, device=device)
                    skill_indices = torch.stack([
                        torch.tensor(batch_skill_rows, dtype=torch.int64, device=device),
                        torch.tensor(batch_skill_cols, dtype=torch.int64, device=device)
                    ])
                    
                    # Create sparse tensor
                    sparse_skill = torch.sparse_coo_tensor(
                        skill_indices, skill_values, 
                        (current_batch_size, n_skills),
                        device=device
                    )
                    
                    # Add directly from sparse indices
                    for i in range(skill_indices.shape[1]):
                        row_idx = skill_indices[0, i].item()
                        col_idx = skill_indices[1, i].item()
                        skill_rows.append(start_idx + row_idx)
                        skill_cols.append(col_idx)
                
                if member_nnz > 0:
                    member_values = torch.ones(member_nnz, dtype=torch.int8, device=device)
                    member_indices = torch.stack([
                        torch.tensor(batch_member_rows, dtype=torch.int64, device=device),
                        torch.tensor(batch_member_cols, dtype=torch.int64, device=device)
                    ])
                    
                    # Create sparse tensor
                    sparse_member = torch.sparse_coo_tensor(
                        member_indices, member_values, 
                        (current_batch_size, n_members),
                        device=device
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
                del skill_values, skill_indices, sparse_skill
                if 'member_values' in locals():
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
            (skill_data, (skill_rows, skill_cols)),
            shape=(n_teams, n_skills)
        )
        
        # Clear temporary variables to free memory
        del skill_data, skill_rows, skill_cols
        gc.collect()
        
        combined_nnz = len(member_rows)
        member_data = np.ones(combined_nnz, dtype=np.int8)
        
        member_matrix = csr_matrix(
            (member_data, (member_rows, member_cols)),
            shape=(n_teams, n_members)
        )
        
        # Clear temporary variables to free memory
        del member_data, member_rows, member_cols
        gc.collect()
        
        tprint(f"Created sparse matrices with {combined_nnz} skill entries and {len(member_matrix.data)} member entries")
        
        # Return results
        return {
            'id': id_vector,
            'skill': skill_matrix,
            'member': member_matrix
        }
    
    @classmethod
    def process_team_batches(cls, teams, indexes, batch_size, n_threads):
        """
        Process teams in batches using CPU multi-threading
        
        Args:
            teams: List of teams
            indexes: Dictionary mapping skills and members to indices
            batch_size: Batch size for processing
            n_threads: Number of threads to use
            
        Returns:
            Dictionary of sparse vectors
        """
        from scipy.sparse import lil_matrix, vstack
        from functools import partial
        import multiprocessing
        import gc
        
        # Extract indexes
        skill_index = indexes['s2i']
        member_index = indexes['c2i']
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
        process_func = partial(cls.process_team_batch_with_offset, skill_index=skill_index, member_index=member_index)
        
        # Process batches in parallel
        with multiprocessing.Pool(n_threads) as pool:
            # Use a smaller chunksize for more frequent updates
            chunksize = max(1, min(5, len(team_batches) // n_threads))
            
            # Process with progress tracking
            with tqdm(total=len(team_batches), desc="Processing batches") as pbar:
                for batch_offset, batch_result in pool.imap(process_func, team_batches, chunksize=chunksize):
                    # Update results
                    batch_size = len(batch_result['id'])
                    batch_end = batch_offset + batch_size
                    
                    # Update ID vector
                    id_vector[batch_offset:batch_end] = batch_result['id']
                    
                    # Update sparse matrices
                    for i in range(batch_size):
                        row = batch_offset + i
                        
                        # Update skill matrix
                        for j in batch_result['skill_indices'][i]:
                            skill_matrix[row, j] = 1
                        
                        # Update member matrix
                        for j in batch_result['member_indices'][i]:
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
        
        return {
            'id': id_vector,
            'skill': skill_matrix,
            'member': member_matrix
        }
    
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
            'id': id_vector,
            'skill_indices': skill_indices,
            'member_indices': member_indices
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