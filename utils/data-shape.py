import pickle
import sys
import json
import csv
import os
import io
from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import re
from collections import Counter, defaultdict
import scipy.sparse

# Suppress the deprecation warning about sparse matrices
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message="Please use.*sparse` namespace"
)


# Custom module handling for pickle loading
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Special case for known module issues
        if module == "cmn.movie":
            print(f"Handling missing module: {module}.{name}")

            # Create a dummy movie class with minimal functionality
            class DummyMovie:
                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs

                def __getattr__(self, attr):
                    return lambda *a, **kw: None

                def __str__(self):
                    return f"DummyMovie({self.args}, {self.kwargs})"

                def __repr__(self):
                    return self.__str__()

            return DummyMovie

        # Try the normal approach
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, ImportError, AttributeError) as e:
            # Create a custom dummy class for this module
            print(f"Creating dummy class for {module}.{name}")

            # Define a dummy class with the right name
            DummyClass = type(
                name,
                (),
                {
                    "__init__": lambda self, *args, **kwargs: setattr(
                        self, "args", (args, kwargs)
                    ),
                    "__getattr__": lambda self, attr: lambda *a, **kw: None,
                    "__str__": lambda self: f"Dummy{name}",
                    "__repr__": lambda self: f"Dummy{name}",
                },
            )
            return DummyClass


class TqdmProgressFileReader:
    """Wraps a file reader to display progress using tqdm."""

    def __init__(self, file_path, pbar=None):
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.progress = 0
        self.pbar = pbar
        if self.pbar is None:
            self.pbar = tqdm(
                total=self.file_size,
                unit="B",
                unit_scale=True,
                desc=f"Loading {os.path.basename(file_path)}",
            )

    def __enter__(self):
        self.file = open(self.file_path, "rb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        self.pbar.close()

    def read(self, size=-1):
        data = self.file.read(size)
        self.progress += len(data)
        self.pbar.update(len(data))
        return data

    def tell(self):
        return self.file.tell()

    def seek(self, offset, whence=0):
        self.file.seek(offset, whence)


def load_file(file_path, file_format=None, batch_size=None, pbar=None, encoding=None):
    """Load data from file."""
    # Convert file_path to a Path object if it's a string
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path

    # Get file name and extension
    file_name = file_path.stem
    file_ext = file_path.suffix.lower()

    # Update progress bar with initial status
    if pbar:
        pbar.set_description(f"Opening {os.path.basename(file_path)}")
        pbar.update(5)  # Small initial progress

    # Check if file exists
    if not file_path.exists():
        print(f"Error: File {file_path} not found")
        return None

    # Get file size for progress tracking
    try:
        file_size = file_path.stat().st_size
    except Exception:
        file_size = 0

    # Determine format
    if file_format:
        fmt = file_format.lower()
    else:
        if file_ext == ".pkl" or file_ext == ".pickle":
            fmt = "pickle"
        elif file_ext == ".csv":
            fmt = "csv"
        elif file_ext == ".tsv":
            fmt = "tsv"
        elif file_ext == ".json":
            fmt = "json"
        else:
            fmt = "unknown"

    if pbar:
        pbar.set_description(f"Loading {os.path.basename(file_path)} as {fmt}")
        pbar.update(5)  # Progress after format detection

    # List of encodings to try (in order of preference)
    encodings_to_try = (
        [encoding] if encoding else ["utf-8", "latin1", "cp1252", "ISO-8859-1"]
    )

    try:
        # Handle different file formats
        if fmt == "pickle":
            # Load pickle file (binary, no encoding needed)
            if pbar:
                pbar.set_description(f"Reading pickle data")
            with open(file_path, "rb") as f:
                try:
                    # First try normal unpickling
                    data = pickle.load(f)
                except (ModuleNotFoundError, ImportError) as e:
                    # If a module is missing, try with custom unpickler
                    print(
                        f"Warning: {str(e)} - Attempting to load with custom module handling"
                    )
                    f.seek(0)  # Reset file pointer
                    try:
                        data = CustomUnpickler(f).load()
                    except Exception as inner_e:
                        print(
                            f"Error loading pickle with custom handler: {str(inner_e)}"
                        )
                        raise
            if pbar:
                pbar.update(90)  # Complete progress
            return data
        elif fmt == "csv":
            # For CSV files, try different encodings
            last_exception = None
            for enc in encodings_to_try:
                try:
                    if pbar:
                        pbar.set_description(f"Trying to read CSV with {enc} encoding")

                    # For large CSV files, use chunked reading if batch_size is specified
                    if batch_size:
                        if pbar:
                            pbar.update(10)
                        data = pd.read_csv(
                            file_path, chunksize=batch_size, encoding=enc
                        )
                        if pbar:
                            pbar.set_description(
                                f"Successfully loaded CSV with {enc} encoding"
                            )
                            pbar.update(80)  # Complete progress for setup
                        return data
                    else:
                        # Use progress bar if available
                        if pbar:
                            pbar.update(10)
                        data = pd.read_csv(file_path, encoding=enc)
                        if pbar:
                            pbar.set_description(
                                f"Successfully loaded CSV with {enc} encoding"
                            )
                            pbar.update(80)  # Complete progress
                        return data
                except UnicodeDecodeError as e:
                    last_exception = e
                    continue
                except Exception as e:
                    # For other exceptions, try the next encoding
                    last_exception = e
                    continue

            # If we get here, all encodings failed
            raise last_exception or ValueError(
                "Failed to read CSV file with any encoding"
            )

        elif fmt == "tsv":
            # For TSV files, try different encodings
            last_exception = None
            for enc in encodings_to_try:
                try:
                    if pbar:
                        pbar.set_description(f"Trying to read TSV with {enc} encoding")

                    # For large TSV files, use chunked reading if batch_size is specified
                    if batch_size:
                        if pbar:
                            pbar.update(10)
                        data = pd.read_csv(
                            file_path,
                            sep="\t",
                            chunksize=batch_size,
                            encoding=enc,
                            low_memory=False,
                        )
                        if pbar:
                            pbar.set_description(
                                f"Successfully loaded TSV with {enc} encoding"
                            )
                            pbar.update(80)  # Complete progress for setup
                        return data
                    else:
                        # Use progress bar if available
                        if pbar:
                            pbar.update(10)
                        data = pd.read_csv(
                            file_path, sep="\t", encoding=enc, low_memory=False
                        )
                        if pbar:
                            pbar.set_description(
                                f"Successfully loaded TSV with {enc} encoding"
                            )
                            pbar.update(80)  # Complete progress
                        return data
                except UnicodeDecodeError as e:
                    last_exception = e
                    continue
                except Exception as e:
                    # For other exceptions, try the next encoding
                    last_exception = e
                    continue

            # If we get here, all encodings failed
            raise last_exception or ValueError(
                "Failed to read TSV file with any encoding"
            )

        elif fmt == "json":
            # For JSON files, try different encodings
            last_exception = None

            for enc in encodings_to_try:
                try:
                    if pbar:
                        pbar.set_description(f"Trying JSON with {enc} encoding")
                        pbar.update(5)

                    # For JSON Files, try different approaches depending on format
                    # First check if it's a JSON Lines format by reading first line
                    with open(file_path, "r", encoding=enc) as f:
                        try:
                            first_line = f.readline().strip()
                            f.seek(0)

                            # Check if first line is valid JSON
                            try:
                                json.loads(first_line)
                                is_jsonl = True
                                if pbar:
                                    pbar.set_description(
                                        f"Detected JSON Lines format (encoding: {enc})"
                                    )
                                    pbar.update(5)
                            except json.JSONDecodeError:
                                is_jsonl = False
                                if pbar:
                                    pbar.set_description(
                                        f"Detected standard JSON format (encoding: {enc})"
                                    )
                                    pbar.update(5)

                            if is_jsonl:
                                # Process and return JSON Lines format
                                # ... (Rest of JSON Lines processing code, using the current encoding)
                                if pbar:
                                    pbar.set_description(
                                        f"Counting lines in JSONL file (encoding: {enc})"
                                    )
                                    pbar.update(5)

                                # For very large files, estimate line count instead of counting exactly
                                if file_size > 1024 * 1024 * 100:  # 100MB
                                    # Sample first 1MB to estimate line count
                                    with open(file_path, "r", encoding=enc) as sample_f:
                                        sample_data = sample_f.read(
                                            1024 * 1024
                                        )  # Read 1MB
                                        line_count_sample = sample_data.count("\n")
                                        lines_per_mb = line_count_sample / (
                                            len(sample_data) / (1024 * 1024)
                                        )
                                        estimated_lines = int(
                                            file_size / (1024 * 1024) * lines_per_mb
                                        )
                                        if pbar:
                                            pbar.set_description(
                                                f"Estimated {estimated_lines} lines in large JSONL file"
                                            )
                                            pbar.update(10)
                                else:
                                    # Count exact lines for smaller files
                                    estimated_lines = sum(
                                        1 for _ in open(file_path, "r", encoding=enc)
                                    )
                                    if pbar:
                                        pbar.set_description(
                                            f"Found {estimated_lines} lines in JSONL file"
                                        )
                                        pbar.update(10)
                                # ... (Continue with rest of the JSONL processing logic)

                                # Successfully processed with this encoding
                                return data
                            else:
                                # Standard JSON format
                                if pbar:
                                    pbar.set_description(
                                        f"Reading standard JSON (encoding: {enc})"
                                    )
                                    pbar.update(10)

                                # Reset file pointer and load the whole file
                                f.seek(0)
                                data = json.load(f)

                                if pbar:
                                    pbar.set_description(
                                        f"JSON data loaded with {enc} encoding"
                                    )
                                    pbar.update(70)  # Complete progress
                                return data
                        except json.JSONDecodeError as je:
                            last_exception = je
                            continue
                except UnicodeDecodeError as e:
                    last_exception = e
                    continue
                except Exception as e:
                    last_exception = e
                    continue

            # If we get here, all encodings failed
            raise last_exception or ValueError(
                f"Failed to read JSON file with any encoding"
            )
        else:
            print(f"Unsupported file format: {file_ext}")
            return None
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        if pbar:
            pbar.set_description(f"Error: {str(e)[:30]}...")
        return None


def parse_count_filters(count_str):
    """Parse comma-separated count filters into a list of (field, value, should_split, split_by, unique_by, count_all) tuples."""
    if not count_str:
        return []

    filters = []
    for item in count_str.split(","):
        item = item.strip()
        if not item:
            continue

        if "=" in item:
            field, value = item.split("=", 1)
            field = field.strip()
            value = value.strip()

            # Check if there's a .split modifier
            should_split = False
            split_by = None
            unique_by = None
            count_all = False

            # Determine if we should count all occurrences or just unique values
            if value.lower() == "all":
                count_all = True
                value = "all"  # Normalize

            if value.lower().endswith(".split"):
                should_split = True
                value = value[:-6]  # Remove the .split suffix
                # Check if the base value is 'all' to determine counting mode
                if value.lower() == "all":
                    count_all = True
                    value = "all"  # Normalize
            elif ".splitby(" in value.lower():
                # Extract the field name to split by
                match = re.search(r"\.splitBy\(([^)]+)\)", value, re.IGNORECASE)
                if match:
                    split_by = match.group(1).strip()
                    should_split = False
                    # Get the base value to determine counting mode
                    base_value = value[: value.lower().find(".splitby")]
                    if base_value.lower() == "all":
                        count_all = True
                        value = "all"  # Normalize
                    else:
                        value = "unique"  # Default to unique for splitBy
            elif ".uniqueby(" in value.lower():
                # Extract the field name to check uniqueness by
                match = re.search(r"\.uniqueBy\(([^)]+)\)", value, re.IGNORECASE)
                if match:
                    unique_by = match.group(1).strip()
                    base_value = value[: value.lower().find(".uniqueby")]
                    if base_value.lower() == "all":
                        count_all = True
                        value = "all"  # Normalize
                    else:
                        value = "unique"  # Default to unique
            elif value.lower().startswith("uniqueby("):
                # Handle uniqueBy as the main operation
                match = re.search(r"uniqueBy\(([^)]+)\)", value, re.IGNORECASE)
                if match:
                    unique_by = match.group(1).strip()
                    value = "unique"  # Special marker for uniqueBy operation
                    count_all = False  # Force unique mode
            elif value.lower().startswith("allby("):
                # Handle allBy as the main operation (count all occurrences, not just unique)
                match = re.search(r"allBy\(([^)]+)\)", value, re.IGNORECASE)
                if match:
                    unique_by = match.group(1).strip()
                    value = "all"  # Special marker for allBy operation
                    count_all = True  # Force all mode

            filters.append((field, value, should_split, split_by, unique_by, count_all))
        else:
            # Simple field name only - count all unique values
            filters.append((item, "unique", False, None, None, False))

    return filters


def count_matches(data, count_filters):
    """Count items matching the specified filters."""
    print(f"Starting count operation with {len(count_filters)} filter types")

    counts = defaultdict(int)
    unique_values = defaultdict(set)  # For tracking unique values
    all_values_count = defaultdict(Counter)  # For counting all occurrences
    nested_values = defaultdict(lambda: defaultdict(set))  # For splitBy functionality
    unique_by_values = defaultdict(
        lambda: defaultdict(set)
    )  # For uniqueBy functionality
    unique_by_tracking = defaultdict(set)  # Track unique values for uniqueBy

    # Track record counts for debugging
    total_records = 0
    records_with_title = 0
    records_with_empty_title = 0
    records_with_null_title = 0

    # Split filters into 'unique/all' filters and specific filters
    special_filters = [
        (field, value, should_split, split_by, unique_by, count_all)
        for field, value, should_split, split_by, unique_by, count_all in count_filters
        if value.lower() in ["unique", "all"]
    ]
    specific_filters = [
        (field, value, should_split, split_by, unique_by, count_all)
        for field, value, should_split, split_by, unique_by, count_all in count_filters
        if value.lower() not in ["unique", "all"]
    ]

    # Check if we're counting titles
    counting_title = any(
        field.lower() == "title" for field, _, _, _, _, _ in special_filters
    )

    # Count unique filters vs all filters
    unique_filters = [f for f in special_filters if f[1].lower() == "unique"]
    all_filters = [f for f in special_filters if f[1].lower() == "all"]

    # Print filter information
    print(
        f"Processing {len(unique_filters)} unique value filters, {len(all_filters)} all value filters, and {len(specific_filters)} specific value filters"
    )

    # Function to check if a record matches all specific filters
    def matches_specific_filters(record):
        if not specific_filters:
            return True
        return all(
            field in record
            and (
                (
                    should_split
                    and "," in str(record[field])
                    and value.lower()
                    in [v.strip().lower() for v in str(record[field]).split(",")]
                )
                or (
                    not should_split
                    and not split_by
                    and not unique_by
                    and str(record[field]).lower() == value.lower()
                )
            )
            for field, value, should_split, split_by, unique_by, count_all in specific_filters
        )

    # Function to process a value that might need to be split (unique values)
    def process_value(field_name, field_value, should_split, count_all=False):
        nonlocal records_with_empty_title, records_with_null_title

        # Debug: check for empty or null titles
        if field_name.lower() == "title":
            if field_value is None:
                records_with_null_title += 1
                # Replace None with "[EMPTY]" instead of skipping
                field_value = "[EMPTY]"
            elif str(field_value).strip() == "":
                records_with_empty_title += 1
                # Replace empty string with "[EMPTY]" instead of skipping
                field_value = "[EMPTY]"
            elif str(field_value).strip() == r"\N":
                records_with_null_title += 1
                # Replace \N with "[EMPTY]" instead of skipping
                field_value = "[EMPTY]"

        if should_split and field_value is not None and "," in str(field_value):
            for part in str(field_value).split(","):
                part = part.strip()
                if part == "":
                    part = "[EMPTY]"
                if part == r"\N":
                    part = "[EMPTY]"

                # Convert to lowercase for case-insensitive counting
                part_lower = part.lower()
                if count_all:
                    all_values_count[field_name][part_lower] += 1
                else:
                    unique_values[field_name].add(part_lower)
        else:
            if field_value is not None:
                # Handle \N values
                value = str(field_value)
                if value.strip() == r"\N":
                    value = "[EMPTY]"
                elif value.strip() == "":
                    value = "[EMPTY]"

                # Convert to lowercase for case-insensitive counting
                value_lower = value.lower()
                if count_all:
                    all_values_count[field_name][value_lower] += 1
                else:
                    unique_values[field_name].add(value_lower)

    # Function to process nested structures with splitBy
    def process_nested_value(field_name, field_value, split_by, count_all=False):
        if field_value is None or not isinstance(field_value, list):
            return

        for item in field_value:
            if isinstance(item, dict) and split_by in item:
                key = str(item[split_by])
                if not key or key == r"\N":
                    key = "[EMPTY]"

                # Convert to lowercase for case-insensitive counting
                key_lower = key.lower()
                if count_all:
                    all_values_count[field_name][key_lower] += 1
                else:
                    nested_values[field_name][key_lower].add(key_lower)

    # Function to process uniqueBy values
    def process_unique_by_value(
        field_name, field_value, unique_by_field, count_all=False
    ):
        if field_value is None or not isinstance(field_value, list):
            return

        for item in field_value:
            if isinstance(item, dict):
                if unique_by_field in item:
                    unique_key = str(item[unique_by_field])
                    if not unique_key or unique_key == r"\N":
                        unique_key = "[EMPTY]"

                    # Convert to lowercase for unique key tracking
                    unique_key_lower = unique_key.lower()
                    # Only count if we haven't seen this unique_by value before
                    if unique_key_lower not in unique_by_tracking[field_name]:
                        unique_by_tracking[field_name].add(unique_key_lower)
                        # If the target field exists, count it
                        if field_name in item:
                            target_value = str(item[field_name])
                            if not target_value or target_value == r"\N":
                                target_value = "[EMPTY]"

                            # Convert to lowercase for case-insensitive counting
                            target_value_lower = target_value.lower()
                            if count_all:
                                all_values_count[field_name][target_value_lower] += 1
                            else:
                                unique_by_values[field_name][target_value_lower].add(
                                    unique_key_lower
                                )

    # Process data based on type
    if hasattr(data, "get_chunk"):
        # Handle batched data
        try:
            # Count batches for progress
            print("Estimating number of batches for progress tracking...")
            try:
                # Try to get chunk count non-destructively
                batch_count = sum(1 for _ in data)
                print(f"Found {batch_count} batches to process")
                # Reset the iterator (may not work for all iterators)
                data = load_file(file_path, None, batch_size)
            except Exception as e:
                print(f"Could not count batches: {str(e)}")
                batch_count = None
                print("Will process with indeterminate progress bar")

        except Exception as e:
            print(f"Error setting up batch processing: {str(e)}")
            batch_count = None

        print("Processing batched data...")
        with tqdm(
            total=batch_count,
            desc="Processing batches",
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as batch_pbar:
            batch_num = 0
            for batch in data:
                batch_num += 1
                batch_pbar.set_description(f"Processing batch {batch_num}")

                batch_records = batch.to_dict("records")
                rows_processed = 0

                for record in batch_records:
                    total_records += 1
                    rows_processed += 1

                    # Debug: Check if record has a title field
                    if counting_title:
                        if "title" in record:
                            records_with_title += 1

                    if rows_processed % 10000 == 0:
                        batch_pbar.set_postfix(rows=rows_processed)

                    if matches_specific_filters(record):
                        # Process unique and all filters
                        for (
                            field,
                            value,
                            should_split,
                            split_by,
                            unique_by,
                            count_all,
                        ) in special_filters:
                            if field in record:
                                if unique_by:
                                    process_unique_by_value(
                                        field, record[field], unique_by, count_all
                                    )
                                elif split_by:
                                    process_nested_value(
                                        field, record[field], split_by, count_all
                                    )
                                else:
                                    process_value(
                                        field, record[field], should_split, count_all
                                    )

                        # Process specific filters
                        for (
                            field,
                            value,
                            should_split,
                            split_by,
                            unique_by,
                            count_all,
                        ) in specific_filters:
                            if field in record:
                                if should_split and "," in str(record[field]):
                                    if value.lower() in [
                                        v.strip().lower()
                                        for v in str(record[field]).split(",")
                                    ]:
                                        counts[f"{field}={value}"] += 1
                                elif split_by and isinstance(record[field], list):
                                    for item in record[field]:
                                        if (
                                            isinstance(item, dict)
                                            and split_by in item
                                            and str(item[split_by]).lower()
                                            == value.lower()
                                        ):
                                            counts[f"{field}={value}"] += 1
                                elif unique_by and isinstance(record[field], list):
                                    process_unique_by_value(
                                        field, record[field], unique_by, count_all
                                    )
                                elif (
                                    not split_by
                                    and not unique_by
                                    and str(record[field]).lower() == value.lower()
                                ):
                                    counts[f"{field}={value}"] += 1

                # Update progress after each batch
                batch_pbar.update(1)
                batch_pbar.set_postfix(rows_total=rows_processed)

    elif isinstance(data, list):
        # For list data, set up a progress bar
        print(f"Processing list data with {len(data)} items...")
        with tqdm(
            total=len(data),
            desc="Processing records",
            unit="record",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as record_pbar:
            for record_idx, record in enumerate(data):
                total_records += 1

                # Debug: Check if record has a title field
                if counting_title:
                    if isinstance(record, dict) and "title" in record:
                        records_with_title += 1

                # Update progress bar description periodically
                if record_idx % 10000 == 0:
                    record_pbar.set_description(f"Processing record {record_idx}")

                if isinstance(record, dict) and matches_specific_filters(record):
                    # Process unique and all filters
                    for (
                        field,
                        value,
                        should_split,
                        split_by,
                        unique_by,
                        count_all,
                    ) in special_filters:
                        if field in record:
                            if split_by and isinstance(record[field], list):
                                process_nested_value(
                                    field, record[field], split_by, count_all
                                )
                            else:
                                process_value(
                                    field, record[field], should_split, count_all
                                )

                    # Process specific filters
                    for (
                        field,
                        value,
                        should_split,
                        split_by,
                        unique_by,
                        count_all,
                    ) in specific_filters:
                        if field in record:
                            if should_split and "," in str(record[field]):
                                if value.lower() in [
                                    v.strip().lower()
                                    for v in str(record[field]).split(",")
                                ]:
                                    counts[f"{field}={value}"] += 1
                            elif split_by and isinstance(record[field], list):
                                for item in record[field]:
                                    if (
                                        isinstance(item, dict)
                                        and split_by in item
                                        and str(item[split_by]).lower() == value.lower()
                                    ):
                                        counts[f"{field}={value}"] += 1
                            elif (
                                not split_by
                                and not unique_by
                                and str(record[field]).lower() == value.lower()
                            ):
                                counts[f"{field}={value}"] += 1
                record_pbar.update(1)

    elif isinstance(data, dict):
        # For dict data, create a top-level progress bar
        print(f"Processing dictionary data with {len(data)} keys...")
        with tqdm(
            total=len(data),
            desc="Processing items",
            unit="item",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as item_pbar:
            for key, item in data.items():
                item_pbar.set_description(f"Processing {key}")

                if isinstance(item, dict) and matches_specific_filters(item):
                    # Process unique and all filters
                    for (
                        field,
                        value,
                        should_split,
                        split_by,
                        unique_by,
                        count_all,
                    ) in special_filters:
                        if field in item:
                            if split_by and isinstance(item[field], list):
                                process_nested_value(
                                    field, item[field], split_by, count_all
                                )
                            else:
                                process_value(
                                    field, item[field], should_split, count_all
                                )

                    # Process specific filters
                    for (
                        field,
                        value,
                        should_split,
                        split_by,
                        unique_by,
                        count_all,
                    ) in specific_filters:
                        if field in item:
                            if should_split and "," in str(item[field]):
                                if value.lower() in [
                                    v.strip().lower()
                                    for v in str(item[field]).split(",")
                                ]:
                                    counts[f"{field}={value}"] += 1
                            elif split_by and isinstance(item[field], list):
                                for sub_item in item[field]:
                                    if (
                                        isinstance(sub_item, dict)
                                        and split_by in sub_item
                                        and str(sub_item[split_by]).lower()
                                        == value.lower()
                                    ):
                                        counts[f"{field}={value}"] += 1
                            elif (
                                not split_by
                                and not unique_by
                                and str(item[field]).lower() == value.lower()
                            ):
                                counts[f"{field}={value}"] += 1
                elif isinstance(item, list):
                    # For list items, use a nested progress bar
                    with tqdm(
                        total=len(item),
                        desc=f"Processing {key} list",
                        unit="record",
                        leave=False,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                    ) as subitem_pbar:
                        for record in item:
                            if isinstance(record, dict) and matches_specific_filters(
                                record
                            ):
                                # Process unique and all filters
                                for (
                                    field,
                                    value,
                                    should_split,
                                    split_by,
                                    unique_by,
                                    count_all,
                                ) in special_filters:
                                    if field in record:
                                        if split_by and isinstance(record[field], list):
                                            process_nested_value(
                                                field,
                                                record[field],
                                                split_by,
                                                count_all,
                                            )
                                        else:
                                            process_value(
                                                field,
                                                record[field],
                                                should_split,
                                                count_all,
                                            )

                                # Process specific filters
                                for (
                                    field,
                                    value,
                                    should_split,
                                    split_by,
                                    unique_by,
                                    count_all,
                                ) in specific_filters:
                                    if field in record:
                                        if should_split and "," in str(record[field]):
                                            if value.lower() in [
                                                v.strip().lower()
                                                for v in str(record[field]).split(",")
                                            ]:
                                                counts[f"{field}={value}"] += 1
                                        elif split_by and isinstance(
                                            record[field], list
                                        ):
                                            for sub_item in record[field]:
                                                if (
                                                    isinstance(sub_item, dict)
                                                    and split_by in sub_item
                                                    and str(sub_item[split_by]).lower()
                                                    == value.lower()
                                                ):
                                                    counts[f"{field}={value}"] += 1
                                        elif (
                                            not split_by
                                            and not unique_by
                                            and str(record[field]).lower()
                                            == value.lower()
                                        ):
                                            counts[f"{field}={value}"] += 1
                            subitem_pbar.update(1)

                item_pbar.update(1)

    # Process unique values counts
    if unique_values:
        # Count occurrences of each unique value
        print(
            f"Processing {sum(len(values) for values in unique_values.values())} unique values..."
        )
        with tqdm(
            desc="Counting unique values",
            total=len(unique_values),
            unit="field",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as count_pbar:
            for field, values in unique_values.items():
                count_pbar.set_description(f"Counting {field} values ({len(values)})")
                for value in values:
                    counts[f"{field}={value}"] += 1
                count_pbar.update(1)

    # Process all values counts (counts all occurrences)
    if all_values_count:
        # Add all counts from Counter
        print(
            f"Processing counts for all occurrences ({len(all_values_count)} fields)..."
        )
        with tqdm(
            desc="Processing all counts",
            total=len(all_values_count),
            unit="field",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as all_count_pbar:
            for field, value_counter in all_values_count.items():
                all_count_pbar.set_description(
                    f"Counting {field} all values ({len(value_counter)})"
                )
                for value, count in value_counter.items():
                    counts[f"{field}={value}"] = count
                all_count_pbar.update(1)

    # Process nested values from splitBy operations (these are already unique by design)
    if nested_values:
        print(
            f"Processing {sum(len(values) for values in nested_values.values())} nested values..."
        )
        with tqdm(
            desc="Processing nested values",
            total=len(nested_values),
            unit="field",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as nested_pbar:
            for field, nested_dict in nested_values.items():
                nested_pbar.set_description(
                    f"Processing {field} nested values ({len(nested_dict)})"
                )
                for key, _ in nested_dict.items():
                    counts[f"{field}={key}"] += 1
                nested_pbar.update(1)

    # Process uniqueBy values
    if unique_by_values:
        print(
            f"Processing {sum(len(values) for values in unique_by_values.values())} uniqueBy values..."
        )
        with tqdm(
            desc="Processing uniqueBy values",
            total=len(unique_by_values),
            unit="field",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as unique_by_pbar:
            for field, value_dict in unique_by_values.items():
                unique_by_pbar.set_description(
                    f"Processing {field} uniqueBy values ({len(value_dict)})"
                )
                for value, unique_keys in value_dict.items():
                    # Each unique key represents a distinct occurrence
                    counts[f"{field}={value}"] = len(unique_keys)
                unique_by_pbar.update(1)

    if counting_title:
        # Get total title count (including duplicates) and unique title count
        title_total_count = 0
        empty_title_count = 0

        # Create field_counts from the counts dictionary
        field_counts = defaultdict(list)
        for key, count in counts.items():
            if "=" in key:
                field, value = key.split("=", 1)
                field_counts[field].append((value, count))

        # Calculate total title count from the field_counts
        for field, values in field_counts.items():
            if field.lower() == "title":
                title_total_count = sum(count for _, count in values)
                # Check for [EMPTY] titles
                for value, count in values:
                    if value.lower() == "[empty]":
                        empty_title_count = count
                        break

        print("\n=== TITLE FIELD STATISTICS ===")
        print(f"Total records processed: {total_records}")
        print(f"Records with title field: {records_with_title}")
        print(f"Records without title field: {total_records - records_with_title}")
        print(
            f"Records with null or \\N title: {records_with_null_title} (counted as '[EMPTY]')"
        )
        print(
            f"Records with empty title: {records_with_empty_title} (counted as '[EMPTY]')"
        )
        print(
            f"Total '[EMPTY]' titles: {empty_title_count if empty_title_count > 0 else 'N/A'}"
        )
        print(f"Unique title values: {len(unique_values.get('title', set()))}")
        print(f"Total title occurrences: {title_total_count} (includes duplicates)")

        if title_total_count < total_records:
            print(
                f"\nNote: There are {total_records - title_total_count} records that don't have titles."
            )
        if title_total_count != records_with_title:
            print(
                f"\nNote: The difference between records with title field ({records_with_title}) and"
            )
            print(
                f"total title occurrences ({title_total_count}) indicates there are duplicate titles."
            )

        print(
            "\nNote: All null, \\N, and empty title values are now counted as '[EMPTY]'."
        )
        print(
            "      The total occurrences count includes all entries (including duplicates)."
        )

    print(f"\nCount operation complete. Found {len(counts)} distinct count values.")
    return counts


def filter_by_count_criteria(data, count_filters, max_rows=3):
    """Filter data to display rows matching count filters."""
    # First separate display filters from count-only filters
    # Count-only filters are those with value='unique' or 'all'
    display_filters = [
        (field, value, should_split, split_by, unique_by, count_all)
        for field, value, should_split, split_by, unique_by, count_all in count_filters
        if value.lower() not in ["unique", "all"]
    ]

    # Get counts
    counts = count_matches(data, count_filters)

    # Get field counts organized by field
    field_counts = defaultdict(list)
    with tqdm(
        total=len(counts), desc="Organizing count results", unit="count"
    ) as count_pbar:
        for key, count in counts.items():
            if "=" in key:
                field, value = key.split("=", 1)
                field_counts[field].append((value, count))
            else:
                field_counts[key].append((key, count))
            count_pbar.update(1)

    # If there are no display filters, just return the counts
    if not display_filters:
        return None, max_rows

    # Filter data to show rows matching specific filters
    rows = []
    with tqdm(desc="Filtering data by criteria", leave=False) as filter_pbar:
        if hasattr(data, "get_chunk"):
            # For batched data, filter each batch
            for batch in data:
                records = batch.to_dict("records")
                for record in records:
                    if all(
                        field in record
                        and (
                            (
                                should_split
                                and "," in str(record[field])
                                and value.lower()
                                in [
                                    v.strip().lower()
                                    for v in str(record[field]).split(",")
                                ]
                            )
                            or (
                                not should_split
                                and not split_by
                                and not unique_by
                                and str(record[field]).lower() == value.lower()
                            )
                        )
                        for field, value, should_split, split_by, unique_by, count_all in display_filters
                    ):
                        rows.append(record)
                        if len(rows) >= max_rows:
                            return rows, max_rows
                filter_pbar.update(len(records))
        elif isinstance(data, list):
            # For list data, filter directly
            for record in data:
                if isinstance(record, dict) and all(
                    field in record
                    and (
                        (
                            should_split
                            and "," in str(record[field])
                            and value.lower()
                            in [
                                v.strip().lower() for v in str(record[field]).split(",")
                            ]
                        )
                        or (
                            not should_split
                            and not split_by
                            and not unique_by
                            and str(record[field]).lower() == value.lower()
                        )
                    )
                    for field, value, should_split, split_by, unique_by, count_all in display_filters
                ):
                    rows.append(record)
                    if len(rows) >= max_rows:
                        return rows, max_rows
                filter_pbar.update(1)
        elif isinstance(data, dict):
            # For dict data, check each item
            for key, item in data.items():
                if isinstance(item, dict) and all(
                    field in item
                    and (
                        (
                            should_split
                            and "," in str(item[field])
                            and value.lower()
                            in [v.strip().lower() for v in str(item[field]).split(",")]
                        )
                        or (
                            not should_split
                            and not split_by
                            and not unique_by
                            and str(item[field]).lower() == value.lower()
                        )
                    )
                    for field, value, should_split, split_by, unique_by, count_all in display_filters
                ):
                    rows.append(item)
                    if len(rows) >= max_rows:
                        return rows, max_rows
                elif isinstance(item, list):
                    # If the item is a list, check each subitem
                    for record in item:
                        if isinstance(record, dict) and all(
                            field in record
                            and (
                                (
                                    should_split
                                    and "," in str(record[field])
                                    and value.lower()
                                    in [
                                        v.strip().lower()
                                        for v in str(record[field]).split(",")
                                    ]
                                )
                                or (
                                    not should_split
                                    and not split_by
                                    and not unique_by
                                    and str(record[field]).lower() == value.lower()
                                )
                            )
                            for field, value, should_split, split_by, unique_by, count_all in display_filters
                        ):
                            rows.append(record)
                            if len(rows) >= max_rows:
                                return rows, max_rows
                    filter_pbar.update(len(item))
                filter_pbar.update(1)

    return rows, len(rows)


def analyze_data_shape(data):
    """Analyze the shape and structure of the data."""
    # Define colors for better readability
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    PURPLE = "\033[95m"
    ENDC = "\033[0m"

    # Track structure for visualization at the end
    structure_info = {
        "type": None,
        "shape": None,
        "keys": [],
        "example_values": {},
        "is_teamsvecs": False,
    }

    # Define a recursive function to process nested structures
    def process_item(item, prefix=""):
        # Handle different data types
        if isinstance(item, dict):
            print(f"{prefix}{BOLD}Dictionary with {len(item)} keys{ENDC}")
            if prefix == "":  # Top-level
                structure_info["type"] = "dict"
                structure_info["keys"] = list(item.keys())

                # Check if this looks like teamsvecs format
                if "id" in item and "skill" in item and "member" in item:
                    structure_info["is_teamsvecs"] = True

            for k, v in item.items():
                # Print key with type
                # Adjust spacing for id, skill, and member keys
                if k == "id":
                    process_item(v, prefix=f"{prefix}  {CYAN}{k}{ENDC}:\t\t")
                    structure_info["example_values"]["id"] = (
                        v[:3].tolist()
                        if hasattr(v, "tolist")
                        else v[:3] if hasattr(v, "__getitem__") else v
                    )
                elif k == "skill" or k == "member":
                    process_item(v, prefix=f"{prefix}  {CYAN}{k}{ENDC}:\t")
                    if k == "skill":
                        structure_info["example_values"]["skill"] = v
                    else:
                        structure_info["example_values"]["member"] = v
                else:
                    process_item(v, prefix=f"{prefix}  {CYAN}{k}{ENDC}: ")
        elif isinstance(item, list):
            print(f"{prefix}{BOLD}List with {len(item)} items{ENDC}")
            if prefix == "":  # Top-level
                structure_info["type"] = "list"
                structure_info["shape"] = len(item)

            if len(item) > 0:
                print(
                    f"{prefix}  First item type: {BLUE}{type(item[0]).__name__}{ENDC}"
                )
                if len(item) <= 3:  # Only process small lists in detail
                    for i, v in enumerate(item):
                        process_item(v, prefix=f"{prefix}  [{i}]: ")
        elif isinstance(item, (np.ndarray, pd.DataFrame)):
            shape_str = f"{item.shape}"
            print(
                f"{prefix}{BOLD}{type(item).__name__}{ENDC} with shape {YELLOW}{shape_str}{ENDC}"
            )

            if prefix.strip().startswith("id"):
                structure_info["example_values"]["id_shape"] = item.shape
                structure_info["example_values"]["id_sample"] = (
                    item[:3].tolist() if hasattr(item, "tolist") else item[:3]
                )

            if isinstance(item, np.ndarray):
                print(f"{prefix}  dtype: {BLUE}{item.dtype}{ENDC}")
                if len(item.shape) == 1:
                    print(f"{prefix}  Example values: {item[:3]}")
                else:
                    # For 2D+ arrays, print sample of values
                    print(
                        f"{prefix}  Non-zero elements: {GREEN}{np.count_nonzero(item):,}{ENDC}"
                    )
        elif isinstance(item, scipy.sparse.spmatrix):
            # For sparse matrices, show more detailed information with better formatting
            print(
                f"{prefix}{BOLD}{YELLOW}┌─────────────────────────────────────────────────┐{ENDC}"
            )
            print(f"{prefix}{BOLD}{YELLOW}│ {type(item).__name__}{ENDC}")

            # Format shape info - highlight rows and columns
            rows, cols = item.shape
            print(
                f"{prefix}{YELLOW}│ Dimensions: {BOLD}{rows:,}{ENDC}{YELLOW} rows × {BOLD}{cols:,}{ENDC}{YELLOW} columns{ENDC}"
            )

            # If this is a skill or member matrix, store info for visualization
            if "skill" in prefix.lower():
                structure_info["example_values"]["skill_shape"] = item.shape
                # Get a small sample of the matrix for visualization
                if item.shape[0] > 0:
                    sample_row = min(3, item.shape[0])
                    structure_info["example_values"]["skill_sample"] = (
                        item[:sample_row].toarray().tolist()
                        if hasattr(item, "toarray")
                        else None
                    )
            elif "member" in prefix.lower():
                structure_info["example_values"]["member_shape"] = item.shape
                # Get a small sample of the matrix for visualization
                if item.shape[0] > 0:
                    sample_row = min(3, item.shape[0])
                    structure_info["example_values"]["member_sample"] = (
                        item[:sample_row].toarray().tolist()
                        if hasattr(item, "toarray")
                        else None
                    )

            # Display density information
            density = (item.nnz / np.prod(item.shape)) * 100
            density_color = RED if density < 0.1 else (YELLOW if density < 1 else GREEN)
            print(
                f"{prefix}{YELLOW}│ Non-zero elements: {BOLD}{item.nnz:,}{ENDC}{YELLOW} ({density_color}{density:.4f}%{ENDC}{YELLOW} density){ENDC}"
            )
            print(
                f"{prefix}{YELLOW}│ Memory usage: {BOLD}{item.data.nbytes/1024/1024:.2f}{ENDC}{YELLOW} MB{ENDC}"
            )

            # For team data, if this looks like a skill or member matrix, provide interpretation
            if item.shape[0] > 0 and item.shape[1] > 0:
                if "skill" in prefix.lower() or "member" in prefix.lower():
                    interpret = "skills" if "skill" in prefix.lower() else "experts"
                    print(
                        f"{prefix}{YELLOW}│ Interpretation: {BOLD}{item.shape[0]:,}{ENDC}{YELLOW} teams with up to {BOLD}{item.shape[1]:,}{ENDC}{YELLOW} {interpret}{ENDC}"
                    )

                # Show distribution of non-zeros per row (teams)
                if item.shape[0] > 0:
                    # Count non-zeros in each row
                    row_counts = np.diff(item.indptr)
                    if len(row_counts) > 0:
                        print(
                            f"{prefix}{YELLOW}│ Min {interpret} per team: {BOLD}{np.min(row_counts)}{ENDC}"
                        )
                        print(
                            f"{prefix}{YELLOW}│ Max {interpret} per team: {BOLD}{np.max(row_counts)}{ENDC}"
                        )
                        print(
                            f"{prefix}{YELLOW}│ Avg {interpret} per team: {BOLD}{np.mean(row_counts):.2f}{ENDC}"
                        )
            print(
                f"{prefix}{BOLD}{YELLOW}└─────────────────────────────────────────────────┘{ENDC}"
            )
        else:
            type_name = type(item).__name__
            # Try to get string representation for small items
            try:
                if hasattr(item, "__len__") and len(item) < 100:
                    item_str = str(item)
                    if len(item_str) > 100:
                        item_str = item_str[:100] + "..."
                    print(f"{prefix}<{BLUE}{type_name}{ENDC}> {item_str}")
                else:
                    print(f"{prefix}<{BLUE}{type_name}{ENDC}>")
            except:
                print(f"{prefix}<{BLUE}{type_name}{ENDC}>")

    print(f"\n{'='*60}")
    print(f"               {BOLD}DATA STRUCTURE ANALYSIS{ENDC}")
    print(f"{'='*60}")
    # Start processing from the top level
    process_item(data)
    print(f"{'='*60}\n")

    # Add visualization of data structure
    if structure_info["is_teamsvecs"]:
        print(f"\n{'='*60}")
        print(f"               {BOLD}DATA VISUALIZATION{ENDC}")
        print(f"{'='*60}")

        # Get shapes
        id_sample = structure_info["example_values"].get("id_sample", [1, 2, 3])
        skill_shape = structure_info["example_values"].get("skill_shape", (0, 0))
        member_shape = structure_info["example_values"].get("member_shape", (0, 0))

        # Create a compact visualization of the structure and example values
        print(f"\n{BOLD}Structure of teamsvecs:{ENDC}")
        print(
            f"A dictionary with {len(structure_info['keys'])} keys: {', '.join(structure_info['keys'])}"
        )

        # Show simplified structure
        print(f"\n{BOLD}Conceptual structure:{ENDC}")
        print(f"{{")
        print(f'   "id": [{id_sample[0]}, {id_sample[1]}, {id_sample[2]}, ...],')
        print(
            f'   "skill": sparse_matrix({skill_shape[0]}, {skill_shape[1]}),  // teams × skills'
        )
        print(
            f'   "member": sparse_matrix({member_shape[0]}, {member_shape[1]})   // teams × experts'
        )
        print(f"}}")

        # Show the actual shape with example values
        skill_n = min(5, skill_shape[1]) if skill_shape else 0
        member_m = min(5, member_shape[1]) if member_shape else 0

        print(f"\n{BOLD}Example values:{ENDC}")
        print(f"[")

        # If we have skill and member samples, use them
        skill_samples = structure_info["example_values"].get("skill_sample", [])
        member_samples = structure_info["example_values"].get("member_sample", [])

        # Create example rows with sample data (or placeholder if not available)
        for i in range(min(3, len(id_sample))):
            id_val = id_sample[i] if i < len(id_sample) else "?"

            # Get skill sample for this row or generate a placeholder
            if i < len(skill_samples):
                skill_sample = skill_samples[i]
                if len(skill_sample) > skill_n:
                    skill_str = f"[{', '.join(map(str, skill_sample[:skill_n]))}, ... {len(skill_sample)-skill_n} more]"
                else:
                    skill_str = f"[{', '.join(map(str, skill_sample))}]"
            else:
                skill_str = f"[binary array with {skill_shape[1]} elements]"

            # Get member sample for this row or generate a placeholder
            if i < len(member_samples):
                member_sample = member_samples[i]
                if len(member_sample) > member_m:
                    member_str = f"[{', '.join(map(str, member_sample[:member_m]))}, ... {len(member_sample)-member_m} more]"
                else:
                    member_str = f"[{', '.join(map(str, member_sample))}]"
            else:
                member_str = f"[binary array with {member_shape[1]} elements]"

            # Print the row
            if i < 2:  # Not the last example
                print(
                    f'   {{ "id": {id_val}, "skill": {skill_str}, "member": {member_str} }},'
                )
            else:  # Last example
                print(
                    f'   {{ "id": {id_val}, "skill": {skill_str}, "member": {member_str} }}'
                )

        print(f"   ...")
        print(f"]")

        # Add explanation of what these structures represent
        print(f"\n{BOLD}Explanation:{ENDC}")
        print(f"- Each row represents a team")
        print(
            f"- The {YELLOW}skill{ENDC} matrix uses sparse representation for efficiency"
        )
        print(f"  - A value of 1 means the team (row) has that skill (column)")
        print(f"  - Each team can have multiple skills (multiple 1s in a row)")
        print(f"- The {YELLOW}member{ENDC} matrix also uses sparse representation")
        print(f"  - A value of 1 means the team (row) includes that expert (column)")
        print(f"  - Each team can have multiple experts (multiple 1s in a row)")
        print(f"{'='*60}\n")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Analyze data file structure",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input file to analyze")
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        default=3,
        help="Number of rows to display (default: 3)",
    )
    parser.add_argument(
        "-f", "--format", help="Force file format (pickle, csv, tsv, json)"
    )
    parser.add_argument(
        "-p", "--path", help="Additional files to analyze", action="append"
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of threads to use (default: number of CPUs)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        help="Process files in batches of this size (for large files)",
    )
    parser.add_argument(
        "-c",
        "--count",
        help="Count items matching field=value expressions (comma-separated)",
    )
    parser.add_argument(
        "-e",
        "--encoding",
        help="Specify file encoding (default: tries utf-8, latin1, cp1252, ISO-8859-1 in that order)",
    )

    # Add help text
    help_text = """
Required:
   -i INPUT, --input INPUT
	Input file to analyze

Optional:
   -r ROWS, --rows ROWS
	Number of rows to display (default: 3)

   -f FORMAT, --format FORMAT
	Force file format (pickle, csv, tsv, json)

   -p PATH, --path PATH
	Additional files to analyze

   -t THREADS, --threads THREADS
	Number of threads to use (default: number of CPUs)

   -b BATCH_SIZE, --batch-size BATCH_SIZE
	Process files in batches of this size (for large files)

   -e ENCODING, --encoding ENCODING
	Specify file encoding (default: tries utf-8, latin1, cp1252, ISO-8859-1 in that order)

   -c COUNT, --count COUNT
	Count items matching field=value expressions (comma-separated)
	All counting is case-insensitive (e.g., "Machine Learning" = "machine learning")
	Use "field=unique" to count only unique values for a field
	Use "field=all" to count all occurrences (including duplicates)
	Use "field=unique.split" to split comma-separated values and count each unique part
	Use "field=all.split" to split comma-separated values and count all occurrences
	Use "field=unique.splitBy(name)" to iterate objects in an array and count unique object["name"]
	Use "field=all.splitBy(name)" to iterate objects in an array and count all object["name"]
	Use "field=uniqueBy(raw)" to count unique values where object["raw"] is unique
	Use "field=allBy(raw)" to count all occurrences grouped by object["raw"]
	Use "field" (without =value) as shorthand for "field=unique"
	Examples: 
	  -c "title=unique,authors=unique.splitBy(name),year"  # Count unique titles & author names
	  -c "title=all,authors=all.splitBy(name),year"        # Count all titles & author names
	  -c "venue=uniqueBy(raw)"  # Count unique venues for each unique raw value
	  -c "venue=allBy(raw)"     # Count all occurrences of venues grouped by raw value
"""

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        print(help_text, file=sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Show arguments summary
    print(f"Starting analysis with {args.threads} threads")
    if args.batch_size:
        print(f"Using batch size: {args.batch_size}")
    if args.count:
        print(f"Count filters: {args.count}")

    # Set the number of threads
    if args.threads:
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            # Method already set, that's fine
            pass
        multiprocessing.cpu_count = lambda: args.threads

    # Parse count filters if provided
    count_filters = parse_count_filters(args.count) if args.count else []

    # Load file with progress bar
    print(f"\n[STAGE 1/3] Loading and analyzing {args.input}...")
    with tqdm(
        total=100,
        desc="Loading data",
        unit="%",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:
        data = load_file(args.input, args.format, args.batch_size, pbar, args.encoding)

    if data is None:
        print("Failed to load data.")
        return

    print(f"\n[STAGE 2/3] Analyzing structure...")
    # If batch processing, analyze the first batch
    if hasattr(data, "get_chunk"):
        try:
            batch = next(iter(data))
            print(f"Analyzing first batch ({len(batch)} rows)...")
            analyze_data_shape(batch)

            # Reset iterator for further processing
            print("Reinitializing data iterator for counting...")
            data = load_file(args.input, args.format, args.batch_size)
            if data is None:
                print("Failed to reload data for counting.")
                return
        except StopIteration:
            print("No data in file.")
            return
        except Exception as e:
            print(f"Error analyzing batch: {str(e)}")
            return
    else:
        # Analyze data shape
        analyze_data_shape(data)

    # Apply count filters if requested
    if count_filters:
        print(f"\n[STAGE 3/3] Processing counts and filters...")
        print(f"Analyzing data according to count criteria...")

        # Show progress for the counting process
        with tqdm(
            total=100,
            desc="Counting matches",
            unit="%",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as count_pbar:
            # Update at start
            count_pbar.update(5)

            # Calculate counts
            count_pbar.set_description("Processing data for counts")
            counts = count_matches(data, count_filters)
            count_pbar.update(65)

            # Run the filter criteria
            count_pbar.set_description("Filtering data by criteria")
            display_data, max_rows = filter_by_count_criteria(
                data, count_filters, args.rows
            )
            count_pbar.update(30)

        # Display counts
        print("\n=== COUNTS ===")

        # Organize counts by field
        field_counts = defaultdict(list)
        with tqdm(
            total=len(counts),
            desc="Organizing count results",
            unit="item",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as organize_pbar:
            for key, count in counts.items():
                if "=" in key:
                    field, value = key.split("=", 1)
                    field_counts[field].append((value, count))
                else:
                    field_counts[key].append((key, count))
                organize_pbar.update(1)

        # Display counts grouped by field
        print("Preparing final count display...")
        with tqdm(
            total=len(field_counts),
            desc="Formatting results",
            unit="field",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as format_pbar:
            for field, values in field_counts.items():
                # Sort values
                # For years or numeric values, try to sort numerically
                try:
                    if all(v[0].isdigit() for v in values):
                        # If all values are numeric, sort numerically
                        sorted_values = sorted(values, key=lambda x: int(x[0]))
                    else:
                        # Otherwise sort by count in descending order
                        sorted_values = sorted(values, key=lambda x: x[1], reverse=True)
                except:
                    # Fallback to sorting by count
                    sorted_values = sorted(values, key=lambda x: x[1], reverse=True)

                # Check if this field is a splitBy field or normal field
                split_by_field = any(
                    (f == field and sb) for f, v, s, sb, ub, ca in count_filters
                )

                # Calculate total occurrences for this field (sum of all values)
                total_occurrences = sum(count for _, count in sorted_values)

                # For normal fields, show total count
                if field.lower() == "year" or field.lower().endswith("year"):
                    # Special format for years - show in ascending order as a list
                    print(f"\n{field} (total occurrences: {total_occurrences}):")
                    year_list = ", ".join([v[0] for v in sorted_values])
                    print(f"values: {year_list}")
                else:
                    # Format for regular fields and splitBy fields
                    # Get the count of unique values
                    unique_count = len(sorted_values)

                    # Display field name with total occurrence count (not unique count)
                    print(
                        f"\n{field} (total occurrences: {total_occurrences}, unique values: {unique_count}):"
                    )

                    # Limit display to first 5 items
                    display_limit = 5
                    has_more = len(sorted_values) > display_limit

                    if split_by_field:
                        # For fields with splitBy, show as a list of names
                        first_n = sorted_values[:display_limit]
                        name_list = ", ".join([v[0] for v in first_n])

                        # Add indicator if we're limiting the display
                        if has_more:
                            print(f"names (first {display_limit}): {name_list}")
                            print(
                                f"  ... and {unique_count - display_limit} more unique values"
                            )
                        else:
                            print(f"names: {name_list}")
                    else:
                        # For normal fields, show with counts
                        for i, (value, count) in enumerate(sorted_values):
                            if i >= display_limit:
                                print(
                                    f"  ... and {unique_count - display_limit} more unique values"
                                )
                                break
                            print(f"  {value}: {count}")

                format_pbar.update(1)

        # Display filtered rows if any
        if display_data:
            # Display a sample of matching rows
            print(
                f"\n=== MATCHING ROWS (showing {len(display_data)} of {len(display_data)}) ==="
            )
            for i, row in enumerate(display_data):
                print(f"\nRow {i+1}:")
                for key, value in row.items():
                    print(f"  {key}: {value}")

        print("\nAnalysis complete!")
    else:
        print("\nAnalysis complete! (No count filters specified)")

    # Load and analyze additional files if provided
    if args.path:
        for path in args.path:
            print(f"\nAnalyzing additional file: {path}")
            with tqdm(
                total=100,
                desc="Loading data",
                unit="%",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as pbar:
                additional_data = load_file(
                    path, args.format, args.batch_size, pbar, args.encoding
                )

            if additional_data is None:
                print(f"Failed to load {path}.")
                continue

            # If batch processing, analyze the first batch
            if hasattr(additional_data, "get_chunk"):
                try:
                    batch = next(iter(additional_data))
                    print(f"Analyzing first batch ({len(batch)} rows)...")
                    analyze_data_shape(batch)
                except StopIteration:
                    print("No data in file.")
                    continue
                except Exception as e:
                    print(f"Error analyzing batch: {str(e)}")
                    continue
            else:
                # Analyze data shape
                analyze_data_shape(additional_data)

            # Apply count filters if requested
            if count_filters:
                print(f"\nFiltering {os.path.basename(path)} by criteria...")
                filtered_data, _ = filter_by_count_criteria(
                    additional_data, count_filters, args.rows
                )

                # Display a sample of matching rows
                if filtered_data:
                    print(
                        f"\n=== MATCHING ROWS (showing {len(filtered_data)} of {len(filtered_data)}) ==="
                    )
                    for i, row in enumerate(filtered_data):
                        print(f"\nRow {i+1}:")
                        for key, value in row.items():
                            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
