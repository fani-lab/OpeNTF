import os, json
import argparse
from time import time
import multiprocessing
import sys
import pickle
from datetime import datetime
import importlib.util
import shutil
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from shutil import copyfile
import scipy.sparse
import torch

# Import utility functions
from utils.tprint import tprint
from utils.set_gpus import set_gpus
from utils.format_time import format_time
from utils.create_unique_output_path import create_unique_output_path
from utils.create_toy_logs_folder import create_toy_logs_folder
from utils.create_evaluation_splits import create_evaluation_splits

# Import project modules
from cmn.tools import NumpyArrayEncoder, popular_nonpopular_ratio
from cmn_v3.dblp import Publication
from cmn_v3.gith import Repository
from utils.parse_nthreads import parse_nthreads

# Import model classes
from mdl.fnn import Fnn
from mdl.bnn import Bnn
from mdl.rnd import Rnd
from mdl.nmt import Nmt
from mdl.team2vec.team2vec import Team2Vec

from src.param import settings


def aggregate(output):
    files = list()
    for dirpath, dirnames, filenames in os.walk(output):
        if not dirnames:
            files += [
                os.path.join(os.path.normpath(dirpath), file).split(os.sep)
                for file in filenames
                if file.endswith("pred.eval.mean.csv")
            ]

    # concate the year folder to setting for temporal baselines
    for file in files:
        if file[3].startswith("t"):
            file[4] += "/" + file[5]
            del file[5]

    files = pd.DataFrame(
        files, columns=["", "", "domain", "baseline", "setting", "rfile"]
    )
    rfiles = files.groupby("rfile")
    for rf, r in rfiles:
        dfff = pd.DataFrame()
        rdomains = r.groupby("domain")
        for rd, rr in rdomains:
            names = ["metrics"]
            dff = pd.DataFrame()
            df = rdomains.get_group(rd)
            hr = False
            for i, row in df.iterrows():
                if not hr:
                    dff = pd.concat(
                        [
                            dff,
                            pd.read_csv(
                                f"{output}{rd}/{row['baseline']}/{row['setting']}/{rf}",
                                usecols=[0],
                            ),
                        ],
                        axis=1,
                        ignore_index=True,
                    )
                    hr = True
                dff = pd.concat(
                    [
                        dff,
                        pd.read_csv(
                            f"{output}{rd}/{row['baseline']}/{row['setting']}/{rf}",
                            usecols=[1],
                        ),
                    ],
                    axis=1,
                    ignore_index=True,
                )
                names += [row["baseline"] + "." + row["setting"]]
            dff.set_axis(names, axis=1, inplace=True)
            dff.to_csv(f"{output}{rd}/{rf.replace('.csv', '.agg.csv')}", index=False)


def dynamic_import(output_folder):
    """
    Dynamically import a module from the given output folder

    Args:
        output_folder: Path to the output folder where param_copy.py is located

    Returns:
        The imported param module with settings
    """
    # Check if output_folder is a directory or file
    output_path = Path(output_folder)
    if output_path.is_dir():
        # If it's a directory, look for param_copy.py in it
        param_copy_path = output_path / "param_copy.py"
    else:
        # If it's already a file path, use it as is
        param_copy_path = output_path

    # Make sure the file exists
    if not param_copy_path.exists():
        # If the file doesn't exist, create a blank param_copy.py with default settings
        tprint(
            f"Warning: param_copy.py not found at {param_copy_path}, creating default settings"
        )
        from shutil import copyfile

        try:
            param_src = os.path.join(os.path.dirname(__file__), "param.py")
            os.makedirs(output_path, exist_ok=True)
            copyfile(param_src, param_copy_path)
        except Exception as e:
            tprint(f"Error creating param_copy.py: {e}")
            # Return the original param module as fallback
            import param

            return param

    # Dynamically import the copied module
    spec = importlib.util.spec_from_file_location("param_copy", str(param_copy_path))
    if spec is None:
        tprint(
            f"Error: Could not create spec for {param_copy_path}, falling back to default param"
        )
        import param

        return param

    param_copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(param_copy)

    return param_copy


def run(
    data_list,
    domain_list,
    fair,
    filter,
    future,
    model_list,
    gpus,
    output,
    settings,
):
    overall_start_time = time()

    filter_str = (
        f".filtered.mt{settings['data']['filters']['common']['min_team_size']}.ts{settings['data']['filters']['common']['min_teams_per_expert']}"
        if filter
        # TODO: remove this once we have a new filter logic setup, currently use the -o flag to specify the output folder name
        else ""
    )

    datasets = {}
    models = {}

    if "dblp" in domain_list:
        datasets["dblp"] = Publication
    # if "imdb" in domain_list:
    #     datasets["imdb"] = Movie
    # if "uspt" in domain_list:
    #     datasets["uspt"] = Patent
    if "gith" in domain_list:
        datasets["gith"] = Repository

    # model names starting with 't' means that they will follow the streaming scenario
    # model names ending with _a1 means that they have one 1 added to their input for time as aspect learning
    # model names having _dt2v means that they learn the input embedding with doc2vec where input is (skills + year)

    # Extract models, excluding None if present
    models_to_use = [m for m in model_list if m is not None]

    # Check if we're in preprocessing-only mode
    preprocessing_only = len(models_to_use) == 0

    # Process each dataset
    for d_name, d_cls in datasets.items():
        datapath = data_list[domain_list.index(d_name)]

        # Base preprocessed data directory
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed")
        )

        # The domain directory (e.g., 'dblp')
        domain_dir = os.path.join(base_dir, d_name)

        # If output is specified, use it as a subfolder name
        if output:
            # Create a folder name combining the output value with filter specifications
            output_folder = f"{output}.{filter_str}"
            # Full output path
            base_output_path = os.path.join(domain_dir, output_folder)
            # Create a unique path if it already exists
            output_path = create_unique_output_path(base_output_path)
        else:
            # If no output specified, use the dataset filename with filter specifications
            dataset_filename = os.path.basename(datapath)
            base_output_path = os.path.join(
                domain_dir, f"{dataset_filename}.{filter_str}"
            )
            # Create a unique path if it already exists
            output_path = create_unique_output_path(base_output_path)

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # copy settings to output path
        shutil.copy2(
            os.path.join(os.path.dirname(__file__), "param.py"),
            os.path.join(output_path, "param_copy.py"),
        )

        # Dynamically import the copied param_copy.py module from the output folder
        param_copy_module = dynamic_import(output_path)
        settings = param_copy_module.settings

        # Set domain name in settings for proper logging
        settings["data"]["domain"] = d_name

        # Start time for this dataset's preprocessing
        dataset_start_time = time()
        tprint(f"Starting preprocessing for {d_name} dataset from {datapath}")
        tprint(f"Output will be saved to {os.path.abspath(output_path)}")

        # Generate sparse vectors
        vecs, indexes = d_cls.generate_sparse_vectors_v3(
            datapath, output_path, gpus=gpus
        )

        # Log the actual output directory used
        tprint(f"Data saved to {os.path.abspath(output_path)}")

        # Apply filters to the teamsvecs data
        tprint("Applying filters to teamsvecs data...")

        # Get filter configuration from settings
        filter_config = settings["data"].get("filters", {})

        # Path to the apply_filters.py script
        apply_filters_script = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "cmn_v3",
                "helper_functions",
                "apply_filters.py",
            )
        )

        # Path to the teamsvecs.pkl file
        teamsvecs_path = os.path.join(output_path, "teamsvecs.pkl")

        # Run the apply_filters.py script
        if os.path.exists(apply_filters_script):
            tprint(f"Running filter script: {apply_filters_script}")

            # Create a temporary JSON config file for the filters
            filter_config_path = os.path.join(output_path, "filter_config.json")
            with open(filter_config_path, "w") as f:
                json.dump(filter_config, f, indent=2)

            # Build the command
            filter_cmd = f"python {apply_filters_script} -i {teamsvecs_path} -o {output_path} -c {filter_config_path} -d {d_name}"

            try:
                tprint(f"Running command: {filter_cmd}")
                os.system(filter_cmd)
                tprint(f"Filters applied to teamsvecs data")

                # Reload the filtered teamsvecs data
                with open(teamsvecs_path, "rb") as f:
                    vecs = pickle.load(f)
            except Exception as e:
                tprint(f"Error applying filters: {str(e)}")
        else:
            tprint(
                f"Warning: apply_filters.py script not found at {apply_filters_script}"
            )

        # Ensure that the 'id' key exists in vecs for evaluation splits
        if "id" not in vecs:
            tprint(
                f"WARNING: 'id' key not found in vecs. Creating a default 'id' based on the 'skill' matrix."
            )
            if "skill" in vecs:
                vecs["id"] = np.arange(vecs["skill"].shape[0])
            else:
                raise KeyError("Both 'id' and 'skill' keys are missing from vecs!")

        # Process year indices for temporal data
        year_idx = []
        if "i2y" in indexes.keys():
            for i in range(1, len(indexes["i2y"])):
                if (
                    indexes["i2y"][i][0] - indexes["i2y"][i - 1][0]
                    > settings["model"]["nfolds"]
                ):
                    year_idx.append(indexes["i2y"][i - 1])
            year_idx.append(indexes["i2y"][-1])
            indexes["i2y"] = year_idx

        # Create evaluation splits - this creates splits.json
        splits = create_evaluation_splits(
            vecs["id"].shape[0] if "id" in vecs and vecs["id"].shape[0] > 0 else 0,
            settings["model"]["nfolds"],
            settings["model"]["train_test_split"],
            (
                indexes["i2y"]
                if future and "i2y" in indexes and len(indexes["i2y"]) > 0
                else None
            ),
            output=f"{output_path}",
            step_ahead=settings["model"]["step_ahead"],
        )

        # Calculate elapsed time
        dataset_elapsed_time = time() - dataset_start_time
        formatted_time = format_time(dataset_elapsed_time)

        tprint(f"Data preprocessing completed for {d_name} dataset in {formatted_time}")
        tprint(f"Created files:")
        tprint(f"  - {output_path}/teams.pkl")
        tprint(f"  - {output_path}/indexes.pkl")
        tprint(f"  - {output_path}/teamsvecs.pkl")
        tprint(f"  - {output_path}/splits.json")

        # Load teams regardless of raw_logs setting, as we'll need them for toy dataset generation
        with open(os.path.join(output_path, "teams.pkl"), "rb") as f:
            teams = pickle.load(f)

        # Generate counts directory (renamed from experts-skills-counts)
        # Only generate these files if raw_logs is True in the settings
        if settings["data"]["processing"].get("raw_logs", False):
            counts_dir = os.path.join(output_path, "counts")
            os.makedirs(counts_dir, exist_ok=True)

            # Extract and count skills
            all_skills = []
            for team in teams:
                all_skills.extend(team.skills)

            skill_counts = {}
            for skill in all_skills:
                if skill in skill_counts:
                    skill_counts[skill] += 1
                else:
                    skill_counts[skill] = 1

            # Sort skills by count (descending)
            sorted_skills = sorted(
                skill_counts.items(), key=lambda x: x[1], reverse=True
            )

            # Write skills to file
            skills_file = os.path.join(counts_dir, f"skills_{len(sorted_skills)}.log")
            with open(skills_file, "w", encoding="utf-8") as f:
                for skill, count in sorted_skills:
                    f.write(f"{count}\t{skill}\n")

            # Extract and count experts (members)
            all_members = {}
            for team in teams:
                for member in team.members:
                    member_id = member.id
                    # Store the member object, not just the ID
                    if member_id in all_members:
                        all_members[member_id][0] += 1
                    else:
                        all_members[member_id] = [1, member]

            # Sort experts by count (descending)
            sorted_members = sorted(
                all_members.items(), key=lambda x: x[1][0], reverse=True
            )

            # Write experts to file
            experts_file = os.path.join(
                counts_dir, f"experts_{len(sorted_members)}.log"
            )
            with open(experts_file, "w", encoding="utf-8") as f:
                for member_id, (count, member) in sorted_members:
                    # Use member.name for DBLP, member.login for GitHub
                    member_name = getattr(
                        member, "name", getattr(member, "login", member_id)
                    )
                    f.write(f"{count}\t{member_name}\n")

            # Add teams count file
            teams_file = os.path.join(counts_dir, f"teams_{len(teams)}.log")
            with open(teams_file, "w", encoding="utf-8") as f:
                for team in teams:
                    # Use team.title or other appropriate attribute for team name, or "EMPTY" if none exist
                    team_title = getattr(
                        team,
                        "title",
                        getattr(team, "name", getattr(team, "id", "EMPTY")),
                    )
                    f.write(f"{1}\t{team_title}\n")

            tprint(f"Generated counts files:")
            tprint(f"  - {skills_file}")
            tprint(f"  - {experts_file}")
            tprint(f"  - {teams_file}")
        else:
            tprint(
                "Skipping generation of counts files - raw_logs is disabled in settings"
            )

        # Generate reports directory
        stats_reports_dir = os.path.join(output_path, "reports")
        os.makedirs(stats_reports_dir, exist_ok=True)

        # Call generate_reports.py script
        teamsvecs_path = os.path.join(output_path, "teamsvecs.pkl")
        gpu_param = f"gpu={gpus}" if gpus is not None else "cpu"
        reports_script = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "cmn_v3",
                "helper_functions",
                "generate_reports.py",
            )
        )

        if os.path.exists(reports_script):
            tprint(f"Generating data reports using {reports_script}")
            reports_cmd = f"python {reports_script} -i {teamsvecs_path} -o {stats_reports_dir} -mode {gpu_param}"

            try:
                tprint(f"Running command: {reports_cmd}")
                os.system(reports_cmd)
                tprint(f"Data reports generated in {stats_reports_dir}")
            except Exception as e:
                tprint(f"Error generating data reports: {str(e)}")
        else:
            tprint(f"Warning: generate_reports.py script not found at {reports_script}")

        # Generate toy dataset if enabled in settings
        if settings["data"]["processing"].get("make_toy_data", False):
            toy_data_size = settings["data"]["processing"].get("toy_data_size", 100)
            tprint(f"Generating toy dataset with {toy_data_size} teams")

            # Note: We already loaded the teams variable earlier, no need to load it again

            # Extract main dataset folder name to use as the base for toy dataset
            main_output_folder = os.path.basename(output_path)

            # Create toy output path based on main dataset folder name
            toy_output_folder = f"{main_output_folder}_toy"
            base_toy_output_path = os.path.join(domain_dir, toy_output_folder)
            toy_output_path = create_unique_output_path(base_toy_output_path)
            os.makedirs(toy_output_path, exist_ok=True)

            tprint(
                f"Toy dataset with {toy_data_size} teams will be saved to {os.path.abspath(toy_output_path)}"
            )

            # Create a subset of teams
            toy_teams = teams[:toy_data_size] if len(teams) > toy_data_size else teams

            # Save toy teams
            with open(os.path.join(toy_output_path, "teams.pkl"), "wb") as f:
                pickle.dump(toy_teams, f)

            # Build indexes for toy dataset
            toy_indexes = d_cls.build_indexes(toy_teams)

            # Save toy indexes
            with open(os.path.join(toy_output_path, "indexes.pkl"), "wb") as f:
                pickle.dump(toy_indexes, f)

            # Generate sparse vectors for toy dataset
            # Extract the subset of vectors
            toy_vecs = {}
            for key in vecs:
                if hasattr(vecs[key], "shape") and vecs[key].shape[0] > toy_data_size:
                    toy_vecs[key] = vecs[key][:toy_data_size]
                else:
                    toy_vecs[key] = vecs[key]

            # Save toy teamsvecs
            with open(os.path.join(toy_output_path, "teamsvecs.pkl"), "wb") as f:
                pickle.dump(toy_vecs, f)

            # Apply filters to the toy teamsvecs data
            tprint("Applying filters to toy teamsvecs data...")

            # Get filter configuration from settings
            filter_config = settings["data"].get("filters", {})

            # Path to the apply_filters.py script
            apply_filters_script = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "cmn_v3",
                    "helper_functions",
                    "apply_filters.py",
                )
            )

            # Path to the toy teamsvecs.pkl file
            toy_teamsvecs_path = os.path.join(toy_output_path, "teamsvecs.pkl")

            # Run the apply_filters.py script
            if os.path.exists(apply_filters_script):
                tprint(f"Running filter script for toy dataset: {apply_filters_script}")

                # Create a temporary JSON config file for the filters
                toy_filter_config_path = os.path.join(
                    toy_output_path, "filter_config.json"
                )
                with open(toy_filter_config_path, "w") as f:
                    json.dump(filter_config, f, indent=2)

                # Build the command
                toy_filter_cmd = f"python {apply_filters_script} -i {toy_teamsvecs_path} -o {toy_output_path} -c {toy_filter_config_path} -d {d_name}"

                try:
                    tprint(f"Running command: {toy_filter_cmd}")
                    os.system(toy_filter_cmd)
                    tprint(f"Filters applied to toy teamsvecs data")

                    # Reload the filtered toy teamsvecs data
                    with open(toy_teamsvecs_path, "rb") as f:
                        toy_vecs = pickle.load(f)
                except Exception as e:
                    tprint(f"Error applying filters to toy dataset: {str(e)}")
            else:
                tprint(
                    f"Warning: apply_filters.py script not found at {apply_filters_script}"
                )

            # Create evaluation splits for toy dataset
            toy_splits = create_evaluation_splits(
                (
                    toy_vecs["id"].shape[0]
                    if "id" in toy_vecs and toy_vecs["id"].shape[0] > 0
                    else 0
                ),
                settings["model"]["nfolds"],
                settings["model"]["train_test_split"],
                (
                    toy_indexes["i2y"]
                    if future and "i2y" in toy_indexes and len(toy_indexes["i2y"]) > 0
                    else None
                ),
                output=f"{toy_output_path}",
                step_ahead=settings["model"]["step_ahead"],
            )

            tprint(f"Toy dataset created with {len(toy_teams)} teams")
            tprint(f"Created toy files:")
            tprint(f"  - {toy_output_path}/teams.pkl")
            tprint(f"  - {toy_output_path}/indexes.pkl")
            tprint(f"  - {toy_output_path}/teamsvecs.pkl")
            tprint(f"  - {toy_output_path}/splits.json")

            # Create logs folder with entries_processed.log and skills.log
            create_toy_logs_folder(toy_output_path, toy_teams)

            # Generate counts directory for toy dataset (renamed from experts-skills-counts)
            # Only generate these files if raw_logs is True in the settings
            if settings["data"]["processing"].get("raw_logs", False):
                toy_counts_dir = os.path.join(toy_output_path, "counts")
                os.makedirs(toy_counts_dir, exist_ok=True)

                # Extract and count skills for toy dataset
                toy_all_skills = []
                for team in toy_teams:
                    toy_all_skills.extend(team.skills)

                toy_skill_counts = {}
                for skill in toy_all_skills:
                    if skill in toy_skill_counts:
                        toy_skill_counts[skill] += 1
                    else:
                        toy_skill_counts[skill] = 1

                # Sort skills by count (descending)
                toy_sorted_skills = sorted(
                    toy_skill_counts.items(), key=lambda x: x[1], reverse=True
                )

                # Write skills to file
                toy_skills_file = os.path.join(
                    toy_counts_dir, f"skills_{len(toy_sorted_skills)}.log"
                )
                with open(toy_skills_file, "w", encoding="utf-8") as f:
                    for skill, count in toy_sorted_skills:
                        f.write(f"{count}\t{skill}\n")

                # Extract and count experts (members) for toy dataset
                toy_all_members = {}
                for team in toy_teams:
                    for member in team.members:
                        member_id = member.id
                        # Store the member object, not just the ID
                        if member_id in toy_all_members:
                            toy_all_members[member_id][0] += 1
                        else:
                            toy_all_members[member_id] = [1, member]

                # Sort experts by count (descending)
                toy_sorted_members = sorted(
                    toy_all_members.items(), key=lambda x: x[1][0], reverse=True
                )

                # Write experts to file
                toy_experts_file = os.path.join(
                    toy_counts_dir, f"experts_{len(toy_sorted_members)}.log"
                )
                with open(toy_experts_file, "w", encoding="utf-8") as f:
                    for member_id, (count, member) in toy_sorted_members:
                        # Use member.name for DBLP, member.login for GitHub
                        member_name = getattr(
                            member, "name", getattr(member, "login", member_id)
                        )
                        f.write(f"{count}\t{member_name}\n")

                # Add teams count file
                toy_teams_file = os.path.join(
                    toy_counts_dir, f"teams_{len(toy_teams)}.log"
                )
                with open(toy_teams_file, "w", encoding="utf-8") as f:
                    for team in toy_teams:
                        # Use team.title or other appropriate attribute for team name, or "EMPTY" if none exist
                        team_title = getattr(
                            team,
                            "title",
                            getattr(team, "name", getattr(team, "id", "EMPTY")),
                        )
                        f.write(f"{1}\t{team_title}\n")

                tprint(f"Generated counts files for toy dataset:")
                tprint(f"  - {toy_skills_file}")
                tprint(f"  - {toy_experts_file}")
                tprint(f"  - {toy_teams_file}")
            else:
                tprint(
                    "Skipping generation of counts files for toy dataset - raw_logs is disabled in settings"
                )

            # Generate reports for toy dataset
            toy_stats_reports_dir = os.path.join(toy_output_path, "reports")
            os.makedirs(toy_stats_reports_dir, exist_ok=True)

            # Call generate_reports.py script for toy dataset
            toy_teamsvecs_path = os.path.join(toy_output_path, "teamsvecs.pkl")

            if os.path.exists(reports_script):
                tprint(f"Generating data reports for toy dataset")
                toy_reports_cmd = f"python {reports_script} -i {toy_teamsvecs_path} -o {toy_stats_reports_dir} -mode {gpu_param}"

                try:
                    tprint(f"Running command: {toy_reports_cmd}")
                    os.system(toy_reports_cmd)
                    tprint(f"Toy data reports generated in {toy_stats_reports_dir}")
                except Exception as e:
                    tprint(f"Error generating toy data reports: {str(e)}")
            else:
                tprint(
                    f"Warning: generate_reports.py script not found at {reports_script}"
                )

        # Calculate overall elapsed time
        overall_elapsed_time = time() - overall_start_time
        formatted_overall_time = format_time(overall_elapsed_time)

        tprint(f"All preprocessing completed in {formatted_overall_time}")
        tprint(f"Exiting as no models were specified.")
        return

    if preprocessing_only:
        tprint(f"Preprocessing only completed in {formatted_overall_time}")
        tprint(f"Exiting as no models were specified.")
        return

    # Only get here if models are specified
    # Load the models
    for model_name in models_to_use:
        if model_name.startswith("nmt"):
            models[model_name] = Nmt()
        elif model_name == "random":
            models["random"] = Rnd()
        elif model_name == "fnn":
            models["fnn"] = Fnn()
        elif model_name == "bnn":
            models["bnn"] = Bnn()
        # Add other model types as needed

    # Ensure we have at least one model
    assert len(models) > 0, "No valid models were specified!"

    for d_name, d_cls in datasets.items():
        datapath = data_list[domain_list.index(d_name)]
        prep_output = f"./../data/preprocessed/{d_name}/{os.path.split(datapath)[-1]}"
        vecs, indexes = d_cls.generate_sparse_vectors_v3(
            datapath, f"{prep_output}{filter_str}", filter, settings["data"], gpus=gpus
        )
        # Ensure that the 'id' key exists in vecs for evaluation splits.
        if "id" not in vecs:
            tprint(
                "WARNING: 'id' key not found in vecs. Creating a default 'id' based on the 'skill' matrix."
            )
            if "skill" in vecs:
                vecs["id"] = np.arange(vecs["skill"].shape[0])
            else:
                raise KeyError("Both 'id' and 'skill' keys are missing from vecs!")

        year_idx = []
        # do only if i2y exists in data
        if "i2y" in indexes.keys():
            for i in range(1, len(indexes["i2y"])):
                if (
                    indexes["i2y"][i][0] - indexes["i2y"][i - 1][0]
                    > settings["model"]["nfolds"]
                ):
                    year_idx.append(indexes["i2y"][i - 1])
            year_idx.append(indexes["i2y"][-1])
        indexes["i2y"] = year_idx
        splits = create_evaluation_splits(
            vecs["id"].shape[0],
            settings["model"]["nfolds"],
            settings["model"]["train_test_split"],
            indexes["i2y"] if future else None,
            output=f"{prep_output}{filter_str}",
            step_ahead=settings["model"]["step_ahead"],
        )

        for m_name, m_obj in models.items():
            vecs_ = vecs.copy()
            if m_name.find("_emb") > 0:
                t2v = Team2Vec(
                    vecs,
                    indexes,
                    "dt2v" if m_name.find("_dt2v") > 0 else "skill",
                    f"./../data/preprocessed/{d_name}/{os.path.split(datapath)[-1]}{filter_str}",
                )
                emb_setting = settings["model"]["baseline"]["emb"]
                t2v.train(
                    emb_setting["d"],
                    emb_setting["w"],
                    emb_setting["dm"],
                    emb_setting["e"],
                )
                vecs_["skill"] = t2v.dv()

            if m_name.endswith("a1"):
                vecs_["skill"] = lil_matrix(
                    scipy.sparse.hstack(
                        (
                            vecs_["skill"],
                            lil_matrix(np.ones((vecs_["skill"].shape[0], 1))),
                        )
                    )
                )

            # Kap: added to indicate if GPU is available and to get the last GPU
            set_gpus(settings["gpus"])

            baseline_name = (
                m_name.lstrip("t")
                .replace("_emb", "")
                .replace("_dt2v", "")
                .replace("_a1", "")
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tprint(f"Using device: {device}")
            tprint(f"Running for (dataset, model): ({d_name}, {m_name}) ... ")

            base_output_path = f"{output}{os.path.split(datapath)[-1]}{filter_str}/{m_name}/t{vecs_['skill'].shape[0]}.s{vecs_['skill'].shape[1]}.m{vecs_['member'].shape[1]}.{baseline_name}"
            output_path = create_unique_output_path(base_output_path)
            tprint(f"Output will be saved to {os.path.abspath(output_path)}")

            # Kap: don't copy if model name starts with "nmt", I have a handler in nmt.py
            if not m_name.startswith("nmt"):
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)
                # param.py is already copied to the output folder at the beginning
                # No need to copy it again, just copy from the output folder to the model folder
                copyfile(
                    f"{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'temp', output)}/param_copy.py",
                    f"{output_path}/param_copy.py",
                )

            m_obj.run(
                splits,
                vecs_,
                indexes,
                f"{output_path}",
                baseline_name,
                settings["model"]["cmd"],
                settings["fair"],
                merge_skills=False,
            )
    if "agg" in settings["model"]["cmd"]:
        aggregate(output)


def addargs(parser):
    """Parse and Set Arguments."""

    # Define our custom help text
    help_text = """OpeNTF: Open Neural Team Formation

Required:
   -i INPUT, --input INPUT
\tINPUT FOLDER where to look for teamsvecs.pkl or raw file (default: None)

   -d DOMAIN, --domain DOMAIN
\tDomain of the dataset. Options: dblp, gith, imdb, uspt (default: None)


Optionals:
   -m MODEL, --model MODEL
\tModel to perform the task, or the type of the experiments to run, e.g., random, heuristic, expert, etc. If not provided, process will stop after data loading. (default: None)

   -train TRAIN, --train TRAIN
\tWhether to train the model (default: 0)

   -filter FILTER, --filter FILTER
\tWhether to filter data: zero: no filtering, one: filter zero degree nodes, two: filter one degree nodes (default: 0)

   -future FUTURE, --future FUTURE
\tForecast future teams: zero: no need to forecast future teams, one: predict future teams (default: 0)

   -fair FAIR, --fair FAIR
\tApply fairness to model (default: 0)

   -o OUTPUT, --output OUTPUT
\tOUTPUT FOLDER NAME only (the full path is hardcoded, this specifies the folder name only). If the specified folder exists, a number in brackets will be added (e.g., 'v3_filtered1 (1)') (default: None)

   -gpus GPUS, --gpus GPUS
\tCUDA Visible GPUs (default: None)

   -t THREADS, --threads THREADS
\tNumber of threads to use for parallel processing. Options:
\t  - Specific number (e.g., 16): Use exactly that many threads
\t  - 0: Use value from param.py (defaults to all threads)
\t  - In param.py, you can set a percentage (0.5 = 50% of threads) or specific count

   -b BATCH_SIZE, --batch-size BATCH_SIZE
\tBatch size for processing large datasets (default: IMDB: 10000, DBLP: 10000, GITH: 1000, USPT: 5000)
"""

    # Override the help option to print our custom help text
    parser.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )

    # Required arguments group
    required = parser.add_argument_group("Required")
    required.add_argument(
        "-i",
        "--input",  # Updated to match help menu
        dest="data",  # Keep the original destination
        help=argparse.SUPPRESS,
        required=True,
        metavar="INPUT",
    )

    required.add_argument(
        "-d",
        "--domain",  # Updated to match help menu
        dest="domain",
        help=argparse.SUPPRESS,
        required=True,
        metavar="DOMAIN",
    )

    # Optional arguments group
    optionals = parser.add_argument_group("Optionals")
    optionals.add_argument(
        "-m",
        "--model",  # Updated to match help menu
        dest="model",
        help=argparse.SUPPRESS,
        required=False,
        default=None,
        metavar="MODEL",
    )

    optionals.add_argument(
        "-train",
        "--train",
        dest="train",
        help=argparse.SUPPRESS,
        default=0,
        type=int,
        metavar="TRAIN",
    )

    optionals.add_argument(
        "-filter",
        "--filter",
        dest="filter",
        help=argparse.SUPPRESS,
        default=0,
        type=int,
        metavar="FILTER",
    )

    optionals.add_argument(
        "-future",
        "--future",
        dest="future",
        help=argparse.SUPPRESS,
        default=0,
        type=int,
        metavar="FUTURE",
    )

    optionals.add_argument(
        "-fair",
        "--fair",
        dest="fair",
        help=argparse.SUPPRESS,
        default=0,
        type=int,
        metavar="FAIR",
    )

    optionals.add_argument(
        "-o",
        "--output",  # Updated to match help menu
        dest="output",
        help=argparse.SUPPRESS,
        default=None,
        type=str,
        metavar="OUTPUT",
    )

    optionals.add_argument(
        "-gpus",
        "--gpus",
        dest="gpus",
        help=argparse.SUPPRESS,
        default=settings["gpus"],
        type=str,
        metavar="GPUS",
    )

    optionals.add_argument(
        "-t",
        "--threads",
        dest="threads",
        help=argparse.SUPPRESS,
        default=parse_nthreads(settings),
        type=int,
        metavar="THREADS",
    )

    optionals.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        help=argparse.SUPPRESS,
        default=0,  # 0 means use domain-specific defaults
        type=int,
        metavar="BATCH_SIZE",
    )

    # Override the print_help method to print our custom help text
    def custom_print_help(file=None):
        print(help_text, file=file)

    parser.print_help = custom_print_help

    return parser


# python -u main.py -data ../data/raw/dblp/toy.dblp.v12.json
# 						  ../data/raw/imdb/toy.title.basics.tsv
# 						  ../data/raw/uspt/toy.patent.tsv
#                         ../data/raw/gith/toy.data.csv
# 					-domain dblp imdb uspt gith
# 					-model random
# 					       fnn fnn_emb bnn bnn_emb nmt
# 					       tfnn tbnn tnmt tfnn_emb tbnn_emb tfnn_a1 tbnn_a1 tfnn_emb_a1 tbnn_emb_a1 tfnn_dt2v_emb tbnn_dt2v_emb
#                   -filter 1

# To run on compute canada servers you can use the following command: (time is in minutes)
# sbatch --account=def-hfani --mem=96000MB --time=2880 cc.sh


def main():
    overall_start_time = time()

    tprint("Starting processing pipeline")

    # parse the arguments
    parser = argparse.ArgumentParser(
        add_help=False, description="OpeNTF: Open Neural Team Formation"
    )
    parser = addargs(parser)
    args = parser.parse_args()

    if args.gpus is not None:
        set_gpus(args.gpus)
    else:
        tprint(f"No GPU index specified. Using CPU.")

    # Run the experiment
    run_start_time = time()
    run(
        data_list=[args.data],
        domain_list=[args.domain],
        fair=args.fair,
        filter=args.filter,
        future=args.future,
        model_list=[args.model] if args.model is not None else [],
        gpus=args.gpus,
        output=args.output,
        settings=settings,
    )
    run_end_time = time()

    # Calculate and display total execution time
    overall_end_time = time()
    run_duration = run_end_time - run_start_time
    total_duration = overall_end_time - overall_start_time

    tprint("=" * 80)
    tprint("Execution Summary:")
    tprint(f"  Processing time: {format_time(run_duration)} (HH:MM:SS)")
    tprint(f"  Total execution time: {format_time(total_duration)} (HH:MM:SS)")
    tprint("=" * 80)


if __name__ == "__main__":
    main()
