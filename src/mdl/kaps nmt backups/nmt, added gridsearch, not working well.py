import logging
import yaml, pickle
import subprocess, os, re
import shlex
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from eval.metric import *
from mdl.ntf import Ntf
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer


class Nmt(Ntf):
    def __init__(self):
        super(Ntf, self).__init__()

    def prepare_data(self, vecs):
        input_data = []
        output_data = []
        for i, id in enumerate(vecs["id"]):
            skills = [
                f"s{str(skill_idx)}" for skill_idx in vecs["skill"][i].nonzero()[1]
            ]
            members = [
                f"m{str(member_idx)}" for member_idx in vecs["member"][i].nonzero()[1]
            ]
            if skills:
                input_data.append(skills)
            else:
                input_data.append(["empty_skill"])  # or handle appropriately
            if members:
                output_data.append(members)
            else:
                output_data.append(["empty_member"])  # or handle appropriately
        return input_data, output_data

    def build_vocab(self, input_data, output_data, splits, settings, model_path):
        endl = "\n"
        input_data = np.array(
            ["{}{}".format(_, endl) for _ in [" ".join(_) for _ in input_data] if _]
        )
        output_data = np.array(
            ["{}{}".format(_, endl) for _ in [" ".join(_) for _ in output_data] if _]
        )

        for foldidx in splits["folds"].keys():
            fold_path = f"{model_path}/fold{foldidx}"
            if not os.path.isdir(fold_path):
                os.makedirs(fold_path)

            with open(f"{fold_path}/src-train.txt", "w") as src_train:
                src_train.writelines(input_data[splits["folds"][foldidx]["train"]])
            with open(f"{fold_path}/src-valid.txt", "w") as src_val:
                src_val.writelines(input_data[splits["folds"][foldidx]["valid"]])
            with open(f"{fold_path}/tgt-train.txt", "w") as tgt_train:
                tgt_train.writelines(output_data[splits["folds"][foldidx]["train"]])
            with open(f"{fold_path}/tgt-valid.txt", "w") as tgt_val:
                tgt_val.writelines(output_data[splits["folds"][foldidx]["valid"]])

            settings["data"]["corpus_1"]["path_src"] = f"{fold_path}/src-train.txt"
            settings["data"]["corpus_1"]["path_tgt"] = f"{fold_path}/tgt-train.txt"
            settings["data"]["valid"]["path_src"] = f"{fold_path}/src-valid.txt"
            settings["data"]["valid"]["path_tgt"] = f"{fold_path}/tgt-valid.txt"
            settings["src_vocab"] = f"{fold_path}/vocab.src"
            settings["tgt_vocab"] = f"{fold_path}/vocab.tgt"
            settings["save_data"] = f"{fold_path}/"
            settings["save_model"] = f"{fold_path}/model"
            settings["world_size"] = 1
            settings["gpu_ranks"] = [0] if torch.cuda.is_available() else []
            settings["save_checkpoint_steps"] = 500

            with open(f"{fold_path}/config.yml", "w") as outfile:
                yaml.safe_dump(settings, outfile)

            cli_cmd = f"onmt_build_vocab -config {fold_path}/config.yml -n_sample {len(input_data)}"
            logging.info(f"Executing command: {cli_cmd}")
            try:
                subprocess.run(shlex.split(cli_cmd), check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error executing command: {e}")
                logging.error(f"Command output: {e.stdout}")
                logging.error(f"Command error: {e.stderr}")

        with open(f"{model_path}/src-test.txt", "w") as src_test:
            src_test.writelines(input_data[splits["test"]])
        with open(f"{model_path}/tgt-test.txt", "w") as tgt_test:
            tgt_test.writelines(output_data[splits["test"]])

        return model_path

    def learn(self, splits, indexes, vecs, settings, param, output):
        # Ensure 'train' key exists in settings
        if "train" not in settings:
            settings["train"] = {}

        # Define the hyperparameter grid
        param_grid = {
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": [4, 8, 16],
            "dropout": [0.3, 0.5, 0.7],
        }

        # Create a scorer function

        def scorer(true, pred):
            scores = []
            for t, p in zip(true, pred):
                if len(p) == 0:
                    scores.append(float("inf"))  # Penalize empty predictions
                else:
                    try:
                        # Ensure the true index is within the valid range for pred
                        if t < len(p):
                            scores.append(-np.mean(np.log([p[int(t)]])))
                        else:
                            logging.error(
                                f"IndexError: true={t}, pred={p} (true index out of range)"
                            )
                            scores.append(float("inf"))  # Penalize out-of-range indices
                    except IndexError:
                        logging.error(f"IndexError: true={t}, pred={p}")
                        scores.append(float("inf"))
            return np.mean(scores)

        # Iterate through the hyperparameter grid
        best_score = float("inf")
        best_params = None
        for params in ParameterGrid(param_grid):
            logging.info(f"Training with params: {params}")
            settings["train"]["learning_rate"] = params["learning_rate"]
            settings["train"]["batch_size"] = params["batch_size"]
            settings["train"]["dropout"] = params["dropout"]

            for foldidx in splits["folds"].keys():
                cli_cmd = f"onmt_train -config {output}/fold{foldidx}/config.yml"
                subprocess.Popen(shlex.split(cli_cmd)).wait()

                # Evaluate on validation set
                val_cmd = f"onmt_translate -model {output}/fold{foldidx}/model_step_{settings['train_steps']}.pt -src {output}/fold{foldidx}/src-valid.txt -output {output}/fold{foldidx}/pred.txt -gpu 0"
                subprocess.Popen(shlex.split(val_cmd)).wait()

                # Calculate the score
                with open(f"{output}/fold{foldidx}/pred.txt") as f:
                    preds = [line.strip() for line in f]
                preds = [
                    [float(y[1:]) for y in x.split()] for x in preds
                ]  # Extract numeric part

                # Debugging: print out types and sample data
                logging.info(
                    f"Type of splits['folds'][foldidx]['valid']: {type(splits['folds'][foldidx]['valid'])}"
                )
                logging.info(
                    f"Sample data from splits['folds'][foldidx]['valid']: {splits['folds'][foldidx]['valid'][:5]}"
                )

                # Ensure valid data is in list format
                if isinstance(splits["folds"][foldidx]["valid"], np.ndarray):
                    valid_data = splits["folds"][foldidx]["valid"].tolist()
                elif isinstance(splits["folds"][foldidx]["valid"], list):
                    valid_data = splits["folds"][foldidx]["valid"]
                else:
                    logging.error(
                        f"Unexpected type for splits['folds'][foldidx]['valid']: {type(splits['folds'][foldidx]['valid'])}"
                    )
                    continue

                true = [
                    int(y[1:]) if isinstance(y, str) and y.startswith("m") else int(y)
                    for y in valid_data
                ]

                score = scorer(true, preds)

                if score < best_score:
                    best_score = score
                    best_params = params

        logging.info(f"Best params: {best_params}, Best score: {best_score}")

    # todo: per_trainstep => per_epoch
    # todo: eval on prediction files

    def test(self, splits, path, per_epoch):
        for foldidx in splits["folds"].keys():
            fold_path = f"{path}/fold{foldidx}"
            if len(splits["folds"][foldidx]["train"]) == 0:
                raise ValueError(f"Training split for fold {foldidx} is empty.")
            with open(f"{fold_path}/config.yml") as infile:
                cfg = yaml.safe_load(infile)
            modelfiles = [f"{fold_path}/model_step_{cfg['train_steps']}.pt"]
            if per_epoch:
                modelfiles += [
                    f"{fold_path}/{_}"
                    for _ in os.listdir(fold_path)
                    if re.match(f"model_step_\\d+.pt", _)
                ]
            for model in modelfiles:
                epoch = model.split(".")[-2].split("/")[-1].split("_")[-1]
                pred_path = f"{fold_path}/test.fold{foldidx}.epoch{epoch}.pred.csv"
                cli_cmd = "onmt_translate "
                cli_cmd += f"-model {model} "
                cli_cmd += f"-src {path}/src-test.txt "
                cli_cmd += f"-output {pred_path} "
                cli_cmd += "-gpu 0 " if torch.cuda.is_available() else ""
                cli_cmd += "--min_length 2 "
                cli_cmd += "-verbose "
                print(f"Executing command: {cli_cmd}")

                try:
                    result = subprocess.run(
                        shlex.split(cli_cmd), check=True, capture_output=True, text=True
                    )
                    print(f"Command output: {result.stdout}")
                    print(f"Command error (if any): {result.stderr}")
                except subprocess.CalledProcessError as e:
                    print(f"Error executing command: {e}")
                    print(f"Command output: {e.stdout}")
                    print(f"Command error: {e.stderr}")

    def eval(self, splits, path, member_count, y_test, per_epoch):
        fold_mean = pd.DataFrame()
        test_size = y_test.shape[0]
        for foldidx in splits["folds"].keys():
            fold_path = f"{path}/fold{foldidx}"
            with open(f"{fold_path}/config.yml") as infile:
                cfg = yaml.safe_load(infile)
            modelfiles = [f"{fold_path}/model_step_{cfg['train_steps']}.pt"]

            # per_epoch depends on "save_checkpoint_steps" param in nmt_config.yaml since it simply collect the checkpoint files
            # so, per_epoch => per_checkpoints
            if per_epoch:
                modelfiles += [
                    f"{fold_path}/{_}"
                    for _ in os.listdir(fold_path)
                    if re.match(f"model_step_\\d+.pt", _)
                ]

            for model in modelfiles:
                epoch = model.split(".")[-2].split("/")[-1].split("_")[-1]
                pred_path = f"{fold_path}/test.fold{foldidx}.epoch{epoch}.pred.csv"
                pred_csv = pd.read_csv(f"{pred_path}", header=None)
                # tgt_csv = pd.read_csv(f'{path}/tgt-test.txt', header=None)

                Y_ = np.zeros((test_size, member_count))
                # Y = np.zeros((test_size, member_count))
                for i in range(test_size):
                    yhat_list = (
                        (pred_csv.iloc[i])[0]
                        .replace("m", "")
                        .replace("<unk>", "")
                        .split()
                    )
                    yhat_count = len(yhat_list)
                    if yhat_count != 0:
                        for pred in yhat_list:
                            Y_[i, int(pred)] = 1 / yhat_count

                    # y_list = (tgt_csv.iloc[i])[0].replace('m', '').split(' ')
                    # y_count = len(y_list)
                    # for tgt in y_list:
                    #     Y[i, int(tgt)] = 1
                df, df_mean, (fpr, tpr) = calculate_metrics(y_test, Y_, False)
                df_mean.to_csv(
                    f"{fold_path}/test.fold{foldidx}.epoch{epoch}.pred.eval.mean.csv"
                )
                with open(f"{path}/f{foldidx}.test.pred.eval.roc.pkl", "wb") as outfile:
                    pickle.dump((fpr, tpr), outfile)
                fold_mean = pd.concat([fold_mean, df_mean], axis=1)
        fold_mean.mean(axis=1).to_frame("mean").to_csv(
            f"{path}/test.epoch{epoch}.pred.eval.mean.csv"
        )

    def run(self, splits, vecs, indexes, output, settings, cmd, *args, **kwargs):
        with open(settings["base_config"]) as infile:
            base_config = yaml.safe_load(infile)

        encoder_type = base_config["encoder_type"]
        learning_rate = base_config["learning_rate"]
        word_vec_size = base_config["word_vec_size"]
        batch_size = base_config["batch_size"]
        epochs = base_config["train_steps"]
        team_count = vecs["skill"].shape[0]
        skill_count = vecs["skill"].shape[1]
        member_count = vecs["member"].shape[1]

        if encoder_type == "cnn":
            layer_size = base_config["cnn_size"]
        elif encoder_type == "rnn":
            layer_size = base_config["rnn_size"]
        elif encoder_type == "transformer":
            layer_size = base_config["transformer_ff"]

        model_path = f"{output}/t{team_count}.s{skill_count}.m{member_count}.et{encoder_type}.l{layer_size}.wv{word_vec_size}.lr{learning_rate}.b{batch_size}.e{epochs}"
        if not os.path.isdir(output):
            os.makedirs(output)

        y_test = vecs["member"][splits["test"]]

        if "train" in cmd:
            input_data, output_data = self.prepare_data(vecs)
            model_path = self.build_vocab(
                input_data, output_data, splits, base_config, model_path
            )
            self.learn(splits, indexes, vecs, base_config, None, model_path)
        if "test" in cmd:
            self.test(splits, model_path, per_epoch=True)
        if "eval" in cmd:
            self.eval(splits, model_path, member_count, y_test, per_epoch=True)
        if "plot" in cmd:
            self.plot_roc(model_path, splits, False)