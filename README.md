# (UPDATE IN PROGRESS)
# `OpeNTF`: Team Recommendation via Translation Approach

This readme is specifically for the neural machine translation models, specically, the models supported by the OpenNMT-py framework. In this readme, you'd be able to setup and run the dataset using either the existing NMT models or a model of your own.

## 0. Workflow Overview
This repository utilizes the following workflow:

1. Create a new model by duplicating one of the three templates available in `src/mdl/nmt_models/` folder:
   1. `_transformer_template.yml`
   2. `_rnn_template.yml`
   3. `_cnn_template.yml`
3. Edit the model config file to adjust the hyperparameters and other settings you wish to modify such as data paths. Give your model a unique and meaningful name (ie. `my_transformer_model_1.yml`).
4. **[UP TO HERE]** To be continued...

## 1. Setup

For the most convenience, this setup is heavily reliant on usage of Docker. Therefore,

**Step 1.** Download and install Docker from here: [Get Docker | Docker Docs](https://docs.docker.com/get-started/get-docker/)

**Step 2.** Pull a ready-made image we've prepared from the Docker Hub. It's about 26 GB.

In a terminal, run the following command:

```
docker pull kmthang/opennmt:3.0.4-torch1.10.1
```

**Step 3.** Once the image is on your system, let's create a container from it and then run it with the following command:

###### Note: before running the following command, be in the following folder: `/OpeNTF`, so the volume mapping won't confuse you.

```
docker run -it --name <DESIRED_CONTAINER_NAME> --gpus all -v $(pwd):/OpeNTF kmthang/opennmt:3.0.4-torch1.10.1
```

Example:

```
docker run -it --name opennmt --gpus all -v $(pwd):/OpeNTF kmthang/opennmt:3.0.4-torch1.10.1
```

You're now inside the Docker container. Your terminal should look something like this:

```
root@759fe234ae0f:/OpeNTF#
```

Feel free to run the following two commands to see if the container has access to the host's dedicated GPUs.
Check 1: `root@759fe234ae0f:/OpeNTF# nvidia-smi`
Check 2: `root@759fe234ae0f:/OpeNTF# nvcc --version`

If both return proper results and not errors, you're good to run the models.

## 2. Quickstart

The design of how the NMT models are ran is heavily script-based, ran using `.sh` files.
The basic flow of creating a new model and running it on a dataset follows this flow:

> #### Why use scripts?
>
> Scripting allows unattending sequential test runs such as running the same dataset with several different models, or run several datasets sequentially with different models and as well as capture operation durations and easy repetition (no need to input long argument commands)

**Step 1.** Create a `.yaml` model config file with desired hyperparameters and other settings

**Step 2.** Create a `.sh` script file and mention the above model name

**Step 3.** Run the `.sh` script file to start the process

---

#### Step 1:

1. Create a new `.ymal` file in `/OpeNTF/src/mdl/nmt_models` folder and prefix the file with `nmt_`, like `nmt_MODEL_VARIANT.yaml`

   > example: `nmt_transformer_model1.yaml`

2. Edit your newly created `.yaml` file to adjust the hyperparameters and other settings you wish to modify such as data paths.
   > You can use the provided `nmt_transformer-template.yaml` for a quick test run

---

#### Step 2:

1. Create a new `.sh` file. You can copy the provided `_template.sh` file inside `/run_scripts` folder and rename the file name to something else.

   > i.e., `run20.sh`

2. Edit the `.sh` file you've just created. If you've copied from the `_template.sh` file, then you'll see that several things are already set up (such as the datasets). All you have to do is edit the code betwen the following two dividers:
   > If you had named your model file as `nmt_transformer-model1.yaml` and you're running on the dblp dataset, then the configuration would be as below
   > So far we have the script name as `run20.sh` and model name as `nmt_transformer-model1.yaml`

```
# ------------------------------------------------------------------------------
# CONFIGURATIONS
# ------------------------------------------------------------------------------

# Run number
run_num=20

# Array of variants, ie. variants=("dblp1" "dblp2" "dblp3")
variants=("model1")

# Select dataset (see available datasets above)
dataset=$dblp
dataset_path=$dblp_path
is_toy=false

# Select model (nmt_convs2s, nmt_rnn, or nmt_transformer)
model=$nmt_transformer

# ------------------------------------------------------------------------------
# END CONFIGURATIONS
# ------------------------------------------------------------------------------
```

#### Step 3:

1. Navigate to scripts folder if you aren't already there.

```
cd /OpeNTF/run_scripts
```

2. Run the script

```
nohup ./run20.sh > run20.log 2>&1 &
```

> The above command will detach this process from your current terminal and run in the background while all the normal and error output are stored in `run20.log` file in the same folder as `run20.sh` file (which is in `/OpeNTF/run_scripts` folder)

3. You'll see two `.log` files generated inside `/OpeNTF/run_logs` folder with the following names:

```
run20_dblp_nmt_transformer_model1.log
run20_dblp_nmt_transformer_model1_errors.log
```

> These two file output regular and error messages from the training, testing, validation processes.
