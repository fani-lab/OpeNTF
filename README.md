# `OpeNTF`: Team Recommendation via Translation Approach

This readme is specifically for the neural machine translation models, specifically the models supported by the OpenNMT-py framework. In this readme, you'd be able to set up and run the dataset using either the existing NMT models or a model of your own.

## 0. Workflow Overview
This repository utilizes the following workflow:

1. Set up and run docker.
5. Create and run a new docker container from our image.
7. Create a new NMT model config file from a template.
8. Create a new bash script to automate the test from a template.
9. Run your bash script.
10. Collect the results in the `output/dataset_name` folder.

## 1. Setup docker

For the most convenience, this setup is heavily reliant on the usage of Docker. Therefore,

**### Step 1. ###** Download and install Docker from here: [Get Docker | Docker Docs](https://docs.docker.com/get-started/get-docker/)

**### Step 2. ###** Once your docker is installed, run it and then in a terminal, run the following command to download a ready-made image we've prepared from the Docker Hub. It's about 26 GB.:
```
docker pull kmthang/opennmt:3.0.4-torch1.10.1
```

**### Step 3. ###** Once the image is on your system, let's create a container from it and then run it with the following command:

###### Note: before running the following command, be in the following folder: `/OpeNTF`, so the volume mapping won't confuse you.

```
docker run -it -d --name nmt_container --hostname nmt_host --gpus all -v $(pwd):/OpeNTF kmthang/opennmt:3.0.4-torch1.10.1
```

You can change the `nmt_container` and `nmt_host` to whatever you want.

**### Step 4. ###** Connect to the container by running the following command:
```
docker attach nmt_container
```

You're now inside the Docker container. Your terminal should look something like this:

```
nmt_host@759fe234ae0f:/OpeNTF#
```

Feel free to run the following two commands to see if the container has access to the host's dedicated GPUs.
Check 1: `root@759fe234ae0f:/OpeNTF# nvidia-smi`
Check 2: `root@759fe234ae0f:/OpeNTF# nvcc --version`

You're good to run the models if both return proper results and no errors.

## 2. Create a new NMT model from a template

**### Step 1. ###** Duplicate one of the three templates available in `src/mdl/nmt_models/` folder and prefix your model with `nmt_`:
- `_template_transformer.yml`
- `_template_rnn.yml`
- `_template_cnn.yml`
> example: `nmt_mycnn_model.yaml`

**### Step 2. ###** Modify the hyperparameters and other settings you wish to modify, such as data paths, such as training steps, batch size, etc.



## 3. Create a new bash script to automate the test

**### Step 1. ###** Duplicate a bash script template in `/run_scripts` folder and rename the file name to something else:
- `_template.sh`
> example: `mycnn_model.sh`
>

**### Step 2. ###** In the bash script, edit the configurations to match your model name and dataset name. You only need to edit the three variables in the bash script file that's under the `# CONFIGURATIONS` section.

example:
```bash
# ------------------------------------------------------------------------------
# CONFIGURATIONS
# ------------------------------------------------------------------------------

models=("mode1" "model2")
datasets=("imdb dblp")
gpus="6,7"
```
In this example, this script will run two models (`mode1` and `model2`) on two datasets (`imdb` and `dblp`) using GPUs `6` and `7`. So `model1` will be run on `imdb` and `dblp` datasets and `model2` will also run on both same datasets. Note: If you don't have a dedicated GPU, your CPU will be used, and if you have only one, then it'll default to be used instead of the 6th and 7th, as in the example.

## 4. Run the bash script

**### Step 1. ###** Navigate to the scripts folder (`/run_scripts`) while inside the docker and run the bash script:

example:
```
nmt_host@759fe234ae0f:/OpeNTF# cd /run_scripts
nmt_host@759fe234ae0f:/OpeNTF/run_scripts# ./mycnn_model.sh
```

**Possible issues and fixes:**
- If the bash script doesn't run, check its permissions and ensure it's executable.
example:

```
nmt_host@759fe234ae0f:/OpeNTF/run_scripts# chmod +x mycnn_model.sh
```


**### Misc: ###** 
- process logs are stored in `/OpeNTF/run_logs` folder
- results are stored in `/OpeNTF/output` folder

## 5. Automating multiple models

**### Step 1. ###** As you may have noticed, use of multiple models, yes you can create multiple model files and mention them in the bash script as instructed in.
