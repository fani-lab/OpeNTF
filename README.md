# `OpeNTF`: Translative Neural Team Recommendation

This repository contains the code implementation for our research paper "Translative Neural Team Recommendation" (SIGIR 2025). The paper proposes a novel approach to team recommendation by treating it as a neural machine translation problem, where individual member skills and characteristics are translated into optimal team compositions. Our method leverages state-of-the-art sequence-to-sequence (seq2seq) neural machine translation (NMT) architectures such as the Transformer and RNN with attention and convolutional model, and the [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) framework to capture complex relationships between team members and project requirements, leading to more effective team formation recommendations.
<br/>
<br/>
> ![Overview of the sequence-2-sequence architecture.](./newflow_v3.jpg)
> Overview of the sequence-to-sequence architecture.

<br/>
<br/>


## Workflow Overview 

1. Set up environment
2. Run the models.
3. Collect the results.

<br/>

## 1. Set up environment

Three ways to run our pipeline:

- 1.1. The Docker approach (recommended)
- 1.2. The virtual environment approach (i.e., `venv` or conda `env`)
- 1.3. Without any of the above

<br/>

### 1.1. The Docker approach (recommended)

1.1.1. Download and install Docker from here: [Get Docker | Docker Docs](https://docs.docker.com/get-started/get-docker/)

1.1.2. Once your docker is up and running, pull our a ready-made image from the Docker Hub with this command. It's about 26 GB.
```
docker pull kmthang/opennmt
```
> Note: you may need to login first using `docker login` command.

<br/>

1.1.3. Once downloaded, create a container from it with:
```
docker run -it -d --name container_name --hostname host_name --gpus all -v $(pwd):/OpeNTF kmthang/opennmt
```
> Note: run the above command while in root repository which is the `/OpeNTF` folder. You can change the `container_name` and `host_name` to whatever you like.

<br/>

1.1.4. Connect to the container by this command:
```
docker attach container_name
```

You're now inside the Docker container. You can now run your bash script inside the `/OpeNTF/run_scripts` folder. See section for how to create a new bash script.

> Example
```
container_name@hostname:/OpeNTF# cd /run_scripts
container_name@hostname:/OpeNTF/run_scripts# ./nmt_model.sh
```

<br/>


### 1.2. The virtual environment approach (i.e., `venv` or conda `env`)

- 1.2.1. Python venv

    - 1.2.1.1. Create a new virtual environment with:
        ```
        python -m venv venv_name
        ```
        > Note: be in the `/OpeNTF` folder when running the above command. You can change the `venv_name` to whatever you like.

    - 1.2.1.2. Activate the virtual environment with:
        ```
        source venv_name/bin/activate
        ```

    <br />

- 1.2.2. Conda env

    - 1.2.2.1. Create a new conda environment with:
        ```
        conda create -n venv_name python=3.8
        ```

    <br />


- 1.2.3. Install the dependencies either one of the following:
    ```
    pip install -r requirements.txt
    ```
    or run the setup_nmt_env.sh script:
    ```
    ./setup_nmt_env.sh
    ```


<br />






## 2. Creating a new NMT model and bash script

2.1. Duplicate one of the three templates available in `src/mdl/nmt_models/` folder and prefix your model with `nmt_`:
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
