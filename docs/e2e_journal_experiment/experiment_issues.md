Experiments on different [variations of fnn](https://docs.google.com/spreadsheets/d/1jt4Pvdz58qs0LyAnSjYr0MTpzy5ZfNonw5bBtfbIvKY/edit?gid=212563191#gid=212563191) with mhot vector as an input:
**training phase only**
- We initially ran 3 models on cuda:1 and 3 models on cuda:3, resulting in a training time of ~3 hours per epoch.
- Running 6 models in parallel led to CPU usage [exceeding 97%](https://github.com/mahdis-saeedi/OpeNTF/blob/main/docs/e2e_journal_experiment/cpu%26gpu_usage/cpu_6models_2gpus.txt) across all 224 cores.
- We terminated 3 runs on cuda:3, which reduced CPU usage to [~6%](https://github.com/mahdis-saeedi/OpeNTF/blob/main/docs/e2e_journal_experiment/cpu%26gpu_usage/cpu_3models_1gpu_not_stable.txt) per core, but it was not stable and increased to more than 95% again.
- We then launched a new run on cuda:3, increasing the batch size from 1,000 to 10,000, and started monitoring GPU and CPU utilization.
As a result:
GPU memory usage increased [5×.](https://github.com/mahdis-saeedi/OpeNTF/blob/main/docs/e2e_journal_experiment/cpu%26gpu_usage/gpu_memory.txt)
CPU usage again rose above 97% across all cores.
- We ran a test with 1 epoch and batch size = 50,000 on coda:3 ( it was the only run on coda:3. coda:1 was involved in running one model), while skipping the merging step of skills, which resulted in a memory error. Then reducing the batch size to 25,000 allowed the model to train successfully in approximately 4 minutes.
- When training two models in parallel on different GPUs, the average training time per epoch is about 10 minutes for both models (batch size 1000 and 25,000)
- Training a single model with batch size = 1,000 resulted in an average epoch time of ~5 minutes.
Note: We have not yet tested training a single model with batch size = 25,000!


