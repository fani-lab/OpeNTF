import json
import matplotlib.pyplot as plt
import os 

datasets = []
datasets += ['../output/dblp.v12.json.filtered.mt75.ts3']
datasets += ['../output/title.basics.tsv.filtered.mt75.ts3']
datasets += ['../output/patent.tsv.filtered.mt75.ts3']
models = ['bnn', 'bnn_emb']
colors = ['blue', 'orange', 'deeppink', 'maroon', 'royalblue', 'darkviolet', 'aqua', 'darkorange', 'chocolate', 'black', 'rosybrown', 'orange', 'blue', 'deeppink', 'maroon', 'royalblue', 'darkviolet', 'chocolate', 'darkorange', 'maroon', 'aqua', 'black']

def plot_train_val_loss(datasets, models):
    for dataset in datasets:
        for model in models:
            files = []
            dirs = f'{dataset}/{model}/'
            for dirpath, dirnames, filenames in os.walk(dirs):
                files += [os.path.join(os.path.normpath(dirpath), file).split(os.sep) for file in filenames if file.endswith(".json")]
            files = [f for f in files if "l[128].lr0.1.b128.e20.nns3.nsunigram_b." in f[4]]
            i = 0
            plt.figure(figsize=(4,4)) 
            for file in files:
                train_loss, valid_loss = [], []
                with open(f"../output/{file[2]}/{file[3]}/{file[4]}/train_valid_loss.json", 'r') as infile:
                    train_valid_loss = json.load(infile)
                for foldidx in train_valid_loss.keys():
                    train_loss.append(train_valid_loss[foldidx]['train'])
                    valid_loss.append(train_valid_loss[foldidx]['valid'])
                
                train_l = [sum(sub_list) / len(sub_list) for sub_list in zip(*train_loss)]
                valid_l = [sum(sub_list) / len(sub_list) for sub_list in zip(*valid_loss)]
                i += 1
                plt.plot(train_l, label=f"Train - b{file[4].split('.')[-1]}", color=colors[int(file[4].split('.')[-1].replace('s',''))], linewidth=2)
                plt.plot(valid_l, label=f"Valid - b{file[4].split('.')[-1]}", color=colors[int(file[4].split('.')[-1].replace('s',''))], linestyle=':', linewidth=2)
            plt.legend(loc='upper right')
            plt.title(f"Train/Val loss of {file[3]} on {str(str(dataset.split('.')[2]).split('/')[-1])}")
            plt.savefig(f'../output/{file[2]}/{file[3]}/l[128].lr0.1.b128.e20.nns3.nsunigram_b.train_valid_loss.pdf', dpi=100, bbox_inches='tight')
            plt.show()

# plot_train_val_loss(datasets, models)

def subplot_train_val_loss(datasets, models):
    p = 0
    fig, axs = plt.subplots(len(datasets), len(models), figsize=(12,12))
    axs = axs.flatten()
    for dataset in datasets:
        for model in models:
            i = 0
            files = []
            dirs = f'{dataset}/{model}/'
            for dirpath, dirnames, filenames in os.walk(dirs):
                files += [os.path.join(os.path.normpath(dirpath), file).split(os.sep) for file in filenames if file.endswith(".json")]
            files = [f for f in files if "l[128].lr0.1.b128.e20.nns3.nsunigram_b." in f[4]]  
            
            for file in files:
                train_loss, valid_loss = [], []
                with open(f"../output/{file[2]}/{file[3]}/{file[4]}/train_valid_loss.json", 'r') as infile:
                    train_valid_loss = json.load(infile)
                for foldidx in train_valid_loss.keys():
                    train_loss.append(train_valid_loss[foldidx]['train'])
                    valid_loss.append(train_valid_loss[foldidx]['valid'])
                
                train_l = [sum(sub_list) / len(sub_list) for sub_list in zip(*train_loss)]
                valid_l = [sum(sub_list) / len(sub_list) for sub_list in zip(*valid_loss)]
                
                axs[p].plot(train_l, label=f"Train - b{file[4].split('.')[-1]}", color=colors[int(file[4].split('.')[-1].replace('s',''))], linewidth=2)
                axs[p].plot(valid_l, label=f"Valid - b{file[4].split('.')[-1]}", color=colors[int(file[4].split('.')[-1].replace('s',''))], linestyle=':', linewidth=2)
                i += 1  
            axs[p].legend(loc='upper right')
            axs[p].title.set_text(f"Train/Val loss of {file[3]} on {str(str(dataset.split('.')[2]).split('/')[-1])}")
            p += 1
    plt.savefig(f'../output/l[128].lr0.1.b128.e20.nns3.nsunigram_b.train_valid_loss.pdf', dpi=100, bbox_inches='tight')
    plt.show()

subplot_train_val_loss(datasets, models)