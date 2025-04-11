from dal.load_dblp_data import nn_t2v_dataset_generator
from dal.embedding import Embedding
import dal.load_dblp_data as dblp
from VAE import VAE
from eval.evaluation import Evaluation

import csv
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


class TeamFormationLayer:
    """A class used to control the data flow of the library
    Any other class instantiations or function calls to access the
    VAE architecture made by the user are done using the DataAccessLayer
    ----------
    database_name : string
        The user-provided dataset name that the VAE architecture
        will be applied on
    database_path : string
        The user-provided dataset path that the VAE architecture
        will be applied on
    embeddings_save_path : string, optional (default='output/Models/T2V/')
        The user-provided/default path where the embedded verions of
        the dataset will be stored
    """
    def __init__(self, database_name, database_path, embeddings_save_path):
        self.database_name = database_name
        self.database_path = database_path
        self.embeddings_save_path = embeddings_save_path
        self.embedding_model = Embedding(self.database_name, self.database_path)
        self.t2v_model = ''
        self.embedding_dim = 0
        self.seed = 7
        self.x_train = ''
        self.x_test = ''
        self.y_train = ''
        self.y_test = ''
        self.x_train_indices = ''
        self.x_test_indices = ''
        self.y_train_indices = ''
        self.y_test_indices = ''
        print("An instance of the DAL has been created.")
        print("")

    def get_database_name(self):
        return self.database_name

    def get_database_path(self):
        return self.database_path

    def embeddings_save_path(self):
        return self.embeddings_save_path

    def generate_embeddings(self):
        """Generates embeddings and saves it locally
        Creates skills and experts embeddings in order for the VAE to
        easily process the dataset
        """
        print("Beginning Step#1: Embeddings Generation")
        if self.embeddings_save_path == 'default':
            self.embedding_model = Embedding(self.database_name, self.database_path)
        else:
            self.embedding_model = Embedding(self.database_name, self.database_path, self.embeddings_save_path)
        if self.embedding_model.generate_embeddings():
            print("Embeddings generation is complete.")
        print("")

    def get_x_train_data(self):
        return self.x_train

    def get_x_test_data(self):
        return self.x_test

    def get_y_train_data(self):
        return self.y_train

    def get_y_test_data(self):
        return self.x_test

    # TODO (Step 2) - DONE:
    def generate_t2v_dataset(self):
        print("Beginning Step#2: T2V Dataset Generation")
        self.t2v_model = self.embedding_model.load_model("output/Models/dblp/model_d100_w2_m0_tSkill")
        self.embedding_dim = self.embedding_model.model.vector_size
        preprocessed_dataset = dblp.load_preprocessed_dataset(self.database_path)
        nn_t2v_dataset_generator(self.embedding_model, preprocessed_dataset, output_file_path='dataset/dblp_ae_t2v_dim{}_tSkill_dataset.pkl'.format(self.embedding_dim), mode='skill')
        print("T2V dataset generation is complete.")
        print("")

    # TODO (Step 3) - DONE:
    def train_test_split_data(self):
        """Generate train/test split data for the model
        Split the data points intro train and test sets and write its
        indices to a file
        """
        print("Beginning Step#3: Train/Test Data Split")
        with open('dataset/dblp_ae_t2v_dim100_tSkill_dataset.pkl', 'rb') as f:  # opens the t2v generated dataset from previous step
            data = pickle.load(f)
        x = np.array([item[1] for item in data])  # skill-set array
        y = np.array([(np.array(item[2])[0]) for item in data])  # predicted experts array
        x_indices = np.array([index for index, item in enumerate(data)])  # skill-set array indices
        y_indices = np.array([index for index, item in enumerate(data)])  # predicted experts array indices

        # split the data in train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.15, random_state=self.seed)
        # split the data in train and test set indices
        self.x_train_indices, self.x_test_indices, self.y_train_indices, self.y_test_indices = train_test_split(x_indices, y_indices, test_size=0.15, random_state=self.seed)

        train_indices = self.x_train_indices
        test_indices = self.x_test_indices
        #print(train_indices)
        #print(test_indices)

        # write train indices to local file
        # opening the csv file in 'w' mode
        file = open('output/train_test_indices/dblp_train_indices.csv', 'w+', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(["train_indices"])
            for index in train_indices:
                writer.writerow([index])
        file.close()

        # write test indices to local files
        # opening the csv file in 'w' mode
        file = open('output/train_test_indices/dblp_test_indices.csv', 'w+', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(["test_indices"])
            for index in test_indices:
                writer.writerow([index])
        file.close()

        print("Train/test data split is complete.")
        print("")

    # TODO (Step 4) - DONE:
    def generate_VAE(self):
        """Perform prediction through the Variational Autoencoder model
        Use the train/test split data to train the VAE model and predict
        the user/skill indices
        """
        print("Beginning Step#4: VAE Training/Testing")
        vae = VAE(self.x_train, self.x_test, self.y_train, self.y_test)  # apply the VAE to the train and test sets
        vae.vae_training()  # train the VAE model
        vae.vae_prediction()  # save the predictions
        print("VAE training/testing complete.")
        print("")

    # TODO (Step 5) - DONE:
    def evaluate_results(self, main_prediction_dataset, comparison_prediction_dataset, max_k=20, save_graphs=False):
        """Perform evaluation on predicted output
        Use measures such as Recall, MRR, MAP, and NDCG and correlations
        to evaluate performance of the model
        Parameters
        ----------
        main_prediction_dataset : string
            The prediction dataset that the evaluation measures would be
            performed upon
        comparison_prediction_dataset : string, optional (default=None)
            The prediction dataset that the correlation comparison will be
            performed with
        max_k : int
            The top k of predictions to perform evaluation on
        save_graphs : int, optional (default=False)
            Whether to save the evaluation plots or not
        """
        print("Beginning Step#5: Evaluation")

        # dataset_v1
        print("Dataset evaluation:")
        # apply the evaluation of the VAE model on the user dataset
        eval_1 = Evaluation(main_prediction_dataset)
        eval_1.split_predicted_true_indices()
        eval_1.print_metrics()
        #eval_1.metric_visualization(max_k, save_graphs)
        eval_1_predicted_indices = eval_1.get_predicted_indices()

        # only compute correlation if second dataset is provided
        #print(comparison_prediction_dataset)
        #if comparison_prediction_dataset is not None:
        #    # dataset_v2
        #    # apply the evaluation of another model on the user dataset
        #    eval_2 = Evaluation(comparison_prediction_dataset)
        #    eval_2.split_predicted_true_indices()
        #    eval_2_predicted_indices = eval_2.get_predicted_indices()
        #
        #    # compute correlation between the two models
        #    print("Correlation between 2 methods:")
        #    correlation = eval_1.correlation(eval_1_predicted_indices, eval_2_predicted_indices, 10)
        #    print("Correlation =", correlation)

        print("Evaluation complete.")
        print("")
