from TFL import TeamFormationLayer


def main_team_formation():
    print("---------------------------------------------------------")
    print("This library is a Team Formation tool that uses user database "
          "to predict the best teams to match a specific skill requirement.")
    print("NOTE: the database you provide to this library must be in a one-hot "
          "vector data frame format consisting of the following 3 parts:")
    print("1. ID")
    print("2. Skills")
    print("3. Experts")
    print("---------------------------------------------------------")
    # database_name = input("Please enter the name of your database: ")
    # database_path = input("Please provide the path of your database: ")
    # embeddings_save_path = input("Please enter the path you want to save the "
    #                              "embeddings (type 'default' to save it to a "
    #                              "default path: ")
    print(" ")
    '''
    while not os.path.isdir(embeddings_save_path):
    embeddings_save_path = input("This path does not exist. Please enter a "
                                 "valid path: ")
    '''

    # Create an instance of the TeamFormationLayer
    # TFL = TeamFormationLayer('dblp', 'dataset/dblp_preprocessed_dataset.pkl', 'default')
    TFL = TeamFormationLayer('dblp', 'dataset/radin_sparse_dblp_mt30_ts3.pkl', 'default')

    # 1 - Generate dictionaries and embedding files

    TFL.generate_embeddings()
    # 2 - Create vectors to associate ids, teams, and skills
    TFL.generate_t2v_dataset()
    # 3 - Split the dataset into train and test sets
    TFL.train_test_split_data()
    # 4 - Pass the data through the VAE
    TFL.generate_VAE()
    # 5 - Evaluate the results and compute correlation with another model
    TFL.evaluate_results("output/predictions/S_VAE_O_output_dblp_mt30_ts3.csv", "output/predictions/correlation_baseline_output10-3.csv", 10, True)


main_team_formation()
