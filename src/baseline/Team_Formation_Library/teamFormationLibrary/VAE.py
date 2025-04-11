from util.variational import *
import eval.evaluation as dblp_eval

from keras.losses import mse
from keras.layers import Lambda
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from tensorflow.python.framework.ops import disable_eager_execution
import time
import csv

disable_eager_execution()

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, min_delta=0.0001)

# running settings
dataset_name = 'DBLP'
method_name = 'S_VAE_O'

# eval settings
k_max = 100  # cut_off for eval
evaluation_k_set = np.arange(1, k_max+1, 1)

# nn settings
epochs = 2
back_propagation_batch_size = 64
training_batch_size = 6000
min_skill_size = 0
min_member_size = 0
latent_dim = 2


class VAE:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.input_dim = self.x_train.shape[1]
        # print(self.x_train)
        self.output_dim = self.y_train.shape[1]
        # print(self.y_train)
        print("Input/output Dimensions:  ", self.input_dim, self.output_dim)

        # this is our input placeholder
        # network parameters
        intermediate_dim_encoder = self.input_dim
        intermediate_dim_decoder = self.output_dim

        # VAE model = encoder + decoder
        # build encoder model

        inputs = Input(shape=(self.input_dim,), name='encoder_input')
        x = Dense(intermediate_dim_encoder, activation='relu')(inputs)
        self.z_mean = Dense(latent_dim, name='z_mean')(x)
        self.z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use re-parameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [self.z_mean, self.z_log_var, z], name='encoder')
        encoder.summary()
        # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim_decoder, activation='relu')(latent_inputs)
        outputs = Dense(self.output_dim, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        self.variational_autoencoder = Model(inputs, outputs, name='vae_mlp')

        models = (encoder, decoder)

        self.variational_autoencoder.compile(optimizer='adam', loss=self.vae_loss)
        self.variational_autoencoder.summary()


    def vae_loss(self, y_true, y_pred):
        """Compute loss of the VAE model
        Compute loss between the true indices and the predicted indices
        using a combination of reconstruction loss and Kullback-Leibler
        loss
        Parameters
        ----------
        y_true : array-like, shape=(true_indices,)
            The true indices
        y_pred : array-like, shape=(predicted_indices,)
            The predicted indices
        """
        reconstruction_loss = mse(y_true, y_pred)

        reconstruction_loss *= self.output_dim
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    def vae_training(self):
        """Train the VAE model
        Train the VAE model using the train set and validate the data
        using the test set
        """
        self.variational_autoencoder.fit(self.x_train, self.y_train,
                        epochs=epochs,
                        batch_size=back_propagation_batch_size,
                        callbacks=[es],
                        shuffle=True,
                        verbose=2,
                        validation_data=(self.x_test, self.y_test))
        # Cool down GPU
        # time.sleep(300)

    def vae_prediction(self):
        """Generate the predictions from the model
        Extract predictions from the test list and save the results
        to a CSV file
        """
        true_indices = []
        pred_indices = []

        result_output_name = "output/predictions/{}_output_dblp_mt30_ts3.csv".format(method_name)
        with open(result_output_name, 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ['Method Name', '# Predictions', '# Truth', 'Computation Time (ms)',
                 'Prediction Indices', 'True Indices'])
            for sample_x, sample_y in zip(self.x_test, self.y_test):
                start_time = time.time()
                # probability is in the following variable:
                sample_prediction = self.variational_autoencoder.predict(np.asmatrix(sample_x))
                end_time = time.time()
                elapsed_time = (end_time - start_time)*1000
                pred_index, true_index = dblp_eval.find_indices(sample_prediction, [sample_y])
                true_indices.append(true_index[0])
                pred_indices.append(pred_index[0])
                writer.writerow([method_name, len(pred_index[0][:k_max]), len(true_index[0]),
                                 elapsed_time] + pred_index[0][:k_max] + true_index[0])


# re-parameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Re-parameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
