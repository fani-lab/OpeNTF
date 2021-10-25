import eval.ranking as ranking

import numpy as np, matplotlib.pyplot as pl
import csv


class Evaluation:
    def __init__(self, results_path):
        self.results_path = results_path
        self.predicted_indices = []
        self.true_indices = []
        self.k = 10

    def get_database_name(self):
        """Returns the database name provided by the user
        """
        return self.results_path

    def split_predicted_true_indices(self):
        """Read the predicted and true indices
        Open predicted CSV files to read in predicted and true
        indices into lists
        """
        with open(self.results_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for count, row in enumerate(reader):
                if count != 0:
                    predicted_indices_start = 4
                    predicted_indices_end = predicted_indices_start + int(row[1])
                    true_indices_start = predicted_indices_end
                    true_indices_end = true_indices_start + int(row[2])
                    self.predicted_indices.append(row[predicted_indices_start:predicted_indices_end])
                    self.true_indices.append(row[true_indices_start:true_indices_end])
        # print(self.predicted_indices)
        # print(self.true_indices)

    def get_predicted_indices(self):
        """Returns the predicted indices determined by the VAE
        """
        return self.predicted_indices

    def r_at_k(self):
        """Compute Recall
        """
        all_recall = []
        for pred_indices, t_indices in zip(self.predicted_indices, self.true_indices):
            recall = 0
            for t_index in t_indices:
                if t_index in pred_indices[:self.k]:
                    recall += 1
            all_recall.append(recall / len(t_indices))
        return np.mean(all_recall), all_recall

    def cal_relevance_score(self):
        """Compute CAL Relevance Score
        """
        rs = []
        for p, t in zip(self.predicted_indices, self.true_indices):
            r = []
            for p_record in p[:self.k]:
                if p_record in t:
                    r.append(1)
                else:
                    r.append(0)
            rs.append(r)
        return rs

    def mean_reciprocal_rank(self, rs):
        """Compute Mean Reciprocal Rank
        """
        rs = (np.asarray(r).nonzero()[0] for r in rs)
        return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

    def print_metrics(self):
        """Print all the metrics for evaluation
        """
        # Recall:
        Recall = format(self.r_at_k()[0] * 100, '.2f')
        print("Recall =", str(Recall) + "%")
        # MRR:
        rs = self.cal_relevance_score()
        MRR = format(self.mean_reciprocal_rank(rs) * 100, '.2f')
        print("MRR =", str(MRR) + "%")
        # MAP:
        MAP = format(ranking.mean_average_precision(self.predicted_indices, self.true_indices) * 100, '.2f')
        print("MAP =", str(MAP) + "%")
        # NDCG:
        NDCG = format(ranking.ndcg_at(self.predicted_indices, self.true_indices, self.k) * 100, '.2f')
        print("NDCG =", str(NDCG) + "%")
        print("")

    def metric_visualization(self, max_k, save_graphs):
        """Generate metric visualizations
        Generates visualizations for various different metric
        measures including recall, mrr, map, and ndcg
        Parameters
        ----------
        max_k : integer
            The upper limit on the top-k for the evaluation
        save_graphs : boolean
            Whether to save the graphs or not
        """
        x = np.arange(0, max_k + 1, max_k/min(max_k, 10), dtype=int)[1:]
        # print(x)
        recall = []
        mrr = []
        map = []
        ndcg = []

        fig, axs = pl.subplots(2, 2)

        # Recall plot:
        for value in x:
            self.k = value
            recall.append(self.r_at_k()[0])
        # print(recall)
        axs[0, 0].set_ylim([min(recall) * 0.85, max(recall) * 1.05])
        axs[0, 0].scatter(x, recall, c='red')
        axs[0, 0].plot(x, recall)
        axs[0, 0].grid()
        axs[0, 0].set(xlabel="top-k", ylabel="recall")

        # MRR plot:
        for value in x:
            self.k = value
            rs = self.cal_relevance_score()
            mrr.append(self.mean_reciprocal_rank(rs))
        # print(mrr)
        axs[0, 1].set_ylim([min(mrr) * 0.85, max(mrr) * 1.05])
        axs[0, 1].scatter(x, mrr, c='red')
        axs[0, 1].plot(x, mrr)
        axs[0, 1].grid()
        axs[0, 1].set(xlabel="top-k", ylabel="mrr")

        # MAP plot:
        for value in x:
            self.k = value
            # print([item[:self.k] for item in self.predicted_indices])
            map.append(ranking.mean_average_precision([item[:self.k] for item in self.predicted_indices], self.true_indices))
        # print(map)
        axs[1, 0].set_ylim([min(map) * 0.85, max(map) * 1.05])
        axs[1, 0].scatter(x, map, c='red')
        axs[1, 0].plot(x, map)
        axs[1, 0].grid()
        axs[1, 0].set(xlabel="top-k", ylabel="map")

        # NDCG plot:
        for value in x:
            self.k = value
            if self.k == 0:
                ndcg.append(0)
            else:
                ndcg.append(ranking.ndcg_at(self.predicted_indices, self.true_indices, self.k))
        # print(ndcg)
        axs[1, 1].set_ylim([min(ndcg) * 0.85, max(ndcg) * 1.05])
        axs[1, 1].scatter(x, ndcg, c='red')
        axs[1, 1].plot(x, ndcg)
        axs[1, 1].grid()
        axs[1, 1].set(xlabel="top-k", ylabel="ndcg")

        # Show plot
        pl.tight_layout()
        metrics_fig = pl.gcf()
        pl.show()
        pl.draw()

        # save graphs only if save_graphs is set to True
        if save_graphs:
            self.save_metric_visualization(metrics_fig)

    def correlation(self, predicted_indices_1, predicted_indices_2, k):
        """Compute correlation with another model
        Parameters
        ----------
        predicted_indices_1 : array-like, shape=(predicted_indices,)
            Predictions made by the VAE model
        predicted_indices_2 : array-like, shape=(predicted_indices,)
            Predictions made by another model (i.e. Sapienza)
        """
        top_k_predicted_indices_1 = [item[:k] for item in predicted_indices_1]
        top_k_predicted_indices_2 = [item[:k] for item in predicted_indices_2]
        # handle error when sizes are not the same
        if len(top_k_predicted_indices_1) != len(top_k_predicted_indices_2):
            print("This correlation cannot be computed. The number of rows in each file must be the same.")
            return
        else:
            num_of_prediction = len(top_k_predicted_indices_1)
            num_of_common_authors = []
            for x in range(num_of_prediction):
                num_of_common_authors.append(len(set(top_k_predicted_indices_1[x]).intersection(set(top_k_predicted_indices_2[x])))/k)
            correlation_value = str(format(np.mean(num_of_common_authors) * 100, '.2f')) + "%"
        return correlation_value

    def save_metric_visualization(self, metrics_fig):
        """Save evaluation metric visualizations to local location
        Parameters
        ----------
        metrics_fig : matplotlib object
            The plot to be saved locally
        """
        # "output/diagrams/test.png"
        figure_save_location = input("Enter the location to save the metric figure (type 'default' "
                                     "to save at the default location): ")
        if figure_save_location == 'default':
            metrics_fig.savefig("output/diagrams/metric_fig.png")
        else:
            metrics_fig.savefig(figure_save_location + "/metric_fig.png")
        print("Metric figure saved.")
        print(" ")


def find_indices(prediction, true, min_true=1):
    """Extracts the prediction and training indices
    ----------
    prediction : array-like, shape=(predicted_indices,)
        Extracts the predicted indices
    true : array-like, shape=(true_indices,)
        Extracts the true indices
    min_true : integer, optional (default=1)
        The minimum true valeus to be extracted
    """
    preds = []
    trues = []
    for pred, t in zip(prediction, true):
        t = np.asarray(t)
        pred = np.asarray(pred)
        t_indices = np.argwhere(t)
        if t_indices.__len__() == 0:
            continue
        pred_indices = pred.argsort()[:][::-1]  # sorting checkup
        pred_indices = list(pred_indices)
        pred_indices = [i for i in pred_indices if i in np.argwhere(pred)]
        if len(pred_indices) == 0:
            pred_indices.append(-1)
        if len(t_indices) >= min_true:
            preds.append(pred_indices)
            trues.append([int(t) for t in t_indices])
    return preds, trues


# ----------------------------------------------------------------------------------------------------------------------
# Assertion tests:
eval_assert = Evaluation("output/predictions/S_VAE_O_output.csv")

# r_at_k assertion tests:
eval_assert.predicted_indices = [[0.3, 0.1, 0, 0.5, 0, 0, 0.2]]
eval_assert.true_indices = [[1, 1, 0, 0, 0, 0, 1]]
# test #1
eval_assert.k = 1
assert(eval_assert.r_at_k() == (0.0, [0.0]))
# test #2
eval_assert.k = 2
assert(eval_assert.r_at_k() == (0.0, [0.0]))
# test #3
eval_assert.k = 3
assert(eval_assert.r_at_k() == (0.5714285714285714, [0.5714285714285714]))
# test #4
eval_assert.k = 4
assert(eval_assert.r_at_k() == (0.5714285714285714, [0.5714285714285714]))

# cal_relevance_score assertion tests:
eval_assert.predicted_indices = [[0.3, 0.1, 0, 0.5]]
eval_assert.true_indices = [[0.1, 0.5, 0]]
# test #1
assert(eval_assert.cal_relevance_score() == [[0, 1, 1, 1]])
# test #2
eval_assert.predicted_indices = [[0.3, 0.1, 0, 0.5, 0, 0, 0.2]]
eval_assert.true_indices = [[1, 1, 0, 0, 0, 0, 1]]
assert(eval_assert.cal_relevance_score() == [[0, 0, 1, 0]])
# ----------------------------------------------------------------------------------------------------------------------
