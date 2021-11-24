# Evaluation Folder
This folder contains various classes and functions to evaluate the 
performance of the model through metrics including: Recall, MRR, MAP,
and NDCG. It can also compute correlations with other models. 
Furthermore, diagrams can also be generated to display metric 
performance.
 
Example of usage:
```python
r_at_k() #to compute recall
mean_reciprocal_rank() #to compute MRR
print_metrics() #to print the above metrics along with others
metric_visualization() #generate metric performance diagrams
correlation() #compute the correlation between two models
save_metric_visualization() #save diagrams to a local location
```
