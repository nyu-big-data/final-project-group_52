## Data splitting


## Baseline Model
the result of baseline model for small dataset is in baseline_model_small.ipynb file

the result of baseline model for large dataset is in baseline_model_large.ipynb file

## Recommendation Model

we run hyper tuning on the ALS model and found that for the when rank is 5 and regParam is 1, the evalution critierion is the best

for validation dataset, the evaluation metris results is:

MAP is 0.03924001680359271
precision at 100 is 0.03299999999999998
ndcg at 100 is 0.1531479011240107

for test dataset, the evaluation metris results is:
MAP is 0.04010300296956644
precision at 100 is 0.032224080267558526
ndcg at 100 is 0.15098518359546934