## Data splitting

train : val : test = 6 : 2 : 2 by userid

50% val ---> train.append(50% val)
    the rest is held out in val
50% test ---> train.append(50% test)
    the rest is held out in test

## Baseline Model
the result of baseline model for small dataset is in baseline_model_small.ipynb file

the result of baseline model for large dataset is in baseline_model_large.ipynb file

## Recommendation Model
we first run the jupyter notebook from above to save the splitted data into scratch folder, then put the files into hdfs for spark usage.

we run hyper tuning on the ALS model and found that for the when rank is 5 and regParam is 1, the evalution critierion is the best

for validation dataset, the evaluation metris results is:

MAP is 0.03924001680359271

precision at 100 is 0.03299999999999998

ndcg at 100 is 0.1531479011240107

for test dataset, the evaluation metris results is:

MAP is 0.04010300296956644

precision at 100 is 0.032224080267558526

ndcg at 100 is 0.15098518359546934