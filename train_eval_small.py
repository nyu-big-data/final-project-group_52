#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Final Project code for train ALS model using Spark.
Usage:
    $ spark-submit train_eval_small.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass
from re import T

# And pyspark.sql to get the spark session
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import collect_set, flatten

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    lines = spark.read.text(f'hdfs:/user/{netID}/train_small.txt').rdd
    parts = lines.map(lambda row: row.value.split(" "))
    trainRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                        rating=float(p[2]), timestamp=p[3]))
    train_small = spark.createDataFrame(trainRDD)

    lines = spark.read.text(f'hdfs:/user/{netID}/val_small.txt').rdd
    parts = lines.map(lambda row: row.value.split(" "))
    valRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                        rating=float(p[2]), timestamp=p[3]))
    val_small = spark.createDataFrame(valRDD)

    lines = spark.read.text(f'hdfs:/user/{netID}/test_small.txt').rdd
    parts = lines.map(lambda row: row.value.split(" "))
    testRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                        rating=float(p[2]), timestamp=p[3]))
    test_small = spark.createDataFrame(testRDD)

    best_rank = 0
    best_regParam = 0
    best_MAP = 0

    for rank in [5]:
        for regParam in [0.05, 0.1, 0.5]:
            print(f"rank={rank}, regParam={regParam}")

            als = ALS(rank=rank, maxIter=5, regParam=regParam, userCol="userId", itemCol="movieId", ratingCol="rating",
                        nonnegative=True, implicitPrefs=True, coldStartStrategy="drop", seed=42)
            model = als.fit(train_small)
            # model.save("als_model_small")

            # recommend 100 items for all users
            predictions = model.recommendForAllUsers(100)
            pred_movie_ids = predictions.groupby("userId").agg(flatten(collect_set("recommendations.movieId")).alias("pred_movie_ids"))
            pred_movie_ids.createOrReplaceTempView("pred_movie_ids")

            # test predictions on validation dataset
            actual_movie_id = val_small.groupby("userId").agg(collect_set("movieId").alias('actual_movie_id'))
            actual_movie_id.createOrReplaceTempView("actual_movie_id")
            total = spark.sql("SELECT userId, actual_movie_id, pred_movie_ids FROM actual_movie_id JOIN pred_movie_ids USING (userId)")
            data = total.selectExpr("pred_movie_ids", "actual_movie_id")
            rdd = data.rdd.map(tuple)
            metrics = RankingMetrics(rdd)

            print("For validation dataset")
            print(f"MAP is {metrics.meanAveragePrecision}")
            print(f"precision at 100 is {metrics.precisionAt(100)}")
            print(f"ndcg at 100 is {metrics.ndcgAt(100)}")
            if metrics.meanAveragePrecision > best_MAP:
                best_MAP = metrics.meanAveragePrecision
                best_rank = rank
                best_regParam = regParam

            # test predictions on test dataset
            actual_movie_id = test_small.groupby("userId").agg(collect_set("movieId").alias('actual_movie_id'))
            actual_movie_id.createOrReplaceTempView("actual_movie_id")
            total = spark.sql("SELECT userId, actual_movie_id, pred_movie_ids FROM actual_movie_id JOIN pred_movie_ids USING (userId)")
            data = total.selectExpr("pred_movie_ids", "actual_movie_id")
            rdd = data.rdd.map(tuple)
            metrics = RankingMetrics(rdd)

            print("For test dataset")
            print(f"MAP is {metrics.meanAveragePrecision}")
            print(f"precision at 100 is {metrics.precisionAt(100)}")
            print(f"ndcg at 100 is {metrics.ndcgAt(100)}")
    
    print(f"best rank is {best_rank}")
    print(f"best regParam is {best_regParam}")


    spark.stop()


# Only enter this block if we're in main
if __name__ == "__main__":

    config = pyspark.SparkConf().setAll([('spark.executor.memory', '32g'),
                                        ('spark.driver.memory', '32g'),
                                        ('spark.blacklist.enabled', False)])
    
    # Create the spark session object
    spark = SparkSession.builder.appName('final-project-group52').config(conf=config).getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)