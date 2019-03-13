from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from pyod.utils.data import generate_data, get_outliers_inliers
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer

def train_dt_model(ml_df):# decision tree with pipeline

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(ml_df)
    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(ml_df)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = ml_df.randomSplit([0.7, 0.3])

    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model.  This also runs the indexers.
    dt_model = pipeline.fit(trainingData)
    return dt_model

def get_outliers(train_features, train_labels):
    # outlier detection
    x_outliers, x_inliers = get_outliers_inliers(train_features, train_labels)
    for i in x_outliers[0:10]:
        print("src_ip: {srcip}, src_port: {srcport}, dst_ip: {dstip}, dst_port: {dstport}".format(srcip=long2ip(i[0]), srcport=i[1], dstip=long2ip(i[2]), dstport=i[3]))

# kmeans clustering unsupervised
def get_kmeans_model(train_features_df):
    kmeans = KMeans(k=10, seed=1)
    kmeans_model = kmeans.fit(train_features_df.select('features'))

    return kmeans_model

# random forest classifier
def get_rf_model(train_features_df):

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(train_features_df)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(train_features_df)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = train_features_df.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    rf_model = pipeline.fit(trainingData)
    return rf_model

