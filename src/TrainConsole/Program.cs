using System;
using Microsoft.ML;
using Shared;

namespace TrainConsole
{
    class Program
    {
        private static string TRAIN_DATA_FILEPATH = @"C:\Dev\true_car_listings.csv";

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // Load training data
            Console.WriteLine("Loading data...");
            IDataView trainingData = mlContext.Data.LoadFromTextFile<ModelInput>(path: TRAIN_DATA_FILEPATH, hasHeader: true, separatorChar: ',');

            // Split the data into a train and test set
            var trainTestSplit = mlContext.Data.TrainTestSplit(trainingData, testFraction: 0.2);

            // Create data transformation pipeline
            var dataProcessPipeline =
                mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "MakeEncoded", inputColumnName: "Make")
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ModelEncoded", inputColumnName: "Model"))
                    .Append(mlContext.Transforms.Concatenate("Features", "Year", "Mileage", "MakeEncoded", "ModelEncoded"))
                    .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                    .AppendCacheCheckpoint(mlContext);

            // Choose an algorithm and add to the pipeline
            var trainer = mlContext.Regression.Trainers.LbfgsPoissonRegression();
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Train the model
            Console.WriteLine("Training model...");
            var model = trainingPipeline.Fit(trainTestSplit.TrainSet);
        }
    }
}
