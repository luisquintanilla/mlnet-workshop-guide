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
        }
    }
}
