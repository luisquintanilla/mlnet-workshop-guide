using System;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace AutoMLTrainConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<CarData>("./true_car_listings.csv", hasHeader: true, separatorChar: ',');

            var dropColumnsTransform = context.Transforms.DropColumns("Vin", "State", "City");

            var newData = dropColumnsTransform.Fit(data).Transform(data);

            var settings = new RegressionExperimentSettings
            {
                MaxExperimentTimeInSeconds = 60
            };

            var experiment = context.Auto().CreateRegressionExperiment(settings);

            var progress = new Progress<RunDetail<RegressionMetrics>>(p =>
            {
                if (p.ValidationMetrics != null)
                {
                    Console.WriteLine($"Current Run - {p.TrainerName}. R^2 -                                                                 {p.ValidationMetrics.RSquared}. MAE - {p.ValidationMetrics.MeanAbsoluteError}");
                }
            });

            var run = experiment.Execute(newData, labelColumnName: "Price", progressHandler: progress);

            var bestModel = run.BestRun.Model;

            var predictionEngine = context.Model.CreatePredictionEngine<CarData, CarPrediction>(bestModel);

            var carData = new CarData
            {
                Model = "FusionS",
                Make = "Ford",
                Mileage = 61515f,
                Year = 2012f
            };

            var prediction = predictionEngine.Predict(carData);

            Console.WriteLine($"Prediction - {prediction.PredictedPrice:C}");
        }
    }
}
