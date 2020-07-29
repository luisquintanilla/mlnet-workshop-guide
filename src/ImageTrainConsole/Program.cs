using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Vision;
using Shared;

namespace ImageTrainConsole
{
    class Program
    {
        // update this with your file's path where you saved it
        private static string TRAIN_DATA_FILEPATH = @"C:\Dev\mlnet-workshop\data\preprocessed\index.csv";
        private static string IMAGE_DATA_DIRECTORY = @"C:\Dev\mlnet-workshop\data\preprocessed\image";
        private static string MODEL_FILEPATH = @"C:\Dev\ImageModel.zip";

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // Load data in index.csv file
            var imageData = LoadImageData(TRAIN_DATA_FILEPATH, IMAGE_DATA_DIRECTORY);

            // Split data into training and validation sets
            var trainImages = imageData.Where(image => image.Subset == "T");
            var validationImages = imageData.Where(image => image.Subset == "V");

            // Load data into IDataViews
            var trainImagesDV = mlContext.Data.LoadFromEnumerable(trainImages);
            var validationImagesDV = mlContext.Data.LoadFromEnumerable(validationImages);

            var dataLoadPipeline = mlContext.Transforms.LoadRawImageBytes(outputColumnName: "ImageBytes", imageFolder: null, inputColumnName: "ImagePath");

            // Set the options for the image classification trainer
            var trainerOptions = new ImageClassificationTrainer.Options
            {
                FeatureColumnName = "ImageBytes",
                LabelColumnName = "EncodedLabel",
                WorkspacePath = "workspace",
                Arch = ImageClassificationTrainer.Architecture.InceptionV3,
                ReuseTrainSetBottleneckCachedValues = true,
                MetricsCallback = (metrics) => Console.WriteLine(metrics.ToString())
            };

            // Define training pipeline
            var trainingPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "EncodedLabel", inputColumnName: "DamageClass")
                .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(trainerOptions))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));

            var trainer = dataLoadPipeline.Append(trainingPipeline);

            // Train the model
            var model = trainer.Fit(trainImagesDV);

            // Use the model to make predictions on the validation images
            var predictionsDV = model.Transform(validationImagesDV);

            // Evaluate the model
            var evaluationMetrics = mlContext.MulticlassClassification.Evaluate(predictionsDV, labelColumnName: "EncodedLabel");
            Console.WriteLine($"Train Set Macro Accuracy: {evaluationMetrics.MacroAccuracy}");

            // Convert IDataView to IEnumerable
            var predictions = mlContext.Data.CreateEnumerable<ImageModelOutput>(predictionsDV, reuseRowObject: true);

            //Iterate over predictions and display actual vs predicted values
            foreach (var prediction in predictions)
            {
                var fileName = Path.GetFileName(prediction.ImagePath);
                Console.WriteLine($"Image: {fileName} | Actual: {prediction.DamageClass} | Predicted: {prediction.PredictedLabel}");
            }

            //Save the model
            mlContext.Model.Save(model, trainImagesDV.Schema, MODEL_FILEPATH);
            Console.WriteLine("Saved image classification model");
        }

        static IEnumerable<ImageModelInput> LoadImageData(string path, string imageFilePath, char separator = ',', bool hasHeader = true)
        {
            // Choose how many rows to skip
            var skipRows = hasHeader ? 1 : 0;

            // Local function to get full image path
            Func<string, string> getFilePath = imagePath => Path.Combine(imageFilePath, imagePath.Split('/')[1]);

            // Load file and create IEnumerable<ImageModelInput>
            var imageData =
                File.ReadAllLines(path)
                    .Skip(skipRows)
                    .Select(line =>
                    {
                        var columns = line.Split(separator);
                        return new ImageModelInput
                        {
                            ImagePath = getFilePath(columns[0]),
                            DamageClass = columns[1],
                            Subset = columns[2]
                        };
                    });

            return imageData;
        }
    }
}
