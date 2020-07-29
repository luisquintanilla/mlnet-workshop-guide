using System;
using Microsoft.ML;
using Shared;

namespace ONNXConsole
{
    class Program
    {
        // Update this with the location where you saved the model
        private static string ONNX_MODEL_FILEPATH = @"C:\Dev\mlnet-workshop\models\model.onnx";
        private static string PIPELINE_FILEPATH = @"C:\Dev\ONNXModel.zip";

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            var pipeline =
                mlContext.Transforms.LoadImages("Image", null, "ImagePath")
                .Append(mlContext.Transforms.ResizeImages("ResizedImage", 320, 320, "Image"))
                .Append(mlContext.Transforms.ExtractPixels("data", "ResizedImage"))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    outputColumnNames: new string[] { "detected_boxes", "detected_scores", "detected_classes" },
                    inputColumnNames: new string[] { "data" },
                    modelFile: ONNX_MODEL_FILEPATH));

            // Build the pipeline
            var emptyDV = mlContext.Data.LoadFromEnumerable(new ONNXInput[] { });
            var model = pipeline.Fit(emptyDV);

            // Save the pipeline
            Console.WriteLine("Saving pipeline...");
            mlContext.Model.Save(model, emptyDV.Schema, PIPELINE_FILEPATH);
        }
    }
}
