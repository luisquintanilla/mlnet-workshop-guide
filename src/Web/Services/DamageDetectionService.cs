using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using Microsoft.Extensions.ML;
using Web.Models;
using Shared;

namespace Web.Services
{
    public class DamageDetectionService : IDamageDetectionService
    {
        private readonly PredictionEnginePool<ONNXInput, ONNXOutput> _damagePredictionEnginePool;

        public DamageDetectionService(PredictionEnginePool<ONNXInput, ONNXOutput> damagePredictionEnginePool)
        {
            _damagePredictionEnginePool = damagePredictionEnginePool;
        }

        public IEnumerable<BoundingBox> DetectDamage(ONNXInput input, float threshold)
        {
            // Use prediction engine to make predictions
            ONNXOutput modelOutput = _damagePredictionEnginePool.Predict(modelName: "DamageDetection", example: input);

            // Map detected_boxes, detected_scores, detected_classes to BoundingBox objects
            var boxes = modelOutput.DetectedBoxes;
            var boundingBoxes = new List<BoundingBox>();
            for (var i = 0; i < boxes.Length; i += 4)
            {
                var boxIdx = i / 4;
                var boundingBox = new BoundingBox
                {
                    BoxDimensions = new Dimensions { P1 = new PointF(boxes[i], boxes[i + 1]), P2 = new PointF(boxes[i + 2], boxes[i + 3]) },
                    Confidence = modelOutput.DetectedScores[boxIdx],
                    DamageCategory = modelOutput.DetectedClasses[boxIdx]
                };
                boundingBoxes.Add(boundingBox);
            }

            // Return bounding boxes that are at or above the threshold
            var topBoundingBoxes = boundingBoxes.Where(box => box.Confidence >= threshold).OrderByDescending(box => box.Confidence);
            return topBoundingBoxes;
        }

        public string AnnotateBase64Image(string imagePath, IEnumerable<BoundingBox> boundingBoxes)
        {
            using (var image = Image.FromFile(imagePath))
            {
                var originalImageWidth = image.Width;
                var originalImageHeight = image.Height;

                foreach (var box in boundingBoxes)
                {
                    var left = (int)Math.Max(box.BoxDimensions.P1.X * originalImageWidth, 0);
                    var top = (int)Math.Max(box.BoxDimensions.P1.Y * originalImageHeight, 0);
                    var width = (int)Math.Min(originalImageWidth - left, box.BoxDimensions.Width * originalImageWidth);
                    var height = (int)Math.Min(originalImageHeight - top, box.BoxDimensions.Height * originalImageHeight);

                    using (Graphics thumbnailGraphic = Graphics.FromImage(image))
                    {
                        thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                        thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                        thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

                        Pen pen = new Pen(Color.Red, 3.2f);

                        // Draw bounding box on image
                        thumbnailGraphic.DrawRectangle(pen, left, top, width, height);
                    }
                }

                using (var ms = new MemoryStream())
                {
                    image.Save(ms, ImageFormat.Jpeg);
                    var base64Image = Convert.ToBase64String(ms.ToArray());
                    return $"data:image/jpg;base64,{base64Image}";
                }
            }
        }

        public float CalculateDamageTotalCost(IEnumerable<BoundingBox> boundingBoxes)
        {
            float total = 0;

            if (boundingBoxes.Count() == 0) return total;

            foreach (var boundingBox in boundingBoxes)
            {
                switch (boundingBox.DamageCategory)
                {

                    case 0L: // Cracked windshield
                        total += 200;
                        break;
                    case 1L: // Dent
                        total += 100;
                        break;
                    default:
                        break;
                }
            }
            return total;
        }


    }
}
