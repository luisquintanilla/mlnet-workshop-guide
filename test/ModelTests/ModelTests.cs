using FluentAssertions;
using Microsoft.ML;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Shared;

namespace ModelTests
{
    [TestClass]
    public class ModelTests
    {
        [TestMethod]
        public void Given_LowRangeCar_ShouldEstimatePriceWithinRange()
        {
            //Arrange
            var predictionEngine = GetPredictionEngine();

            var input = new ModelInput
            {
                Year = 2006,
                Mileage = 182248,
                Make = "Chevrolet",
                Model = "TrailBlazer4dr"
            };

            //Act
            var pricePrediction = predictionEngine.Predict(input).Score;

            //Assert
            pricePrediction.Should().BeInRange(2000, 6000);
        }

        [TestMethod]
        public void Given_MidRangeCar_ShouldEstimatePriceWithinRange()
        {
            //Arrange
            var predictionEngine = GetPredictionEngine();

            var input = new ModelInput
            {
                Year = 2013,
                Mileage = 38343,
                Make = "Acura",
                Model = "TSX5-Speed"
            };

            //Act
            var pricePrediction = predictionEngine.Predict(input).Score;

            //Assert
            pricePrediction.Should().BeInRange(13000, 18000);
        }

        [TestMethod]
        public void Given_HighRangeCar_ShouldEstimatePriceWithinRange()
        {
            //Arrange
            var predictionEngine = GetPredictionEngine();

            var input = new ModelInput
            {
                Year = 2016,
                Mileage = 20422,
                Make = "Lexus",
                Model = "GX"
            };

            //Act
            var pricePrediction = predictionEngine.Predict(input).Score;

            //Assert
            pricePrediction.Should().BeInRange(47000, 54000);
        }

        private PredictionEngine<ModelInput, ModelOutput> GetPredictionEngine()
        {
            var modelPath = MLConfiguration.GetModelPath();

            var mlContext = new MLContext();

            var model = mlContext.Model.Load(modelPath, out var schema);

            return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model, schema);
        }
    }
}
