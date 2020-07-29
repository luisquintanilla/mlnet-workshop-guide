using Microsoft.ML.Data;

namespace AutoMLTrainConsole
{
    public class CarPrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice { get; set; }
    }
}
