using Microsoft.ML.Data;

namespace AutoMLTrainConsole
{
    public class CarData
    {
        [LoadColumn(0)]
        public float Price { get; set; }
        [LoadColumn(1)]
        public float Year { get; set; }
        [LoadColumn(2)]
        public float Mileage { get; set; }
        [LoadColumn(3)]
        public string City { get; set; }
        [LoadColumn(4)]
        public string State { get; set; }
        [LoadColumn(5)]
        public string Vin { get; set; }
        [LoadColumn(6)]
        public string Make { get; set; }
        [LoadColumn(7)]
        public string Model { get; set; }
    }
}
