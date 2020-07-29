using Microsoft.ML.Data;

namespace Shared
{
    public class ImageModelInput
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }

        [LoadColumn(1)]
        public string DamageClass { get; set; }

        [LoadColumn(2)]
        public string Subset { get; set; }
    }
}
