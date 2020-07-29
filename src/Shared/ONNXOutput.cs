using Microsoft.ML.Data;

namespace Shared
{
    public class ONNXOutput
    {
        [ColumnName("detected_boxes")]
        public float[] DetectedBoxes { get; set; }

        [ColumnName("detected_scores")]
        public float[] DetectedScores { get; set; }

        [ColumnName("detected_classes")]
        public long[] DetectedClasses { get; set; }
    }
}
