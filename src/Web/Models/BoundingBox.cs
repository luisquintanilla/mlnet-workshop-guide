using System.Drawing;

namespace Web.Models
{
    public class BoundingBox
    {
        public Dimensions BoxDimensions { get; set; }
        public float Confidence { get; set; }
        public long DamageCategory { get; set; }
    }

    public class Dimensions
    {
        public PointF P1 { get; set; }
        public PointF P2 { get; set; }
        public float Width { get { return P2.X - P1.X; } }
        public float Height { get { return P2.Y - P1.Y; } }
    }
}
