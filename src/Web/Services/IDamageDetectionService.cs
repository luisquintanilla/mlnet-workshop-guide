using System.Collections.Generic;
using Web.Models;
using Shared;

namespace Web.Services
{
    public interface IDamageDetectionService
    {
        IEnumerable<BoundingBox> DetectDamage(ONNXInput input, float threshold);
        string AnnotateBase64Image(string imagePath, IEnumerable<BoundingBox> boundingBoxes);
        float CalculateDamageTotalCost(IEnumerable<BoundingBox> boundingBoxes);
    }
}
