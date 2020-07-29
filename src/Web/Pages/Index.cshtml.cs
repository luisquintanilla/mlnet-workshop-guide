using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.AspNetCore.Mvc.TagHelpers;
using Microsoft.Extensions.Logging;
using Web.Models;
using Web.Services;
using Microsoft.Extensions.ML;
using Shared;

namespace Web.Pages
{
    public class IndexModel : PageModel
    {
        private readonly IWebHostEnvironment _env;
        private readonly ILogger<IndexModel> _logger;
        private readonly PredictionEnginePool<ModelInput, ModelOutput> _pricePredictionEnginePool;

        private readonly IEnumerable<CarModelDetails> _carModelService;
        private readonly IDamageDetectionService _damageDetectionService;

        public bool ShowPrice { get; private set; } = false;
        public bool ShowImage { get; private set; } = false;

        [BindProperty]
        public CarDetails CarInfo { get; set; }

        [BindProperty]
        public int CarModelDetailId { get; set; }

        [BindProperty]
        public IFormFile ImageUpload { get; set; }

        public SelectList CarYearSL { get; } = new SelectList(Enumerable.Range(1930, (DateTime.Today.Year - 1929)).Reverse());
        public SelectList CarMakeSL { get; }

        public IndexModel(IWebHostEnvironment env, ILogger<IndexModel> logger, ICarModelService carFileModelService, PredictionEnginePool<ModelInput, ModelOutput> pricePredictionEnginePool, IDamageDetectionService damageDetectionService)
        {
            _env = env;
            _logger = logger;
            _carModelService = carFileModelService.GetDetails();
            CarMakeSL = new SelectList(_carModelService, "Id", "Model", default, "Make");
            _pricePredictionEnginePool = pricePredictionEnginePool;
            _damageDetectionService = damageDetectionService;
        }

        public void OnGet()
        {
            _logger.LogInformation("Got page");
        }

        public async Task OnPostAsync()
        {
            var selectedMakeModel = _carModelService.Where(x => CarModelDetailId == x.Id).FirstOrDefault();

            CarInfo.Make = selectedMakeModel.Make;
            CarInfo.Model = selectedMakeModel.Model;

            ModelInput input = new ModelInput
            {
                Year = (float)CarInfo.Year,
                Mileage = (float)CarInfo.Mileage,
                Make = CarInfo.Make,
                Model = CarInfo.Model
            };

            ModelOutput prediction = _pricePredictionEnginePool.Predict(modelName: "PricePrediction", example: input);

            CarInfo.Price = prediction.Score;

            if (ImageUpload != null)
            {
                var boundingBoxes = await ProcessUploadedImageAsync(ImageUpload);
                var damageCost = _damageDetectionService.CalculateDamageTotalCost(boundingBoxes);
                CarInfo.Price = Math.Max(0, CarInfo.Price - damageCost);
                ShowImage = true;
            }
            ShowPrice = true;
        }

        private async Task<IEnumerable<BoundingBox>> ProcessUploadedImageAsync(IFormFile uploadedImage)
        {
            // Save uploaded image
            var fileName = Path.Combine(_env.ContentRootPath, "ImageTemp", $"{Guid.NewGuid().ToString()}.jpg");
            using (var fs = new FileStream(fileName, FileMode.Create))
            {
                //Copy image to memory stream
                await uploadedImage.CopyToAsync(fs);
            }

            // Create ONNX input
            var imageInput = new ONNXInput
            {
                ImagePath = fileName
            };

            // Detect damage
            var boundingBoxes = _damageDetectionService.DetectDamage(imageInput, 0.7f);
            CarInfo.Base64Image = _damageDetectionService.AnnotateBase64Image(fileName, boundingBoxes);

            return boundingBoxes;
        }
    }
}
