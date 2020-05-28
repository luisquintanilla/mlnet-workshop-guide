using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks;
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
        private readonly ILogger<IndexModel> _logger;

        private readonly IEnumerable<CarModelDetails> _carModelService;

        private readonly PredictionEnginePool<ModelInput, ModelOutput> _predictionEnginePool;

        public bool ShowPrice { get; private set; } = false;

        [BindProperty]
        public CarDetails CarInfo { get; set; }

        [BindProperty]
        public int CarModelDetailId { get; set; }

        public SelectList CarYearSL { get; } = new  SelectList(Enumerable.Range(1930, (DateTime.Today.Year-1929)).Reverse());
        public SelectList CarMakeSL { get; }

        public IndexModel(ILogger<IndexModel> logger, ICarModelService carFileModelService, PredictionEnginePool<ModelInput, ModelOutput> predictionEnginePool)
        {
            _logger = logger;
            _carModelService = carFileModelService.GetDetails();
            CarMakeSL = new SelectList(_carModelService, "Id", "Model", default, "Make");
            _predictionEnginePool = predictionEnginePool;
        }

        public void OnGet()
        {
            _logger.LogInformation("Got page");
        }

        public void OnPost()
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

            ModelOutput prediction = _predictionEnginePool.Predict(input);
            CarInfo.Price = prediction.Score;
            ShowPrice = true;
        }
    }
}
