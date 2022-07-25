using Microsoft.ML;
using Ml_Cs_MultiClass_classification.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ml_Cs_MultiClass_classification.ML
{
    public class Ml_Steps
    {
        public static string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        public static string _trainDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        public static string _testDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        public static string _modelPath = Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        public static MLContext _mlContext = new MLContext(seed: 0);
        public static PredictionEngine<IssueModel, IssuePrediction> _predEngine;
        public static ITransformer _trainedModel;
        public static IDataView _trainingDataView;
        public static IEstimator<ITransformer> ProccesData()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText("TitleFeaturized", "Title"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext);//for cache using (only for small or midlle datasets)

            return pipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView TrainDataset, IEstimator<ITransformer> pipeline)
        {
            Console.WriteLine("Build and Training");
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(TrainDataset);
            _predEngine = _mlContext.Model.CreatePredictionEngine<IssueModel, IssuePrediction>(_trainedModel);

            IssueModel issue = new IssueModel()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var prediction = _predEngine.Predict(issue);

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
            return trainingPipeline;
        }

        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testDataView = _mlContext.Data.LoadFromTextFile<IssueModel>(_testDataPath, hasHeader: true);

            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));


            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");
        }

        public static void SaveModelAsFile(MLContext mLContext, DataViewSchema trainingDataSchema, ITransformer model)
        {
            mLContext.Model.Save(model, trainingDataSchema, _modelPath);
        }

        public static void PredictIssue()
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            IssueModel singleIssue = new IssueModel() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };

            _predEngine = _mlContext.Model.CreatePredictionEngine<IssueModel, IssuePrediction>(loadedModel);

            var prediction = _predEngine.Predict(singleIssue);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }

        public static void Ml_Testing(IssueModel testmodel)
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
            _predEngine = _mlContext.Model.CreatePredictionEngine<IssueModel, IssuePrediction>(loadedModel);
            var prediction = _predEngine.Predict(testmodel);
            Console.WriteLine("predict ~=   " +prediction.Area);
        }
    }
}
