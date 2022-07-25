using System;
using System.IO;
using Microsoft.ML;
using Ml_Cs_MultiClass_classification.Models;

namespace Ml_Cs_MultiClass_classification
{
    class Program
    {
        private static string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        private static string _testDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        private static string _modelPath = Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext= new MLContext(seed: 0);
        private static PredictionEngine<IssueModel, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        private static IDataView _trainingDataView;
        static void Main(string[] args)
        {
           // MLContext mLContext = new MLContext(seed: 0);

            _trainingDataView = _mlContext.Data.LoadFromTextFile<IssueModel>(_trainDataPath, hasHeader: true);

            var pipeline = ProccesData();

            var trainingPipelane = BuildAndTrainModel(_trainingDataView, pipeline);

            Evaluate(_trainingDataView.Schema);

           // SaveModelAsFile(,_trainingDataView,_modelPath);
        }

        public static IEstimator<ITransformer> ProccesData()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText("TitleFeaturized", "Title"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext);//for cache using (only for small or midlle datasets)

            return pipeline;
        }

        private static IEstimator<ITransformer> BuildAndTrainModel(IDataView TrainDataset,IEstimator<ITransformer> pipeline)
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

        private static void Evaluate(DataViewSchema trainingDataViewSchema)
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

        private static void SaveModelAsFile(MLContext mLContext,DataViewSchema trainingDataSchema,ITransformer model)
        {
            mLContext.Model.Save(model, trainingDataSchema, _modelPath);
        }
    }
}
