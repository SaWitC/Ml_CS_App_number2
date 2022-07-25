using System;
using System.IO;
using Microsoft.ML;
using Ml_Cs_MultiClass_classification.Models;

namespace Ml_Cs_MultiClass_classification
{
    class Program
    {
      
        static void Main(string[] args)
        {
            Console.WriteLine("Operations \n1) deffault start \n2)testing");
            int variable = 0;
            while (variable == 0)
            {
                int.TryParse(Console.ReadLine(),out variable);
            }
            switch (variable)
            {
                case 1:
                    ML.Ml_Steps._trainingDataView = ML.Ml_Steps._mlContext.Data.LoadFromTextFile<IssueModel>(ML.Ml_Steps._trainDataPath, hasHeader: true);

                    var pipeline = ML.Ml_Steps.ProccesData();

                    var trainingPipelane = ML.Ml_Steps.BuildAndTrainModel(ML.Ml_Steps._trainingDataView, pipeline);

                    ML.Ml_Steps.Evaluate(ML.Ml_Steps._trainingDataView.Schema);

                    ML.Ml_Steps.SaveModelAsFile(ML.Ml_Steps._mlContext, ML.Ml_Steps._trainingDataView.Schema, ML.Ml_Steps._trainedModel);

                    //ML.Ml_Steps.PredictIssue();
                    break;

                case 2:
                    IssueModel model = new IssueModel();
                    Console.WriteLine("Test model Title");
                    model.Title = Console.ReadLine();
                    Console.WriteLine("Test model Description");
                    model.Description = Console.ReadLine();
                    ML.Ml_Steps.Ml_Testing(model);
                   break;
            }
           

        }

      
    }
}
