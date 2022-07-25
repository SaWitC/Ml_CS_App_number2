using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ml_Cs_MultiClass_classification.Models
{
    public class IssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area;
    }
}
