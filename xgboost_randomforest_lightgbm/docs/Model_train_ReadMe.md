## Missing Values in Dataset

Missing values in km_till_OMS comes from:

the computation: df$km_till_OMS = df$nextOMSPerf - df$TotalPerformanceSnapshot where nexOMSPerf is the TotalPerformanceSnapshot of the next OMS. When there is no next OMS in the sequences, then there are missing values for the last sequence within the same trainset.

LightGBM-model gives prediction for all the data points, including those which y eller x is missing.
