1. short time series: monthly data. Perhaps no more than 2 years  
2. Model a GP to the time series  
3. iterate over series, leaving one idx out.  
4. check the percentile of the point left out, based on the trace of the GP
