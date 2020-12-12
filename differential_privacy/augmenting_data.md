# Augmenting data: maintaining differential privacy 

## Objective  
The goal of this post is to imagine a scenario where small datasets can be published and are able to maintain some variant of differential privacy.  
The main idea here is to take one privacy mechanism and based on the adequate magnitude of noise, generate extra records. This will affect some count based queries by a known factor.  


## Brief introduction to differential privacy  
Differential privacy comes in main two forms: global DP and local DP.  
In global DP data is kept raw in a database and noise is added to it for specific queries and contingent on a privacy budget (the amount of information you can extract from the dataset).  
Local DP, on the other hand, adds noise directly at the insertion of the records in the database. This typically adds noise of higher magnitude than the globabl scenario. One could in principle publish datasets without fearing disclosing personal information about individuals.  

1. ε-differential privacy:  
This definition of DP considers one parameter, ε, which is essentially telling us that eε is how much confidence we can gain about whether one user has been added/removed from the data, by looking at the output.   

2. (ε, δ)-differential privacy:  
A relaxation of the previous definition is now described: essentialy we have a certain probability that n*δ individuals are identified in the dataset. δ can be controlled, but it is not neglibible with the size of the dataset.  


### Example 1  
If we apply differential privacy to a table containing information

The noise magnitude is calculated based on the sensitivity to the query: take count and sum.  
For count, if we exclude one record, the difference is 1.
For sum, the difference is the record that will make the biggest difference - in this case, 10.  
 
