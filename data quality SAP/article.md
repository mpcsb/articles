# Building a data quality pipeline using SAP Data Services and SAP Information Steward

This article will explain with a few examples what are the main data quality issues and it will sketch a rough arquitecture of a SAP based solution that not only identifies data quality issues, but also acts on it.  

The folk wisdom of Data Quality divides issues in seven dimensions, and we will follow that structure: 
1. Completeness: it captures instances where there are missing data and on which columns it's incomplete;
2. Accuracy: it identifies how well the data you collected reflects the actual occurence. One example would be a date for a past event recorded as 01-01-9999;
3. Consistency: how consistent is a specific piece of information across your IT landscape? For instance, if the marital status of a customer consistent in two databases;
4. Uniqueness: it detects duplicate entries. For instance, when there's two records for the same customer;
5. Integrity: it ensures that data isn’t missing important relationship linkages. One example can be if data is stored in a shorter form in different systems, making it harder to establish a direct connection across systems. 
6. Timeliness: it captures instances where data isn't sufficiently up to date for some task. One example is data from the previous business day is loaded at noon of the following day, and we have a report is being executed at 10am;
7. Validity: it identifies data which does not necessarily obey the rules we expect. For instance, a telephone number should not contain letters;

[source/inspiration](https://help.sap.com/viewer/f0171f9321f243838f94a0032f829065/4.2.14/en-US/57ada9fc6d6d1014b3fc9283b0e91070.html)


