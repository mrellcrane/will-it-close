Hi there!

If you would like to re-run the featurization and modeling process on your own computer, please use this process:
## Overview
1. Clone this repo to your computer
2. Enter the *src* directory
3. *15-20 minutes (optional):* If you would like to re-run the featurization of the dataframe, using near-raw data stored in s3 buckets:
  - Run *featurize.py*
    - This will save the featurized dataframe to a local csv file
    - **Note** - if you run into any errors when running featurize.py, skip it and just run model.py instead  
4. *5-10 minutes:* Run *model.py* to see detailed performance metrics of the model for predicting restaurant closure. It will also show performance using different subsets of the data, and feature importances. If you didn't already run featurize.py,
it will get the featurized dataframe from an existing s3 bucket
5. Check out the figures that will be saved in the *plots* directory after you've run *model.py*
