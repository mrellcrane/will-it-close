# Predicting Restaurant Closure
## [Watch the presentation on Youtube](https://www.youtube.com/watch?v=te4h5v2pU8M)

## About this project
I ***love*** Mongolian grill. So I was heartbroken to find out that my favorite Mongolian grill restaurant had closed.
<!-- ![Ruzhen Closed](images/1_ruzhen_closed.png) -->
<img src="images/1_ruzhen_closed.png" width="600"/>
But from this tragedy I found an opportunity.
<!-- ![Opportunity](images/2_the_problem.png) -->
<img src="images/2_the_problem.png" width="600"/>
I thought that if I could accurately predict restaurant closure, that would provide valuable information to stakeholders, such as investors, that need a risk assessment of a restaurant. The closing probability predictions were made on restaurant data from May 2018.
<br />
<br />
If you would like to repeat this project on your computer, follow the readme: *DIRECTIONS.MD*

## Overview
- Data sources:
	- Yelp academic dataset
	- Google Maps API
	- Data scraped from US Census website
- Data capture
	- Selenium
	- Requests
- Data storage:
	- MongoDB
	- AWS S3 Buckets
- Data manipulation:
	- Pandas
	- TFIDF
- Modeling - all done with Sklearn
	- Train test split (validation)
	- Random Forest, Gradient Boost, KNeighbors, and Logistic Regression.
	- Pickle (saving model)
	- ROC curve, AUC, classification report
- Exploratory Data Analysis
	- Matplotlib

*Continue reading if you are curious about the details of the project, otherwise feel free to check out the code, the website, or email me at <ellcrane@gmail.com>*

## Data Sources

### Starting point
<!-- ![Starting point](images/3_starting_point.png) -->
<img src="images/3_starting_point.png" width="600"/>

My starting point for this project was a Yelp Academic Dataset from January 2018 on ~175,000 businesses in different industries around the world.

<!-- ![Closed restaurants](images/5_map.png) -->
<img src="images/5_map.png" width="600"/>

After slicing this data down to only those businesses I was interested in: 1) restaurants 2) in the US 3) that were still in business/open, I had ~35,000 businesses. The restaurants were in 6 major cities, but the final model made predictions on restaurants in any location.

<!-- ![Closed restaurants](images/4_700_closed.png) -->
<img src="images/4_700_closed.png" width="600"/>

I then used the Google Maps API to get the updated information on if the restaurants were still open or closed. About 2% or 700 restaurants had closed in that 4 month period from January 2018 to May 2018. I used the Google Maps API instead of Yelp because Google Maps had a 150,000/day query limit, while Yelp had a 5,000/day limit.

### Yelp Academic Dataset - Basic Information
<!-- ![Closed restaurants](images/6_yelp.png) -->
<img src="images/6_yelp.png" width="600"/>

This included review count, star rating, price level, category (Chinese, Sandwiches, etc.), attributes (has wifi, noise level, bike parking, etc.). The categorical and attributes data had to be cleaned and procured from the columns using pandas apply/lambda.

### Yelp Academic Dataset - Reviews
<!-- ![Yelp reviews](images/7_yelp_reviews.png) -->
<img src="images/7_yelp_reviews.png" width="600"/>
<!-- ![Review buckets](images/9_buckets.png) -->
<img src="images/9_buckets.png" width="600"/>

In order to get more granular features, I put the reviews into 1-star, 2-4 star, and 5-star buckets and then did TFIDF, so each feature was titled with this format: [# of stars]: [word] (5-star: “service”, etc.). In order to limit the total number of features, I only used the most frequent 100 words from each star bucket for open restaurants, then did the same for closed restaurants. This resulted in a total of 322 features.

<!-- ![Review top features](images/10_review_top_features.png) -->
<img src="images/10_review_top_features.png" width="600"/>

To limit this feature count further, I put exclusively these features into a gradient boosted classifier to predict if the restaurant were open/closed, and then used only the top 100 features by feature importance in the final model.

<!-- ![Census data](images/8_census.png) -->
<img src="images/8_census.png" width="600"/>

I thought that economic data about the location in which the restaurant was located would be important, so I went to the census website. Their data was very poorly documented and difficult to read, so I decided I would scrape the data from the census factfinder.census.gov, which was organized by zip code. Some of the information I collected was average income, average age, percentage living in poverty, and the number of veterans.

### Google Maps Nearby Data/API
<!-- ![Google maps](images/11_google_maps_nearby.png) -->
<img src="images/11_google_maps_nearby.png" width="600"/>
I also thought that information on a restaurant’s competitiveness, such as the restaurant density in that area, or the average price/star-rating of its neighbors would be relevant. I used the Google Maps Places API to get that data, and also engineered features such as relative restaurant rating (restaurant rating minus the average rating of nearby restaurants).

## Modeling
<!-- ![Models used](images/12_modeling.png) -->
<img src="images/12_modeling.png" width="600"/>

For modeling, I used Random Forest, Gradient Boost, KNeighbors, and Logistic Regression.

<!-- ![Gradient boosting ROC curve](images/13_gb_roc.png) -->
<img src="images/13_gb_roc.png" width="600"/>

With a .66/.33 train/test split, gradient boost performed the best, with an AUC of .73.

For the predictions on the website, only the Yelp - Basic Information features were used, as those were the easiest to obtain for new data, as it could be obtained only by using the Google Maps API and Yelp API. The AUC only using this basic information was .67.

Based on these AUCs, predicting restaurant closure in 4 months is still very difficult to do, but the model does provide some signal.

## Feature Overview
The final model had the following feature count by category:

<!-- ![Features by category](images/14_features_by_category.png) -->
<img src="images/14_features_by_category.png" width="600"/>

Here’s a breakdown of feature importance, or how often the model used each feature:

<!-- ![Feature importance by source](images/15_feature_importance_by_source.png) -->
<img src="images/15_feature_importance_by_source.png" width="600"/>

As you can see, the review-based features were used most often, which isn’t surprising since review-based features were tied for the most number of features. It is interesting that the Google Maps restaurant competitiveness based features were used second most, even though there were only 6 of those features. I think this suggests that those features were very important, and additional features that compare a restaurant to its surroundings would be useful.

## Potential Improvements
- More features
	- I think the biggest improvement in this project would come from improving the predictive power of the model. Data from other restaurant websites could be useful, such as FourSquare and TripAdvisor.
- Improve website
	- I would like to add functionality so that a user can enter in any US restaurant on Yelp and could get a probability for that restaurant. Currently www.willitclose.com has a static database of 35,000 restaurants in the US.
	- I would like to engineer the predictions so that the predictions on current restaurants use all the features from the train/test split, since the AUC using all features is higher than when just using the Yelp Basic Information
- More business categories
	- Businesses such as retail and services that provide a lot of relevant data on their Yelp and Google Maps pages likely could also get predictions

## Me - please contact with any questions or comments!
Elliott Crane

Data Scientist

[www.willitclose.com](http://www.willitclose.com)

[linkedin.com/in/ellcrane](https://www.linkedin.com/in/ellcrane/)

[github.com/ellcrane](https://github.com/ellcrane)
