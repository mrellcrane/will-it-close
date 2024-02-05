# The purpose of this script is to clean and create features for the dataframe

import pandas as pd
from collections import Counter
import boto3
from sklearn.feature_extraction.text import TfidfVectorizer
import os


def get_dataframe_from_s3_json(bucket_name, key_name, multi_line=True):
    '''
    INPUT: s3 bucket name and key name
    OUTPUT: dataframe of from s3 .json file
    '''
    client = boto3.client("s3")
    result = client.get_object(Bucket=bucket_name, Key=key_name)
    # Read the object (not compressed):
    dataframe = pd.read_json(result["Body"].read().decode(), lines=multi_line)
    return dataframe


def is_food(item):
    '''
    INPUT: cell from pandas dataframe
    OUTPUT: boolean
    '''
    restaurants_and_related_categories = ['Restaurants', 'Italian',
                                          'Food', 'Bars', 'Fast Food', 'Coffee & Tea', 'Sandwiches']
    if len(set(restaurants_and_related_categories) & set(item)) >= 1:
        return True
    else:
        return False


def flatten_dict(row):
    '''
    INPUT: cell from pandas dataframe with nested dictionairies
    OUTPUT: non-nested dictionairies
    '''
    out = {}
    for key, value in row.items():
        if type(value) != dict:
            out[key] = value
        else:
            sub_key = key
            for k, v in value.items():
                out[sub_key + "|" + k] = v
    return out


def make_exists_function(key):
    '''
    INPUT: cell from pandas dataframe
    OUTPUT: dictionary value
    '''
    def get_key_if_exists(row):
        if key in row:
            return row[key]
        else:
            return "N/A"
    return get_key_if_exists


def add_restaurant_count_column(dataframe):
    '''
    INPUT: dataframe
    OUTPUT: dataframe with column that shows frequency that restaurant is
    in the dataset
    '''
    restaurant_frequency = dataframe.groupby(
        ['name']).count().sort_values('address', ascending=False)

    restaurant_frequency = pd.DataFrame(restaurant_frequency['address'])

    restaurant_frequency.columns = ['restaurant_count']

    restaurant_frequency['name'] = restaurant_frequency.index

    restaurant_frequency = restaurant_frequency[['name', 'restaurant_count']]

    return previously_open_US_restaurants.merge(restaurant_frequency, how='left', left_on='name', right_on='name')


def closed_on_google(row):
    '''
    INPUT: cell from pandas dataframe
    OUTPUT: parses through google maps api data to get closed status
    '''
    try:
        return row[0]['permanently_closed']
    except:
        return False


def fix_percent(row):
    '''
    INPUT: cell from pandas dataframe as percentage
    OUTPUT: float of percentage without mod symbol
    '''
    row = str(row).strip('%')
    row = float(row)
    return row/100


def summaries_from_google(dataframe, key, default_val=0):
    '''
    INPUT: dataframe with data on nearby restaurants (from google maps api)
    OUTPUT: summary data of nearby restaurants
    '''
    summaries = []
    key_errors = 0
    for i in range(len(dataframe)):
        total = 0
        count = 0
        for j in range(len(dataframe['results'][i])):
            try:
                total += dataframe['results'][i][j][key]
                count += 1
            except KeyError:
                key_errors += 1
        try:
            summaries.append(
                {'business_id': dataframe['yelp_business_id'][i], 'avg_'+key: (total / count)})
        except ZeroDivisionError:
            summaries.append(
                {'business_id': dataframe['yelp_business_id'][i], 'avg_'+key: default_val})
    return pd.DataFrame(summaries)


def get_price(row):
    '''
    INPUT: cell from pandas dataframe that is a dictionary
    OUTPUT: price value in dictionairy, or 1.5, which is the average price
    '''
    try:
        return row['RestaurantsPriceRange2']
    except KeyError:
        return 1.5


def concat_unique_columns(df1, df2, suffix):
    '''
    INPUT: two dataframes and a suffix for column names. Dataframes should have
    some of the same columns.
    OUTPUT: a dataframe with the unique columns from the dataframes, with
    the suffix attached.
    '''
    cols = list(set(list(df1.columns) + list(df2.columns)))
    df_dict = {'df1': [], 'df2': []}
    for col in cols:
        if col in df1.columns:
            df_dict['df1'].append(col)
        else:
            df_dict['df2'].append(col)
    combined_df = pd.concat([df1[df_dict['df1']], df2[df_dict['df2']]], axis=1)
    combined_df.columns = [suffix + str(col) for col in combined_df.columns]
    return combined_df


def get_zipped_postcode_data_from_s3_bucket(postcodes):
    '''
    INPUT: list of zip codes
    OUTPUT: list of dictionairies with economic data on each zip code
    *Note: This bucket only has the postcodes used for these restaurants, which
    is about 600 zip codes.
    '''
    s3 = boto3.client('s3')
    zip_code_data = []
    for code in postcodes:
        response = s3.get_object(Bucket='zip-code-economic-data', Key=f'zip_code: {code}')
        body = response['Body'].read()
        df = pd.read_html(body)[0][pd.read_html(body)[0]['Measure'].map(
            type) == str][['Description', 'Measure']]
        keys = [str(x) for x in list(df['Description'].values)]
        vals = [str(x) for x in list(df['Measure'].values)]
        zipped = dict(zip(keys, vals))
        zipped['Zip Code'] = code
        zip_code_data.append(zipped)
    return zip_code_data


def str_to_num(row):
    '''
    INPUT: cell from pandas dataframe
    OUTPUT: int or float value of cell
    '''
    try:
        return int(row)
    except ValueError:
        return float(row)


if __name__ == "__main__":
    if not os.path.exists('../data'):
        os.makedirs('../data')
        print("Data folder created.")

    print("Reading basic business data from s3 bucket...")
    yelp_business_data = get_dataframe_from_s3_json("businesspredictiondata",
                                                    "business.json", multi_line=True)

    print("Cutting down yelp data to only open US restaurants...")
    # filters businesses that were open when this dataset was published Jan. 2018
    open_businesses = yelp_business_data[yelp_business_data['is_open'] == 1].drop_duplicates([
                                                                                             'name', 'address'])

    # creates column that says if business is restaurant and creates df of just open restaurants
    open_businesses['is_food'] = open_businesses['categories'].apply(is_food)
    open_restaurants = open_businesses[open_businesses['is_food'] == True].copy()

    # creates column that says if business is in USA and creates df of just
    # restaurants open in the US as of January 2018

    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
              "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
              "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
              "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
              "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

    open_restaurants['in_US'] = open_restaurants['state'].isin(states)
    mask = (open_restaurants['in_US'] == True) & (open_restaurants['longitude'] < -20)
    previously_open_US_restaurants = open_restaurants.loc[mask].copy()

    print("Creating dummy columns for restaurant attributes...")
    # creates dummy columns for 0
    previously_open_US_restaurants['flat_attributes'] = previously_open_US_restaurants.loc[:, 'attributes'].apply(
        flatten_dict)
    all_attributes = []

    for row in previously_open_US_restaurants['flat_attributes']:
        all_attributes.extend(row.keys())
    unique_attributes = list(dict(Counter(all_attributes).most_common(50)).keys())

    for key in unique_attributes:
        f = make_exists_function(key)
        previously_open_US_restaurants['Attribute|' + key +
                                       ' value:'] = previously_open_US_restaurants['flat_attributes'].apply(f)

    all_categories = []
    [all_categories.extend(item) for item in list(previously_open_US_restaurants['categories'])]

    most_common_categories = list(dict(Counter(all_categories).most_common(50)).keys())

    for key in most_common_categories:
        previously_open_US_restaurants[f"Category|{key}_true"] = previously_open_US_restaurants['categories'].apply(
            lambda x: key in x)

    previously_open_US_restaurants = add_restaurant_count_column(previously_open_US_restaurants)

    previously_open_US_restaurants['restaurant_count > 1'] = previously_open_US_restaurants['restaurant_count'] > 1
    previously_open_US_restaurants['restaurant_count > 5'] = previously_open_US_restaurants['restaurant_count'] > 5
    previously_open_US_restaurants['restaurant_count > 25'] = previously_open_US_restaurants['restaurant_count'] > 25

    print("Downloading updated open/closed on google data...")
    updated_google_df = get_dataframe_from_s3_json(
        "businesspredictiondata", "updated_google_data.json", multi_line=False)

    updated_google_df['closed_on_google'] = updated_google_df['results'].apply(closed_on_google)

    restaurants_with_google_data = previously_open_US_restaurants.merge(
        updated_google_df, how='inner', left_on='business_id', right_on='yelp_business_id')

    # removes rows without any matching data from Google
    restaurants_with_google_data = restaurants_with_google_data[restaurants_with_google_data['results'].map(
        len) > 0]

    # gets the valid postal codes from the dataframe.
    postcodes = list(previously_open_US_restaurants['postal_code'].unique())

    postcodes = [x for x in postcodes if len(x) > 2]

    print("Downloading zipcode data...")
    # grabs the zip code data from the s3 bucket and turns it into a dataframe
    zip_code_dicts = get_zipped_postcode_data_from_s3_bucket(postcodes)

    zip_code_df = pd.DataFrame(zip_code_dicts)

    zip_code_df['Zip Code'] = zip_code_df['Zip Code'].apply(str)

    restaurants_with_economic_data = restaurants_with_google_data.merge(
        zip_code_df, how='left', left_on='postal_code', right_on='Zip Code')

    restaurants_with_economic_data.iloc[:, -
                                        19:] = restaurants_with_economic_data.iloc[:, -19:].fillna(0).copy()

    percent_columns = [
        'Educational Attainment: Percent high school graduate or higher', 'Individuals below poverty level']
    for col in percent_columns:
        restaurants_with_economic_data[col] = restaurants_with_economic_data[col].apply(fix_percent)

    num_columns = ['2016 ACS 5-Year Population Estimate',
                   'American Indian and Alaska Native alone',
                   'Asian alone',
                   'Black or African American alone',
                   'Census 2010 Total Population',
                   'Foreign Born Population',
                   'Hispanic or Latino (of any race)',
                   'Median Age',
                   'Median Household Income',
                   'Native Hawaiian and Other Pacific Islander alone',
                   'Some Other Race alone',
                   'Total housing units',
                   'Two or More Races',
                   'Veterans',
                   'White alone']

    for col in num_columns:
        restaurants_with_economic_data[col] = restaurants_with_economic_data[col].apply(str_to_num)

    print("Downloading google nearby data...")
    google_nearby_df = get_dataframe_from_s3_json("businesspredictiondata",
                                                  "google_nearby.json",
                                                  multi_line=False)
    google_nearby_df['num_nearby_restaurants'] = google_nearby_df['results'].apply(len)

    google_nearby_df.reset_index(inplace=True)
    nearby_prices = summaries_from_google(google_nearby_df, 'price_level', 1.5)
    nearby_ratings = summaries_from_google(google_nearby_df, 'rating', 3.5)
    nearby_prices_and_rating = nearby_prices.merge(nearby_ratings, how='outer', on='business_id')
    nearby_prices_rating_num = nearby_prices_and_rating.merge(
        google_nearby_df, how='outer', left_on='business_id', right_on='yelp_business_id')
    trimmed_nearby_data = nearby_prices_rating_num[[
        'business_id', 'avg_price_level', 'avg_rating', 'num_nearby_restaurants']]

    restaurants_with_nearby_data = restaurants_with_economic_data.merge(
        trimmed_nearby_data, how='left', on='business_id')

    restaurants_with_nearby_data['relative rating'] = restaurants_with_nearby_data['stars'] - \
        restaurants_with_nearby_data['avg_rating']

    restaurants_with_nearby_data['price_level'] = restaurants_with_nearby_data['attributes'].apply(
        get_price)

    restaurants_with_nearby_data['relative_price'] = restaurants_with_nearby_data['price_level'] - \
        restaurants_with_nearby_data['avg_price_level']

    print("Downloading review data...")
    reviews_df = get_dataframe_from_s3_json("businesspredictiondata",
                                            "small_review_df.json",
                                            multi_line=False)

    restaurants_with_stars = restaurants_with_nearby_data.merge(
        reviews_df, how='left', on='business_id')

    restaurants_with_stars['one_star_text'] = restaurants_with_stars['one_star_text'].apply(str)
    restaurants_with_stars['two_to_four_star_text'] = restaurants_with_stars['two_to_four_star_text'].apply(
        str)
    restaurants_with_stars['five_star_text'] = restaurants_with_stars['five_star_text'].apply(str)

    closed_restaurants = restaurants_with_stars[restaurants_with_stars['closed_on_google'] == True]
    open_restaurants = restaurants_with_stars[restaurants_with_stars['closed_on_google'] == False]

    print("Vectorizing one star text...")
    tfidf_one_closed = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_one_closed.fit(closed_restaurants['one_star_text'])
    feature_matrix = tfidf_one_closed.transform(restaurants_with_stars['one_star_text'])
    tfidf_one_closed_df = pd.DataFrame(
        feature_matrix.toarray(), columns=tfidf_one_closed.get_feature_names())

    tfidf_one_open = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_one_open.fit(open_restaurants['one_star_text'])
    feature_matrix = tfidf_one_open.transform(restaurants_with_stars['one_star_text'])
    tfidf_one_open_df = pd.DataFrame(feature_matrix.toarray(),
                                     columns=tfidf_one_open.get_feature_names())

    print("Vectorizing two to four star text...")
    tfidf_two_to_four_closed = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_two_to_four_closed.fit(closed_restaurants['two_to_four_star_text'])
    feature_matrix = tfidf_two_to_four_closed.transform(restaurants_with_stars['one_star_text'])
    tfidf_two_to_four_closed_df = pd.DataFrame(
        feature_matrix.toarray(), columns=tfidf_two_to_four_closed.get_feature_names())

    tfidf_two_to_four_open = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_two_to_four_open.fit(open_restaurants['two_to_four_star_text'])
    feature_matrix = tfidf_two_to_four_open.transform(
        restaurants_with_stars['two_to_four_star_text'])
    tfidf_two_to_four_open_df = pd.DataFrame(
        feature_matrix.toarray(), columns=tfidf_two_to_four_open.get_feature_names())

    print("Vectorizing five star text...")
    tfidf_five_closed = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_five_closed.fit(closed_restaurants['five_star_text'])
    feature_matrix = tfidf_five_closed.transform(restaurants_with_stars['one_star_text'])
    tfidf_five_closed_df = pd.DataFrame(
        feature_matrix.toarray(), columns=tfidf_five_closed.get_feature_names())

    tfidf_five_open = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_five_open.fit(open_restaurants['five_star_text'])
    feature_matrix = tfidf_five_open.transform(restaurants_with_stars['five_star_text'])
    tfidf_five_open_df = pd.DataFrame(feature_matrix.toarray(),
                                      columns=tfidf_five_open.get_feature_names())

    print("Removing non-unique features")
    unique_one_star_df = concat_unique_columns(tfidf_one_closed_df,
                                               tfidf_one_open_df,
                                               'one_star: ')
    unique_two_to_four_star_df = concat_unique_columns(
        tfidf_two_to_four_closed_df,
        tfidf_two_to_four_open_df,
        'two_to_four_star: ')

    unique_five_star_df = concat_unique_columns(
        tfidf_five_closed_df, tfidf_five_open_df, 'five_star: ')

    all_tfidf_reviews_df = pd.concat(
        [unique_one_star_df, unique_two_to_four_star_df, unique_five_star_df], axis=1)

    restaurants_with_reviews = pd.concat([restaurants_with_stars, all_tfidf_reviews_df], axis=1)

    restaurants_with_reviews.replace('N/A', value=False, inplace=True)

    restaurants_with_reviews.replace('no', value=False, inplace=True)

    restaurants_with_reviews.drop_duplicates(['name', 'address'], inplace=True)

    columns_to_remove = ['attributes', 'categories', 'hours', 'flat_attributes',
                         'one_star_text', 'two_to_four_star_text',
                         'five_star_text',
                         'categories', 'White alone, Not Hispanic or Latino',
                         'Attribute|Alcohol value:',
                         'Attribute|RestaurantsAttire value:',
                         'Attribute|NoiseLevel value:',
                         'Attribute|WiFi value:',
                         'Attribute|Smoking value:']

    restaurants_with_reviews.drop(columns_to_remove, axis=1, inplace=True)

    restaurants_with_reviews.to_json('../data/featurized_dataframe.json')

    print("Data downloaded, features engineeered, and dataframe saved!")
