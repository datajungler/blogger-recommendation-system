
import pandas as pd

df = pd.read_csv("us_hiking_review_list.csv")

print(df.shape[0] / df['reviewer_name'].unique().shape[0])
print(df['product_id'].unique().shape[0])


# content-based cf
num_rating = pd.pivot_table(df[['product_id', 'review_comments_count']], index='product_id', aggfunc=len).reset_index().rename(columns={'review_comments_count': 'review_count'})
df = pd.merge(df, num_rating, on='product_id', how='left')

# create a column to capture rating of the products
df['rating'] = df['review_rating'].str[:3].astype(float)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
content = df.groupby(['product_id'])['review_content'].apply(lambda x: '.'.join(x)).reset_index()
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(content['review_content'])

def get_rating_similar(product_id, matrix):
    similar = matrix.corrwith(matrix[product_id])
    corr_similar = pd.DataFrame(similar, columns=['rating_similiarity'])
    return corr_similar.reset_index()


def get_review_similar(product_id, tfidf_matrix):
    cosine_similar = linear_kernel(tfidf_matrix, tfidf_matrix)
    cosine_similar = pd.DataFrame(index=content['product_id'].values, columns=content['product_id'].values, data=cosine_similar)
    review_similar = cosine_similar[[product_id]].reset_index().rename(columns={'index': 'product_id', product_id: 'review_similiarity'})
    return review_similar[review_similar['product_id']!=product_id]


def get_similar_product(product_id, df, tfidf_matrix, n_ratings_filter=100, n_recommendations=5, weight_rating_frac=0.5):
    matrix = pd.pivot_table(df, index='reviewer_name', columns='product_id', values='rating')
    content = df.groupby(['product_id'])['review_content'].apply(lambda x: '.'.join(x)).reset_index()
    rating_similar = get_rating_similar(product_id=product_id, matrix=matrix)
    review_similar = get_review_similar(product_id=product_id, tfidf_matrix=tfidf_matrix)
    orig = df.copy()
    agg_similar = pd.merge(orig, rating_similar, on='product_id')
    agg_similar = pd.merge(agg_similar, review_similar, on='product_id')
    agg_similar['rating_similiarity'].fillna(0, inplace=True)
    agg_similar['agg_similarity'] = agg_similar['rating_similiarity'] * weight_rating_frac + agg_similar['review_similiarity'] * (1-weight_rating_frac)
    agg_similar = agg_similar[['product_id', 'review_count', 'agg_similarity']].drop_duplicates().reset_index(drop=True)
    result = agg_similar[agg_similar['review_count'] > n_ratings_filter].sort_values(by='agg_similarity', ascending=False)
    return result[:n_recommendations]
	

result = get_similar_product(product_id='B07B31BJQJ', df=df, tfidf_matrix=tfidf_matrix, n_ratings_filter=0, n_recommendations=10, weight_rating_frac=0.5)
print(result)

