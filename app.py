
 # This is basically the heart of my flask
 
# import Libraries
import pickle
from flask import Flask, request, render_template
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

tf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))

lm = WordNetLemmatizer()
nltk.download('stopwords')

# Initialise the Flask and load pickle files.
app = Flask(__name__)
model = pickle.load(open('./Models/model.pkl', 'rb'))
FinalRating = pickle.load(open('./Models/FinalRating.pkl', 'rb'))
data = pickle.load(open('./Models/processedData.pkl', 'rb'))
features = pickle.load(open('./Models/features.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')  # Render html page for UI


@app.route('/Predict', methods=['GET', 'POST'])
def predict():
    inputFromUser = request.form['UserName']  # Fetch the username from Frontend
    inputFromUser = inputFromUser.lower()  # Lower the userid

    prediction = recommend(inputFromUser)
    if type(prediction) == str:  # compare if output is a string or a dataframe
        return render_template('index.html', prediction_text='{}'.format(prediction))
    else:  # Output obtained is a dataframe
        return (render_template('index.html',
                                prediction_text='1. {}\n 2. {}\n 3. {}\n 4. {}\n 5. {}'.format(prediction[0],
                                                                                               prediction[1],
                                                                                               prediction[2],
                                                                                               prediction[3],
                                                                                               prediction[4])))


def recommend(username):
    try:
        # Filter top 20  recommendations
        userRecommendations = FinalRating.loc[username].sort_values(ascending=False)[0:20]
        vectorizer = TfidfVectorizer(vocabulary=features)
        newdf = data[data.name.isin(userRecommendations.index)]
        X = vectorizer.fit_transform(newdf['Review'])
        newdf = newdf[['name']]
        # Make prediction
        newdf['prediction'] = model.predict(X)
        final_df = newdf.groupby('name').sum()
        # Compute the most positive response
        final_df['positive_percent'] = final_df.apply(lambda x: x['prediction'] / sum(final_df['prediction']), axis=1)
        final_list = final_df.sort_values('positive_percent', ascending=False).iloc[:5, :].index
        return final_list

    except:
        return "No User Available /Zero Recommendations for valid user."


if __name__ == "__main__":
    app.run(debug=True)
