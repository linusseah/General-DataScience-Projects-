###############
# Loading Dependencies
# Libraries & Pickle files
################

from flask import Flask, render_template, jsonify, request
from wtforms import Form, TextAreaField, validators #for the text input forms

#libraries to import for the model 
#vader, textblob, spacy, logreg, pickle nltk, countvect, pandas, numpy, matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk 
from textblob import TextBlob
import matplotlib.pyplot as plt

vader = SentimentIntensityAnalyzer()

#import libraries for clean text function 
from bs4 import BeautifulSoup   
import regex as re
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

# Instantiate lemmatizer 
lemmatizer = WordNetLemmatizer()

#########################
#Loading the Pickle files
#########################

#prediction model
lr_baseline_model = pickle.load(open("./static/pickle-files/lr_baseline_model","rb"))

#fitted and transformed count vectorizer
cvec = pickle.load(open("./static/pickle-files/cvec","rb"))

#text cleaning function
#clean_text = pickle.load(open("./static/pickle-files/clean_text","rb"))


###########
#FLASK
###########

app = Flask(__name__)

class ReviewForm(Form):
    moviereview = TextAreaField('',
            [validators.DataRequired(), validators.length(min=15)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        emotions = emo_machine(review)[0]

        return render_template('results.html', 
                            content=review,
                            tables=[emotions.to_html()], titles = ['emotional breakdown'],
                            score= emo_machine(review)[1])

    return render_template('reviewform.html', form=form)


###################
#Generating Charts
###################
import io
#import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

#Emotion pie chart 
@app.route('/emo_pie.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    global emo_perc
    axis.pie(emo_perc, labels =emo_perc.index, autopct='%1.1f%%')
    return fig


#Vader Sentiment line graph
@app.route('/vader_graph.png')
def plot2_png():
    fig2 = create2_figure()
    output2 = io.BytesIO()
    FigureCanvas(fig2).print_png(output2)
    return Response(output2.getvalue(), mimetype='image2/png')

def create2_figure():    
    fig2 = Figure()
    axis2 = fig2.add_subplot(1, 1, 1)
    global vader_scores
    xs = range(len(vader_scores))
    ys = vader_scores
    axis2.plot(xs, ys)
    return fig2

###################


##########
# MODEL
##########

def emo_machine(text_string_input):    

    #defining this is as a global variable  
    global vader_scores

    #instantiating a dataframe to which we will add columns of data
    emo_summary = pd.DataFrame()

    #a list to store vader scores for each sentence from the input
    vader_scores = []

    #list to add textblob noun-phrases
    bnp = []

    #list to add textblob POS tags
    b_pos_tags = []
    pos_subset = ['JJ','JJR','JJS','MD'] 
    #pre-define which POS tags you would like to select for; in this case, only adjectives & Verbs 
    
    def clean_text(raw_post):
         #1. Remove HTML.
        review_text = BeautifulSoup(raw_post).get_text()
    
        # 2. Remove non-letters.
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    
         #3. Convert to lower case, split into individual words.
        words = letters_only.lower().split()
        # Notice that we did this in one line!
    
        # 4. In Python, searching a set is much faster than searching
        # a list, so convert the stop words to a set.
        stops = set(stopwords.words('english'))
    
        # 5. Remove stop words.
        meaningful_words = [w for w in words if not w in stops]
    
        # 6. Lematize 
        lem_meaningful_words = [lemmatizer.lemmatize(i) for i in meaningful_words]
    
        # 7. Join the words back into one string separated by space, 
        # and return the result.
        return(" ".join(lem_meaningful_words))

    # Initialize an empty list to hold the clean sent tokens.
    clean_text_input_sent = []
    
    #tokenizing the input into indiv. sentences
    #uncleaned text input for vader scoring (because it scores better on uncleaned text)
    text_input_sent = nltk.tokenize.sent_tokenize(text_string_input)
    
    #sent tokenizing the cleaned text input for modelling and textblob extractions
    for i in text_input_sent:
        
        # Clean each sent token, then append to clean_text_input_sent.
        clean_text_input_sent.append(clean_text(i))

    #Vectorizing the list of indiv. cleaned sentences for model prediction 
    X_input_cvec = cvec.transform(clean_text_input_sent).todense() 
    
    #predicting for emotions based on the trained model 
    predictions = lr_baseline_model.predict(X_input_cvec)   
    
    #adding vader scores for indiv sentences
    for sent in text_input_sent:
        vader_scores.append(vader.polarity_scores(sent)['compound']) #i am only appending the compound scores here

    #adding textblob noun-phrases and pos_tags
    for sent in clean_text_input_sent:    
        blob = TextBlob(sent)
        bnp.append(blob.noun_phrases) 

        sent_pos_tags = [] #creating indiv. lists to collect pos_tags for each indiv. sentence 
        for i in blob.pos_tags:
            if i[1] in pos_subset: #note that for strings, the command is 'in' and not '.isin()' as with series or lists
                sent_pos_tags.append(i[0])
            else:
                pass
        b_pos_tags.append(sent_pos_tags)

    #adding columns to the dataframe
    emo_summary['emotions'] = predictions
    emo_summary['sentences'] = clean_text_input_sent
    emo_summary['sentiment'] = vader_scores
    emo_summary['key phrases'] = bnp
    emo_summary['key words'] = b_pos_tags

    #Only after constructing the emo_summary dataframe do we go about calculating overall avg valence
    #defining which emotions should be getting negative valences 
    neg_emo = ['sadness','anger','fear','worry']
    neg_criteria = ((emo_summary['emotions'].isin(neg_emo)) & (emo_summary['sentiment']<0))
    
    #defining which emotions should be getting positive/neutral valences 
    pos_emo = ['joy','love','surprise','neutral']
    pos_criteria = ((emo_summary['emotions'].isin(pos_emo)) & (emo_summary['sentiment']>=0))
    
    #combining both criteria
    all_criteria = (neg_criteria | pos_criteria)
    clean_vader_scores = emo_summary[(all_criteria)]['sentiment']
    
    #subsetting vader values using the defined boolean conditions above to calculate the overall valence scores 
    avg_vader_score = np.round(np.sum(clean_vader_scores)/len(clean_vader_scores),3)  
    
    #for pie chart breakdown of emotions
    global emo_perc 
    emo_perc = emo_summary['emotions'].value_counts(normalize=True)

    return emo_summary, avg_vader_score

if __name__ == "__main__":
    app.run(debug=True)


