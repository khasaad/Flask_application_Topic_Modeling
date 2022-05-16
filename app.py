from model import *
from flask import Flask, render_template, request, redirect, url_for
from sklearn.metrics.pairwise import euclidean_distances
import re
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt


connection = sqlite3.connect('database.db', check_same_thread=False)

if connection:
    print("Connected Successfully")
else:
    print("Connection Not Established")

cursor = connection.cursor()

app = Flask(__name__)

# Read data
data_lemmatized = pd.read_csv('C:/Users/khale/Topic modeling/data_npr_cleaning.csv', parse_dates=[0],
                              infer_datetime_format=True)

nlp = spacy.blank("en")

dict_topics = {0: 'USA_politic', 1: 'General_security', 2: 'Entertainment', 3: 'Health_care',
               4: 'Elections', 5: 'Women_rights', 6: 'Well_being', 7: 'Education',
               8: 'Social_security', 9: 'World_politics'}


def add_comma(match):
    return match.group(0) + ','


@app.route('/', methods=['POST', 'GET'])
def index(top_n=1):
    global df_conf
    if request.method == 'POST':
        text = request.form['article']
        text = [text]
        cleaned_text, topic, scores = predict_topic(text)
        text = [cleaned_text]
        dists = euclidean_distances(scores.reshape(1, -1), lda_output_GS)[0]
        doc_ids = np.argsort(dists)[:top_n]

        conf = np.round(lda_output_GS[doc_ids], 1)
        s = re.sub(r'\[[0-9\.\s]+\]', add_comma, str(conf))
        s = re.sub(r'([0-9\.]+)', add_comma, s)
        confidence_scores = eval(s)

        confidence = {}
        for i, val in enumerate(confidence_scores[0]):
            confidence[dict_topics[list(dict_topics.keys())[i]]] = val

        sorted_dict = sorted(confidence.items(), key=lambda kv: kv[1], reverse=True)
        df_conf = pd.DataFrame(sorted_dict, columns=['Topic', 'Score'])

        new_dict = {}
        for r in range(len(sorted_dict)):
            new_dict[sorted_dict[r][0]] = sorted_dict[r][1]

        art = text[0]
        USA_politic = new_dict['USA_politic']
        General_security = new_dict['General_security']
        Entertainment = new_dict['Entertainment']
        Health_care = new_dict['Health_care']
        Elections = new_dict['Elections']
        Women_rights = new_dict['Women_rights']
        Well_being = new_dict['Well_being']
        Education = new_dict['Education']
        Social_security = new_dict['Social_security']
        World_politics = new_dict['World_politics']

        # Insert data to topic_modeling table
        if len(art) >= 3:
            cursor.execute("INSERT INTO topic_modeling (article, USA_politic, General_security, Entertainment,"
                           " Health_care,Elections, Women_rights, Well_being, Education, Social_security , "
                           "World_politics) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (art, USA_politic, General_security, Entertainment, Health_care, Elections, Women_rights,
                            Well_being, Education, Social_security, World_politics))

            connection.commit()

            # Plot bar chart
            plt.figure(figsize=(10, 5))
            plt.barh(df_conf["Topic"], df_conf["Score"])
            plt.gca().invert_yaxis()
            plt.ylabel("Score", fontsize=11)
            plt.xlabel("Topic", fontsize=6)
            plt.title("Predict topics of an article")
            plt.xticks(rotation=45)
            plt.yticks(fontsize=6)
            plt.savefig('static/images/plot.png')

            return render_template('submit.html', text=text, tables=[df_conf.to_html(classes='data')],
                                   titles=df_conf.columns.values, url='/static/images/plot.png')
        else:
            return render_template('index.html')
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
