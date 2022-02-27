import pickle

import pandas as pd
import plotly.graph_objs as go
from sqlalchemy import create_engine

classification_model = pickle.load(open('models/classifier.pkl', 'rb'))

classes = ['related', 'request', 'offer', 'aid_related', 'medical_help',
       'medical_products', 'search_and_rescue', 'security', 'military',
       'water', 'food', 'shelter', 'clothing', 'money', 'missing_people',
       'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport',
       'buildings', 'electricity', 'tools', 'hospitals', 'shops',
       'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
       'storm', 'fire', 'earthquake', 'cold', 'other_weather',
       'direct_report']


def return_predicted_labels(text):
    return (classes, classification_model.predict(text)[0])


def return_figures():

    df = load_data()

    graph_one = []
    graph_one.append(
        go.Bar(
        x = df.iloc[:, 4:].sum().sort_values(),
        y = df.iloc[:, 4:].sum().sort_values().index,
        orientation='h'
    ))

    layout_one = dict(
        title = 'Number of past occurencies',
        autosize=False,
        width=1300,
        height=1000,
        margin=dict(
            l=150,
            r=50,
            b=100,
            t=100,
            pad=4)
    )

    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))

    return figures


def load_data():
    engine = create_engine('sqlite:///data/messages_db.db')
    df = pd.read_sql_table('messages', engine)
    return df
