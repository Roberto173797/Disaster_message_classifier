import json

from flask import render_template, request
import plotly

from classification_messages_app import app
from classification_messages_app.data_wrangling import return_predicted_labels, return_figures

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():

    # GET request for the main page
    figures = return_figures()
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                            figuresJSON = figuresJSON,
                            ids=ids)

@app.route('/go', methods=['GET', 'POST'])
def go():

    query = request.args.get('query', '')

    classes, classification_labels = return_predicted_labels([query])
    classification_result = dict(zip(classes, classification_labels))
    return render_template('go.html',
                            query=query,
                            classification_result=classification_result)
