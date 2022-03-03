# DESCRIPTION
This simple program is built on top of classifier model which, given a text message, can funnel it down in the most proper categories.

It is thought to be used in emergency situations of various kind, when there are posted thousands of messages indicating the nature of a specific need or trouble, or sended directely.

Classify quickly every message can help to speed up the process of helping the needing people. If every organization receive only the messages regarding their own area, each one has to read less messages and faster knowing how and where to act.

After this first automated clean the messages still need a final human check, but it helps.


# INSTALLATION
### REQUIREMENTS

> pyhton (>= 3.8)  
> pandas (>= 1.4.0)  
> numpy (>= 1.22.2)  
> scikit-learn (>= 1.0.2)  
> SQLAlchemy (>= 1.4.31)  
> plotly (>= 5.5.0)  
> nltk (>= 3.6.7)  

###### (optionally - used to perform unit tests)
> pytest (>= 7.0.1)
###### (optionally - used to run the app locally)
> flask (>= 2.0.2)



# ISTRUCTIONS
This is a web app, so it needs to be run on a web server.  
You can run it using Flask (in this case you must install it locally beforehand) or deploying on a web server (like Heroku, for instances).

#### RUN LOCALLY
To run this app locally, from Terminal or Prompt you must use the following command:
```shell
cd project_folder
python run.py
```
after having run the last line, the terminal windows will still run the web app as a local server, and the last line of the message will be the address at which you can access the app, so type it on a web browser or copy and paste it.

To stop running the local server, from the terminal windows press CTRL+C to interrupt the program.

#### DEPLOY ON A WEB SERVER
If you want to deploy the app elsewhere, probably you'll have to remove the following command from the run.py file before submitting the app:
```python
app.run(host='0.0.0.0', port=3001, debug=True)
```
You'll probably have to add other configuration files, too, required from the web server you choose.


# FILES IN THE REPOSITORY

ROOT/  
├───classification_messages_app/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───templates/ &nbsp;&nbsp;# html pages  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───\_\_init\_\_.py &nbsp;&nbsp;  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───data_wrangling.py &nbsp;&nbsp;# load data from db and prepare plots for frontend  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└───routes.py &nbsp;&nbsp;# joins web links to html templates  
├───data/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───disaster_categories.csv &nbsp;&nbsp;  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───disaster_messages.csv &nbsp;&nbsp;   
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───messages_db.db &nbsp;&nbsp;  # sql database  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└───process_data.py &nbsp;&nbsp;# load from csv files, clean data and create sql database  
├───models/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───classifier.pkl &nbsp;&nbsp;  # the classifier used from frontend calls  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└───train_classifier.py &nbsp;&nbsp; # defines the classification model, trains and tests it  
├───tests/ &nbsp;&nbsp; # unit tests  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───\_\_init\_\_.py &nbsp;&nbsp;   
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───test_etl.py &nbsp;&nbsp;  # tests for the module "process_data.py"  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└───test_train_classifier.py &nbsp;&nbsp; # tests for the module "train_classifier.py"  
└───utils/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───\_\_init\_\_.py   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└───utils.py &nbsp;&nbsp; # utility functions used in other modules


# CONTRIBUTING
This program has not further scheduled updates.  
If you're interested in contributing to this idea and software, please ask.


# LICENSE
MIT License

Copyright (c) 2022 Roberto De Monte

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
