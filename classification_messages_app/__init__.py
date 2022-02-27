from flask import Flask

app = Flask(__name__)

from classification_messages_app import routes
