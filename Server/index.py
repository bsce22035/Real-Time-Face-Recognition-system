from flask import Flask
from flask_cors import CORS
from routes import routes

app = Flask(__name__)
CORS(app)  # Allow React frontend to connect
# Serve static images

app.register_blueprint(routes.video_feed_route)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9000, debug=True)
