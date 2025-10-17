from flask import Blueprint, jsonify, Response, send_from_directory
import sys
import os
import re


video_feed_route = Blueprint('video_feed_route', __name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

@video_feed_route.route("/video-feed")
def video_feed():
    from model import video_generator
    return Response(
        (b'--frame\r\n'
         b'Content-Type: image/jpeg\r\n\r\n' + frm + b'\r\n'
         for frm in video_generator(mode='multi')),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
