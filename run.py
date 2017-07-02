# from eventlet import wsgi
# import eventlet

# Run app on server
from app import app
app.run(host='0.0.0.0', debug=True)
# wsgi.server(eventlet.listen(('', 5000)), app)
