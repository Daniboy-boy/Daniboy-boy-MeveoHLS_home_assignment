from flask import Flask, jsonify, request, render_template
from moveoHLS_task import group_claims
# creating a Flask app 
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('flask_frontend.html')


@app.route('/groups', methods=['POST'])
def groups():
    data = request.json
    patent_path = data.get('patent_path')
    n_clusters = data.get('n_clusters')
    result = group_claims(patent_path, int(n_clusters))
    return jsonify({'result': result})


if __name__ == '__main__':
    """
    The Flask app run locally. To tun it I went to the development server it crated at http://127.0.0.1:5000
    (path for me at least)
    """
    app.run(debug=True)