# app.py
from flask import Flask, jsonify, request, render_template_string
from recommend import recommend_for_user_hybrid, movie_info, get_top_n_similar_items_itemcf, get_top_n_similar_items_content
import json

app = Flask(__name__)

# Simple homepage UI to test
INDEX_HTML = """
<!doctype html>
<title>Netflix Project - Demo</title>
<h2>Netflix-style Recommendation Demo</h2>
<form id="frm">
  <label>User ID: <input name="userId" id="userId" value="" /></label>
  <button type="button" onclick="getRec()">Get Recommendations</button>
</form>
<div id="out"></div>
<script>
async function getRec(){
  const uid = document.getElementById('userId').value;
  if(!uid){ alert('enter user id'); return; }
  const res = await fetch('/api/recommend?userId='+encodeURIComponent(uid));
  const data = await res.json();
  const out = document.getElementById('out');
  if(data.error){ out.innerHTML = '<pre>'+JSON.stringify(data, null, 2)+'</pre>'; return; }
  let html = '<h3>Recommendations for user '+uid+'</h3><ol>';
  data.recommendations.forEach(r=>{
    html += '<li><b>'+r.title+'</b> (movieId: '+r.movieId+') — score: '+r.score.toFixed(3)+'</li>';
  });
  html += '</ol>';
  out.innerHTML = html;
}
</script>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/api/recommend')
def api_recommend():
    userId = request.args.get('userId', type=int)
    if userId is None:
        return jsonify({"error":"userId (integer) required as query param"}), 400
    try:
        recs = recommend_for_user_hybrid(userId, top_n=10, alpha=0.6)
    except Exception as e:
        return jsonify({"error":str(e)}), 500
    res = []
    from recommend import movie_info
    for mid, score in recs:
        info = movie_info(mid)
        res.append({"movieId": mid, "title": info.get('title',''), "genres": info.get('genres',''), "score": float(score)})
    return jsonify({"userId": userId, "recommendations": res})

@app.route('/api/similar/item/<int:movieId>')
def api_similar_item(movieId):
    sims = get_top_n_similar_items_itemcf(movieId, top_n=10)
    out = []
    for mid, score in sims:
        out.append({"movieId": int(mid), "title": movie_info(mid).get('title',''), "score": float(score)})
    return jsonify({"movieId": movieId, "similar_itemcf": out})

@app.route('/api/similar/content/<int:movieId>')
def api_similar_content(movieId):
    sims = get_top_n_similar_items_content(movieId, top_n=10)
    out = []
    for mid, score in sims:
        out.append({"movieId": int(mid), "title": movie_info(mid).get('title',''), "score": float(score)})
    return jsonify({"movieId": movieId, "similar_content": out})

if __name__ == "__main__":
    app.run(debug=True)
