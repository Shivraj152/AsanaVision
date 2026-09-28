from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AsanaVision — Real-Time Yoga Posture AI</title>
    <style>
        body {
            font-family: system-ui, -apple-system, sans-serif;
            background-color: #F6F0E4;
            color: #183B2E;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            box-sizing: border-box;
        }
        .container {
            background-color: #FFFFFF;
            border: 2px solid #E9DDC8;
            border-radius: 20px;
            padding: 40px;
            max-width: 650px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(24, 59, 46, 0.1);
        }
        .header-box {
            background: linear-gradient(135deg, #183B2E 0%, #0F281E 100%);
            color: #FFFFFF;
            padding: 24px;
            border-radius: 14px;
            margin-bottom: 24px;
        }
        h1 {
            font-size: 2.2rem;
            margin: 0 0 8px 0;
            letter-spacing: 2px;
        }
        p {
            font-size: 1.05rem;
            line-height: 1.6;
            color: #3A3028;
        }
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #B8664A 0%, #985137 100%);
            color: #FFFFFF !important;
            font-weight: 800;
            font-size: 1.1rem;
            padding: 14px 32px;
            border-radius: 12px;
            text-decoration: none;
            margin-top: 20px;
            box-shadow: 0 4px 15px rgba(184, 102, 74, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-box">
            <h1>🪷 ASANAVISION</h1>
            <div>Real-Time Yoga Posture Detection & Biomechanical Feedback</div>
        </div>
        <h2>Vercel Build Status: Active & Operational</h2>
        <p>AsanaVision uses MediaPipe & OpenCV for real-time webcam pose detection.</p>
        <p>To use the interactive live camera application, visit Streamlit Community Cloud below:</p>
        <a class="btn" href="https://share.streamlit.io/" target="_blank">Launch Live App 🚀</a>
    </div>
</body>
</html>"""
        self.wfile.write(html_content.encode('utf-8'))
