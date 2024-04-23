from flask import Flask, render_template
from werkzeug.exceptions import HTTPException
import os
import shutil

def create_app():
    app = Flask(__name__)
    app.secret_key = 'DgYdiILpcV'
    UPLOAD_FOLDER = './uploads'
    STATIC_FOLDER = './web/static/images'
    
    app.config['STATIC_FOLDER'] = STATIC_FOLDER
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'mp4', 'flac'}
    
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    
    for filename in os.listdir(STATIC_FOLDER):
        if filename != 'loading.gif':
            file_path = os.path.join(STATIC_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    
    
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    

    return app

def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    return render_template('error.html', e=e), code

