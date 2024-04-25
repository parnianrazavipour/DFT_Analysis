from flask import Blueprint
from flask import Flask, request, jsonify
from flask import render_template, redirect, url_for
from flask import current_app as app
from werkzeug.utils import secure_filename
import os
from .process_file import process_file
from .process_file import load_audio
from .process_file import process_brute_force_dft
from .process_file import create_combined_signal
import librosa
from flask import abort, Response
import os
from flask import render_template, session, redirect, url_for
from flask import session, request, jsonify
import logging

from datetime import datetime
import soundfile as sf

import numpy as np


ORIGIN = ['']

main = Blueprint('main', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@main.route('/')
def index():
    session.pop('waves', None)
    session.pop('current_signal', None)
    session.pop('current_plot', None)
    session.pop('start', None)
    session.pop('end', None)

    waves_info = "No waves added"
    return render_template('index.html', waves_info=waves_info)


@main.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify(error="No file provided"), 400
    if not allowed_file(file.filename):
        return jsonify(error="File type not supported"), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    signal, sr = librosa.load(filepath, sr=None)  
    data = signal.tolist() 
    return jsonify(filename=filename, signal=data, sampleRate=sr)




@main.route('/figures', methods=['GET'])
def figures():
    if 'waves' in session and session['waves'] and ('current_signal' in session) and session['current_signal'] :
        rate, name = session['current_signal']
        # signal, rate = create_combined_signal(session['waves'])
        filename = "combined_signal_"+name+".wav"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # signal, rate = load_audio(file_path)
        
        time_domain_image, frequency_domain_image = process_file(file_path)
        return render_template('figures.html', time_image=time_domain_image, freq_image=frequency_domain_image)
    return 'No waves to process', 400


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['wav', 'mp3', 'flac']


from flask import send_from_directory
@main.route('/process/<filename>', methods=['GET'])
def process(filename):
    start_sec = request.args.get('start', type=float)
    end_sec = request.args.get('end', type=float)

    print("double checking:",start_sec,end_sec )
    try:
        session['start']= start_sec
        session['end'] = end_sec
    except Exception as e:
        print("error start end!!")


    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print("hi ! filename",filename,"file_path", file_path )
    time_image, freq_image = process_file(file_path,start_sec,end_sec )
    return render_template('upload_figures.html', filename=filename, time_image=time_image, freq_image=freq_image)




@main.route('/process_xcorr', methods=['GET'])
def process_xcorr():
    if 'current_signal' in session and session['current_signal'] :
        rate , name  = session['current_signal']  
        filename = "combined_signal_"+name+".wav"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        signal, rate = load_audio(file_path)


        signal = np.array(signal)
        
        plot_path = os.path.join(app.config['STATIC_FOLDER'], 'signal_created_xcorr_'+name+'.png')

        
        if not os.path.exists(plot_path):
            try:
                
                process_brute_force_dft(signal, rate, plot_path)
            except Exception as e:
                app.logger.error(f"Error processing the XCorr plot: {str(e)}")
                abort(500, description="Error processing the XCorr plot")
        
        if os.path.exists(plot_path):
            with open(plot_path, 'rb') as f:
                img = f.read()
            return Response(img, mimetype='image/png')
        else:
            app.logger.error("Processed XCorr plot file not found")
            abort(404, description="Processed XCorr plot file not found")
    
    abort(404, description="No current signal to process")





@main.route('/add_wave', methods=['POST'])
def add_wave():
    try:
        freq = float(request.form['freq'])
        sampling_rate = int(request.form['sampling_rate'])
        duration = float(request.form['duration'])
        weight = float(request.form['weight'])
    except ValueError as e:
        return jsonify({'error': 'Invalid input data', 'details': str(e)}), 400

    new_wave = {'freq': freq, 'rate': sampling_rate, 'duration': duration, 'weight': weight}

    if 'waves' not in session:
        session['waves'] = []

    session['waves'].append(new_wave)
    session.modified = True  

    waves_info = "<br>".join([f"Freq: {w['freq']}, Rate: {w['rate']}, Duration: {w['duration']}, Weight: {w['weight']}" for w in session['waves']])
    formula = " + ".join([f"sin({w['freq']}Hz) * {w['weight']}" for w in session['waves']])
    full_info = f"{waves_info}<br>Combined Signal Formula: {formula}"

    return jsonify({'waves_info': full_info})




@main.route('/create_signal', methods=['GET'])
def create_signal():
    if 'waves' in session and session['waves']:
        print("last signal created:", session['waves'])
        signal, rate = create_combined_signal(session['waves'])
        name = datetime.now().strftime("%Y%m%d%H%M%S%f")
        session['current_signal'] = (rate, name)

        filename = "combined_signal_"+name+".wav"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        from scipy.io.wavfile import write

        # sf.write(file_path, signal, rate)
        write(file_path, rate, signal)
        
        return jsonify({'signal': signal.tolist(), 'success': True})
    return jsonify({'error': 'No waves to process', 'success': False}), 400



@main.route('/reset_waves', methods=['POST'])
def reset_waves():
    session.pop('waves', None)
    session.pop('current_signal', None)
    session.pop('current_plot', None)
    return jsonify({'message': 'Waves and plots reset'}), 200

@main.route('/process_xcorr_uploaded/<filename>', methods=['GET'])
def process_xcorr_uploaded(filename):
    print("filename:", filename)
    start_sec = session.get('start')
    end_sec = session.get('end')
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    signal, rate = load_audio(file_path)
    
    if start_sec is not None and end_sec is not None:
        start_sample = int(float(start_sec) * rate)
        end_sample = int(float(end_sec) * rate)
        signal = signal[start_sample:end_sample]
        plot_filename = f"xcorr_plot_{start_sec}_{end_sec}_{filename}.png"
    else:
        plot_filename = f"xcorr_plot_{filename}.png"

    plot_path = os.path.join(app.config['STATIC_FOLDER'], plot_filename)
    
    if not os.path.exists(plot_path):
        try:
            process_brute_force_dft(signal, rate, plot_path)
        except Exception as e:
            app.logger.error(f"Error processing the XCorr plot: {str(e)}")
            abort(500, description="Error processing the XCorr plot")
    
    if os.path.exists(plot_path):
        with open(plot_path, 'rb') as f:
            img = f.read()
        return Response(img, mimetype='image/png')
    
    abort(404, description="XCorr plot file not found")
