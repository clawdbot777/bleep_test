import os
import sys
import logging
import json
import glob
import magic
import subprocess
import uuid
import time
import threading
import torch
import traceback
import shutil
import tempfile
import string
import re
from flask import request, jsonify, send_file
from app import app
from werkzeug.utils import secure_filename
# import whisperx

"""
Section 1 - Configuration related settings
1. Logging
2. Tracking job statusses
3. Variables
4. Directory setup
5. Confirming dependencies
6. Configuring file sizes and types
"""

# 1. Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 2. Track job statusses
job_status = {}

# 3. Variables
UPLOAD_FOLDER = '/tmp/uploads'
TEMP_FOLDER = '/tmp/uploads/tmp'
PROCESSED_FOLDER = '/tmp/uploads/processed'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 80 * 1024 * 1024 * 1024
CHUNK_SIZE = 20 * 1024 * 1024

# 4. Directory setup - ensure necessary directories exist
for folder in [UPLOAD_FOLDER, TEMP_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)
tempfile.tempdir = TEMP_FOLDER

# 5. Configure dependecies - confirm cuda availability
if torch.cuda.is_available():
    logger.debug(f"CUDA available: {torch.cuda.is_available()}")
    logger.debug(f"CUDA device: {torch.cuda.get_device_name(0)}")

# 6. Configure file sizes and types - allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

"""
Section 2 - Define functions
1. Save file
2. Finalize upload file
3. Get config details
4. Update config details
5. Channel mapping
6. Analyze media file
7. Extract audio from media file
8. Align file
9. Process file
10. Get file
11. Delete file
12. Get job status
13. Get job details
14. Get job config
15. Update job config
16. Get job config details
"""
# 1. Save file function
def save_file(temp_path, final_path, job_id):
    try:
        shutil.move(temp_path, final_path)
        job_status[job_id] = 'completed'
        logger.info(f"file saved successfully: {final_path} - job id: {job_id}")
    except Exception as e:
        job_status[job_id] = 'failed'
        logger.error(f"error saving file: {str(e)} - job id: {job_id}")

# 2. Finalize upload file
def finalize_file(temp_path, final_path, job_id):
    try:
        os.rename(temp_path, final_path)
        logger.info(f"File saved successfully: {final_path}")
        # Additional processing can be done here if needed
    except Exception as e:
        logger.error(f"Error finalizing file: {str(e)}")

# 3. Get config details
def get_config(job_id):
    config_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for job_id {job_id}: {str(e)}")
            return None
    else:
        logger.error(f"Config file not found for job_id: {job_id}")
        return None

# 4. Update config details
def update_config(job_id, updates):
    config = get_config(job_id)
    if config:
        config.update(updates)
        config_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)

# 5. Channel mapping
def get_channel_names():
    return {
        "mono": ["FC"],
        "stereo": ["FL", "FR"],
        "2.1": ["FL", "FR", "LFE"],
        "3.0": ["FL", "FR", "FC"],
        "3.0(back)": ["FL", "FR", "BC"],
        "4.0": ["FL", "FR", "FC", "BC"],
        "quad": ["FL", "FR", "BL", "BR"],
        "quad(side)": ["FL", "FR", "SL", "SR"],
        "3.1": ["FL", "FR", "FC", "LFE"],
        "5.0": ["FL", "FR", "FC", "BL", "BR"],
        "5.0(side)": ["FL", "FR", "FC", "SL", "SR"],
        "4.1": ["FL", "FR", "FC", "LFE", "BC"],
        "5.1": ["FL", "FR", "FC", "LFE", "BL", "BR"],
        "5.1(side)": ["FL", "FR", "FC", "LFE", "SL", "SR"],
        "6.0": ["FL", "FR", "FC", "BC", "SL", "SR"],
        "6.0(front)": ["FL", "FR", "FLC", "FRC", "SL", "SR"],
        "6.1": ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"],
        "6.1(back)": ["FL", "FR", "FC", "LFE", "BL", "BR", "BC"],
        "6.1(front)": ["FL", "FR", "LFE", "FLC", "FRC", "SL", "SR"],
        "7.0": ["FL", "FR", "FC", "BL", "BR", "SL", "SR"],
        "7.0(front)": ["FL", "FR", "FC", "FLC", "FRC", "SL", "SR"],
        "7.1": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
        "7.1(wide)": ["FL", "FR", "FC", "LFE", "BL", "BR", "FLC", "FRC"],
    }

# 6. Analyze media and select the most appropriate audio stream
def analyze_and_select_audio_stream(job_id):
    config = get_config(job_id)
    input_file = config.get('input_filename')
    if not input_file:
        raise ValueError(f"No input filename found in config for job_id: {job_id}")

    input_path = os.path.join(UPLOAD_FOLDER, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Use ffprobe to get stream information
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-show_format', input_path
    ]
    logger.debug("ffprobe command: " + ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe command failed: {result.stderr}")

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse ffprobe output: {e}\\nOutput: {result.stdout}")

    if 'streams' not in info:
        raise KeyError(f"No 'streams' key in ffprobe output. Full output: {info}")

    audio_streams = [stream for stream in info['streams'] if stream.get('codec_type') == 'audio']

    if not audio_streams:
        raise ValueError("No audio streams found in the input file.")

    # Apply prioritization rules
    def stream_priority(stream):
        codec = stream['codec_name']
        language = stream.get('tags', {}).get('language', '').lower()
        bit_rate = int(stream.get('bit_rate', 0))
        sample_rate = int(stream.get('sample_rate', 0))
        channels = int(stream.get('channels', 0))
        priority = 0
        if language == 'eng':
            priority += 1000
        if codec == 'ac3':
            if channels == 2:
                priority += 1
            else:
                priority += 500
        elif codec == 'dts':
            priority += 400
        elif codec == 'truehd':
            priority += 200
        elif codec == 'eac3':
            priority += 100

        # Add bit_rate priority if available
        if bit_rate:
            try:
                priority += float(bit_rate) / 1000000  # Convert to Mbps
            except ValueError:
                logger.warning(f"Invalid bit_rate value: {bit_rate}")

        # Add sample_rate priority if available
        if sample_rate:
            try:
                priority += float(sample_rate) / 1000  # Convert to kHz
            except ValueError:
                logger.warning(f"Invalid sample_rate value: {sample_rate}")
        logger.debug(f"priority: {priority}")
        return priority

    selected_stream = max(audio_streams, key=stream_priority)
    logger.debug(f"selected_stream: {selected_stream}")

    # Calculate audio_stream_index_nr
    audio_stream_index_nr = audio_streams.index(selected_stream)
    update_config(job_id, {'audio_stream_index_nr': audio_stream_index_nr})
    logger.debug(f"selected_stream: {selected_stream}")
    return selected_stream

# 7. Extract audio stream and center channel
def extract_audio_stream(job_id):
    config = get_config(job_id)
    input_file = config.get('input_filename')
    #selected_audio_stream = config.get('selected_audio_stream')
    audio_stream_index_nr = config.get('audio_stream_index_nr')
    logger.debug(f"DEBUG: input_file: {input_file}")
    logger.debug(f"DEBUG: audio_stream_index_nr: {audio_stream_index_nr}")
    if not input_file or audio_stream_index_nr is None:
        raise ValueError("Input file or selected audio stream not found in config")

    input_path = os.path.join(UPLOAD_FOLDER, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Get updated stream info for the converted file
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-select_streams', f'a:{audio_stream_index_nr}',
        input_path
    ]
    logger.debug("ffprobe command: " + ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe command failed: {result.stderr}")
    else:
        tmp_stream_info = json.loads(result.stdout)
        logger.debug(f"DEBUG: tmp_stream_info: {tmp_stream_info}")

    tmp_audio_stream = tmp_stream_info['streams'][0]  # Assume the first stream is our audio stream
    codec_name = tmp_audio_stream.get('codec_name')
    bit_rate = int(tmp_audio_stream.get('bit_rate', 0))
    sample_rate = int(tmp_audio_stream.get('sample_rate', 0))
    channels = int(tmp_audio_stream.get('sample_rate', 0))

    if codec_name == 'truehd':
        # Convert TrueHD to DTS
        audio_output_file = f"{os.path.splitext(input_file)[0]}_audio.dts"
        audio_output_path = os.path.join(UPLOAD_FOLDER, audio_output_file)
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", input_path,
            "-map", f"0:a:{audio_stream_index_nr}",
            "-c:a", "dca",
            "-sn",
            "-strict", "-2"
        ]
        if bit_rate:
            cmd.extend(["-b:a", f"{bit_rate}"])
        else:
            cmd.extend(["-b:a", "1509k"])
        if sample_rate:
            cmd.extend(["-ar", f"{sample_rate}"])
        cmd.extend([
            "-ac", "6",
            audio_output_path
        ])
        logger.debug("ffmpeg command to convert truehd to dts: " + ' '.join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio stream extraction FFmpeg command failed: {e.stderr}")
            raise
        update_config(job_id, {'audio_filename': audio_output_file})

    else:
        audio_output_file = f"{os.path.splitext(input_file)[0]}_audio.{codec_name}"
        audio_output_path = os.path.join(UPLOAD_FOLDER, audio_output_file)
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", input_path,
            "-map", f"0:a:{audio_stream_index_nr}",
            "-c:a", "copy",
            "-strict", "-2",
            audio_output_path
        ]
        logger.debug("ffmpeg audio extraction command: " + ' '.join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg command failed: {e.stderr}")
            raise
        update_config(job_id, {'audio_filename': audio_output_file})

    # Updated audio stream info from extracted audio stream
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams', '-show_format',
        '-select_streams', 'a:0',
        audio_output_path
    ]
    logger.debug("ffprobe command: " + ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe command failed: {result.stderr}")
    try:
        audio_stream_info = json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse ffprobe output: {result.stdout}")
        raise ValueError("Failed to parse ffprobe output")

    logger.debug(f"Raw ffprobe output: {result.stdout}")
    logger.debug(f"Parsed audio stream info: {audio_stream_info}")

    if 'streams' in audio_stream_info and audio_stream_info['streams']:
        update_config(job_id, {'audio_stream_info': audio_stream_info})
        logger.debug(f"Config after updating: {config}")
        logger.debug(f"Final audio stream info: {audio_stream_info}")
    else:
        logger.error("No streams found in audio_stream_info")
        raise ValueError("Failed to get audio stream info: No streams found")

    # Extract center channel from extracted audio stream
    tmp_audio_stream = audio_stream_info

    if not tmp_audio_stream or 'streams' not in tmp_audio_stream or not tmp_audio_stream['streams']:
        logger.error("Invalid or missing audio stream info in config")
        raise ValueError("Invalid or missing audio stream info in config")

    audio_stream_config = audio_stream_info['streams'][0]
    codec_name = audio_stream_config.get('codec_name')
    bit_rate = int(audio_stream_config.get('bit_rate', 0))
    sample_rate = int(audio_stream_config.get('sample_rate', 0))

    if codec_name == 'ac3':
        channel_output_file = f"{os.path.splitext(input_file)[0]}_center.{codec_name}"
    elif codec_name == 'dts':
        channel_output_file = f"{os.path.splitext(input_file)[0]}_center.wav"
    else:
        channel_output_file = f"{os.path.splitext(input_file)[0]}_center.ac3"

    channel_output_path = os.path.join(UPLOAD_FOLDER, channel_output_file)

    # Extract center channel command
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", audio_output_path,
        "-filter_complex", "[0:a]pan=mono|c0=FC[center]",
        "-map", "[center]"
    ]
    if codec_name == "dts":
        cmd.extend([
            "-c:a", "pcm_s32le"
        ])
    if bit_rate:
        cmd.extend([
            "-b:a", f"{bit_rate}"
        ])
    if sample_rate:
        cmd.extend([
            "-ar", f"{sample_rate}"
        ])
    cmd.extend([
        "-strict", "-2",
        channel_output_path
    ])
    logger.debug("ffmpeg command to extract center channel: " + ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Center channel extraction FFmpeg command failed: {e.stderr}")
        raise
    update_config(job_id, {'center_channel_file': channel_output_file})

    # Determine loudness of center channel, to be used to normalize later
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", channel_output_path,
        "-filter:a", "loudnorm=print_format=json",
        "-f", "null",
        "-"
    ]

    logger.debug(f"ffmpeg loudness command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg loudness command failed: {e.stderr}")
        raise

    # Check both stdout and stderr for JSON output
    output = result.stdout if result.stdout else result.stderr

    try:
        # Find the JSON part in the output
        json_start = output.find('{')
        json_end = output.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = output[json_start:json_end]
            loudness_info = json.loads(json_str)
        else:
            raise ValueError("No JSON data found in ffmpeg output")
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse loudness JSON: {e}")
        logger.debug(f"Raw output: {output}")
        raise

    input_i = loudness_info.get('input_i')

    if input_i is None:
        raise ValueError("input_i not found in loudness information")

    # Store all loudness information in the config
    update_config(job_id, {'loudness_info': input_i})
    logger.debug(f"Loudness info: {loudness_info}")

    return loudness_info

# 8. Transcribe center channel audio
def transcribe_audio(job_id):
    config = get_config(job_id)

    input_file = config.get('center_channel_file')
    logger.debug(f"DEBUG: input_file: {input_file}")
    if not input_file:
        return jsonify({'error': 'Transcription input filename not set. Did you complete the previous step to extract the audio stream?'}), 404

    input_path = os.path.join(UPLOAD_FOLDER, input_file)
    if not os.path.exists(input_path):
        return jsonify({'error': f'Input file not found: {input_file}'}), 404

    whisperx_settings = request.json
    device = whisperx_settings.get('device', "cuda" if torch.cuda.is_available() else "cpu")
    language = whisperx_settings.get('language', "en")
    batch_size = whisperx_settings.get('batch_size', 20)
    compute_type = whisperx_settings.get('compute_type', "float16")
    model = whisperx_settings.get('model', "large-v3")
    align_model = whisperx_settings.get('align_model', "WAV2VEC2_ASR_LARGE_LV60K_960H")

    json_file = f"{os.path.splitext(input_file)[0]}.json"
    srt_file = f"{os.path.splitext(input_file)[0]}.srt"

    cmd = [
        "whisperx",
        "--device", str(device),
        "--model", str(model),
        "--align_model", str(align_model),
        "--batch_size", str(batch_size),
        "--compute_type", str(compute_type),
        "--language", str(language),
        "--output_dir", str(UPLOAD_FOLDER),
        str(input_path)
    ]

    try:
        logger.debug(f"Starting transcribe command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug("Transcribe command completed successfully")
        # logger.debug(f"Stdout: {result.stdout}")
        # logger.debug(f"Stderr: {result.stderr}")

        updates = {'transcription_json': json_file, 'transcription_srt': srt_file}
        update_config(job_id, updates)
    except subprocess.CalledProcessError as e:
        logger.error(f"Transcode command failed: {e.stderr}")
        raise

    return True

# 9. Redact audio
def identify_profanity_timestamps(timestamps_data, profanity):
    profanity_timestamps = []
    for segment in timestamps_data["segments"]:
        for word in segment["words"]:
            word_without_punct = ''.join(char for char in word["word"].lower() if char not in set(string.punctuation))
            if word_without_punct in (sw.lower().strip(string.punctuation) for sw in profanity):
                start_time = word["start"] - 0.10
                end_time = word["end"] + 0.10
                profanity_timestamps.append({"start": start_time, "end": end_time})
    return profanity_timestamps

def get_non_profanity_intervals(profanity_timestamps, duration):
    non_profanity_intervals = []
    if not profanity_timestamps:
        return [{'start': 0, 'end': duration}]
    if profanity_timestamps[0]['start'] > 0:
        non_profanity_intervals.append({'start': 0, 'end': profanity_timestamps[0]['start']})
    for i in range(len(profanity_timestamps) - 1):
        non_profanity_intervals.append({
            'start': profanity_timestamps[i]['end'],
            'end': profanity_timestamps[i + 1]['start']
        })
    if profanity_timestamps[-1]['end'] < duration:
        non_profanity_intervals.append({'start': profanity_timestamps[-1]['end'], 'end': duration})
    return non_profanity_intervals

def create_ffmpeg_filter(profanity_timestamps, non_profanity_intervals, duration):
    dipped_vocals_conditions = '+'.join([f"between(t,{b['start']},{b['end']})" for b in profanity_timestamps])
    dipped_vocals_filter = f"[0]volume=0:enable='{dipped_vocals_conditions}'[main]"

    no_bleeps_conditions = '+'.join([f"between(t,{segment['start']},{segment['end']})" for segment in non_profanity_intervals[:-1]])

    if non_profanity_intervals:
        last_interval = non_profanity_intervals[-1]
        if last_interval['end'] == duration:
            no_bleeps_conditions += f"+gte(t,{last_interval['start']})"
        else:
            no_bleeps_conditions += f"+between(t,{last_interval['start']},{last_interval['end']})"

    dipped_bleep_filter = f"sine=f=800,volume=0.4,aformat=channel_layouts=mono+stereo,volume=0:enable='{no_bleeps_conditions}'[beep]"

    amix_filter = "[main][beep]amix=inputs=2:duration=first"

    filter_complex = ';'.join([
        dipped_vocals_filter,
        dipped_bleep_filter,
        amix_filter
    ])
    logger.debug(f"FFmpeg filter complex: {filter_complex}")
    return filter_complex

def load_srt(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def replace_words_in_srt(lines, profanity):
    # Separate single words and phrases
    single_words = [word for word in profanity if ' ' not in word and '"' not in word]
    phrases = [phrase.strip('"') for phrase in profanity if ' ' in phrase or '"' in phrase]

    # Sort profanity list by length (longest first)
    single_words = sorted(single_words, key=len, reverse=True)
    phrases = sorted(phrases, key=len, reverse=True)

    # Create patterns for single words and phrases
    word_pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in single_words) + r')\b', re.IGNORECASE)
    phrase_pattern = re.compile('|'.join(re.escape(phrase) for phrase in phrases), re.IGNORECASE)

    modified_lines = []
    redacted_words_count = 0
    for i, line in enumerate(lines, 1):
        if not line.strip().isdigit() and '-->' not in line:
            original_line = line
            # Replace phrases first
            line = phrase_pattern.sub(lambda m: '*' * len(m.group()), line)
            # Then replace individual words
            line = word_pattern.sub(lambda m: '*' * len(m.group()), line)
            if line != original_line:
                word_count = len(word_pattern.findall(original_line)) + len(phrase_pattern.findall(original_line))
                redacted_words_count += word_count
        modified_lines.append(line)

    logger.debug(f"Redacted words count: {redacted_words_count}")
    logger.debug(f"Modified lines: {modified_lines}")
    return modified_lines

def save_srt(output_path, lines):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

def apply_complex_filter_to_audio(center_channel_file, filter_complex, bitrate=None, sample_rate=None, codec=None):
    channel_file_name, channel_file_ext = os.path.splitext(os.path.basename(center_channel_file))
    output_file = f"{channel_file_name}_redacted{channel_file_ext}"
    output_path = os.path.join(UPLOAD_FOLDER, output_file)

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", center_channel_file,
        "-filter_complex", filter_complex,
        "-bitexact", "-ac", "1",
        "-strict", "-2"
    ]
    if bitrate:
        cmd.extend([
            "-b:a", str(bitrate)
        ])

    if sample_rate:
        cmd.extend([
            "-ar", str(sample_rate)
        ])

    if codec:
        cmd.extend([
            "-c:a", codec
        ])
    #cmd.extend(["-af", "loudnorm=I=-23:LRA=7:TP=-2:measured_I=-33.47:linear=true:print_format=summary"])
    cmd.extend([
        "-max_interleave_delta", "0", output_path
    ])

    try:
        logger.debug(f"Apply complext filter to center channel audio: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug("Applying complex filter to center channel completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Applying complext filter to center channel failed: {e.stderr}")
        raise
    return output_file

def normalize_center_audio(modified_audio_file, bitrate=None, sample_rate=None, codec=None, loudness_info=None):
    modified_audio_path = os.path.join(UPLOAD_FOLDER, modified_audio_file)
    normalized_channel_file_name, normalized_channel_file_ext = os.path.splitext(os.path.basename(modified_audio_file))
    normalized_output_file = f"{normalized_channel_file_name}_normalized{normalized_channel_file_ext}"
    normalized_output_path = os.path.join(UPLOAD_FOLDER, normalized_output_file)
    logger.debug(f"normalized_output_path: {modified_audio_path}")
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", modified_audio_path,
        "-bitexact", "-ac", "1",
        "-strict", "-2",
        "-af", f"loudnorm=I=-24:LRA=7:TP=-2:measured_I={loudness_info}:linear=true:print_format=summary,volume=0.90"
    ]

    if bitrate:
        cmd.extend([
            "-b:a", str(bitrate)
        ])

    if sample_rate:
        cmd.extend([
            "-ar", str(sample_rate)
        ])

    if codec:
        cmd.extend([
            "-c:a", codec
        ])
    cmd.extend([
        "-max_interleave_delta", "0", normalized_output_path
    ])

    try:
        logger.debug(f"Normalize center channel audio: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug("Normalized center channel completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Normalization to center channel failed: {e.stderr}")
        raise
    return normalized_output_file

def redact_audio(job_id):
    config = get_config(job_id)

    if not config:
        raise ValueError('Config not found for the given job_id')

    json_filename = config.get('transcription_json')
    srt_filename = config.get('transcription_srt')
    audio_filename = config.get('center_channel_file')
    filter_list_filename = os.path.join('/app', 'app', 'filter_list.txt')

    if not all([json_filename, srt_filename, audio_filename]) or not os.path.exists(filter_list_filename):
        raise ValueError('Missing required filenames or filter list not found')

    selected_stream = config.get('audio_stream_info', {})
    audio_stream_info = selected_stream.get('streams', [])[0] if selected_stream.get('streams') else {}
    codec = audio_stream_info.get('codec_name')
    bit_rate = audio_stream_info.get('bit_rate', None)
    sample_rate = audio_stream_info.get('sample_rate', None)
    channels = audio_stream_info.get('channels')
    loudness_info = config.get('loudness_info', None)

    logger.debug(f"codec: {codec}")
    logger.debug(f"bit_rate: {bit_rate}")
    logger.debug(f"sample_rate: {sample_rate}")
    logger.debug(f"channels: {channels}")

    json_path = os.path.join(UPLOAD_FOLDER, json_filename)
    srt_path = os.path.join(UPLOAD_FOLDER, srt_filename)
    audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)

    # Load filter words
    with open(filter_list_filename, 'r') as f:
        profanity = [line.strip() for line in f]

    # Load JSON data
    with open(json_path) as f:
        timestamps_data = json.load(f)

    # Identify swearing timestamps
    profanity_timestamps = identify_profanity_timestamps(timestamps_data, profanity)

    # Get audio duration
    duration = timestamps_data["segments"][-1]["end"]

    # Get non-swearing intervals
    non_profanity_intervals = get_non_profanity_intervals(profanity_timestamps, duration)

    # Create FFmpeg filter
    filter_complex = create_ffmpeg_filter(profanity_timestamps, non_profanity_intervals, duration)

    # Redact subtitles
    srt_lines = load_srt(srt_path)
    modified_lines = replace_words_in_srt(srt_lines, profanity)

    # Generate redact SRT
    srt_base_name = srt_filename.rsplit('.', 1)[0]
    output_srt_filename = f"{srt_base_name}_redacted_subtitle.srt"
    output_srt_path = os.path.join(UPLOAD_FOLDER, output_srt_filename)
    save_srt(output_srt_path, modified_lines)

    update_config(job_id, {'redacted_srt': output_srt_filename})

    # Apply complex filter
    modified_audio_file = apply_complex_filter_to_audio(audio_path, filter_complex, bit_rate, sample_rate, codec)
    normalized_audio_path = normalize_center_audio(modified_audio_file, bit_rate, sample_rate, codec, loudness_info)
    # Update config with redacted audio file
    if channels <= 2:
        update_config(job_id, {'redacted_audio_stream_final': modified_audio_file})
    else:
        update_config(job_id, {'redacted_channel_FC': normalized_audio_path})

    return "Redaction completed successfully"

# 10. Combine audio streams
"""
ffmpeg -i Chosen_test.mkv -i Chosen_center_channel_redacted_normalized.wav -filter_complex "
[0:a]aformat=channel_layouts=5.1(side)[original];
[1:a]aformat=channel_layouts=mono[redacted];
[original][redacted]amerge=inputs=2[merged];
[merged]pan=5.1(side)|FL=c0|FR=c1|FC=c6|LFE=c3|SL=c4|SR=c5[final]
" -map 0:v -map "[final]" -c:v copy -c:a dca -b:a 3266k -strict -2 Chosen_test_output.mkv
"""
def combine_media_file(job_id):
    config = get_config(job_id)

    if not config:
        raise ValueError('Config not found for the given job_id')

    redacted_audio_filename = config.get('redacted_channel_FC')
    redacted_srt = config.get('redacted_srt')
    input_media_filename = config.get('input_filename')
    output_filename, output_file_ext = os.path.splitext(os.path.basename(input_media_filename))
    output_file = f"{output_filename}_final{output_file_ext}"

    if not all([redacted_audio_filename, input_media_filename, output_file]):
        raise ValueError('Missing required filenames')

    redacted_audio_path = os.path.join(UPLOAD_FOLDER, redacted_audio_filename)
    redacted_srt_path = os.path.join(UPLOAD_FOLDER, redacted_srt)
    input_media_path = os.path.join(UPLOAD_FOLDER, input_media_filename)
    output_path = os.path.join(UPLOAD_FOLDER, output_file)

    original_audio_index = config.get('audio_stream_index_nr')
    selected_stream = config.get('audio_stream_info', {})
    audio_stream_info = selected_stream.get('streams', [])[0] if selected_stream.get('streams') else {}
    codec = audio_stream_info.get('codec_name')
    bit_rate = audio_stream_info.get('bit_rate', None)
    sample_rate = audio_stream_info.get('sample_rate', None)

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", input_media_path,
        "-i", redacted_audio_path,
        "-i", redacted_srt_path,
        "-filter_complex",
        "[0:a]aformat=channel_layouts=5.1(side)[original];"
        "[1:a]aformat=channel_layouts=mono[redacted];"
        "[original][redacted]amerge=inputs=2[merged];"
        "[merged]pan=5.1(side)|FL=c0|FR=c1|FC=c6|LFE=c3|SL=c4|SR=c5[final]",
        "-map", "0:v",  # Map video from the original input
        "-map", "[final]",  # Map the final audio after filtering
        f"-map", f"0:a:{original_audio_index}",
        "-map", "2",    # Map redacted subtitles
        "-c:v", "copy",  # Copy the video codec
        "-c:a:0", codec
    ]
    if bit_rate:
         cmd.extend([
             "-b:a", f"{bit_rate}"
         ])
    if sample_rate:
         cmd.extend([
             "-ar", f"{sample_rate}"
         ])
    cmd.extend([
        "-c:a:1", "copy"
    ])
    if output_file_ext == ".mp4":
        cmd.extend([
            "-c:s", "mov_text"
        ])
    else:
        cmd.extend([
            "-c:s", "srt"
        ])
    cmd.extend([
        "-metadata:s:a:0", "title=Family audio",
        "-metadata:s:a:0", "language=eng",
        "-disposition:a:0", "default",
        "-metadata:s:a:1", "title=Original audio",
        "-metadata:s:a:1", "language=eng",
        "-disposition:a:1", "0",
        "-metadata:s:s:0", "title=Redacted subtitles",
        "-metadata:s:s:0", "language=eng",
        "-disposition:s:0", "default",
        "-strict", "-2"
    ])
    if output_file_ext == ".mp4":
        cmd.extend([
            "-f", "mp4"
        ])
    cmd.extend([
        output_path
    ])

    try:
        logger.debug(f"Combine media file: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        update_config(job_id, {'final_output': output_file})
        logger.debug("Combine media file completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Combine media file failed: {e.stderr}")
        raise

    return "Combine media file completed successfully"

def cleanup_job_files(job_id):
    config = get_config(job_id)
    if not config:
        raise ValueError(f"No configuration found for job_id: {job_id}")

    # Get the original filename and the final output filename
    original_filename = config.get('original_filename')
    input_filename = config.get('input_filename')
    input_prefix = f"{os.path.splitext(input_filename)[0]}"
    final_output = config.get('final_output')

    final_output_path = os.path.join(UPLOAD_FOLDER, final_output)
    if not os.path.exists(final_output_path):
        raise FileNotFoundError(f"Final output file not found: {final_output_path}")

    # Rename the final output file to the original filename
    original_filename_path = os.path.join(UPLOAD_FOLDER, original_filename)
    os.rename(final_output_path, original_filename_path)

    # List of file patterns to remove
    patterns_to_remove = [
        f"{job_id}_*",  # All files starting with the job_id
        f"{input_prefix}_center*.*",    # All audio files
        f"{input_prefix}_audio*.*",    # All audio files
    ]

    # Remove files matching the patterns
    for pattern in patterns_to_remove:
        for file in glob.glob(os.path.join(UPLOAD_FOLDER, pattern)):
            if os.path.isfile(file) and file != original_filename_path:
                os.remove(file)
                logger.debug(f"Removed file: {file}")

    logger.info(f"Cleanup completed for job {job_id}. Final file renamed to {original_filename}")
    return original_filename

"""
Section 3 - Define routes
1. List availble media files on server
2. Initialize job and create empty json config file
3. Select media file saved on remote server
4. Get job status
"""
# 1. List available media files on server
@app.route('/api/list_files', methods=['GET'])
def list_files():
    video_files = []

    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            video_files.append(filename)
    logger.debug(f"list_files output: {video_files}")
    return jsonify({'files': video_files})

# 2. Initialize the job and create the empty json config file
@app.route('/api/initialize_job', methods=['POST'])
def initialize_job():
    job_id = str(uuid.uuid4())
    config = {
        'job_id': job_id
    }

    config_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)

    logger.debug(f"successfully initialized new job - {job_id}")
    return jsonify({'job_id': job_id})

# 3. Select media file saved on remote server
@app.route('/api/select_remote_file', methods=['POST'])
def select_remote_file():
    if 'filename' not in request.json or 'job_id' not in request.json:
        return jsonify({'error': 'Missing filename or job_id'}), 400

    filename = request.json['filename']
    job_id = request.json['job_id']

    # Secure the filename
    secure_name = secure_filename(filename)
    file_path = os.path.join(UPLOAD_FOLDER, secure_name)

    # Check if the file exists
    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found on server'}), 404

    # Check if it's a video file
    try:
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        if not file_type.startswith('video/'):
            logger.debug(f"Selected file is not a video")
            return jsonify({'error': 'Selected file is not a video'}), 400
    except Exception as e:
        logger.error(f"Error checking file type: {str(e)}")
        return jsonify({'error': 'Error verifying file type'}), 500

    # If extension is not in allowed list, double-check
    if not allowed_file(secure_name):
        logger.debug(f"File extension not in allowed list, but mime type is video: {secure_name}")
        return jsonify({'error': 'File extension not allowlisted, but mime type is video'})

    updates = {'original_filename': filename, 'input_filename': filename}
    update_config(job_id, updates)
    logger.debug(f"remote file selected: {filename} - job id: {job_id}")
    return jsonify({'remote_file': filename})

# 4. Upload media file
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.error(f"no file part in the request - job id: {job_id}")
        return jsonify({'error': 'no file part'}), 400

    file = request.files['file']
    job_id = request.form.get('job_id')

    if file.filename == '':
        logger.error("no file selected - job id: {job_id}")
        return jsonify({'error': 'No file selected'}), 400

    if file and job_id and allowed_file(file.filename):
        logger.debug(f"file confirmed: {file.filename} - job id: {job_id}")
        filename = secure_filename(f"{job_id}_input_{file.filename}")
        upload_path = os.path.join(UPLOAD_FOLDER, filename)

        updates = {'original_filename': file.filename}
        update_config(job_id, updates)
        logger.debug(f"preparing to save file to: {upload_path} - job id: {job_id}")

        # Write directly to the final destination file
        try:
            with open(upload_path, 'wb') as f:
                while True:
                    chunk = file.read(app.config['CHUNK_SIZE'])
                    if not chunk:
                        break
                    f.write(chunk)

            updates = {'input_filename': filename}
            update_config(job_id, updates)
            logger.info(f"File saved successfully: {upload_path}")
            return jsonify({'job_id': job_id, 'filename': filename}), 200
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return jsonify({'error': 'File upload failed'}), 500

    else:
        logger.error(f"file type not allowed: {file.filename} - job id: {job_id}")
        return jsonify({'error': 'File type not allowed'}), 400

# 5. Analyze media, select and convert audio stream if required
@app.route('/api/analyze_and_select_audio', methods=['POST'])
def analyze_and_select_audio():
    data = request.json
    job_id = data['job_id']

    try:
        config = get_config(job_id)
        if not config:
            raise ValueError(f"No configuration found for job_id: {job_id}")

        if 'input_filename' not in config:
            raise KeyError(f"'input_filename' not found in configuration for job_id: {job_id}")

        selected_stream = analyze_and_select_audio_stream(job_id)
        return jsonify({
            'status': 'success',
            'selected_stream': selected_stream
        })
    except Exception as e:
        error_msg = f"Error in analyze_and_select_audio: {str(e)}\n"
        error_msg += f"Job ID: {job_id}\n"
        error_msg += f"Config: {config}\n"
        error_msg += f"Traceback: {traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': error_msg
        }), 500

# 6. Extract audio channel
@app.route('/api/extract_audio', methods=['POST'])
def extract_audio():
    data = request.json
    job_id = data['job_id']

    try:
        config = get_config(job_id)
        if not config:
            raise ValueError(f"No configuration found for job_id: {job_id}")

        if 'input_filename' not in config:
            raise KeyError(f"'input_filename' not found in configuration for job_id: {job_id}")

        extract_audio_stream(job_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        error_msg = f"Error in extract_audio: {str(e)}\n"
        error_msg += f"Job ID: {job_id}\n"
        error_msg += f"Config: {config}\n"
        error_msg += f"Traceback: {traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': error_msg
        }), 500

# 7. Transcribe center channel audio
@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    job_id = data['job_id']

    try:
        config = get_config(job_id)
        if not config:
            raise ValueError(f"No configuration found for job_id: {job_id}")

        if 'center_channel_file' not in config:
            raise KeyError(f"'channel_output_file' not found in configuration for job_id: {job_id}")
        logger.debug(f"DEBUG: config: {config}")
        transcribe_audio(job_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        error_msg = f"Error transcribing audio: {str(e)}\n"
        error_msg += f"Job ID: {job_id}\n"
        error_msg += f"Config: {config}\n"
        error_msg += f"Traceback: {traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': error_msg
        }), 500

# 8. Redact center channel
@app.route('/api/redact', methods=['POST'])
def api_redact():
    data = request.json
    job_id = request.json['job_id']

    try:
        config = get_config(job_id)
        if not config:
            raise ValueError(f"No configuration found for job_id: {job_id}")

        if 'transcription_json' not in config:
            raise KeyError(f"'transcription_json' not found in configuration for job_id: {job_id}")

        redact_audio(job_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        error_msg = f"Error redacting center channel audio: {str(e)}\n"
        error_msg += f"Job ID: {job_id}\n"
        error_msg += f"Config: {config}\n"
        error_msg += f"Traceback: {traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({
            'error': "error",
            'details': str(e),
            'job_id': error_msg
        }), 500

# 9. Combine redacted center channel with original audio stream
@app.route('/api/combine_media', methods=['POST'])
def combine_media():
    data = request.json
    job_id = request.json['job_id']

    try:
        config = get_config(job_id)
        if not config:
            raise ValueError(f"No configuration found for job_id: {job_id}")

        combine_media_file(job_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        error_msg = f"Error combining audio and video media files: {str(e)}\n"
        error_msg += f"Job ID: {job_id}\n"
        error_msg += f"Config: {config}\n"
        error_msg += f"Traceback: {traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({
            'error': "error",
            'details': str(e),
            'job_id': error_msg
        }), 500

# 10. Clean-up temporary files
@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    data = request.json
    job_id = request.json['job_id']

    try:
        config = get_config(job_id)
        if not config:
            raise ValueError(f"No configuration found for job_id: {job_id}")

        cleanup_job_files(job_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        error_msg = f"Error cleaning up temporary files: {str(e)}\n"
        error_msg += f"Job ID: {job_id}\n"
        error_msg += f"Config: {config}\n"
        error_msg += f"Traceback: {traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({
            'error': "error",
            'details': str(e),
            'job_id': error_msg
        }), 500
