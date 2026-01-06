# app.py - 统一的后端服务（适配Render部署）
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import datetime
import random
import json
import os
import sys
import io
import logging
import struct
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import re
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq


# ============================================================
# 配置和初始化
# ============================================================

# 彻底解决编码问题
def setup_encoding():
    """设置编码以避免字符问题"""
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
        os.environ['PYTHONIOENCODING'] = 'utf-8'


setup_encoding()

# 禁用所有不需要的日志
werkzeug_log = logging.getLogger('werkzeug')
werkzeug_log.disabled = True
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

# 创建Flask应用
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 数据源基础URL
BASE_URL = "http://58.57.159.186:30200"

# IMU相关配置
SAMPLE_RATE = 50  # Hz
FRAME_HEADER = b'\x55\xaa'
FRAME_LEN = 160
WINDOW_SIZE = 10 * 60 * SAMPLE_RATE  # 10分钟窗口大小


# ============================================================
# 通用工具函数
# ============================================================

def format_response(status="success", data=None, message=None, **kwargs):
    """标准化响应格式"""
    response = {
        "status": status,
        "timestamp": datetime.datetime.now().isoformat(),
        **kwargs
    }
    if data is not None:
        response["data"] = data
    if message is not None:
        response["message"] = message
    return response


def validate_time_format(time_str):
    """验证时间格式是否为YYYYMMDDHHMM"""
    try:
        datetime.datetime.strptime(time_str, "%Y%m%d%H%M")
        return True
    except ValueError:
        return False


# ============================================================
# 风浪数据功能（原getwindwavedata.py）
# ============================================================

def generate_mock_wind_wave_data(st1, st2, dataname):
    """生成模拟风浪数据"""
    try:
        start_dt = datetime.datetime.strptime(st1, "%Y%m%d%H%M")
        end_dt = datetime.datetime.strptime(st2, "%Y%m%d%H%M")

        data = []
        current_dt = start_dt

        while current_dt <= end_dt:
            if dataname == "wind":
                data.append({
                    "sdt": current_dt.strftime("%Y%m%d%H%M"),
                    "df": 5 + random.random() * 10,
                    "wd": random.random() * 360,
                    "ws": 5 + random.random() * 10
                })
            elif dataname == "wave":
                data.append({
                    "sdt": current_dt.strftime("%Y%m%d%H%M"),
                    "avgH": 0.5 + random.random() * 2,
                    "maxH": 1 + random.random() * 3
                })
            else:
                data.append({
                    "sdt": current_dt.strftime("%Y%m%d%H%M"),
                    "value": random.random() * 100
                })

            current_dt += datetime.timedelta(hours=1)

        return data
    except Exception as e:
        print(f"生成模拟数据错误: {e}")
        return []


def get_wind_wave_data(st1: str, st2: str, classic: int, dataname: str):
    """获取风浪数据"""
    try:
        url = f"{BASE_URL}/getdata/getwindwavedata"
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0'
        }

        payload = {
            "sdt1": st1,
            "sdt2": st2,
            "classic": classic,
            "dataname": dataname
        }

        r = requests.post(url, json=payload, headers=headers, timeout=15)

        if r.status_code == 200:
            try:
                response_data = r.json()
                data = response_data.get("data", [])
                return {
                    "status": "success",
                    "source": "api",
                    "count": len(data),
                    "data": data
                }
            except json.JSONDecodeError:
                # 使用模拟数据
                mock_data = generate_mock_wind_wave_data(st1, st2, dataname)
                return {
                    "status": "warning",
                    "source": "mock",
                    "count": len(mock_data),
                    "data": mock_data
                }
        else:
            # 使用模拟数据作为备用
            mock_data = generate_mock_wind_wave_data(st1, st2, dataname)
            return {
                "status": "warning",
                "source": "mock",
                "count": len(mock_data),
                "data": mock_data
            }

    except Exception as e:
        print(f"获取{dataname}数据失败: {e}")
        mock_data = generate_mock_wind_wave_data(st1, st2, dataname)
        return {
            "status": "error",
            "source": "mock",
            "count": len(mock_data),
            "data": mock_data
        }


# ============================================================
# IMU数据功能（原getimudata.py）
# ============================================================

def get_gnss_data_names(year, month, day, hour, classic=None):
    """获取GNSS数据文件名列表"""
    try:
        r = requests.post(
            f"{BASE_URL}/getdata/getgnssdatanames",
            json={"year": year, "month": month, "day": day, "hour": hour},
            timeout=10
        )
        r.raise_for_status()
        files = r.json().get("files", [])
        return files
    except Exception as e:
        print(f"查询文件名失败: {e}")
        return []


def get_bin_bytes(sdt):
    """获取二进制文件内容"""
    try:
        r = requests.get(f"{BASE_URL}/getdata/getGnssData/{sdt}", timeout=10)
        if r.status_code == 200:
            return r.content
        elif r.status_code == 404:
            print(f"文件不存在: {sdt}")
            return None
        else:
            print(f"获取失败，状态码: {r.status_code}")
            return None
    except Exception as e:
        print(f"获取文件失败: {e}")
        return None


def parse_frame(data: bytes):
    """解析单帧数据"""
    frame_data = {
        'timestamp': struct.unpack_from('<I', data, 3)[0],
        'week': struct.unpack_from('<H', data, 7)[0],
        'accX_m_s2': struct.unpack_from('<i', data, 27)[0] * 0.000001,
        'accY_m_s2': struct.unpack_from('<i', data, 31)[0] * 0.000001,
        'accZ_m_s2': struct.unpack_from('<i', data, 35)[0] * 0.000001,
        'gyroX_rad_s': struct.unpack_from('<i', data, 39)[0] * 0.000001,
        'gyroY_rad_s': struct.unpack_from('<i', data, 43)[0] * 0.000001,
        'gyroZ_rad_s': struct.unpack_from('<i', data, 47)[0] * 0.000001,
        'roll_deg': struct.unpack_from('<i', data, 51)[0] * 0.000001,
        'pitch_deg': struct.unpack_from('<i', data, 55)[0] * 0.000001,
        'yaw_deg': struct.unpack_from('<i', data, 59)[0] * 0.000001,
    }
    return frame_data


def parse_bin_bytes(content: bytes, base_time: datetime.datetime):
    """解析整个二进制数据"""
    frames = []
    i = 0
    frame_count = 0

    while i < len(content) - FRAME_LEN:
        if content[i:i + 2] == FRAME_HEADER:
            frame_data = parse_frame(content[i:i + FRAME_LEN])

            time_offset = frame_count * (1.0 / SAMPLE_RATE)
            frame_time = base_time + datetime.timedelta(seconds=time_offset)
            frame_data['time_str'] = frame_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            frame_data['timestamp_seconds'] = frame_time.timestamp()

            frames.append(frame_data)
            i += FRAME_LEN
            frame_count += 1
        else:
            i += 1

    return pd.DataFrame(frames)


def extract_timestamp_from_filename(filename):
    """从文件名提取时间戳"""
    match = re.search(r'data_(\d{12})\.bin', filename)
    if match:
        timestamp = match.group(1)
        if len(timestamp) == 12 and timestamp.isdigit():
            return timestamp
    return None


def acceleration_to_displacement(acceleration, sample_rate=SAMPLE_RATE):
    """频域双重积分：加速度 -> 位移"""
    n = len(acceleration)
    if n == 0 or np.std(acceleration) < 1e-10:
        return np.zeros_like(acceleration)

    acceleration = acceleration - np.mean(acceleration)
    window = np.hanning(n)
    acceleration_windowed = acceleration * window

    fft_acc = np.fft.fft(acceleration_windowed)
    frequencies = np.fft.fftfreq(n, 1 / sample_rate)
    omega = 2 * np.pi * frequencies

    min_freq = 0.1
    omega_threshold = 2 * np.pi * min_freq
    omega_sq = np.zeros_like(omega, dtype=complex)

    for i, w in enumerate(omega):
        omega_sq[i] = -omega_threshold ** 2 if abs(w) < omega_threshold else -w ** 2

    fft_disp = fft_acc / omega_sq
    fft_disp[0] = 0

    displacement = np.real(np.fft.ifft(fft_disp))

    window_compensation = np.mean(window)
    if window_compensation > 0:
        displacement /= window_compensation

    return displacement


def gaussian(x, a, mu, sigma):
    """高斯函数"""
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def gaussian_fit_displacement(displacement):
    """高斯拟合"""
    if len(displacement) == 0 or np.std(displacement) < 1e-10:
        return 0.0, False

    hist, bin_edges = np.histogram(displacement, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    try:
        initial_guess = [np.max(hist), np.mean(displacement), np.std(displacement)]
        popt, _ = curve_fit(gaussian, bin_centers, hist, p0=initial_guess, maxfev=5000)
        _, _, sigma_fit = popt
        return float(sigma_fit), True
    except:
        return float(np.std(displacement)), False


def extract_dominant_frequency(acceleration, sample_rate=SAMPLE_RATE):
    """提取主频率和周期"""
    n = len(acceleration)
    if n < 10 or np.std(acceleration) < 1e-10:
        return 0.0, 0.0

    acceleration = acceleration - np.mean(acceleration)
    window = np.hanning(n)
    acceleration_windowed = acceleration * window

    fft_values = np.fft.fft(acceleration_windowed)
    frequencies = np.fft.fftfreq(n, 1 / sample_rate)

    magnitude = np.abs(fft_values)
    positive_freq_mask = frequencies > 0
    positive_freqs = frequencies[positive_freq_mask]
    positive_magnitude = magnitude[positive_freq_mask]

    if len(positive_freqs) == 0:
        return 0.0, 0.0

    min_freq_threshold = 0.1
    valid_mask = positive_freqs > min_freq_threshold
    if not np.any(valid_mask):
        return 0.0, 0.0

    valid_freqs = positive_freqs[valid_mask]
    valid_magnitude = positive_magnitude[valid_mask]

    dominant_idx = np.argmax(valid_magnitude)
    dominant_freq = valid_freqs[dominant_idx]
    period = 1.0 / dominant_freq if dominant_freq > 0 else 0.0

    return float(dominant_freq), float(period)


def process_window_data(window_df):
    """处理单个10分钟窗口的数据"""
    if len(window_df) == 0:
        return None

    window_start_time = window_df.iloc[0]['time_str']
    acc_east = window_df['accY_m_s2'].values
    acc_north = window_df['accX_m_s2'].values
    acc_up = window_df['accZ_m_s2'].values

    disp_east = acceleration_to_displacement(acc_east)
    disp_north = acceleration_to_displacement(acc_north)
    disp_up = acceleration_to_displacement(acc_up)

    sigma_east, _ = gaussian_fit_displacement(disp_east)
    sigma_north, _ = gaussian_fit_displacement(disp_north)
    sigma_up, _ = gaussian_fit_displacement(disp_up)

    freq_east, period_east = extract_dominant_frequency(acc_east)
    freq_north, period_north = extract_dominant_frequency(acc_north)
    freq_up, period_up = extract_dominant_frequency(acc_up)

    result = {
        "window_start_time": window_start_time,
        "swing_displacement": {
            "east": round(sigma_east, 6),
            "north": round(sigma_north, 6),
            "up": round(sigma_up, 6)
        },
        "dominant_frequency": {
            "east": round(freq_east, 4),
            "north": round(freq_north, 4),
            "up": round(freq_up, 4)
        },
        "swing_period": {
            "east": round(period_east, 2),
            "north": round(period_north, 2),
            "up": round(period_up, 2)
        }
    }

    return result


def process_imu_data(st1: str, st2: str, classic=None):
    """处理IMU数据"""
    dt_start = datetime.datetime.strptime(st1, "%Y%m%d%H%M")
    dt_end = datetime.datetime.strptime(st2, "%Y%m%d%H%M")

    all_files_info = []
    current_hour = dt_start.replace(minute=0, second=0, microsecond=0)
    end_hour = dt_end.replace(minute=0, second=0, microsecond=0)

    while current_hour <= end_hour:
        year, month, day, hour = current_hour.year, current_hour.month, current_hour.day, current_hour.hour
        files = get_gnss_data_names(year, month, day, hour)

        for filename in files:
            sdt = extract_timestamp_from_filename(filename)
            if sdt:
                try:
                    file_dt = datetime.datetime.strptime(sdt, "%Y%m%d%H%M")
                    if dt_start <= file_dt <= dt_end:
                        all_files_info.append({
                            "filename": filename,
                            "sdt": sdt,
                            "file_dt": file_dt
                        })
                except Exception as e:
                    print(f"解析文件时间失败: {filename}, 错误: {e}")

        current_hour += datetime.timedelta(hours=1)

    all_files_info.sort(key=lambda x: x["file_dt"])

    if not all_files_info:
        return []

    all_data_frames = []

    for file_info in all_files_info:
        content = get_bin_bytes(file_info["sdt"])
        if content:
            try:
                file_dt = file_info["file_dt"]
                df = parse_bin_bytes(content, file_dt)
                if not df.empty:
                    all_data_frames.append(df)
            except Exception as e:
                print(f"解析文件失败: {file_info['filename']}, 错误: {e}")

    if not all_data_frames:
        return []

    combined_df = pd.concat(all_data_frames, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp_seconds')

    window_results = []
    window_size_samples = WINDOW_SIZE

    for i in range(0, len(combined_df), window_size_samples):
        end_idx = min(i + window_size_samples, len(combined_df))
        window_df = combined_df.iloc[i:end_idx]

        if len(window_df) >= 60 * SAMPLE_RATE:
            result = process_window_data(window_df)
            if result:
                window_results.append(result)

    return window_results


# ============================================================
# API路由
# ============================================================

@app.route("/")
def home():
    """主页"""
    return format_response(
        message="海洋监测平台后端服务",
        version="1.0.0",
        endpoints={
            "wind_wave_data": "POST /api/wind_wave_data - 获取风浪数据",
            "imu_platform_swing": "POST /api/imu_platform_swing - 获取IMU平台晃动数据",
            "imu_file_list": "POST /api/imu_file_list - 获取IMU文件列表",
            "health": "GET /health - 健康检查",
            "test": "GET /test - 测试接口"
        }
    )


@app.route("/health")
def health():
    """健康检查"""
    return format_response(
        status="healthy",
        service="Marine Monitoring Platform API",
        timestamp=datetime.datetime.now().isoformat()
    )


@app.route("/test")
def test():
    """测试接口"""
    return format_response(
        message="后端服务运行正常",
        endpoints={
            "wind_wave_data": {"method": "POST", "path": "/api/wind_wave_data", "params": ["st1", "st2", "classic"]},
            "imu_platform_swing": {"method": "POST", "path": "/api/imu_platform_swing",
                                   "params": ["st1", "st2", "classic"]},
            "imu_file_list": {"method": "POST", "path": "/api/imu_file_list", "params": ["st1", "st2", "classic"]}
        }
    )


@app.route("/api/wind_wave_data", methods=["POST"])
def wind_wave_data():
    """获取风浪数据"""
    try:
        payload = request.json or {}
        st1 = payload.get("st1")
        st2 = payload.get("st2")
        classic = payload.get("classic")

        if not (st1 and st2 and classic):
            return jsonify(format_response("error", None, "缺少参数: st1, st2, classic")), 400

        if not (validate_time_format(st1) and validate_time_format(st2)):
            return jsonify(format_response("error", None, "时间格式错误，应为YYYYMMDDHHMM")), 400

        classic = int(classic)

        wind_result = get_wind_wave_data(st1, st2, classic, "wind")
        wave_result = get_wind_wave_data(st1, st2, classic, "wave")

        response_data = format_response(
            data={
                "wind": wind_result,
                "wave": wave_result,
                "request": {
                    "st1": st1,
                    "st2": st2,
                    "classic": classic
                }
            }
        )

        return jsonify(response_data)

    except Exception as e:
        error_msg = str(e)
        return jsonify(format_response("error", None, f"服务器内部错误: {error_msg}")), 500


@app.route("/api/imu_platform_swing", methods=["POST"])
def imu_platform_swing():
    """获取IMU平台晃动数据"""
    try:
        payload = request.json or {}
        st1 = payload.get("st1")
        st2 = payload.get("st2")
        classic = payload.get("classic")

        if not (st1 and st2):
            return jsonify(format_response("error", None, "缺少参数: st1, st2")), 400

        if not (validate_time_format(st1) and validate_time_format(st2)):
            return jsonify(format_response("error", None, "时间格式错误，应为YYYYMMDDHHMM")), 400

        results = process_imu_data(st1, st2, classic)

        return jsonify(format_response(
            data={
                "parameters": {
                    "start_time": st1,
                    "end_time": st2,
                    "classic": classic,
                    "sample_rate": SAMPLE_RATE,
                    "window_size_minutes": 10
                },
                "total_windows": len(results),
                "data": results
            }
        ))

    except Exception as e:
        return jsonify(format_response("error", None, f"处理失败: {str(e)}")), 500


@app.route("/api/imu_file_list", methods=["POST"])
def imu_file_list():
    """获取IMU文件列表"""
    try:
        payload = request.json or {}
        st1 = payload.get("st1")
        st2 = payload.get("st2")
        classic = payload.get("classic")

        if not (st1 and st2):
            return jsonify(format_response("error", None, "缺少参数: st1, st2")), 400

        dt_start = datetime.datetime.strptime(st1, "%Y%m%d%H%M")
        dt_end = datetime.datetime.strptime(st2, "%Y%m%d%H%M")

        all_files_info = []
        current_hour = dt_start.replace(minute=0, second=0, microsecond=0)
        end_hour = dt_end.replace(minute=0, second=0, microsecond=0)

        while current_hour <= end_hour:
            year, month, day, hour = current_hour.year, current_hour.month, current_hour.day, current_hour.hour
            files = get_gnss_data_names(year, month, day, hour)

            for filename in files:
                sdt = extract_timestamp_from_filename(filename)
                if sdt:
                    try:
                        file_dt = datetime.datetime.strptime(sdt, "%Y%m%d%H%M")
                        if dt_start <= file_dt <= dt_end:
                            all_files_info.append({
                                "filename": filename,
                                "timestamp": sdt,
                                "datetime": file_dt.strftime("%Y-%m-%d %H:%M")
                            })
                    except:
                        pass

            current_hour += datetime.timedelta(hours=1)

        all_files_info.sort(key=lambda x: x["timestamp"])

        return jsonify(format_response(
            data={
                "count": len(all_files_info),
                "files": all_files_info
            }
        ))

    except Exception as e:
        return jsonify(format_response("error", None, f"处理失败: {str(e)}")), 500


# ============================================================
# 启动应用
# ============================================================

if __name__ == "__main__":
    # Render会设置PORT环境变量
    port = int(os.environ.get("PORT", 5000))

    print("========================================")
    print("    海洋监测平台后端服务")
    print("========================================")
    print(f"端口: {port}")
    print(f"环境: {'生产' if port != 5000 else '开发'}")
    print("========================================")
    print("可用端点:")
    print("  GET  /           - 主页")
    print("  GET  /health     - 健康检查")
    print("  GET  /test       - 测试接口")
    print("  POST /api/wind_wave_data - 风浪数据")
    print("  POST /api/imu_platform_swing - IMU数据")
    print("  POST /api/imu_file_list - IMU文件列表")
    print("========================================")

    app.run(host="0.0.0.0", port=port, debug=False)