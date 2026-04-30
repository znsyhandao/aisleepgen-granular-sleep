# world_model.py - 优化版本：数字孪生作为Real2Sim2Real框架的核心


from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import statistics
import json
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from data_validator import DataValidator
from physics_skin_framework import PhysicsEnhancedSkinAI
#from real2sim2real_architecture import Real2Sim2RealSkinFramework
from collections import defaultdict

from typing import Dict, List, Optional
import random
import time
import logging
import requests

import copy
import math
from product_database import ProductDatabase
from product_recommender_merged import ProductRecommender

# 在 world_model.py 开头添加
import sqlite3
import threading
import time

# 和风天气API配置
HEWEATHER_CONFIG = {
    'api_key': '8789e31ababa4c97ad8b5f36bb15d7eb',  # 您的API密钥
    'base_url': 'https://devapi.qweather.com/v7/',
    'default_location': '101010100',  # 北京的location ID
    
    # Android应用限制参数
    # 留空则允许所有Android应用使用当前凭据
    'android_restrictions': {
        'enabled': False,  # 是否启用Android应用限制
        'package_name': '',  # Android应用的包名（Package name）
        'sha1_fingerprint': ''  # Android应用的签名证书SHA-1指纹
    }
}



# 传感器数据获取状态
SENSOR_STATUS = {
    'available': False,
    'last_reading': None,
    'sensor_type': 'virtual'  # 或 'physical', 'api'
}

# 创建线程本地存储
_thread_local = threading.local()

def create_thread_safe_connection(db_path='skin_care.db'):
    """创建线程安全的数据库连接"""
    try:
        # 如果当前线程还没有连接，创建一个新的
        if not hasattr(_thread_local, 'db_connection'):
            _thread_local.db_connection = sqlite3.connect(db_path, check_same_thread=False)
            _thread_local.db_connection.row_factory = sqlite3.Row
        return _thread_local.db_connection
    except Exception as e:
        logger.error(f"创建线程安全数据库连接失败: {e}")
        return None

def get_products_by_category(conn, category=None):
    """线程安全的按类别查询产品"""
    try:
        if conn is None:
            return []
        
        cursor = conn.cursor()
        if category:
            cursor.execute("SELECT * FROM products WHERE category = ?", (category,))
        else:
            cursor.execute("SELECT * FROM products")
        
        products = cursor.fetchall()
        # 将Row对象转换为字典
        result = []
        for row in products:
            result.append(dict(row))
        return result
    except Exception as e:
        logger.error(f"查询产品失败: {e}")
        return []

def get_products_by_concerns(conn, concerns=None):
    """线程安全的按功效查询产品"""
    try:
        if conn is None or not concerns:
            return []
        
        cursor = conn.cursor()
        # 构建查询条件 - 使用LIKE匹配功效
        query = "SELECT * FROM products WHERE "
        conditions = []
        params = []
        
        for concern in concerns[:3]:  # 最多考虑前3个功效
            conditions.append("benefits LIKE ?")
            params.append(f"%{concern}%")
        
        if conditions:
            query += " OR ".join(conditions)
            cursor.execute(query, params)
            products = cursor.fetchall()
            
            # 将Row对象转换为字典
            result = []
            for row in products:
                result.append(dict(row))
            return result
        else:
            return []
    except Exception as e:
        logger.error(f"按功效查询产品失败: {e}")
        return []




logger = logging.getLogger(__name__)

class SleepMeditationDigitalTwin:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance



    def _get_environment_values(context, default_uv=3, default_humidity=60, 
                            default_temp=25, default_aq=50):
        """从上下文中获取环境值"""
        if context and 'environmental_data' in context:
            env = context['environmental_data']
            return {
                'uv': env.get('uv_index', env.get('uv', default_uv)),
                'humidity': env.get('humidity', default_humidity),
                'temperature': env.get('temperature', default_temp),
                'air_quality': env.get('air_quality', env.get('aq', default_aq))
            }
        return {
            'uv': default_uv,
            'humidity': default_humidity,
            'temperature': default_temp,
            'air_quality': default_aq
        }

    """睡眠冥想数字孪生 - Real2Sim2Real框架的核心"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        # 多层次孪生状态
        self.states = {
            'physical': self._initialize_physical_state(),       # 物理状态
            'biological': self._initialize_biological_state(),   # 生物状态
            'psychological': self._initialize_psychological_state(),  # 心理状态
            'meditation': self._initialize_meditation_state(),     # 冥想状态
            'sleep': self._initialize_sleep_state(),             # 睡眠状态
            'cognitive': self._initialize_cognitive_state(),     # 认知状态
            'temporal': self._initialize_temporal_state(),        # 时间状态
            # [OK] 更新后的关键状态层，专注于睡眠和冥想
            'lifestyle': self._initialize_simplified_lifestyle_state(),  # 简化生活方式（睡眠和冥想相关）
            'environmental': self._initialize_simplified_environmental_state()  # 简化环境（专注于睡眠和冥想环境）
        }

        
        # 模拟历史
        self.simulation_history = []
        self.reality_history = []
        self.reality_gaps = []
        
        # 预测模型
        self.prediction_models = {}

    def _initialize_meditation_state(self) -> Dict:
        """初始化冥想状态"""
        return {
            'state_history': [],  # 冥想练习历史
            'average_duration': 0,  # 平均冥想时长（分钟）
            'mindfulness_score': 50,  # 正念评分（0-100）
            'practice_frequency': 0,  # 每周练习频率
            'current_state': 'resting',  # 当前状态：resting, focusing, deep_meditation
            'anatta_awareness': 30,  # 无我意识（0-100）
            'non_attachment_level': 30,  # 非附着水平（0-100）
            'last_session': None,  # 最后一次冥想会话
            'mindfulness_journal': []  # 正念日志
        }

    def _initialize_sleep_state(self) -> Dict:
        """初始化睡眠状态"""
        return {
            'sleep_history': [],  # 睡眠历史
            'average_duration': 7.0,  # 平均睡眠时长（小时）
            'sleep_quality_score': 60,  # 睡眠质量评分（0-100）
            'deep_sleep_percentage': 20,  # 深度睡眠占比（%）
            'REM_sleep_percentage': 25,  # REM睡眠占比（%）
            'sleep_cycles': [],  # 睡眠周期记录
            'last_night': None,  # 最后一晚睡眠数据
            'sleep_hygiene_score': 50  # 睡眠卫生评分（0-100）
        }

        
    def _initialize_simplified_lifestyle_state(self) -> Dict:
        """初始化简化生活方式状态 - 专注于睡眠和冥想"""
        return {
            'critical_factors': {
                'sleep_quality': 60,      # 睡眠质量
                'meditation_consistency': 50,  # 冥想一致性
                'stress_level': 50,       # 压力水平
                'exercise_frequency': 50, # 运动频率
                'screen_time': 50         # 屏幕时间（影响睡眠）
            },
            'sleep_habits': {
                'regular_sleep_schedule': True,  # 规律睡眠时间表
                'pre_sleep_routine': True,       # 睡前常规
                'napping': 'moderate',           # 午休习惯
            },
            'meditation_habits': {
                'daily_practice': False,         # 每日练习
                'preferred_time': 'morning',     # 偏好的练习时间
                'session_duration': 15,          # 会话时长（分钟）
            },
            'last_updated': datetime.now().isoformat()
        }


    
    def _initialize_simplified_environmental_state(self):
        """初始化简化的环境状态 - 专注于睡眠和冥想环境"""
        # 默认的冥想友好环境参数
        return {
            'current_conditions': {
                'temperature': 22,           # 舒适的冥想/睡眠温度 (°C)
                'humidity': 50,             # 舒适的湿度 (%)
                'light_intensity': 30,       # 适中的光照强度 (0-100)
                'noise_level': 35,           # 低噪音水平 (dB)
                'atmospheric_pressure': 1013,# 正常气压 (hPa)
                'air_quality': 80,           # 良好的空气质量指数
                'pm25': 25,                  # 低PM2.5浓度 (μg/m³)
                'time_of_day': datetime.now().hour, # 一天中的时间
                'season': self._get_current_season() # 当前季节
            },
            'meditation_environment': {
                'room_type': 'bedroom',      # 冥想房间类型
                'comfort_level': 70,         # 舒适度 (0-100)
                'ambient_sound': 'silence',  # 环境声音
            },
            'sleep_environment': {
                'darkness_level': 80,        # 黑暗程度 (0-100)
                'mattress_comfort': 70,      # 床垫舒适度 (0-100)
                'pillow_comfort': 70,        # 枕头舒适度 (0-100)
            },
            'last_location': {
                'city': 'unknown',
                'country': 'unknown',
                'last_update': datetime.now().isoformat()
            },
            'environmental_history': [],     # 环境历史数据
            'risk_factors': {
                'poor_air_quality': False,
                'high_noise': False,
                'extreme_temperature': False,
                'bright_light': False
            }
        }


    def fetch_real_time_environment(self, location_id: str = None) -> bool:
        """从和风天气API获取实时环境数据 - 支持Android应用限制"""
        try:
            if not location_id:
                location_id = HEWEATHER_CONFIG['default_location']
            
            # 构建API请求URL
            api_key = HEWEATHER_CONFIG['api_key']
            base_url = HEWEATHER_CONFIG['base_url']
            weather_url = f"{base_url}weather/now?location={location_id}&key={api_key}"
            air_quality_url = f"{base_url}air/now?location={location_id}&key={api_key}"
            
            # 构建请求头
            headers = {
                'User-Agent': 'SleepMeditationApp/1.0'
            }
            
            # 添加Android应用限制参数（如果已启用）
            android_restrictions = HEWEATHER_CONFIG['android_restrictions']
            if android_restrictions['enabled']:
                if android_restrictions['package_name']:
                    headers['X-Android-Package-Name'] = android_restrictions['package_name']
                if android_restrictions['sha1_fingerprint']:
                    headers['X-Android-Cert'] = android_restrictions['sha1_fingerprint']
            
            # 发送API请求
            print(f"正在从和风天气API获取数据...")
            weather_response = requests.get(weather_url, headers=headers)
            air_quality_response = requests.get(air_quality_url, headers=headers)
            
            # 检查响应状态
            weather_response.raise_for_status()
            air_quality_response.raise_for_status()
            
            # 解析响应数据
            weather_data = weather_response.json()
            air_quality_data = air_quality_response.json()
            
            # 检查API返回状态
            if weather_data['code'] != '200':
                raise Exception(f"天气API错误: {weather_data.get('msg', '未知错误')}")
            
            if air_quality_data['code'] != '200':
                raise Exception(f"空气质量API错误: {air_quality_data.get('msg', '未知错误')}")
            
            # 提取环境数据
            current_weather = weather_data['now']
            current_air = air_quality_data['now']
            
            environment_data = {
                'temperature': float(current_weather.get('temp', 22)),
                'humidity': int(current_weather.get('humidity', 50)),
                'atmospheric_pressure': float(current_weather.get('pressure', 1013)),
                'wind_speed': float(current_weather.get('windSpeed', 0)),
                'weather_condition': current_weather.get('text', '未知'),
                'light_intensity': self._estimate_light_intensity(current_weather.get('text', ''), datetime.now().hour),
                'noise_level': self._estimate_noise_level(float(current_weather.get('windSpeed', 0))),
                'air_quality': int(current_air.get('aqi', 50)),
                'pm25': float(current_air.get('pm2p5', 25)),
                'uv_index': float(current_air.get('uvIndex', 3))
            }
            
            # 更新环境状态
            self.states['environmental']['current_conditions'].update(environment_data)
            self.states['environmental']['last_updated'] = datetime.now().isoformat()
            
            print(f"[OK] 环境数据更新成功: {environment_data}")
            return True
            
        except Exception as e:
            print(f"[FAIL] 获取环境数据失败: {e}")
            # 失败时使用虚拟数据作为备选
            self._generate_virtual_environment_data()
            return False
    
    def enable_android_restrictions(self, package_name: str, sha1_fingerprint: str) -> bool:
        """
        启用Android应用限制功能
        
        参数:
            package_name: Android应用的包名
            sha1_fingerprint: Android应用的签名证书SHA-1指纹
            
        返回值:
            bool: 启用成功返回True，失败返回False
        """
        if not package_name or not sha1_fingerprint:
            print("[FAIL] 包名和SHA-1指纹不能为空")
            return False
        
        try:
            # 更新全局配置
            global HEWEATHER_CONFIG
            HEWEATHER_CONFIG['android_restrictions'] = {
                'enabled': True,
                'package_name': package_name,
                'sha1_fingerprint': sha1_fingerprint
            }
            print(f"[OK] Android应用限制已启用")
            print(f"   包名: {package_name}")
            print(f"   SHA-1指纹: {sha1_fingerprint}")
            return True
        except Exception as e:
            print(f"[FAIL] 启用Android应用限制失败: {e}")
            return False
    
    def disable_android_restrictions(self) -> bool:
        """
        禁用Android应用限制功能
        
        返回值:
            bool: 禁用成功返回True，失败返回False
        """
        try:
            # 更新全局配置
            global HEWEATHER_CONFIG
            HEWEATHER_CONFIG['android_restrictions']['enabled'] = False
            print("[OK] Android应用限制已禁用")
            return True
        except Exception as e:
            print(f"[FAIL] 禁用Android应用限制失败: {e}")
            return False
    
    def get_android_restrictions_status(self) -> dict:
        """
        获取当前Android应用限制状态
        
        返回值:
            dict: 包含enabled、package_name和sha1_fingerprint的字典
        """
        global HEWEATHER_CONFIG
        return HEWEATHER_CONFIG['android_restrictions']



    def update_from_real_world(self, real_data: Dict, framework_type: str = None):
        """从现实世界数据更新数字孪生（增强版，包含简化因素）"""
        self.last_updated = datetime.now()
        
        # 原有核心更新逻辑保持不变
        if 'skin_analysis' in real_data:
            self._update_physical_state(real_data['skin_analysis'])
        
        if 'user_profile' in real_data:
            self._update_psychological_state(real_data['user_profile'])
            self._update_economic_state(real_data['user_profile'])
        
        if 'environment' in real_data:
            self._update_risk_state(real_data['environment'])
        
        # [OK] 新增简化因素更新（后置，不破坏原有逻辑）
        if 'lifestyle_data' in real_data:
            self._update_simplified_lifestyle_state(real_data['lifestyle_data'])
        elif 'user_habits' in real_data:  # 兼容旧字段
            self._update_simplified_lifestyle_state(real_data['user_habits'])
        
        if 'environmental_data' in real_data:
            self._update_simplified_environmental_state(real_data['environmental_data'])
        elif 'location_data' in real_data:  # 从位置数据推断环境
            self._infer_environment_from_location(real_data['location_data'])
        
        # 记录现实数据
        self.reality_history.append({
            'timestamp': datetime.now(),
            'data': real_data,
            'framework': framework_type
        })
        
        # 保留最近100条记录
        if len(self.reality_history) > 100:
            self.reality_history = self.reality_history[-100:]

    def _update_simplified_lifestyle_state(self, lifestyle_data: Dict):
        """更新简化生活方式状态 - 仅更新最关键的几个因素"""
        if not lifestyle_data:
            return
            
        critical_factors = self.states['lifestyle']['critical_factors']
        
        # 更新关键因素（带容错处理）
        if 'sleep_quality' in lifestyle_data:
            value = lifestyle_data['sleep_quality']
            if isinstance(value, (int, float)):
                critical_factors['sleep_quality'] = max(0, min(100, value))
        
        if 'water_intake' in lifestyle_data:
            value = lifestyle_data['water_intake']
            if isinstance(value, (int, float)):
                critical_factors['water_intake'] = max(0, min(100, value))
        
        if 'stress_level' in lifestyle_data:
            value = lifestyle_data['stress_level']
            if isinstance(value, (int, float)):
                critical_factors['stress_level'] = max(0, min(100, value))
        
        if 'sun_protection' in lifestyle_data:
            value = lifestyle_data['sun_protection']
            if isinstance(value, (int, float)):
                critical_factors['sun_protection'] = max(0, min(100, value))
        
        # 更新习惯
        habits = self.states['lifestyle']['habits']
        if 'smoking' in lifestyle_data:
            habits['smoking'] = bool(lifestyle_data['smoking'])
        
        if 'alcohol' in lifestyle_data and lifestyle_data['alcohol'] in ['none', 'low', 'medium', 'high']:
            habits['alcohol'] = lifestyle_data['alcohol']
        
        if 'exercise' in lifestyle_data and lifestyle_data['exercise'] in ['none', 'low', 'medium', 'high']:
            habits['exercise'] = lifestyle_data['exercise']
        
        # 更新时间戳
        self.states['lifestyle']['last_updated'] = datetime.now().isoformat()
    
    def _update_simplified_environmental_state(self, environmental_data: Dict):
        """更新简化环境状态 - 增强版，支持新的环境参数"""
        if not environmental_data:
            return
            
        current_conditions = self.states['environment']['current_conditions']
        
        # 更新当前环境条件
        if 'uv_index' in environmental_data:
            value = environmental_data['uv_index']
            if isinstance(value, (int, float)):
                current_conditions['uv_index'] = max(0, value)
        
        if 'temperature' in environmental_data:
            value = environmental_data['temperature']
            if isinstance(value, (int, float)):
                current_conditions['temperature'] = value
        
        if 'humidity' in environmental_data:
            value = environmental_data['humidity']
            if isinstance(value, (int, float)):
                current_conditions['humidity'] = max(0, min(100, value))
        
        if 'air_quality' in environmental_data:
            value = environmental_data['air_quality']
            if isinstance(value, (int, float)):
                current_conditions['air_quality'] = max(0, min(100, value))
        
        # 新增：更新新的环境参数
        if 'light_intensity' in environmental_data:
            value = environmental_data['light_intensity']
            if isinstance(value, (int, float)):
                current_conditions['light_intensity'] = max(0, min(100, value))
        
        if 'noise_level' in environmental_data:
            value = environmental_data['noise_level']
            if isinstance(value, (int, float)):
                current_conditions['noise_level'] = max(0, value)
        
        if 'atmospheric_pressure' in environmental_data:
            value = environmental_data['atmospheric_pressure']
            if isinstance(value, (int, float)):
                current_conditions['atmospheric_pressure'] = max(800, min(1200, value))
        
        if 'pm25' in environmental_data:
            value = environmental_data['pm25']
            if isinstance(value, (int, float)):
                current_conditions['pm25'] = max(0, value)
        
        if 'season' in environmental_data:
            current_conditions['season'] = str(environmental_data['season'])
        
        # 更新时间（即使没有提供，也更新为当前时间）
        current_conditions['time_of_day'] = datetime.now().hour
        if 'time_of_day' in environmental_data and isinstance(environmental_data['time_of_day'], (int, float)):
            current_conditions['time_of_day'] = int(environmental_data['time_of_day']) % 24
        
        # 更新暴露模式
        exposure = self.states['environment']['exposure_patterns']
        if 'indoor_hours' in environmental_data:
            value = environmental_data['indoor_hours']
            if isinstance(value, (int, float)):
                exposure['indoor_hours'] = max(0, min(24, value))
        
        if 'outdoor_hours' in environmental_data:
            value = environmental_data['outdoor_hours']
            if isinstance(value, (int, float)):
                exposure['outdoor_hours'] = max(0, min(24, value))
        
        if 'office_environment' in environmental_data:
            exposure['office_environment'] = str(environmental_data['office_environment'])
        
        # 更新位置信息
        if 'location' in environmental_data:
            location_data = environmental_data['location']
            if isinstance(location_data, dict):
                self.states['environment']['last_location'].update({
                    'city': location_data.get('city', 'unknown'),
                    'country': location_data.get('country', 'unknown'),
                    'last_update': datetime.now().isoformat()
                })
        
        # 更新环境风险因素和记录趋势
        self._update_environmental_risk_factors()
        self._record_environmental_trend()
    
    def _infer_environment_from_location(self, location_data: Dict):
        """从位置数据推断环境条件（简化版）"""
        # 这是一个占位函数，实际应用中可以从天气API获取数据
        # 这里我们只记录位置信息，环境条件可以使用默认值
        if location_data and isinstance(location_data, dict):
            self.states['environment']['last_location'].update({
                'city': location_data.get('city', 'unknown'),
                'country': location_data.get('country', 'unknown'),
                'last_update': datetime.now().isoformat()
            })
            
            # 可以根据位置推断季节（简化逻辑）
            city = location_data.get('city', '').lower()
            country = location_data.get('country', '').lower()
            
            # 简单季节推断（北半球为例）
            month = datetime.now().month
            if month in [12, 1, 2]:
                season = 'winter'
            elif month in [3, 4, 5]:
                season = 'spring'
            elif month in [6, 7, 8]:
                season = 'summer'
            else:
                season = 'autumn'
            
            self.states['environment']['current_conditions']['season'] = season
    

    

    
    def _calculate_lifestyle_score(self) -> float:
        """计算综合生活方式评分 - 修复版"""
        critical_factors = self.states['lifestyle']['critical_factors']
        habits = self.states['lifestyle']['habits']
        
        # 调试信息
        # print(f"DEBUG 生活方式关键因素: {critical_factors}")
        # print(f"DEBUG 生活习惯: {habits}")
        
        # 关键因素计算 (权重: 70%)
        factors_score = (
            critical_factors.get('sleep_quality', 50) * 0.25 +
            critical_factors.get('water_intake', 50) * 0.20 +
            (100 - critical_factors.get('stress_level', 50)) * 0.15 +  # 压力越低越好
            critical_factors.get('sun_protection', 50) * 0.10
        )
        
        # 习惯计算 (权重: 30%)
        habits_score = 0
        if not habits.get('smoking', False):
            habits_score += 15
        
        alcohol = habits.get('alcohol', 'low')
        if alcohol == 'none':
            habits_score += 10
        elif alcohol == 'low':
            habits_score += 7
        elif alcohol == 'medium':
            habits_score += 3
        
        exercise = habits.get('exercise', 'low')
        if exercise == 'high':
            habits_score += 5
        elif exercise == 'medium':
            habits_score += 3
        elif exercise == 'low':
            habits_score += 1
        
        # 一致性评分（如果有）
        consistency = critical_factors.get('consistency_score', 50)
        habits_score += consistency * 0.05  # 增加一致性影响
        
        total_score = factors_score + habits_score
        
        # 确保分数在合理范围内
        return max(0, min(100, total_score))
    
    def get_lifestyle_insights(self) -> Dict:
        """获取生活方式洞察 - 修复版"""
        critical_factors = self.states['lifestyle']['critical_factors']
        habits = self.states['lifestyle']['habits']
        
        # 先计算总分
        overall_score = self._calculate_lifestyle_score()
        
        insights = {
            'actionable_improvements': [],
            'maintain_good_habits': [],
            'risk_factors': [],
            'overall_score': overall_score,
            'priority_level': 'medium'
        }
        
        # 识别可改善的关键因素（阈值调整）
        sleep_quality = critical_factors.get('sleep_quality', 50)
        if sleep_quality < 70:  # 阈值从60提高到70
            priority = 'high' if sleep_quality < 50 else 'medium'
            insights['actionable_improvements'].append({
                'factor': 'sleep_quality',
                'current': sleep_quality,
                'target': 80,
                'suggestion': 'Try to get 7-8 hours of quality sleep each night',
                'priority': priority
            })
        
        water_intake = critical_factors.get('water_intake', 50)
        if water_intake < 70:  # 阈值从60提高到70
            priority = 'high' if water_intake < 50 else 'medium'
            insights['actionable_improvements'].append({
                'factor': 'water_intake',
                'current': water_intake,
                'target': 80,
                'suggestion': 'Increase daily water intake to 2-3 liters',
                'priority': priority
            })
        
        stress_level = critical_factors.get('stress_level', 50)
        if stress_level > 50:  # 阈值从60降低到50
            priority = 'high' if stress_level > 70 else 'medium'
            insights['actionable_improvements'].append({
                'factor': 'stress_level',
                'current': stress_level,
                'target': 30,
                'suggestion': 'Incorporate stress management techniques like meditation',
                'anatta_insight': '通过冥想培养无我的觉察，可以帮助减少自我认同带来的压力',
                'priority': priority
            })
        
        sun_protection = critical_factors.get('sun_protection', 50)
        if sun_protection < 70:  # 阈值从60提高到70
            insights['actionable_improvements'].append({
                'factor': 'sun_protection',
                'current': sun_protection,
                'target': 80,
                'suggestion': 'Use SPF 30+ sunscreen daily, especially when outdoors',
                'priority': 'medium'
            })
        
        # 识别风险习惯
        if habits.get('smoking', False):
            insights['risk_factors'].append({
                'factor': 'smoking',
                'impact': 'high',
                'suggestion': 'Consider smoking cessation programs',
                'resources': ['quitline', 'nicotine_replacement']
            })
        
        alcohol = habits.get('alcohol', 'low')
        if alcohol in ['medium', 'high']:
            insights['risk_factors'].append({
                'factor': 'alcohol_consumption',
                'level': alcohol,
                'impact': 'medium',
                'suggestion': 'Moderate alcohol consumption for skin health'
            })
        
        # 运动建议（现在只在运动不足时给出）
        exercise = habits.get('exercise', 'low')
        if exercise in ['none', 'low']:
            insights['actionable_improvements'].append({
                'factor': 'exercise',
                'level': exercise,
                'target': 'medium',
                'suggestion': 'Incorporate 30 minutes of moderate exercise 3-4 times per week',
                'priority': 'medium'
            })
        elif exercise in ['medium', 'high']:
            insights['maintain_good_habits'].append({
                'factor': 'exercise',
                'level': exercise,
                'benefit': 'Improves circulation and skin health'
            })
        
        # 确定优先级
        high_priority_count = sum(1 for item in insights['actionable_improvements'] 
                                 if item.get('priority') == 'high')
        
        if high_priority_count >= 2:
            insights['priority_level'] = 'high'
        elif high_priority_count == 1:
            insights['priority_level'] = 'medium_high'
        
        return insights
    
    def update_meditation_session(self, session_data: Dict) -> Dict:
        """更新冥想会话数据并调整认知状态"""
        # 更新认知状态中的冥想数据
        meditation_state = self.states['cognitive']['meditation']
        
        # 记录新会话
        session = {
            'timestamp': datetime.now().isoformat(),
            'duration': session_data.get('duration', 0),
            'focus_level': session_data.get('focus_level', 50),
            'mindfulness_score': session_data.get('mindfulness_score', 50),
            'meditation_type': session_data.get('meditation_type', 'default'),
            'anatta_awareness_gain': session_data.get('anatta_awareness_gain', 0)
        }
        
        # 更新状态历史
        meditation_state['state_history'].append(session)
        
        # 更新平均数据
        total_sessions = len(meditation_state['state_history'])
        if total_sessions > 0:
            meditation_state['average_duration'] = round(
                sum(s['duration'] for s in meditation_state['state_history']) / total_sessions
            )
            meditation_state['mindfulness_score'] = round(
                sum(s['mindfulness_score'] for s in meditation_state['state_history']) / total_sessions
            )
        
        # 更新频率（基于最近7天）
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_sessions = [
            s for s in meditation_state['state_history']
            if datetime.fromisoformat(s['timestamp']) >= seven_days_ago
        ]
        meditation_state['practice_frequency'] = len(recent_sessions)
        
        # 更新无我意识
        if session_data.get('anatta_focus', False):
            anatta_gain = session_data.get('anatta_awareness_gain', 1)
            self.states['cognitive']['perception']['anatta_awareness'] = min(
                100, self.states['cognitive']['perception']['anatta_awareness'] + anatta_gain
            )
            self.states['cognitive']['perception']['non_attachment_level'] = min(
                100, self.states['cognitive']['perception']['non_attachment_level'] + anatta_gain * 0.5
            )
        
        # 更新自我感知
        self.states['cognitive']['perception']['self_perception'] = round(
            50 + (self.states['cognitive']['perception']['anatta_awareness'] - 50) * 0.3
        )
        
        # 更新当前状态
        meditation_state['current_state'] = session_data.get('final_state', 'observer')
        meditation_state['last_session'] = session['timestamp']
        
        # 更新最后修改时间
        self.last_updated = datetime.now()
        
        return {
            'success': True,
            'updated_state': {
                'meditation': meditation_state,
                'perception': self.states['cognitive']['perception']
            },
            'session_id': session['timestamp']
        }
    
    def get_meditation_insights(self) -> Dict:
        """获取冥想相关洞察"""
        meditation_state = self.states['cognitive']['meditation']
        perception_state = self.states['cognitive']['perception']
        
        insights = {
            'practice_frequency': meditation_state['practice_frequency'],
            'average_session_duration': meditation_state['average_duration'],
            'mindfulness_score': meditation_state['mindfulness_score'],
            'current_anatta_awareness': perception_state['anatta_awareness'],
            'non_attachment_level': perception_state['non_attachment_level'],
            'recommendations': []
        }
        
        # 根据当前状态提供建议
        if meditation_state['practice_frequency'] < 3:
            insights['recommendations'].append('建议每周至少进行3次冥想练习')
        
        if meditation_state['average_duration'] < 10:
            insights['recommendations'].append('可以尝试逐步增加每次冥想的时长')
        
        if perception_state['anatta_awareness'] < 50:
            insights['recommendations'].append('建议尝试无我冥想练习，培养对自我概念的觉察')
        else:
            insights['recommendations'].append('继续保持良好的无觉察状态，这有助于减少压力和焦虑')
        
        return insights
    
    def get_anatta_insights(self) -> Dict:
        """获取无我心智相关洞察"""
        anatta_awareness = self.states['cognitive']['perception'].get('anatta_awareness', 30)
        non_attachment_level = self.states['cognitive']['perception'].get('non_attachment_level', 30)
        self_perception = self.states['cognitive']['perception'].get('self_perception', 50)
        
        anatta_levels = [
            (0, 30, "初步觉察", "开始意识到自我概念的局限性"),
            (31, 60, "深入理解", "能够观察到思想和情绪的流动性"),
            (61, 80, "体验无我", "能够在冥想中体验到无我的状态"),
            (81, 100, "自然融入", "无我意识成为日常生活的一部分")
        ]
        
        current_level = "初步觉察"
        level_description = "开始意识到自我概念的局限性"
        
        for min_level, max_level, level, desc in anatta_levels:
            if min_level <= anatta_awareness <= max_level:
                current_level = level
                level_description = desc
                break
        
        insights = {
            'anatta_awareness_level': anatta_awareness,
            'anatta_level_category': current_level,
            'level_description': level_description,
            'non_attachment_level': non_attachment_level,
            'self_perception_balance': self_perception,
            'philosophical_insights': [],
            'practical_guidance': []
        }
        
        # 生成哲学洞察
        if anatta_awareness < 40:
            insights['philosophical_insights'].append("'无我'不是否定存在，而是超越对'自我'的执着")
        elif anatta_awareness < 70:
            insights['philosophical_insights'].append("思想、情绪和身体都是不断变化的过程，而非固定不变的实体")
        else:
            insights['philosophical_insights'].append("在当下的体验中，能够观察到'观察者'与'被观察者'的融合")
        
        # 提供实践指导
        if non_attachment_level < 50:
            insights['practical_guidance'].append("在日常生活中，尝试观察自己的想法而不与之认同")
        
        if self_perception > 70:
            insights['practical_guidance'].append("尝试放下对'完美自我'的追求，接受事物的本来面目")
        
        return insights
    
    def analyze_mind_module_relationships(self) -> Dict:
        """分析心智模块与无我心智的关系"""
        # 获取各心智模块状态
        psychological_state = self.states['psychological']
        cognitive_state = self.states['cognitive']
        meditation_state = cognitive_state['meditation']
        perception_state = cognitive_state['perception']
        
        # 分析压力与无我意识的关系
        stress_level = self.states['lifestyle']['critical_factors'].get('stress_level', 50)
        anatta_awareness = perception_state['anatta_awareness']
        
        # 计算心智模块之间的相关性
        correlations = {
            'stress_vs_anatta': -0.3 * (stress_level / 100) + 0.7 * (anatta_awareness / 100),
            'mindfulness_vs_nonattachment': 0.8 * (meditation_state['mindfulness_score'] / 100) * (perception_state['non_attachment_level'] / 100),
            'self_perception_balance': 1.0 - abs((self_perception - 50) / 50)
        }
        
        # 生成整合分析
        analysis = {
            'mind_module_states': {
                'psychological': psychological_state,
                'cognitive': cognitive_state,
                'meditation': meditation_state,
                'perception': perception_state
            },
            'correlations': correlations,
            'integration_insights': [],
            'balance_status': {
                'stress_anatta_balance': 'balanced' if 0.4 < correlations['stress_vs_anatta'] < 0.6 else 'imbalanced',
                'mindfulness_nonattachment_balance': 'balanced' if correlations['mindfulness_vs_nonattachment'] > 0.5 else 'developing',
                'self_perception_balance': 'balanced' if correlations['self_perception_balance'] > 0.8 else 'developing'
            }
        }
        
        # 提供整合洞察
        if stress_level > 60 and anatta_awareness < 40:
            analysis['integration_insights'].append("较高的压力水平与较低的无我意识相关，建议通过无我冥想练习来降低压力")
        
        if meditation_state['practice_frequency'] > 0:
            analysis['integration_insights'].append("规律的冥想练习有助于提升整体心智模块的整合性")
        
        if perception_state['anatta_awareness'] > 60:
            analysis['integration_insights'].append("较高的无我意识水平可以促进不同心智模块之间的和谐运作")
        
        return analysis
    
    def get_real2sim2real_framework_data(self) -> Dict:
        """获取Real2Sim2Real框架的相关数据，将冥想与皮肤健康关联"""
        # 获取相关状态
        meditation_insights = self.get_meditation_insights()
        anatta_insights = self.get_anatta_insights()
        stress_level = self.states['lifestyle']['critical_factors'].get('stress_level', 50)
        sleep_quality = self.states['lifestyle']['critical_factors'].get('sleep_quality', 60)
        
        # 计算冥想对皮肤健康的潜在影响
        meditation_skin_impact = {
            'stress_reduction': max(0, (100 - stress_level) / 100),
            'sleep_improvement': max(0, (sleep_quality - 50) / 50),
            'anatta_awareness_impact': (anatta_insights['anatta_awareness_level'] / 100) * 0.3
        }
        
        total_meditation_impact = sum(meditation_skin_impact.values()) / len(meditation_skin_impact.values())
        
        # 生成Real2Sim2Real框架数据
        framework_data = {
            'real_world_data': {
                'stress_level': stress_level,
                'sleep_quality': sleep_quality,
                'meditation_frequency': meditation_insights['practice_frequency'],
                'anatta_awareness': anatta_insights['anatta_awareness_level']
            },
            'simulation_prediction': {
                'meditation_skin_health_impact': round(total_meditation_impact * 100, 2),
                'projected_stress_reduction': round((100 - stress_level) * 0.2, 2),
                'projected_sleep_improvement': round((sleep_quality - 50) * 0.15, 2)
            },
            'actionable_recommendations': [
                {
                    'type': 'meditation',
                    'suggestion': '增加无我冥想练习频率至每周4-5次',
                    'expected_impact': '可能降低10-15%的压力水平'
                },
                {
                    'type': 'lifestyle',
                    'suggestion': '将冥想与睡前放松习惯结合',
                    'expected_impact': '可能提高8-12%的睡眠质量'
                }
            ],
            'framework_feedback_loop': {
                'real_data_collection': ['stress_monitoring', 'sleep_tracking', 'meditation_session'],
                'simulation_analysis': ['stress_reduction_model', 'anatta_awareness_growth_model'],
                'real_world_intervention': ['personalized_meditation_plan', 'lifestyle_adjustment']
            }
        }
        
        return framework_data
    
    def run_simplified_enhanced_simulation(self, intervention: Dict, duration_days: int = 30) -> Dict:
        """运行简化增强模拟 - 考虑生活方式和环境因素"""
        simulation_id = f"sim_simple_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        simulation_results = {
            'simulation_id': simulation_id,
            'intervention': intervention,
            'duration_days': duration_days,
            'initial_state': self.get_current_state(),
            'daily_progress': [],
            'factor_impact_analysis': {},
            'final_predictions': {}
        }
        
        # 计算因素影响权重
        lifestyle_score = self._calculate_lifestyle_score()
        environmental_factor = self._calculate_environmental_impact_factor()
        
        # 基础权重分配
        weights = {
            'intervention': 0.5,    # 干预措施
            'lifestyle': 0.3,       # 生活方式（根据评分调整）
            'environment': 0.2      # 环境因素
        }
        
        # 根据生活质量调整权重
        if lifestyle_score < 40:
            weights['lifestyle'] = 0.4
            weights['intervention'] = 0.4
        
        simulation_results['factor_weights'] = weights
        
        for day in range(1, duration_days + 1):
            daily_state = self._simplified_daily_simulation(intervention, day, weights)
            simulation_results['daily_progress'].append(daily_state)
        
        # 分析因素影响
        simulation_results['factor_impact_analysis'] = self._analyze_factor_impact(
            simulation_results['daily_progress']
        )
        
        # 提取最终预测
        final_state = simulation_results['daily_progress'][-1] if simulation_results['daily_progress'] else {}
        simulation_results['final_predictions'] = self._extract_predictions(final_state, intervention)
        
        # 添加生活方式和环境建议
        simulation_results['lifestyle_recommendations'] = self.get_lifestyle_insights()
        simulation_results['environmental_recommendations'] = self.get_environmental_recommendations()
        
        # 记录模拟历史
        self.simulation_history.append({
            'timestamp': datetime.now(),
            'simulation': simulation_results,
            'simulation_type': 'simplified_enhanced'
        })
        
        return simulation_results
    

    
    def _get_current_season(self) -> str:
        """获取当前季节"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def _update_environmental_risk_factors(self):
        """更新环境风险因素"""
        conditions = self.states['environment']['current_conditions']
        risk_factors = self.states['environment']['risk_factors']
        
        # 更新各项风险因素
        risk_factors['extreme_temperature'] = conditions['temperature'] > 35 or conditions['temperature'] < 5
        risk_factors['high_uv'] = conditions['uv_index'] > 7
        risk_factors['poor_air_quality'] = conditions['air_quality'] < 30
        risk_factors['high_noise'] = conditions['noise_level'] > 80
        risk_factors['high_pm25'] = conditions['pm25'] > 75
    
    def _record_environmental_trend(self):
        """记录环境趋势数据"""
        conditions = self.states['environment']['current_conditions'].copy()
        conditions['timestamp'] = datetime.now().isoformat()
        
        # 记录历史数据（保留最近24小时）
        history = self.states['environment']['environmental_history']
        history.append(conditions)
        
        # 只保留最近24小时的数据（假设每小时记录一次）
        if len(history) > 24:
            history = history[-24:]
        
        self.states['environment']['environmental_history'] = history
    
    def get_environmental_impact_on_meditation(self) -> Dict:
        """分析当前环境条件对冥想效果的影响"""
        environment = self.states['environment']['current_conditions']
        impact = {}
        
        # 噪音对冥想的影响（高噪音降低专注力）
        if environment['noise_level'] < 40:
            impact['noise'] = {'level': 'low', 'effect': 'positive', 'score': 80}
        elif environment['noise_level'] < 60:
            impact['noise'] = {'level': 'medium', 'effect': 'neutral', 'score': 50}
        else:
            impact['noise'] = {'level': 'high', 'effect': 'negative', 'score': 20}
        
        # 光照对冥想的影响（适度光照有助于专注，过亮或过暗可能影响）
        if 30 <= environment['light_intensity'] <= 70:
            impact['light'] = {'level': 'moderate', 'effect': 'positive', 'score': 85}
        else:
            impact['light'] = {'level': 'extreme', 'effect': 'negative', 'score': 40}
        
        # 温度对冥想的影响（舒适温度提升冥想效果）
        if 20 <= environment['temperature'] <= 28:
            impact['temperature'] = {'level': 'comfortable', 'effect': 'positive', 'score': 90}
        elif 15 <= environment['temperature'] < 20 or 28 < environment['temperature'] <= 32:
            impact['temperature'] = {'level': 'moderate', 'effect': 'neutral', 'score': 60}
        else:
            impact['temperature'] = {'level': 'extreme', 'effect': 'negative', 'score': 30}
        
        # 综合环境影响评分
        total_score = sum(item['score'] for item in impact.values()) / len(impact.values())
        impact['overall'] = {
            'score': round(total_score, 2),
            'description': self._get_environmental_meditation_description(total_score)
        }
        
        return impact
    
    def _get_environmental_meditation_description(self, score: float) -> str:
        """根据环境评分获取冥想环境描述"""
        if score >= 80:
            return "理想的冥想环境，有助于深度冥想和专注"
        elif score >= 60:
            return "良好的冥想环境，基本适合冥想练习"
        elif score >= 40:
            return "一般的冥想环境，可能需要调整或使用辅助工具"
        else:
            return "较差的冥想环境，建议改善环境条件或选择其他时间"
    
    def integrate_real_time_data(self, data_source: str, data: Dict) -> bool:
        """整合来自不同数据源的实时环境数据"""
        try:
            # 验证数据源和数据
            if not data_source or not isinstance(data, dict):
                return False
            
            current_conditions = self.states['environment']['current_conditions']
            
            # 数据映射和验证
            data_mapping = {
                'uv_index': ['uv', 'uv_index'],
                'temperature': ['temp', 'temperature', 'temp_c', 'temperature_celsius'],
                'humidity': ['humidity', 'humidity_percent'],
                'air_quality': ['air_quality', 'aq', 'aqi'],
                'light_intensity': ['light', 'light_intensity', 'lux'],
                'noise_level': ['noise', 'noise_level', 'db'],
                'atmospheric_pressure': ['pressure', 'atmospheric_pressure', 'hpa', 'pressure_hpa'],
                'pm25': ['pm25', 'pm_25', 'particulate_matter_25']
            }
            
            # 融合数据
            for target_param, source_params in data_mapping.items():
                for source_param in source_params:
                    if source_param in data:
                        value = data[source_param]
                        # 验证数据类型和范围
                        if isinstance(value, (int, float)):
                            # 应用合理范围限制
                            if target_param == 'uv_index':
                                value = max(0, min(15, value))
                            elif target_param == 'temperature':
                                value = max(-20, min(50, value))
                            elif target_param == 'humidity':
                                value = max(0, min(100, value))
                            elif target_param == 'air_quality':
                                value = max(0, min(100, value))
                            elif target_param == 'light_intensity':
                                value = max(0, min(100, value))
                            elif target_param == 'noise_level':
                                value = max(0, min(150, value))
                            elif target_param == 'atmospheric_pressure':
                                value = max(800, min(1200, value))
                            elif target_param == 'pm25':
                                value = max(0, min(500, value))
                            
                            current_conditions[target_param] = value
                            break
            
            # 更新时间和季节
            current_conditions['time_of_day'] = datetime.now().hour
            current_conditions['season'] = self._get_current_season()
            
            # 更新风险因素和趋势
            self._update_environmental_risk_factors()
            self._record_environmental_trend()
            
            return True
        except Exception as e:
            logger.error(f"整合实时环境数据失败: {e}")
            return False
    
    def _calculate_environmental_impact_factor(self) -> float:
        """计算环境影响因子（0-1，越高表示环境越不利）- 增强版，考虑更多物理环境参数"""
        conditions = self.states['environment']['current_conditions']
        exposure = self.states['environment']['exposure_patterns']
        
        # 计算环境压力分数（增强版，考虑更多参数）
        uv_impact = min(1.0, conditions['uv_index'] / 12)  # UV影响，12为极高
        
        humidity_impact = 0
        humidity = conditions['humidity']
        if humidity < 30:
            humidity_impact = 0.4  # 过干
        elif humidity > 80:
            humidity_impact = 0.3  # 过湿
        
        air_quality_impact = (100 - conditions['air_quality']) / 200  # 空气质量越差影响越大，但幅度减半
        
        outdoor_impact = min(0.8, exposure['outdoor_hours'] / 10)  # 室外时间影响
        
        # 温度影响（极端温度）
        temperature = conditions['temperature']
        temp_impact = 0
        if temperature > 35 or temperature < 5:
            temp_impact = 0.3
        
        # 新增：噪音影响
        noise_impact = 0
        if conditions['noise_level'] > 80:
            noise_impact = 0.2
        elif conditions['noise_level'] > 60:
            noise_impact = 0.1
        
        # 新增：PM2.5影响
        pm25_impact = min(0.4, conditions['pm25'] / 150)
        
        # 新增：光照影响（极端光照）
        light_impact = 0
        if conditions['light_intensity'] > 90 or conditions['light_intensity'] < 10:
            light_impact = 0.1
        
        # 加权平均（调整权重，考虑新增参数）
        total_impact = (
            uv_impact * 0.30 +
            humidity_impact * 0.15 +
            air_quality_impact * 0.20 +
            outdoor_impact * 0.10 +
            temp_impact * 0.05 +
            noise_impact * 0.10 +
            pm25_impact * 0.07 +
            light_impact * 0.03
        )
        
        # 确保在0-1范围内
        return max(0, min(1, total_impact))
    

    

    
    def _simplified_daily_simulation(self, intervention: Dict, day: int, weights: Dict) -> Dict:
        """简化版每日模拟 - 考虑关键因素（修复改进计算）"""
        daily_state = {}
        
        for param in ['pores', 'sebum', 'redness', 'wrinkles', 'barrier', 'melanin', 'elasticity', 'collagen']:
            param_data = self.states['physical']['skin_parameters'].get(param, {})
            current_score = param_data.get('score', 50) if isinstance(param_data, dict) else 50
            
            # 1. 基础干预效果
            base_improvement = self._calculate_parameter_improvement(param, intervention, day)
            
            # 2. 生活方式影响（基于关键因素）
            lifestyle_impact = self._calculate_lifestyle_impact_on_parameter(param)
            
            # 3. 环境影响
            environmental_impact = self._calculate_environmental_impact_on_parameter(param, day)
            
            # 加权计算（确保权重和为1）
            total_weight = weights.get('intervention', 0.5) + weights.get('lifestyle', 0.3) + weights.get('environment', 0.2)
            if total_weight == 0:
                total_weight = 1
            
            total_improvement = (
                base_improvement * weights.get('intervention', 0.5) +
                lifestyle_impact * weights.get('lifestyle', 0.3) +
                environmental_impact * weights.get('environment', 0.2)
            ) / total_weight
            
            # 限制改进范围
            max_improvement = 5  # 每天最大改进
            total_improvement = max(-2, min(max_improvement, total_improvement))
            
            predicted_score = min(100, max(0, current_score + total_improvement))
            
            daily_state[param] = {
                'day': day,
                'current': current_score,
                'predicted': predicted_score,
                'improvement': total_improvement,
                'factor_breakdown': {
                    'intervention': base_improvement,
                    'lifestyle': lifestyle_impact,
                    'environment': environmental_impact
                }
            }
        
        return daily_state
    

    
    def _calculate_lifestyle_impact_on_parameter(self, parameter: str) -> float:
        """计算生活方式对特定参数的影响 - 优化版"""
        critical_factors = self.states['lifestyle']['critical_factors']
        habits = self.states['lifestyle']['habits']
        
        # 调试信息
        # print(f"DEBUG 计算 {parameter} 的生活方式影响")
        # print(f"DEBUG 关键因素: {critical_factors}")
        # print(f"DEBUG 习惯: {habits}")
        
        impact = 0
        
        # 基础因子计算（标准化到-1到1范围）
        sleep_factor = (critical_factors.get('sleep_quality', 50) - 50) / 50  # -1到1
        water_factor = (critical_factors.get('water_intake', 50) - 50) / 50
        stress_factor = (50 - critical_factors.get('stress_level', 50)) / 50  # 压力越低越好
        sun_factor = (critical_factors.get('sun_protection', 50) - 50) / 50
        
        # 习惯因子
        smoking_penalty = -0.5 if habits.get('smoking', False) else 0
        
        alcohol = habits.get('alcohol', 'low')
        alcohol_penalty = 0
        if alcohol == 'high':
            alcohol_penalty = -0.3
        elif alcohol == 'medium':
            alcohol_penalty = -0.15
        
        exercise = habits.get('exercise', 'low')
        exercise_bonus = 0
        if exercise == 'high':
            exercise_bonus = 0.3
        elif exercise == 'medium':
            exercise_bonus = 0.15
        elif exercise == 'low':
            exercise_bonus = 0.05
        
        # 根据参数类型分配权重（确保每种类型都有影响）
        if parameter in ['hydration', 'barrier']:
            # 水合和屏障主要受水分和睡眠影响
            impact = (
                water_factor * 0.5 +
                sleep_factor * 0.3 +
                stress_factor * 0.1 +
                exercise_bonus * 0.1 +
                smoking_penalty * 0.3 +
                alcohol_penalty * 0.2
            )
        elif parameter in ['redness', 'sebum']:
            # 泛红和皮脂主要受压力和睡眠影响
            impact = (
                stress_factor * 0.4 +
                sleep_factor * 0.3 +
                water_factor * 0.2 +
                smoking_penalty * 0.4 +
                alcohol_penalty * 0.3
            )
        elif parameter in ['collagen', 'wrinkles', 'elasticity']:
            # 胶原蛋白、皱纹和弹性受多种因素影响
            impact = (
                sun_factor * 0.4 +
                sleep_factor * 0.25 +
                water_factor * 0.15 +
                exercise_bonus * 0.1 +
                smoking_penalty * 0.5 +
                alcohol_penalty * 0.3
            )
        elif parameter == 'melanin':
            # 黑色素主要受防晒影响
            impact = (
                sun_factor * 0.7 +
                sleep_factor * 0.2 +
                smoking_penalty * 0.2
            )
        else:  # pores
            # 毛孔受多种因素影响
            impact = (
                water_factor * 0.3 +
                sleep_factor * 0.3 +
                stress_factor * 0.2 +
                smoking_penalty * 0.3 +
                alcohol_penalty * 0.2
            )
        
        # 确保影响在合理范围内并调整幅度
        impact = max(-1, min(1, impact))
        
        # 转换为每日改进分数（-2到2分）
        daily_impact = impact * 2
        
        # print(f"DEBUG {parameter} 生活方式影响: {impact:.2f}, 每日影响: {daily_impact:.2f}")
        
        return daily_impact
    
    def _calculate_environmental_impact_on_parameter(self, parameter: str, day: int) -> float:
        """计算环境对特定参数的影响 - 增强版，考虑更多物理环境参数"""
        conditions = self.states['environment']['current_conditions']
        
        uv_impact = conditions['uv_index'] / 10  # 0-1.5+
        humidity = conditions['humidity']
        air_quality = conditions['air_quality'] / 100  # 0-1
        noise_level = conditions['noise_level']
        pm25 = conditions['pm25']
        light_intensity = conditions['light_intensity']
        
        impact = 0
        
        # UV对黑色素、皱纹、胶原蛋白有显著负面影响
        if parameter in ['melanin', 'wrinkles', 'collagen']:
            impact -= uv_impact * 0.5
        
        # 低湿度对屏障、水合有负面影响
        if parameter in ['barrier', 'hydration'] and humidity < 40:
            impact -= 0.3
        
        # 高湿度可能加重皮脂分泌
        if parameter == 'sebum' and humidity > 70:
            impact += 0.2
        
        # 空气质量差对泛红有影响
        if parameter == 'redness' and air_quality < 0.5:
            impact += 0.2
        
        # 新增：高PM2.5对皮肤屏障和泛红有负面影响
        if parameter in ['barrier', 'redness'] and pm25 > 75:
            impact -= 0.2
        
        # 新增：极端温度对皮肤屏障有负面影响
        if parameter == 'barrier' and (conditions['temperature'] > 35 or conditions['temperature'] < 5):
            impact -= 0.15
        
        return impact
    

    
    def _analyze_factor_impact(self, daily_progress: List[Dict]) -> Dict:
        """分析各因素对改善的贡献 - 修复版"""
        if not daily_progress:
            return {}
        
        # 取最后一天的数据分析
        last_day = daily_progress[-1]
        
        factor_contributions = {}
        total_contributions = {}
        
        for param, data in last_day.items():
            if param == 'factor_contributions':
                continue
                
            if not isinstance(data, dict):
                continue
                
            breakdown = data.get('factor_breakdown', {})
            total_improvement = data.get('improvement', 0)
            
            # 过滤掉无效数据
            if not isinstance(breakdown, dict) or total_improvement == 0:
                continue
            
            # 计算每个因素的贡献百分比
            contributions = {}
            valid_factors = []
            factor_values = []
            
            for factor, value in breakdown.items():
                if isinstance(value, (int, float)):
                    valid_factors.append(factor)
                    factor_values.append(abs(value))  # 使用绝对值
            
            if valid_factors and sum(factor_values) > 0:
                for factor, value in zip(valid_factors, factor_values):
                    percentage = (value / sum(factor_values)) * 100
                    contributions[factor] = round(percentage, 2)
                
                factor_contributions[param] = contributions
        
        # 计算平均贡献（修复计算逻辑）
        if factor_contributions:
            avg_contributions = {}
            factor_sums = {}
            factor_counts = {}
            
            for param, contributions in factor_contributions.items():
                for factor, percentage in contributions.items():
                    if factor not in factor_sums:
                        factor_sums[factor] = 0
                        factor_counts[factor] = 0
                    
                    factor_sums[factor] += percentage
                    factor_counts[factor] += 1
            
            for factor in factor_sums:
                if factor_counts[factor] > 0:
                    avg_contributions[factor] = round(factor_sums[factor] / factor_counts[factor], 2)
            
            total_contributions['average'] = avg_contributions
            total_contributions['by_parameter'] = factor_contributions
        
        return total_contributions
    

    
    def get_current_state(self) -> Dict:
        """获取当前数字孪生状态（包含新状态层）- 修复版"""
        state = {
            'user_id': self.user_id,
            'timestamp': self.last_updated.isoformat(),
            'states': self.states,
            'metadata': {
                'age_days': (datetime.now() - self.created_at).days,
                'update_count': len(self.reality_history),
                'simulation_count': len(self.simulation_history),
                'reality_gap_count': len(self.reality_gaps),
                'lifestyle_score': self._calculate_lifestyle_score(),
                'environmental_impact': self._calculate_environmental_impact_factor()
            }
        }
        
        # 添加可行动洞察（使用修复后的方法）
        try:
            state['actionable_insights'] = {
                'lifestyle': self.get_lifestyle_insights(),
                'environment': self.get_environmental_recommendations()
            }
        except Exception as e:
            # print(f"DEBUG 获取可行动洞察时出错: {e}")
            state['actionable_insights'] = {
                'lifestyle': {'overall_score': 0, 'actionable_improvements': []},
                'environment': {'daily_adjustments': [], 'protective_measures': []}
            }
        
        return state
    
    def get_insights(self) -> Dict:
        """获取数字孪生洞察（增强版）"""
        current_state = self.get_current_state()
        
        insights = {
            'health_trends': self._analyze_health_trends(),
            'optimization_opportunities': self._identify_optimization_opportunities(),
            'risk_assessment': self._assess_risks(),
            'prediction_confidence': self._calculate_prediction_confidence(),
            # 新增可行动洞察
            'lifestyle_insights': self.get_lifestyle_insights(),
            'environmental_insights': self.get_environmental_recommendations(),
            'action_priority': self._determine_action_priority()
        }
        
        return insights
    
    def _determine_action_priority(self) -> List[Dict]:
        """确定行动优先级 - 整合所有洞察"""
        priorities = []
        
        # 从生活方式洞察获取高优先级项
        lifestyle_insights = self.get_lifestyle_insights()
        for improvement in lifestyle_insights.get('actionable_improvements', []):
            if improvement.get('priority') == 'high':
                priorities.append({
                    'type': 'lifestyle',
                    'factor': improvement['factor'],
                    'action': improvement['suggestion'],
                    'priority': 'high',
                    'expected_impact': 'moderate_high'
                })
        
        # 从环境建议获取紧急项
        env_recommendations = self.get_environmental_recommendations()
        if env_recommendations.get('urgency_level') == 'high':
            for measure in env_recommendations.get('protective_measures', []):
                if measure.get('level') in ['extreme', 'high']:
                    priorities.append({
                        'type': 'environment',
                        'factor': measure['type'],
                        'action': measure['actions'][0],  # 取第一个建议
                        'priority': 'high',
                        'expected_impact': 'immediate'
                    })
        
        # 从健康趋势获取关键项
        health_trends = self._analyze_health_trends()
        for param, trend in health_trends.items():
            if trend == 'critical_needs_attention':
                priorities.append({
                    'type': 'skin_health',
                    'factor': param,
                    'action': f'Address {param} immediately',
                    'priority': 'high',
                    'expected_impact': 'high'
                })
        
        # 去重并排序
        unique_priorities = []
        seen_factors = set()
        
        for priority in priorities:
            key = f"{priority['type']}_{priority['factor']}"
            if key not in seen_factors:
                seen_factors.add(key)
                unique_priorities.append(priority)
        
        # 按优先级排序
        priority_order = {'high': 0, 'medium_high': 1, 'medium': 2, 'low': 3}
        unique_priorities.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 3))
        
        return unique_priorities[:5]  # 返回前5个优先级



    def _initialize_physical_state(self) -> Dict:
        """初始化物理状态 - 增强版，包含更多皮肤参数"""
        return {
            'skin_parameters': {
                # 核心皮肤参数（0-100评分，越高越好）
                'pores': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'sebum': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'redness': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'wrinkles': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'barrier': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'melanin': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'elasticity': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'collagen': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                
                # 新增皮肤参数
                'hydration': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'smoothness': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'firmness': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'brightness': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'evenness': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'texture': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'pore_size': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'acne_prone': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                'sensitivity': {'score': 50, 'level': 'normal', 'trend': 'stable', 'confidence': 0.5},
                
                # 新增：参数依赖关系
                'interdependencies': {
                    'barrier_hydration': {'strength': 0.7, 'type': 'positive'},
                    'collagen_elasticity': {'strength': 0.8, 'type': 'positive'},
                    'sebum_acne': {'strength': 0.6, 'type': 'positive'}
                }
            },
            
            # 材料物理属性
            'material_properties': {
                'elasticity_modulus': 2.5,  # 弹性模量 (MPa)
                'thermal_conductivity': 0.21,  # 热导率 (W/m·K)
                'hydration_level': 0.65,  # 水合水平 (0-1)
                'trans_epidermal_water_loss': 15.0,  # 经皮水分流失 (g/m²·h)
                'sebum_secretion_rate': 1.2,  # 皮脂分泌率 (μg/cm²·h)
                'ph_level': 5.5,  # pH值
                'skin_thickness': 1.2  # 皮肤厚度 (mm)
            },
            
            # 纹理和几何特征
            'texture_metrics': {
                'roughness': {'value': 2.5, 'unit': 'μm'},  # 粗糙度
                'smoothness': {'value': 0.7, 'unit': 'index'},  # 光滑度指数
                'uniformity': {'value': 0.65, 'unit': 'index'},  # 均匀度指数
                'porosity': {'value': 0.15, 'unit': 'density'},  # 孔隙率
                'wrinkle_density': {'value': 0.08, 'unit': 'lines/mm²'},  # 皱纹密度
                'line_depth': {'value': 0.03, 'unit': 'mm'}  # 纹路深度
            },
            
            # 区域特征（面部不同区域）
            'regional_characteristics': {
                'forehead': {
                    'sebum': 'high',
                    'hydration': 'medium',
                    'pores': 'medium'
                },
                'cheeks': {
                    'sebum': 'low',
                    'hydration': 'high',
                    'sensitivity': 'high'
                },
                'nose': {
                    'sebum': 'very_high',
                    'pores': 'large',
                    'hydration': 'low'
                },
                'chin': {
                    'sebum': 'medium',
                    'acne_prone': 'high',
                    'texture': 'rough'
                }
            },
            
            # 光学特性
            'optical_properties': {
                'reflectance': 0.35,
                'transparency': 0.25,
                'gloss': 0.4,
                'color_uniformity': 0.7
            },
            
            'last_calibration': datetime.now().isoformat(),
            'data_source_confidence': 0.5
        }

    def _initialize_biological_state(self) -> Dict:
        """初始化生物状态 - 超简化实用版"""
        return {
            # 核心生物状态（可从皮肤外观推断）
            'core_indicators': {
                # 屏障健康度（基于皮肤是否干燥、敏感）
                'barrier_health': {
                    'score': 50,
                    'evidence': [],  # 从图像和问卷收集的证据
                    'recommendation_priority': 'medium'
                },
                
                # 炎症状态（基于泛红、痘痘）
                'inflammation_status': {
                    'score': 50,
                    'visible_signs': 0,  # 可见炎症迹象数量
                    'severity': 'low'
                },
                
                # 老化进程（基于皱纹、纹理）
                'aging_progress': {
                    'relative_age': 50,  # 相对年龄（0=年轻，100=老化）
                    'wrinkle_score': 50,
                    'elasticity_perceived': 50
                }
            },
            
            # 皮肤类型分类（基于可见特征）
            'skin_type_assessment': {
                'baumann_type': 'unknown',  # 鲍曼皮肤类型
                'fitzpatrick_type': 'unknown',  # 菲茨帕特里克皮肤类型
                'visia_parameters': {}  # 类似VISIA分析参数
            },
            
            # 问题区域识别
            'problem_areas': {
                't_zone': {
                    'sebum': 'normal',
                    'pores': 'normal'
                },
                'cheeks': {
                    'dryness': 'normal',
                    'redness': 'normal'
                },
                'forehead': {
                    'wrinkles': 'minimal',
                    'texture': 'smooth'
                }
            },
            
            # 动态变化跟踪
            'dynamic_changes': {
                'weekly_variation': 10,  # 周变化率 (%)
                'seasonal_pattern': 'unknown',  # 季节性模式
                'product_response': 'neutral'  # 产品反应
            },
            
            # 数据质量标记
            'metadata': {
                'data_source': 'camera_questionnaire',
                'collection_method': 'user_provided',
                'accuracy_estimate': 0.7,
                'last_updated': datetime.now().isoformat()
            }
        }

    def _initialize_psychological_state(self) -> Dict:
        """初始化心理状态"""
        return {
            'emotional_state': {
                'mood': 'neutral',
                'stress_level': 0,
                'emotional_stability': 'medium'
            },
            'behavioral_patterns': {
                'routine_consistency': 0,
                'product_adherence': 0,
                'compliance_rate': 0
            },
            'motivational_factors': {
                'intrinsic_motivation': 0,
                'extrinsic_motivation': 0,
                'goal_alignment': 0
            }
        }

    def _initialize_geometric_state(self) -> Dict:
        """初始化几何状态"""
        return {
            'spatial_patterns': {
                'symmetry': 0,
                'fractal_dimension': 0,
                'texture_regularity': 0
            },
            'temporal_patterns': {
                'cycles_detected': [],
                'trend_direction': 'stable',
                'change_rate': 0
            }
        }

    def _initialize_economic_state(self) -> Dict:
        """初始化经济状态"""
        return {
            'resource_allocation': {
                'budget_utilization': 0,
                'time_investment': 0,
                'cost_effectiveness': 0
            },
            'investment_state': {
                'roi': 0,
                'sunk_cost': 0,
                'future_value': 0
            }
        }

    def _initialize_risk_state(self) -> Dict:
        """初始化风险状态"""
        return {
            'extreme_risks': [],
            'risk_profile': {
                'risk_tolerance': 'medium',
                'vulnerability_score': 0,
                'resilience_score': 0
            },
            'mitigation_status': {
                'mitigation_plans': [],
                'effectiveness': 0,
                'coverage': 0
            }
        }

    def _initialize_cognitive_state(self) -> Dict:
        """初始化认知状态 - 扩展包含冥想和无我心智"""
        return {
            'belief_systems': {
                'skincare_beliefs': [],
                'evidence_based': 0,
                'openness_to_change': 0
            },
            'decision_models': {
                'decision_style': 'balanced',
                'risk_perception': 0,
                'information_processing': 'moderate'
            },
            'meditation': {
                'practice_frequency': 0,  # 每周练习频率
                'average_duration': 0,     # 平均每次练习时长（分钟）
                'current_state': 'observer',  # 当前冥想状态：observer, participant, anatta
                'state_history': [],        # 冥想状态历史
                'focus_level': 50,          # 专注水平（0-100）
                'mindfulness_score': 50,    # 正念分数（0-100）
                'last_session': None        # 最后一次练习时间
            },
            'perception': {
                'self_perception': 50,      # 自我感知水平
                'anatta_awareness': 30,     # 无我意识水平（0-100）
                'non_attachment_level': 30  # 不执着程度（0-100）
            }
        }

    def _initialize_temporal_state(self) -> Dict:
        """初始化时间状态"""
        return {
            'historical_trends': {
                'improvement_rate': 0,
                'consistency': 0,
                'seasonal_patterns': []
            },
            'future_projections': {
                'optimistic_scenario': {},
                'pessimistic_scenario': {},
                'most_likely_scenario': {}
            }
        }

    def update_from_real_world(self, real_data: Dict, framework_type: str = None):
        """从现实世界数据更新数字孪生"""
        self.last_updated = datetime.now()
        
        # 根据数据类型更新对应状态层
        if 'skin_analysis' in real_data:
            self._update_physical_state(real_data['skin_analysis'])
        
        if 'user_profile' in real_data:
            self._update_psychological_state(real_data['user_profile'])
            self._update_economic_state(real_data['user_profile'])
        
        if 'environment' in real_data:
            self._update_risk_state(real_data['environment'])
        
        # 记录现实数据
        self.reality_history.append({
            'timestamp': datetime.now(),
            'data': real_data,
            'framework': framework_type
        })
        
        # 保留最近100条记录
        if len(self.reality_history) > 100:
            self.reality_history = self.reality_history[-100:]

    def _update_physical_state(self, skin_analysis: Dict):
        """更新物理状态"""
        for param in ['pores', 'sebum', 'redness', 'wrinkles', 'barrier', 'melanin', 'elasticity', 'collagen']:
            if param in skin_analysis:
                param_data = skin_analysis[param]
                if isinstance(param_data, dict):
                    self.states['physical']['skin_parameters'][param] = param_data
        
        # 更新纹理指标
        if 'texture' in skin_analysis:
            texture_data = skin_analysis['texture']
            if isinstance(texture_data, dict):
                self.states['physical']['texture_metrics'].update(texture_data)

    def _update_psychological_state(self, user_profile: Dict):
        """更新心理状态"""
        # 情绪状态
        if 'emotional_state' in user_profile:
            self.states['psychological']['emotional_state'].update(user_profile['emotional_state'])
        
        # 行为模式
        if 'behavior_patterns' in user_profile:
            self.states['psychological']['behavioral_patterns'].update(user_profile['behavior_patterns'])

    def _update_economic_state(self, user_profile: Dict):
        """更新经济状态"""
        if 'financial' in user_profile:
            financial_data = user_profile['financial']
            self.states['economic']['resource_allocation']['budget_utilization'] = \
                financial_data.get('budget_utilization_rate', 0)

    def _update_risk_state(self, environment: Dict):
        """更新风险状态"""
        # 基于环境数据评估风险
        risks = []
        if environment.get('uv_index', 0) > 8:
            risks.append({
                'type': 'high_uv_exposure',
                'severity': 'high',
                'probability': 0.7
            })
        
        if environment.get('temperature', 25) > 35:
            risks.append({
                'type': 'heat_stress',
                'severity': 'medium',
                'probability': 0.5
            })
        
        self.states['risk']['extreme_risks'] = risks

    def run_simulation(self, intervention: Dict, duration_days: int = 30) -> Dict:
        """运行模拟：预测干预措施的效果"""
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 基于当前状态和干预措施运行模拟
        simulation_results = {
            'simulation_id': simulation_id,
            'intervention': intervention,
            'duration_days': duration_days,
            'initial_state': self.get_current_state(),
            'daily_progress': [],
            'final_predictions': {}
        }
        
        # 简化的逐日模拟
        for day in range(1, duration_days + 1):
            daily_state = self._simulate_daily_progress(intervention, day)
            simulation_results['daily_progress'].append(daily_state)
        
        # 提取最终预测
        final_state = simulation_results['daily_progress'][-1] if simulation_results['daily_progress'] else {}
        simulation_results['final_predictions'] = self._extract_predictions(final_state, intervention)
        
        # 记录模拟历史
        self.simulation_history.append({
            'timestamp': datetime.now(),
            'simulation': simulation_results
        })
        
        return simulation_results

    def _simulate_daily_progress(self, intervention: Dict, day: int) -> Dict:
        """模拟单日进展"""
        daily_state = {}
        
        # 模拟物理参数变化
        for param in ['pores', 'sebum', 'redness', 'wrinkles', 'barrier', 'melanin', 'elasticity', 'collagen']:
            current_score = self.states['physical']['skin_parameters'].get(param, {}).get('score', 50)
            
            # 基于干预效果计算改进
            improvement = self._calculate_parameter_improvement(param, intervention, day)
            predicted_score = min(100, current_score + improvement)
            
            daily_state[param] = {
                'day': day,
                'current': current_score,
                'predicted': predicted_score,
                'improvement': improvement
            }
        
        return daily_state

    def _calculate_parameter_improvement(self, parameter: str, intervention: Dict, day: int) -> float:
        """计算参数改进"""
        # 简化模型：根据参数类型和干预措施计算改进
        base_rates = {
            'barrier': 0.5,    # 屏障修复较快
            'hydration': 0.8,  # 水合作用改善快
            'redness': 0.3,    # 泛红减少较慢
            'wrinkles': 0.1,   # 皱纹改善很慢
            'elasticity': 0.2, # 弹性改善中等
            'collagen': 0.15   # 胶原蛋白生成慢
        }
        
        base_rate = base_rates.get(parameter, 0.2)
        
        # 考虑干预效果
        intervention_efficacy = intervention.get('efficacy', 0.5)
        
        # 考虑时间衰减（边际收益递减）
        time_factor = max(0.1, 1.0 - (day / 30) * 0.5)
        
        # 考虑个体差异（从当前状态推导）
        current_score = self.states['physical']['skin_parameters'].get(parameter, {}).get('score', 50)
        individual_factor = (100 - current_score) / 100  # 分数越低，改进空间越大
        
        return base_rate * intervention_efficacy * time_factor * individual_factor * 10

    def _extract_predictions(self, final_state: Dict, intervention: Dict) -> Dict:
        """从最终状态提取预测"""
        predictions = {
            'overall_improvement': 0,
            'parameter_improvements': {},
            'recommendations': [],
            'confidence_score': 0.7
        }
        
        improvements = []
        for param, data in final_state.items():
            if isinstance(data, dict) and 'improvement' in data:
                improvement = data['improvement']
                predictions['parameter_improvements'][param] = improvement
                improvements.append(improvement)
        
        if improvements:
            predictions['overall_improvement'] = sum(improvements) / len(improvements)
        
        # 基于改进生成推荐
        if predictions['overall_improvement'] > 15:
            predictions['recommendations'].append('Continue current intervention')
        elif predictions['overall_improvement'] > 5:
            predictions['recommendations'].append('Consider adjusting intervention')
        else:
            predictions['recommendations'].append('Re-evaluate intervention strategy')
        
        return predictions

    def compare_with_reality(self, real_data: Dict) -> Dict:
        """比较模拟与现实：计算现实差距"""
        if not self.simulation_history:
            return {'error': 'No simulations to compare'}
        
        latest_simulation = self.simulation_history[-1]['simulation']
        predictions = latest_simulation.get('final_predictions', {})
        
        gap_analysis = {
            'timestamp': datetime.now(),
            'simulation_id': latest_simulation.get('simulation_id'),
            'parameter_gaps': {},
            'overall_gap': 0,
            'model_accuracy': 0,
            'calibration_needed': False
        }
        
        # 比较参数预测
        parameter_gaps = []
        for param, predicted_data in predictions.get('parameter_improvements', {}).items():
            if param in real_data:
                actual_improvement = real_data[param].get('improvement', 0) if isinstance(real_data[param], dict) else 0
                gap = abs(predicted_data - actual_improvement)
                gap_analysis['parameter_gaps'][param] = gap
                parameter_gaps.append(gap)
        
        if parameter_gaps:
            gap_analysis['overall_gap'] = sum(parameter_gaps) / len(parameter_gaps)
            gap_analysis['model_accuracy'] = max(0, 100 - gap_analysis['overall_gap'] * 10)
            gap_analysis['calibration_needed'] = gap_analysis['overall_gap'] > 5
        
        # 记录现实差距
        self.reality_gaps.append(gap_analysis)
        
        return gap_analysis

    def optimize_models(self):
        """基于现实差距优化模型"""
        if not self.reality_gaps:
            return {'status': 'no_data', 'message': 'No reality gaps to optimize from'}
        
        # 分析最近的现实差距
        recent_gaps = self.reality_gaps[-5:] if len(self.reality_gaps) >= 5 else self.reality_gaps
        
        gap_analysis = {
            'total_comparisons': len(recent_gaps),
            'average_gap': 0,
            'parameter_biases': {},
            'optimization_suggestions': []
        }
        
        # 计算平均差距
        gaps = [gap.get('overall_gap', 0) for gap in recent_gaps]
        if gaps:
            gap_analysis['average_gap'] = sum(gaps) / len(gaps)
        
        # 识别参数偏差
        parameter_biases = {}
        for gap in recent_gaps:
            for param, param_gap in gap.get('parameter_gaps', {}).items():
                if param not in parameter_biases:
                    parameter_biases[param] = []
                parameter_biases[param].append(param_gap)
        
        for param, biases in parameter_biases.items():
            avg_bias = sum(biases) / len(biases)
            gap_analysis['parameter_biases'][param] = avg_bias
            
            # 基于偏差提出优化建议
            if avg_bias > 7:
                gap_analysis['optimization_suggestions'].append(
                    f"Adjust {param} improvement model: current bias = {avg_bias:.2f}"
                )
        
        # 如果平均差距较大，建议重新校准
        if gap_analysis['average_gap'] > 8:
            gap_analysis['optimization_suggestions'].append(
                "Consider comprehensive model recalibration"
            )
        
        return gap_analysis



    def get_insights(self) -> Dict:
        """获取数字孪生洞察"""
        current_state = self.get_current_state()
        
        insights = {
            'health_trends': self._analyze_health_trends(),
            'optimization_opportunities': self._identify_optimization_opportunities(),
            'risk_assessment': self._assess_risks(),
            'prediction_confidence': self._calculate_prediction_confidence()
        }
        
        return insights

    def _analyze_health_trends(self) -> Dict:
        """分析健康趋势"""
        # 简化的趋势分析
        trends = {}
        
        # 分析物理参数趋势
        physical_params = self.states['physical']['skin_parameters']
        for param, data in physical_params.items():
            if isinstance(data, dict) and 'score' in data:
                score = data['score']
                if score < 40:
                    trends[param] = 'critical_needs_attention'
                elif score < 60:
                    trends[param] = 'needs_improvement'
                elif score < 80:
                    trends[param] = 'adequate'
                else:
                    trends[param] = 'optimal'
        
        return trends

    def _identify_optimization_opportunities(self) -> List[Dict]:
        """识别优化机会"""
        opportunities = []
        
        # 基于参数状态识别优化机会
        physical_params = self.states['physical']['skin_parameters']
        
        for param, data in physical_params.items():
            if isinstance(data, dict) and 'score' in data:
                score = data['score']
                if score < 60:
                    opportunities.append({
                        'parameter': param,
                        'current_score': score,
                        'target_score': 80,
                        'priority': 'high' if score < 40 else 'medium',
                        'suggested_actions': self._get_optimization_actions(param)
                    })
        
        # 基于经济状态识别优化机会
        budget_utilization = self.states['economic']['resource_allocation'].get('budget_utilization', 0)
        if budget_utilization < 0.5:
            opportunities.append({
                'parameter': 'budget_utilization',
                'current_score': budget_utilization,
                'target_score': 0.8,
                'priority': 'medium',
                'suggested_actions': ['Review spending patterns', 'Consider reallocating budget']
            })
        
        return opportunities

    def _get_optimization_actions(self, parameter: str) -> List[str]:
        """获取优化行动建议"""
        action_map = {
            'barrier': ['Use barrier repair creams', 'Avoid harsh cleansers', 'Incorporate ceramides'],
            'hydration': ['Increase water intake', 'Use humectants like hyaluronic acid', 'Apply moisturizer regularly'],
            'redness': ['Use anti-inflammatory ingredients', 'Avoid triggers', 'Consider professional treatment'],
            'wrinkles': ['Use retinoids', 'Incorporate peptides', 'Sun protection'],
            'elasticity': ['Collagen-boosting products', 'Facial exercises', 'Professional treatments']
        }
        
        return action_map.get(parameter, ['Consult dermatologist for personalized advice'])

    def _assess_risks(self) -> Dict:
        """评估风险"""
        risk_assessment = {
            'immediate_risks': [],
            'long_term_risks': [],
            'resilience_score': 0
        }
        
        # 从风险状态提取
        extreme_risks = self.states['risk'].get('extreme_risks', [])
        for risk in extreme_risks:
            if risk.get('severity') == 'high':
                risk_assessment['immediate_risks'].append(risk.get('type'))
            else:
                risk_assessment['long_term_risks'].append(risk.get('type'))
        
        # 计算韧性分数
        resilience_factors = []
        
        # 屏障功能
        barrier_score = self.states['physical']['skin_parameters'].get('barrier', {}).get('score', 50)
        if barrier_score > 70:
            resilience_factors.append(1.0)
        elif barrier_score > 50:
            resilience_factors.append(0.5)
        else:
            resilience_factors.append(0.2)
        
        # 细胞健康
        oxidative_stress = self.states['biological']['cellular_health'].get('oxidative_stress', 0)
        if oxidative_stress < 30:
            resilience_factors.append(1.0)
        elif oxidative_stress < 60:
            resilience_factors.append(0.5)
        else:
            resilience_factors.append(0.2)
        
        if resilience_factors:
            risk_assessment['resilience_score'] = sum(resilience_factors) / len(resilience_factors)
        
        return risk_assessment

    def _calculate_prediction_confidence(self) -> float:
        """计算预测置信度"""
        if not self.reality_gaps:
            return 0.5  # 默认中等置信度
        
        # 基于现实差距计算置信度
        recent_gaps = self.reality_gaps[-3:] if len(self.reality_gaps) >= 3 else self.reality_gaps
        
        accuracies = []
        for gap in recent_gaps:
            accuracy = gap.get('model_accuracy', 50)
            accuracies.append(accuracy / 100)  # 转换为0-1范围
        
        if accuracies:
            return sum(accuracies) / len(accuracies)
        
        return 0.5

    def get_environmental_recommendations(self):
        """获取环境建议 - 增强版，支持新的环境参数"""
        try:
            # 安全地获取环境数据
            env_state = self.states.get('environment', {})
            
            # 安全地获取所有需要的环境数据
            conditions = env_state.get('current_conditions', {})
            exposure = env_state.get('exposure_patterns', {})
            
            # 如果conditions是空字典，尝试直接使用env_state
            if not conditions and isinstance(env_state, dict):
                # 检查env_state是否本身就是条件数据（没有current_conditions键）
                if 'uv_index' in env_state or 'uv' in env_state:
                    conditions = env_state
            
            # 提取环境参数，全部使用安全的get方法
            uv = conditions.get('uv_index', conditions.get('uv', 3))
            humidity = conditions.get('humidity', 60)
            temperature = conditions.get('temperature', 25)
            air_quality = conditions.get('air_quality', conditions.get('aq', 50))
            pollen_level = conditions.get('pollen_level', 0)
            wind_speed = conditions.get('wind_speed', 0)
            
            # 新增：提取新的环境参数
            noise_level = conditions.get('noise_level', 40)
            pm25 = conditions.get('pm25', 25)
            light_intensity = conditions.get('light_intensity', 50)
            
            # 提取暴露模式数据
            outdoor_hours = exposure.get('outdoor_hours', 0)
            indoor_hours = exposure.get('indoor_hours', 8)
            
            # 生成建议
            recommendations = []
            
            # UV防护建议
            if uv >= 8:
                recommendations.append("使用SPF50+防晒霜，佩戴太阳镜和帽子")
            elif uv >= 6:
                recommendations.append("使用SPF30+防晒霜")
            elif uv >= 3:
                recommendations.append("使用基础防晒保护")
            
            # 空气质量建议
            if air_quality >= 150:
                recommendations.append("避免户外活动，使用空气净化器")
            elif air_quality >= 100:
                recommendations.append("减少户外活动，佩戴口罩")
            elif air_quality >= 50:
                recommendations.append("注意空气质量，敏感人群减少外出")
            
            # 新增：PM2.5建议
            if pm25 > 75:
                recommendations.append("PM2.5浓度较高，外出佩戴N95口罩，使用空气净化器")
            elif pm25 > 35:
                recommendations.append("PM2.5浓度中等，敏感人群减少户外活动")
            
            # 温湿度建议
            if humidity < 30:
                recommendations.append("使用加湿器，加强保湿护肤")
            elif humidity > 75:
                recommendations.append("注意防潮，使用控油护肤品")
            
            if temperature > 35:
                recommendations.append("避免日间外出，注意防暑降温")
            elif temperature < 5:
                recommendations.append("注意保暖，防止皮肤冻伤")
            
            # 花粉建议
            if pollen_level >= 4:
                recommendations.append("高花粉浓度，过敏者避免外出")
            elif pollen_level >= 3:
                recommendations.append("花粉浓度较高，注意防护")
            
            # 新增：噪音建议
            if noise_level > 80:
                recommendations.append("噪音水平较高，建议使用耳塞或降噪耳机保护听力")
            elif noise_level > 60:
                recommendations.append("噪音水平中等，注意避免长时间暴露")
            
            # 新增：光照建议
            if light_intensity > 90:
                recommendations.append("光照过强，建议使用遮阳帘或佩戴太阳镜")
            elif light_intensity < 10:
                recommendations.append("光照不足，建议使用适当的照明设备")
            
            # 暴露模式建议
            if outdoor_hours > 6:
                recommendations.append("长时间户外活动，需加强防护")
            elif outdoor_hours > 3:
                recommendations.append("中等时长户外活动，注意补涂防晒")
            
            # 评估紧急程度
            urgency = self._assess_environmental_urgency_safe({
                'uv': uv,
                'air_quality': air_quality,
                'pollen_level': pollen_level,
                'pm25': pm25,
                'noise_level': noise_level,
                'temperature': temperature
            })
            
            return {
                'daily_adjustments': [],
                'protective_measures': recommendations,
                'urgency_level': urgency
            }
            
        except Exception as e:
            print(f"获取环境建议时出错: {e}")
            return {
                'daily_adjustments': [],
                'protective_measures': ["注意环境变化，做好基础防护"],
                'urgency_level': 'normal'
            }
    
    def _assess_environmental_urgency_safe(self, env_data=None):
        """安全的环境紧急程度评估 - 增强版，考虑新的环境参数"""
        if env_data is None:
            env_data = {}
        
        uv = env_data.get('uv', env_data.get('uv_index', 3))
        air_quality = env_data.get('air_quality', env_data.get('aq', 50))
        pollen_level = env_data.get('pollen_level', 0)
        
        # 新增：获取新的环境参数
        pm25 = env_data.get('pm25', 25)
        noise_level = env_data.get('noise_level', 40)
        temperature = env_data.get('temperature', 25)
        
        risk_score = 0
        
        # UV风险
        if uv >= 11:
            risk_score += 3
        elif uv >= 8:
            risk_score += 2
        elif uv >= 6:
            risk_score += 1
        
        # 空气质量风险
        if air_quality >= 200:
            risk_score += 3
        elif air_quality >= 150:
            risk_score += 2
        elif air_quality >= 100:
            risk_score += 1
        
        # 花粉风险
        if pollen_level >= 4:
            risk_score += 2
        elif pollen_level >= 3:
            risk_score += 1
        
        # 新增：PM2.5风险
        if pm25 > 150:
            risk_score += 3
        elif pm25 > 75:
            risk_score += 2
        elif pm25 > 35:
            risk_score += 1
        
        # 新增：噪音风险
        if noise_level > 100:
            risk_score += 3
        elif noise_level > 80:
            risk_score += 2
        elif noise_level > 60:
            risk_score += 1
        
        # 新增：温度风险
        if temperature > 40 or temperature < -10:
            risk_score += 3
        elif temperature > 35 or temperature < 0:
            risk_score += 2
        elif temperature > 30 or temperature < 5:
            risk_score += 1
        
        if risk_score >= 6:
            return "severe"
        elif risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "normal"

class GeometricDynamicsFramework:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    """几何动力学框架 - 整合群论对称性思想的时空结构分析"""
    
    def __init__(self):
        # 初始化群论对称性分析组件
        self.symmetry_groups = {
            'spatial': ['Z2', 'Euclidean_3D', 'Scale'],  # 空间对称群
            'temporal': ['TimeTranslation', 'Periodic', 'Symplectic']  # 时间对称群
        }

    # 在 GeometricDynamicsFramework 类中添加
    def _calculate_certainty(self, geometry_results):
        """计算几何动力学框架的确定性"""
        try:
            # 根据几何参数计算
            surface_topology = geometry_results.get('surface_topology', 'unknown')
            symmetry_analysis = geometry_results.get('symmetry_analysis', {})
            
            # 简单计算
            topology_score = 0.5
            if surface_topology == 'smooth':
                topology_score = 0.8
            elif surface_topology == 'rough':
                topology_score = 0.3
            
            symmetry_score = symmetry_analysis.get('score', 0.5)
            
            certainty = (topology_score + symmetry_score) / 2
            return max(0.1, min(1.0, certainty))
            
        except Exception as e:
            logger.error(f"几何框架计算确定性失败: {e}")
            return 0.1
    
    async def infer(self, skin_params, context=None):
        """几何动力学框架推理 - 修复版"""
        try:
            # 几何参数分析逻辑
            surface_topology = skin_params.get('surface_topology', 'normal')
            symmetry_analysis = {'score': 0.8}
            
            # 创建结果
            result = {
                'surface_topology': surface_topology,
                'symmetry_analysis': symmetry_analysis,
                'geometric_score': 0.9 if surface_topology == 'smooth' else 0.6
            }
            
            # 关键修复：调用 _calculate_certainty 方法
            result['certainty'] = self._calculate_certainty(result)
            result['framework_status'] = 'success'
            
            return result
            
        except Exception as e:
            logger.error(f"几何框架推理失败: {e}")
            return {
                'surface_topology': 'unknown',
                'symmetry_analysis': {'score': 0.5},
                'certainty': 0.1,
                'framework_status': 'error'
            }
    
    def _analyze_spatial_structure(self, context: Dict) -> Dict:
        """分析空间结构"""
        skin_analysis = context.get('skin_analysis', {})
        
        return {
            'surface_topology': self._analyze_surface_topology(skin_analysis),
            'regional_variations': self._analyze_regional_variations(skin_analysis),
            'spatial_patterns': self._identify_spatial_patterns(skin_analysis),
            'spatial_symmetry': self._analyze_spatial_symmetry(skin_analysis)  # 新增：空间对称性分析
        }
    
    def _analyze_surface_topology(self, skin_analysis: Dict) -> Dict:
        """分析表面拓扑结构"""
        return {
            'roughness': 'moderate',
            'texture': 'uniform',
            'topography': 'smooth'
        }
    
    def _analyze_regional_variations(self, skin_analysis: Dict) -> Dict:
        """分析区域差异"""
        return {
            'forehead': {'sebum': 'high'},
            'cheeks': {'hydration': 'medium'},
            't_zone': {'oiliness': 'high'}
        }
    
    def _identify_spatial_patterns(self, skin_analysis: Dict) -> List[str]:
        """识别空间模式"""
        return ['uniform', 'symmetrical', 'balanced']
    
    def _analyze_spatial_symmetry(self, skin_analysis: Dict) -> Dict:
        """分析空间对称性 - 群论应用：Z₂群、欧几里得群等"""
        # 分析面部双侧镜像对称性（Z₂群）
        z2_symmetry_score = self._calculate_z2_symmetry(skin_analysis)
        
        # 分析3D空间变换对称性（欧几里得群）
        euclidean_symmetry = self._analyze_euclidean_symmetry(skin_analysis)
        
        return {
            'z2_mirror_symmetry': {
                'score': z2_symmetry_score,
                'interpretation': 'high' if z2_symmetry_score > 0.8 else 'moderate' if z2_symmetry_score > 0.5 else 'low',
                'anomalies': self._detect_asymmetrical_anomalies(skin_analysis)
            },
            'euclidean_symmetry': euclidean_symmetry,
            'scale_invariance': self._analyze_scale_invariance(skin_analysis)
        }
    
    def _calculate_z2_symmetry(self, skin_analysis: Dict) -> float:
        """计算面部双侧镜像对称性（Z₂群）得分"""
        # 简化实现：模拟对称性得分计算
        # 在实际应用中，应比较左右脸对应部位的皮肤参数
        return 0.85  # 示例得分：高对称性
    
    def _analyze_euclidean_symmetry(self, skin_analysis: Dict) -> Dict:
        """分析3D空间变换（旋转、平移）下的对称性"""
        return {
            'rotation_invariance': 0.9,  # 旋转不变性得分
            'translation_invariance': 0.85,  # 平移不变性得分
            'equivariant_features': ['texture', 'color_uniformity', 'pore_density']
        }
    
    def _analyze_scale_invariance(self, skin_analysis: Dict) -> Dict:
        """分析尺度变换下的对称性"""
        return {
            'scale_invariance_score': 0.8,
            'scale_equivariant_features': ['wrinkle_pattern', 'skin_tone_gradient', 'moisture_distribution']
        }
    
    def _detect_asymmetrical_anomalies(self, skin_analysis: Dict) -> List[Dict]:
        """检测单侧异常（基于Z₂群对称性破坏）"""
        # 简化实现：模拟检测单侧异常
        return [
            {'region': 'left_cheek', 'type': 'redness', 'severity': 'mild'},
            {'region': 'right_eye', 'type': 'wrinkles', 'severity': 'moderate'}
        ]
    
    def _analyze_temporal_evolution(self, context: Dict) -> Dict:
        """分析时间演化 - 整合时间对称群"""
        return {
            'aging_process': 'gradual',
            'healing_rate': 'normal',
            'cycle_length': '28 days',
            'temporal_symmetry': self._analyze_temporal_symmetry(context)  # 新增：时间对称群分析
        }
    
    def _analyze_temporal_symmetry(self, context: Dict) -> Dict:
        """分析时间对称群 - 如时间平移对称性、周期对称性"""
        # 分析护肤周期的时间平移对称性
        skincare_cycle_symmetry = self._analyze_skincare_cycle_symmetry(context)
        
        # 分析成分作用的时间演化对称性
        ingredient_dynamics_symmetry = self._analyze_ingredient_dynamics_symmetry(context)
        
        return {
            'time_translation_symmetry': {
                'score': 0.75,
                'periodic_patterns': ['28-day_cycle', 'day-night_rhythm', 'weekly_fluctuations']
            },
            'skincare_cycle_symmetry': skincare_cycle_symmetry,
            'ingredient_dynamics_symmetry': ingredient_dynamics_symmetry
        }
    
    def _analyze_skincare_cycle_symmetry(self, context: Dict) -> Dict:
        """分析护肤周期的对称性模式"""
        return {
            'cycle_periodicity': '28 days',
            'symmetry_breaking_points': ['menstrual_cycle_phase', 'seasonal_changes'],
            'optimal_intervention_times': ['day_7', 'day_14', 'day_21']
        }
    
    def _analyze_ingredient_dynamics_symmetry(self, context: Dict) -> Dict:
        """分析成分作用的动力学对称性"""
        return {
            'reaction_network_symmetry': 'approximate',
            'symmetric_ingredient_effects': ['hydration', 'barrier_repair'],
            'asymmetric_ingredient_effects': ['exfoliation', 'pigmentation_reduction']
        }
    
    def _identify_geometric_constraints(self, context: Dict) -> Dict:
        """识别几何约束 - 整合对称性约束"""
        return {
            'physical_constraints': ['surface_area', 'curvature', 'elasticity_limits'],
            'symmetry_constraints': self._identify_symmetry_constraints(context),  # 新增：对称性约束
            'biophysical_invariants': self._identify_biophysical_invariants(context)  # 补充完整
        }
    
    def _identify_symmetry_constraints(self, context: Dict) -> List[str]:
        """识别对称性约束"""
        return [
            'z2_mirror_invariance',  # 双侧镜像不变性约束
            'scale_invariance',  # 尺度不变性约束
            'temporal_periodicity_constraint'  # 时间周期约束
        ]
    
    def _identify_biophysical_invariants(self, context: Dict) -> List[str]:
        """识别生物物理规范不变量"""
        return [
            'barrier_integrity_invariant',  # 皮肤屏障完整性不变量
            'hydration_diffusion_invariant',  # 水分扩散不变量
            'sebum_production_balance'  # 皮脂分泌平衡不变量
        ]
    
    def _analyze_symmetry_properties(self, context: Dict) -> Dict:
        """分析对称性属性 - 群论核心分析"""
        return {
            'spatial_groups': self.symmetry_groups['spatial'],
            'temporal_groups': self.symmetry_groups['temporal'],
            'symmetry_breaking_points': self._detect_symmetry_breaking(context),
            'symmetry_guided_interventions': self._generate_symmetry_guided_interventions(context)
        }
    
    def _detect_symmetry_breaking(self, context: Dict) -> List[Dict]:
        """检测对称性破缺点"""
        return [
            {'region': 'left_cheek', 'type': 'redness', 'severity': 'mild'},
            {'cycle_phase': 'menstrual', 'effect': 'increased_sensitivity'}
        ]
    
    def _generate_symmetry_guided_interventions(self, context: Dict) -> List[Dict]:
        """生成对称性引导的干预措施"""
        return [
            {
                'type': 'symmetrical_treatment',
                'target': 'left_cheek_redness',
                'suggestion': '使用抗炎成分对称性涂抹双侧脸颊'
            },
            {
                'type': 'temporal_symmetry_adjustment',
                'target': 'day-night_rhythm',
                'suggestion': '根据昼夜节律调整护肤产品使用时间'
            }
        ]
    

    
    def _analyze_spatial_structure(self, context: Dict) -> Dict:
        """分析空间结构"""
        skin_analysis = context.get('skin_analysis', {})
        
        return {
            'surface_topology': self._analyze_surface_topology(skin_analysis),
            'regional_variations': self._analyze_regional_variations(skin_analysis),
            'spatial_patterns': self._identify_spatial_patterns(skin_analysis)
        }
    
    def _analyze_surface_topology(self, skin_analysis: Dict) -> Dict:
        """分析表面拓扑结构"""
        return {
            'roughness': 'moderate',
            'texture': 'uniform',
            'topography': 'smooth'
        }
    
    def _analyze_regional_variations(self, skin_analysis: Dict) -> Dict:
        """分析区域差异"""
        return {
            'forehead': {'sebum': 'high'},
            'cheeks': {'hydration': 'medium'},
            't_zone': {'oiliness': 'high'}
        }
    
    def _identify_spatial_patterns(self, skin_analysis: Dict) -> List[str]:
        """识别空间模式"""
        return ['uniform', 'symmetrical', 'balanced']
    
    def _analyze_temporal_evolution(self, context: Dict) -> Dict:
        """分析时间演化"""
        return {
            'aging_process': 'gradual',
            'healing_rate': 'normal',
            'cycle_length': '28 days'
        }
    

    
class Real2Sim2RealFramework:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    """Real2Sim2Real框架 - 包含数字孪生作为核心"""
    
    def __init__(self):
        self.digital_twins = {}  # user_id -> SleepMeditationDigitalTwin
        self.optimization_history = []

    def get_digital_twin(self, user_id: str) -> SleepMeditationDigitalTwin:
        """获取或创建用户的数字孪生"""
        if user_id not in self.digital_twins:
            self.digital_twins[user_id] = SleepMeditationDigitalTwin(user_id)
        
        return self.digital_twins[user_id]

    async def infer(self, context: Dict) -> Dict:
        """Real2Sim2Real推理"""
        user_id = context.get('user_id', 'anonymous')
        digital_twin = self.get_digital_twin(user_id)
        
        # 从现实世界更新数字孪生
        digital_twin.update_from_real_world(context, 'real2sim2real')
        
        # 获取当前状态
        current_state = digital_twin.get_current_state()
        
        # 如果有建议的干预措施，运行模拟
        simulations = {}
        if 'suggested_interventions' in context:
            interventions = context['suggested_interventions']
            for intervention in interventions[:2]:  # 最多模拟2个干预措施
                sim_result = digital_twin.run_simulation(intervention)
                simulations[intervention.get('name', 'unknown')] = sim_result
        
        # 如果有历史数据，比较现实差距
        reality_comparison = {}
        if 'historical_data' in context:
            reality_comparison = digital_twin.compare_with_reality(context['historical_data'])
        
        # 定期优化模型
        optimization_needed = len(digital_twin.reality_gaps) % 10 == 0  # 每10次比较优化一次
        optimization_result = {}
        if optimization_needed:
            optimization_result = digital_twin.optimize_models()
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'user_id': user_id,
                'result': optimization_result
            })
        
        # 获取洞察
        insights = digital_twin.get_insights()
        
        return {
            'digital_twin_state': current_state,
            'simulation_results': simulations,
            'reality_comparison': reality_comparison,
            'optimization_result': optimization_result,
            'insights': insights,
            'certainty': digital_twin._calculate_prediction_confidence()
        }

    async def infer_with_lifestyle(self, context: Dict) -> Dict:
        """增强推理：包含生活方式和环境因素"""
        user_id = context.get('user_id', 'anonymous')
        digital_twin = self.get_digital_twin(user_id)
        
        # 从现实世界更新数字孪生（包含新因素）
        digital_twin.update_from_real_world(context, 'real2sim2real_enhanced')
        
        # 获取当前状态（包含新状态层）
        current_state = digital_twin.get_current_state()
        
        # 运行增强模拟
        simulations = {}
        if 'suggested_interventions' in context:
            interventions = context['suggested_interventions']
            for intervention in interventions[:2]:
                # 使用增强模拟
                sim_result = digital_twin.run_simplified_enhanced_simulation(intervention)
                simulations[intervention.get('name', 'unknown')] = sim_result
        
        # 比较现实差距
        reality_comparison = {}
        if 'historical_data' in context:
            reality_comparison = digital_twin.compare_with_reality(context['historical_data'])
        
        # 获取洞察（包含可行动建议）
        insights = digital_twin.get_insights()
        
        return {
            'digital_twin_state': current_state,
            'simulation_results': simulations,
            'reality_comparison': reality_comparison,
            'insights': insights,
            'action_priorities': insights.get('action_priority', []),
            'certainty': digital_twin._calculate_prediction_confidence()
        }
    
    def get_lifestyle_recommendations(self, user_id: str) -> Dict:
        """获取针对用户的个性化生活方式建议"""
        digital_twin = self.get_digital_twin(user_id)
        return digital_twin.get_lifestyle_insights()
    
    def get_environmental_adaptations(self, user_id: str) -> Dict:
        """获取环境适应建议"""
        digital_twin = self.get_digital_twin(user_id)
        return digital_twin.get_environmental_recommendations()

class LifestyleDataCollector:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    """简化生活方式数据收集器"""
    
    @staticmethod
    def create_simplified_questionnaire() -> Dict:
        """创建简化生活方式问卷（5个关键问题）"""
        return {
            'questions': [
                {
                    'id': 'sleep_quality',
                    'text': 'How would you rate your sleep quality? (1-10)',
                    'type': 'scale',
                    'min': 1,
                    'max': 10,
                    'mapping': lambda x: x * 10  # 转换为0-100
                },
                {
                    'id': 'water_intake',
                    'text': 'How many glasses of water do you drink daily?',
                    'type': 'number',
                    'options': [4, 6, 8, 10],
                    'mapping': lambda x: min(100, x * 12.5)  # 4杯=50分，8杯=100分
                },
                {
                    'id': 'stress_level',
                    'text': 'How stressed have you felt recently? (1-10)',
                    'type': 'scale',
                    'min': 1,
                    'max': 10,
                    'mapping': lambda x: x * 10  # 转换为0-100
                },
                {
                    'id': 'sun_protection',
                    'text': 'How often do you use sunscreen when outdoors?',
                    'type': 'choice',
                    'options': [
                        {'value': 'never', 'score': 0},
                        {'value': 'sometimes', 'score': 50},
                        {'value': 'usually', 'score': 75},
                        {'value': 'always', 'score': 100}
                    ]
                },
                {
                    'id': 'smoking',
                    'text': 'Do you smoke?',
                    'type': 'boolean',
                    'mapping': lambda x: x
                }
            ],
            'estimated_time': '2 minutes',
            'priority': 'high'
        }
    
    @staticmethod
    def process_questionnaire_responses(responses: Dict) -> Dict:
        """处理问卷响应，生成生活方式数据"""
        lifestyle_data = {}
        
        # 处理睡眠质量
        if 'sleep_quality' in responses:
            value = responses['sleep_quality']
            if 1 <= value <= 10:
                lifestyle_data['sleep_quality'] = value * 10
        
        # 处理水分摄入
        if 'water_intake' in responses:
            value = responses['water_intake']
            if isinstance(value, (int, float)):
                lifestyle_data['water_intake'] = min(100, value * 12.5)
        
        # 处理压力水平
        if 'stress_level' in responses:
            value = responses['stress_level']
            if 1 <= value <= 10:
                lifestyle_data['stress_level'] = value * 10
        
        # 处理防晒习惯
        if 'sun_protection' in responses:
            value = responses['sun_protection']
            mapping = {'never': 0, 'sometimes': 50, 'usually': 75, 'always': 100}
            if value in mapping:
                lifestyle_data['sun_protection'] = mapping[value]
        
        # 处理吸烟
        if 'smoking' in responses:
            lifestyle_data['smoking'] = bool(responses['smoking'])
        
        return lifestyle_data


class EnvironmentalDataAdapter:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    """环境数据适配器 - 从各种API获取简化环境数据"""
    
    @staticmethod
    def get_simplified_environment_data(location: Dict = None) -> Dict:
        """获取简化环境数据"""
        environmental_data = {
            'current_conditions': {
                'uv_index': 3,  # 默认值
                'temperature': temperature,
                'humidity': humidity,
                'air_quality': air_quality,
                'season': EnvironmentalDataAdapter._get_current_season()
            }
        }
        
        if location and 'city' in location:
            # 在实际应用中，这里会调用天气API
            # 现在使用简化逻辑
            city = location['city'].lower()
            environmental_data['last_location'] = location
            
            # 简单城市环境推断（实际应用中应该使用API）
            city_profiles = {
                'beijing': {'uv_index': 5, 'air_quality': 40},
                'shanghai': {'uv_index': 4, 'air_quality': 60},
                'shenzhen': {'uv_index': 6, 'air_quality': 70},
                'new york': {'uv_index': 4, 'air_quality': 65},
                'london': {'uv_index': 2, 'air_quality': 70}
            }
            
            for city_name, profile in city_profiles.items():
                if city_name in city:
                    environmental_data['current_conditions'].update(profile)
                    break
        
        return environmental_data
    
    @staticmethod
    def _get_current_season() -> str:
        """根据当前月份推断季节"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'

# 在world_model.py中修改PhysicsFramework类
class PhysicsFramework:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    """物理学框架 - 集成增强的多物理场耦合分析"""
    
    def __init__(self):
        # 集成PhysicsEnhancedSkinAI
        self.physics_ai = PhysicsEnhancedSkinAI()
    
# 合并并优化PhysicsFramework类中的物质特性分析方法

    def _extract_material_properties(self, physics_results: Dict) -> Dict:
        """
        从物理分析结果中提取物质特性
        合并_analyze_material_properties和_extract_material_properties功能
        """
        material_properties = {}
        
        # 1. 从物理分析结果中提取数据
        if 'optics' in physics_results:
            optics_data = physics_results['optics']
            if 'water_content' in optics_data:
                water_content = optics_data['water_content'] * 100  # 转换为百分比
                material_properties['water_content'] = {
                    'score': water_content,
                    'unit': '百分比',
                    'physical_interpretation': self._interpret_physical_property('water_content', water_content)
                }
        
        if 'mechanics' in physics_results:
            mechanics_data = physics_results['mechanics']
            if 'elasticity' in mechanics_data:
                elasticity_score = mechanics_data['elasticity'].get('score', 70.0)
                material_properties['elasticity'] = {
                    'score': elasticity_score,
                    'unit': '牛顿',
                    'physical_interpretation': self._interpret_physical_property('elasticity', elasticity_score)
                }
        
        # 2. 从皮肤分析参数中提取数据（兼容原有功能）
        skin_analysis = physics_results.get('skin_analysis', {})
        parameters = skin_analysis.get('parameters', {})
        
        for param_name, param_data in parameters.items():
            if isinstance(param_data, dict) and 'score' in param_data:
                # 如果该参数尚未从物理分析结果中获取，则从皮肤分析参数中获取
                if param_name not in material_properties:
                    material_properties[param_name] = {
                        'score': param_data['score'],
                        'unit': self._get_property_unit(param_name),
                        'physical_interpretation': self._interpret_physical_property(param_name, param_data['score'])
                    }
        
        # 3. 提供默认值（确保至少包含基本参数）
        default_properties = {
            'barrier_integrity': {
                'score': 75.0,
                'unit': '指数',
                'physical_interpretation': '物理性能良好'
            },
            'sebum_production': {
                'score': 60.0,
                'unit': 'μg/cm²',
                'physical_interpretation': '物理性能一般'
            }
        }
        
        # 合并默认值（仅当参数不存在时）
        for prop_name, prop_value in default_properties.items():
            if prop_name not in material_properties:
                material_properties[prop_name] = prop_value
        
        return material_properties



    def _calculate_certainty(self, physics_results: Dict) -> float:
        """计算物理分析结果的确定性"""
        # 根据可用的物理分析结果计算确定性
        available_analyses = []
        
        if 'thermodynamics' in physics_results:
            available_analyses.append(physics_results['thermodynamics'])
        if 'fluid_dynamics' in physics_results:
            available_analyses.append(physics_results['fluid_dynamics'])
        if 'mechanics' in physics_results:
            available_analyses.append(physics_results['mechanics'])
        if 'optics' in physics_results:
            available_analyses.append(physics_results['optics'])
        
        # 基础确定性基于可用分析的数量
        base_certainty = 0.5 + (len(available_analyses) * 0.1)
        
        # 如果有耦合分析，增加确定性
        if 'coupled_physics' in physics_results:
            coupling_score = physics_results['coupled_physics'].get('coupling_score', 0.5)
            base_certainty += (coupling_score * 0.1)
        
        return min(1.0, base_certainty)  # 限制在0-1之间
    
    async def infer(self, context: Dict) -> Dict:
        """增强的物理推理，集成多物理场耦合分析"""
        skin_analysis = context.get('skin_analysis', {})
        
        # 构建物理分析所需数据
        skin_data = {
            'skin_image': context.get('skin_image'),
            'product_type': context.get('product_type', 'light_lotion'),
            'molecule_type': context.get('molecule_type', 'hyaluronic_acid'),
            'facial_geometry': context.get('facial_geometry', {'area': 500, 'curvature': 0.3}),
            'skin_type': context.get('skin_type', 'normal'),
            'hyperspectral_image': context.get('hyperspectral_image')
        }
        
        # 使用增强的物理分析
        physics_results = self.physics_ai.analyze_skin_physics(skin_data)
        
        # 提取关键信息
        material_properties = self._extract_material_properties(physics_results)
        physical_constraints = self._identify_physical_constraints(context, physics_results)
        energy_flow = self._analyze_energy_flow(physics_results)
        
        return {
            'material_properties': material_properties,
            'physical_constraints': physical_constraints,
            'energy_flow': energy_flow,
            'coupled_analysis': physics_results.get('coupled_physics', {}),
            'ingredient_penetration': physics_results.get('coupled_physics', {}).get('coupled_effects', {}).get('ingredient_penetration', {}),
            'product_experience': physics_results.get('coupled_physics', {}).get('coupled_effects', {}).get('product_experience', {}),
            'certainty': self._calculate_certainty(physics_results)
        }


    def _analyze_material_properties(self, skin_analysis: Dict) -> Dict:
        """分析皮肤的物质特性"""
        material_properties = {}
        
        # 从皮肤分析中提取物理特性
        parameters = skin_analysis.get('parameters', {})
        
        for param_name, param_data in parameters.items():
            if isinstance(param_data, dict) and 'score' in param_data:
                material_properties[param_name] = {
                    'score': param_data['score'],
                    'unit': self._get_property_unit(param_name),
                    'physical_interpretation': self._interpret_physical_property(param_name, param_data['score'])
                }
        
        return material_properties

    def _get_property_unit(self, property_name: str) -> str:
        """获取物理属性的单位"""
        unit_map = {
            'barrier': '指数',
            'hydration': '百分比',
            'elasticity': '牛顿',
            'sebum': 'μg/cm²',
            'temperature': '°C'
        }
        return unit_map.get(property_name, '无量纲')

    def _interpret_physical_property(self, property_name: str, score: float) -> str:
        """解释物理属性的含义"""
        if score < 50:
            return '物理性能较差'
        elif score < 70:
            return '物理性能一般'
        elif score < 90:
            return '物理性能良好'
        else:
            return '物理性能优秀'

    def _identify_physical_constraints(self, context: Dict, physics_results: Dict) -> List[str]:
        """识别物理约束"""
        constraints = []
        
        # 基于环境数据识别物理约束
        environmental_data = context.get('environmental_data', {})
        
        if environmental_data.get('temperature', 0) > 35:
            constraints.append('high_temperature_risk')
        
        if environmental_data.get('humidity', 0) < 30:
            constraints.append('low_humidity_risk')
        
        # 基于物理分析结果识别约束
        if 'thermodynamics' in physics_results:
            if physics_results['thermodynamics'].get('penetration_probability', 0.0) > 0.8:
                constraints.append('high_penetration_risk')
        
        return constraints

    def _analyze_energy_flow(self, physics_results: Dict) -> Dict:
        """分析能量流动"""
        energy_flow = {
            'heat_transfer': 'conduction',
            'fluid_dynamics': 'diffusion',
            'energy_balance': 'equilibrium'
        }
        
        # 基于物理分析结果更新能量流动分析
        if 'thermodynamics' in physics_results:
            energy_flow['heat_transfer'] = 'convection' if physics_results['thermodynamics'].get('temperature_gradient', 0.0) > 5.0 else 'conduction'
        
        if 'fluid_dynamics' in physics_results:
            energy_flow['fluid_dynamics'] = 'advection' if physics_results['fluid_dynamics'].get('flow_speed', 0.0) > 0.5 else 'diffusion'
        
        return energy_flow


class BiologyFramework:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    """生物学框架 - 生命系统机制"""

    def _interpret_biological_significance(self, param_name: str, score: float) -> str:
        """解释生物意义"""
        significance_map = {
            'barrier': '皮肤屏障功能，保护身体免受外界侵害',
            'hydration': '细胞水分含量，维持皮肤弹性和光泽',
            'redness': '炎症反应指标，反映皮肤健康状态',
            'elasticity': '胶原蛋白和弹性蛋白含量，反映皮肤老化程度',
            'sebum': '皮脂分泌水平，影响皮肤油腻度和痤疮风险',
            'texture': '皮肤纹理状况，反映角质层健康',
            'pore_size': '毛孔大小，与皮脂分泌和老化相关',
            'pigmentation': '色素沉着，反映黑色素代谢情况',
            'sensitivity': '皮肤敏感度，影响对外界刺激的反应'
        }
    
    def _analyze_biological_trends(self, skin_analysis: Dict) -> Dict:
        """分析生物趋势"""
        trends = {}
        parameters = skin_analysis.get('parameters', {})
        historical_data = skin_analysis.get('historical_data', {})
        
        for param_name, param_data in parameters.items():
            if isinstance(param_data, dict) and 'score' in param_data:
                current_score = param_data['score']
                
                # 基于历史数据的趋势分析
                if param_name in historical_data and len(historical_data[param_name]) > 1:
                    scores = historical_data[param_name]
                    # 计算趋势斜率
                    slope = self._calculate_trend_slope(scores + [current_score])
                    
                    if slope < -0.5:
                        trends[f'{param_name}_trend'] = 'rapidly_declining'
                    elif slope < 0:
                        trends[f'{param_name}_trend'] = 'declining'
                    elif slope > 0.5:
                        trends[f'{param_name}_trend'] = 'rapidly_improving'
                    elif slope > 0:
                        trends[f'{param_name}_trend'] = 'improving'
                    else:
                        trends[f'{param_name}_trend'] = 'stable'
                else:
                    # 回退到基于当前分数的简单趋势判断
                    if current_score < 60:
                        trends[f'{param_name}_trend'] = 'declining'
                    elif current_score > 80:
                        trends[f'{param_name}_trend'] = 'improving'
                    else:
                        trends[f'{param_name}_trend'] = 'stable'
        
        return trends

    # 在 BiologyFramework 类中添加
    def _calculate_certainty(self, biology_results):
        """计算生物框架的确定性"""
        try:
            # 根据生物参数计算
            collagen_level = biology_results.get('collagen_level', 0)
            biological_state = biology_results.get('biological_state', {})
            
            # 简单计算
            health_score = 0.5
            if biological_state.get('health') == 'good':
                health_score = 0.8
            elif biological_state.get('health') == 'poor':
                health_score = 0.2
            
            certainty = (collagen_level + health_score) / 2
            return max(0.1, min(1.0, certainty))
            
        except Exception as e:
            logger.error(f"生物框架计算确定性失败: {e}")
            return 0.1

    @staticmethod
    def _calculate_trend_slope(scores: List[float]) -> float:
        """计算分数趋势的斜率"""
        if len(scores) < 2:
            return 0.0
        
        n = len(scores)
        x = list(range(n))
        
        # 计算斜率 (简单线性回归)
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(scores)
        
        numerator = sum((x[i] - mean_x) * (scores[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator

    def _analyze_parameter_interactions(self, skin_analysis: Dict) -> Dict:
        """分析生物参数之间的相互关系"""
        interactions = {}
        parameters = skin_analysis.get('parameters', {})
        
        # 示例：分析屏障功能与水分含量的关系
        if 'barrier' in parameters and 'hydration' in parameters:
            barrier_score = parameters['barrier'].get('score', 0)
            hydration_score = parameters['hydration'].get('score', 0)
            
            if barrier_score < 60 and hydration_score < 60:
                interactions['barrier_hydration'] = {
                    'relationship': '协同下降',
                    'impact': '皮肤屏障受损导致水分流失加剧，形成恶性循环',
                    'recommendation': '同时修复屏障和补充水分'
                }
            elif barrier_score > 80 and hydration_score < 60:
                interactions['barrier_hydration'] = {
                    'relationship': '屏障完好但水分不足',
                    'impact': '皮肤锁水能力正常但水分补充不足',
                    'recommendation': '增加水分补充'
                }
        
        # 示例：分析皮脂与炎症的关系
        if 'sebum' in parameters and 'redness' in parameters:
            sebum_score = parameters['sebum'].get('score', 0)
            redness_score = parameters['redness'].get('score', 0)
            
            if sebum_score > 80 and redness_score > 70:
                interactions['sebum_redness'] = {
                    'relationship': '皮脂分泌过多伴随炎症',
                    'impact': '可能存在痤疮或脂溢性皮炎风险',
                    'recommendation': '调节皮脂分泌并控制炎症'
                }
        
        return interactions


    async def infer(self, skin_params, context=None):
        """生物框架推理 - 修复版"""
        try:
            # 生物参数分析逻辑
            collagen_level = skin_params.get('collagen_level', 0.5)
            biological_state = {'health': 'good' if collagen_level > 0.6 else 'average'}
            
            # 创建结果
            result = {
                'collagen_level': collagen_level,
                'biological_state': biological_state,
                'biological_analysis': {
                    'health_score': min(1.0, collagen_level * 1.2),
                    'vitality': 'high' if collagen_level > 0.7 else 'medium'
                }
            }
            
            # 关键修复：调用 _calculate_certainty 方法
            result['certainty'] = self._calculate_certainty(result)
            result['framework_status'] = 'success'
            
            return result
            
        except Exception as e:
            logger.error(f"生物框架推理失败: {e}")
            return {
                'collagen_level': 0.5,
                'biological_state': {'health': 'unknown'},
                'certainty': 0.1,
                'framework_status': 'error',
                'error': str(e)
            }

    def _calculate_dynamic_certainty(self, skin_analysis: Dict) -> float:
        """根据数据质量动态计算确定性"""
        parameters = skin_analysis.get('parameters', {})
        historical_data = skin_analysis.get('historical_data', {})
        
        # 基础确定性
        base_certainty = 0.75
        
        # 根据参数数量调整
        param_count = len(parameters)
        if param_count < 3:
            base_certainty *= 0.8  # 减少参数时降低确定性
        elif param_count > 6:
            base_certainty *= 1.1  # 增加参数时提高确定性
        
        # 根据历史数据量调整
        historical_count = sum(len(data) for data in historical_data.values())
        if historical_count > 10:
            base_certainty *= 1.15  # 有足够历史数据时提高确定性
        
        # 确保确定性在合理范围内
        return max(0.5, min(0.95, base_certainty))


    def _assess_biological_state(self, skin_analysis: Dict) -> Dict:
        """评估生物状态"""
        parameters = skin_analysis.get('parameters', {})
        external_factors = skin_analysis.get('external_factors', {})
        
        biological_state = {}
        
        for param_name, param_data in parameters.items():
            if isinstance(param_data, dict) and 'score' in param_data:
                score = param_data['score']
                
                # 根据外部因素调整分数解释
                adjusted_score = self._adjust_score_for_external_factors(param_name, score, external_factors)
                
                biological_state[param_name] = {
                    'score': score,
                    'adjusted_score': adjusted_score,
                    'biological_significance': self._interpret_biological_significance(param_name, adjusted_score),
                    'external_influences': self._identify_external_influences(param_name, external_factors)
                }
        
        return biological_state

    def _adjust_score_for_external_factors(self, param_name: str, score: float, external_factors: Dict) -> float:
        """根据外部因素调整分数"""
        adjusted_score = score
        
        # 示例：环境湿度对水分含量的影响
        if param_name == 'hydration':
            humidity = external_factors.get('humidity')
            if humidity:
                if humidity < 30:
                    adjusted_score = min(score + 5, 100)  # 干燥环境下水分分数适当提高
                elif humidity > 70:
                    adjusted_score = max(score - 5, 0)  # 潮湿环境下水分分数适当降低
        
        # 示例：紫外线暴露对炎症的影响
        if param_name == 'redness':
            uv_exposure = external_factors.get('uv_exposure')
            if uv_exposure and uv_exposure > 6:
                adjusted_score = min(score + 10, 100)  # 高UV暴露下炎症分数适当提高
        
        return adjusted_score

    def _identify_external_influences(self, param_name: str, external_factors: Dict) -> List[str]:
        """识别影响特定参数的外部因素"""
        influences = []
        
        if param_name in ['hydration', 'barrier']:
            if external_factors.get('humidity', 50) < 30:
                influences.append('环境干燥可能导致皮肤水分流失')
            if external_factors.get('temperature', 25) > 30:
                influences.append('高温可能加剧皮肤水分蒸发')
        
        if param_name in ['redness', 'sensitivity']:
            if external_factors.get('uv_exposure', 0) > 5:
                influences.append('紫外线暴露可能引起皮肤炎症')
            if external_factors.get('pollution', 0) > 7:
                influences.append('环境污染可能刺激皮肤')
        
        return influences


    def _interpret_biological_significance(self, param_name: str, score: float) -> str:
        """解释生物意义"""
        significance_map = {
            'barrier': '皮肤屏障功能，保护身体免受外界侵害',
            'hydration': '细胞水分含量，维持皮肤弹性和光泽',
            'redness': '炎症反应指标，反映皮肤健康状态',
            'elasticity': '胶原蛋白和弹性蛋白含量，反映皮肤老化程度'
        }
        
        significance = significance_map.get(param_name, '生物参数')
        
        if score < 50:
            return f'{significance}：严重不足'
        elif score < 70:
            return f'{significance}：需要改善'
        elif score < 90:
            return f'{significance}：状态良好'
        else:
            return f'{significance}：状态优秀'

    def _analyze_biological_trends(self, skin_analysis: Dict) -> Dict:
        """分析生物趋势"""
        trends = {}
        parameters = skin_analysis.get('parameters', {})
        
        for param_name, param_data in parameters.items():
            if isinstance(param_data, dict) and 'score' in param_data:
                score = param_data['score']
                if score < 60:
                    trends[f'{param_name}_trend'] = 'declining'
                elif score > 80:
                    trends[f'{param_name}_trend'] = 'improving'
                else:
                    trends[f'{param_name}_trend'] = 'stable'
        
        return trends

    def _assess_homeostasis(self, skin_analysis: Dict) -> Dict:
        """评估体内平衡"""
        parameters = skin_analysis.get('parameters', {})
        scores = [param_data['score'] for param_data in parameters.values() if isinstance(param_data, dict) and 'score' in param_data]
        
        if not scores:
            return {'balance': 'unknown'}
        
        avg_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0
        
        if std_score < 15 and avg_score > 70:
            return {'balance': 'balanced'}
        elif std_score > 25 or avg_score < 50:
            return {'balance': 'imbalanced'}
        else:
            return {'balance': 'moderately_balanced'}


class PsychologyFramework:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    """心理学框架 - 心智与行为模式"""

    async def infer(self, context: Dict) -> Dict:
        """心理学推理"""
        return {
            'behavioral_patterns': self._analyze_behavioral_patterns(context),
            'emotional_influences': self._assess_emotional_influences(context),
            'motivation_factors': self._identify_motivation_factors(context),
            'certainty': 0.6
        }

    def _analyze_behavioral_patterns(self, context: Dict) -> Dict:
        """分析行为模式"""
        user_behavior = context.get('user_behavior', {})
        
        return {
            'skincare_routine': user_behavior.get('routine_consistency', 'inconsistent'),
            'product_usage': user_behavior.get('product_adherence', 'low'),
            'environmental_exposure': self._analyze_environmental_exposure(context)
        }

    def _analyze_environmental_exposure(self, context: Dict) -> Dict:
        """分析环境暴露"""
        environmental_data = context.get('environmental_data', {})
        
        return {
            'sun_exposure': self._assess_sun_exposure(environmental_data),
            'pollution_exposure': self._assess_pollution_exposure(environmental_data),
            'climate_exposure': self._assess_climate_exposure(environmental_data)
        }

    def _assess_sun_exposure(self, environmental_data: Dict) -> str:
        """评估阳光暴露"""
        uv_index = environmental_data.get('uv_index', 0)
        if uv_index > 10:
            return 'extreme'
        elif uv_index > 7:
            return 'high'
        elif uv_index > 3:
            return 'moderate'
        else:
            return 'low'

    def _assess_pollution_exposure(self, environmental_data: Dict) -> str:
        """评估污染暴露"""
        aqi = environmental_data.get('air_quality_index', 0)
        if aqi > 200:
            return 'hazardous'
        elif aqi > 150:
            return 'unhealthy'
        elif aqi > 100:
            return 'unhealthy_for_sensitive_groups'
        elif aqi > 50:
            return 'moderate'
        else:
            return 'good'

    def _assess_climate_exposure(self, environmental_data: Dict) -> str:
        """评估气候暴露"""
        humidity = environmental_data.get('humidity', 50)
        if humidity < 30:
            return 'dry'
        elif humidity > 70:
            return 'humid'
        else:
            return 'comfortable'

    def _assess_emotional_influences(self, context: Dict) -> Dict:
        """评估情绪影响"""
        user_profile = context.get('user_profile', {})
        
        return {
            'stress_level': user_profile.get('stress_level', 'medium'),
            'sleep_quality': user_profile.get('sleep_quality', 'medium'),
            'emotional_impact': self._analyze_emotional_impact(context)
        }

    def _analyze_emotional_impact(self, context: Dict) -> Dict:
        """分析情绪影响"""
        return {
            'self_perception': 'positive',
            'confidence': 'moderate',
            'emotional_triggers': ['stress', 'lack_of_sleep']
        }

    def _identify_motivation_factors(self, context: Dict) -> List[str]:
        """识别动机因素"""
        user_goals = context.get('user_goals', [])
        
        motivation_factors = []
        for goal in user_goals:
            if goal.get('priority', 'low') == 'high':
                motivation_factors.append(goal.get('type', 'unknown'))
        
        return motivation_factors


class DecisionTheoryFramework:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    """决策理论框架 - 不确定性下的最优决策"""

    async def infer(self, context: Dict) -> Dict:
        """决策理论推理"""
        return {
            'decision_options': self._generate_decision_options(context),
            'utility_analysis': self._analyze_utility(context),
            'risk_assessment': self._assess_risk(context),
            'optimal_decision': self._determine_optimal_decision(context),
            'certainty': self._calculate_certainty(context)
        }

    def _generate_decision_options(self, context: Dict) -> List[Dict]:
        """生成可能的决策选项"""
        user_goals = context.get('user_goals', [])
        skin_analysis = context.get('skin_analysis', {})
        available_products = context.get('available_products', [])
        
        # 基于用户目标和皮肤状态生成决策选项
        options = []
        
        for goal in user_goals:
            # 根据目标类型生成相关决策选项
            if goal.get('type') == 'hydration':
                options.extend(self._generate_hydration_options(available_products))
            elif goal.get('type') == 'anti-aging':
                options.extend(self._generate_anti_aging_options(available_products))
            elif goal.get('type') == 'acne_treatment':
                options.extend(self._generate_acne_options(available_products))
        
        # 去重
        seen = set()
        unique_options = []
        for option in options:
            key = (option['name'], tuple(option['products']))
            if key not in seen:
                seen.add(key)
                unique_options.append(option)
                
        return unique_options

    def _analyze_utility(self, context: Dict) -> Dict:
        """分析每个决策选项的效用"""
        decision_options = self._generate_decision_options(context)
        user_profile = context.get('user_profile', {})
        skin_analysis = context.get('skin_analysis', {})
        
        utility_scores = {}
        
        for option in decision_options:
            # 计算每个选项的期望效用
            effectiveness = self._calculate_effectiveness(option, skin_analysis)
            satisfaction = self._calculate_satisfaction(option, user_profile)
            convenience = self._calculate_convenience(option, user_profile)
            
            # 构建效用函数：加权求和
            utility = (
                0.5 * effectiveness +  # 效果权重最高
                0.3 * satisfaction +   # 满意度次之
                0.2 * convenience      # 便利性权重较低
            )
            
            utility_scores[option['name']] = utility
        
        return utility_scores

    def _assess_risk(self, context: Dict) -> Dict:
        """评估决策风险"""
        decision_options = self._generate_decision_options(context)
        skin_sensitivity = context.get('skin_analysis', {}).get('sensitivity', 'low')
        
        risk_scores = {}
        
        for option in decision_options:
            # 评估每个选项的风险
            irritation_risk = self._calculate_irritation_risk(option, skin_sensitivity)
            cost_risk = self._calculate_cost_risk(option, context)
            time_risk = self._calculate_time_risk(option, context)
            
            # 综合风险评分
            total_risk = (
                0.6 * irritation_risk +  # 皮肤刺激风险权重最高
                0.2 * cost_risk +        # 成本风险次之
                0.2 * time_risk          # 时间风险权重较低
            )
            
            risk_scores[option['name']] = total_risk
        
        return risk_scores

    def _determine_optimal_decision(self, context: Dict) -> Dict:
        """确定最优决策"""
        utility_scores = self._analyze_utility(context)
        risk_scores = self._assess_risk(context)
        user_profile = context.get('user_profile', {})
        
        # 获取用户风险偏好
        risk_tolerance = self._get_risk_tolerance(user_profile)
        
        # 基于效用和风险的综合评分
        comprehensive_scores = {}
        for option_name in utility_scores.keys():
            utility = utility_scores[option_name]
            risk = risk_scores[option_name]
            
            # 风险调整后的综合评分
            adjusted_score = utility * (1 - risk * (1 - risk_tolerance))
            comprehensive_scores[option_name] = adjusted_score
        
        # 找到最优决策
        if not comprehensive_scores:
            return {'name': 'no_optimal_decision', 'score': 0}
        
        optimal_option = max(comprehensive_scores, key=comprehensive_scores.get)
        
        return {
            'name': optimal_option,
            'score': comprehensive_scores[optimal_option],
            'utility': utility_scores[optimal_option],
            'risk': risk_scores[optimal_option]
        }

    def _calculate_certainty(self, context: Dict) -> float:
        """计算决策的确定性"""
        optimal_decision = self._determine_optimal_decision(context)
        utility_scores = self._analyze_utility(context)
        
        if not utility_scores:
            return 0.5
        
        # 基于效用分布计算确定性
        max_utility = optimal_decision['utility']
        avg_utility = sum(utility_scores.values()) / len(utility_scores)
        
        # 确定性与效用优势成正比
        if avg_utility == 0:
            return 0.5
        
        certainty = 0.5 + (max_utility - avg_utility) / (2 * avg_utility)
        return min(max(certainty, 0.1), 0.9)  # 限制在0.1-0.9之间

    # 辅助方法 - 根据决策类型生成选项
    def _generate_hydration_options(self, products: List[Dict]) -> List[Dict]:
        """生成保湿相关决策选项"""
        hydration_products = [p for p in products if 'hydration' in p.get('benefits', [])]
        
        options = []
        # 简单保湿方案
        if any(p.get('type') == 'moisturizer' for p in hydration_products):
            options.append({
                'name': 'simple_hydration',
                'description': '基础保湿方案',
                'products': [p for p in hydration_products if p.get('type') == 'moisturizer'][:1],
                'time_required': 'low'
            })
        
        # 深层保湿方案
        if any(p.get('type') == 'serum' and 'hydration' in p.get('benefits', []) for p in hydration_products):
            options.append({
                'name': 'deep_hydration',
                'description': '深层保湿方案',
                'products': [p for p in hydration_products if p.get('type') in ['serum', 'moisturizer']][:2],
                'time_required': 'medium'
            })
        
        return options

    def _generate_anti_aging_options(self, products: List[Dict]) -> List[Dict]:
        """生成抗衰老相关决策选项"""
        anti_aging_products = [p for p in products if 'anti-aging' in p.get('benefits', [])]
        
        options = []
        # 预防方案
        if any(p.get('type') == 'sunscreen' for p in anti_aging_products):
            options.append({
                'name': 'prevention_anti_aging',
                'description': '抗衰老预防方案',
                'products': [p for p in anti_aging_products if p.get('type') == 'sunscreen'][:1],
                'time_required': 'low'
            })
        
        # 综合方案
        if len(anti_aging_products) >= 2:
            options.append({
                'name': 'comprehensive_anti_aging',
                'description': '综合抗衰老方案',
                'products': anti_aging_products[:3],
                'time_required': 'high'
            })
        
        return options

    def _generate_acne_options(self, products: List[Dict]) -> List[Dict]:
        """生成 acne 治疗相关决策选项"""
        acne_products = [p for p in products if 'acne' in p.get('benefits', [])]
        
        options = []
        # 温和方案
        if any(p.get('type') == 'cleanser' for p in acne_products):
            options.append({
                'name': 'gentle_acne_treatment',
                'description': '温和 acne 治疗方案',
                'products': [p for p in acne_products if p.get('type') == 'cleanser'][:1],
                'time_required': 'low'
            })
        
        # 强化方案
        if len(acne_products) >= 2:
            options.append({
                'name': 'intensive_acne_treatment',
                'description': '强化 acne 治疗方案',
                'products': acne_products[:2],
                'time_required': 'medium'
            })
        
        return options

    # 辅助方法 - 计算各种指标
    def _calculate_effectiveness(self, option: Dict, skin_analysis: Dict) -> float:
        """计算决策选项的效果"""
        # 基于产品成分、用户皮肤状态计算预期效果
        products = option.get('products', [])
        skin_concerns = skin_analysis.get('concerns', [])
        
        effectiveness = 0.0
        for product in products:
            for concern in skin_concerns:
                if concern in product.get('benefits', []):
                    effectiveness += 0.3  # 每个匹配的关注点增加0.3分
        
        return min(effectiveness, 1.0)  # 限制在0-1之间

    def _calculate_satisfaction(self, option: Dict, user_profile: Dict) -> float:
        """计算用户对决策的满意度"""
        user_preferences = user_profile.get('preferences', {})
        product_types = [p.get('type') for p in option.get('products', [])]
        
        satisfaction = 0.5  # 基础满意度
        
        # 考虑用户对产品类型的偏好
        for ptype in product_types:
            if ptype in user_preferences.get('liked_product_types', []):
                satisfaction += 0.1
            elif ptype in user_preferences.get('disliked_product_types', []):
                satisfaction -= 0.1
        
        return min(max(satisfaction, 0.0), 1.0)  # 限制在0-1之间

    def _calculate_convenience(self, option: Dict, user_profile: Dict) -> float:
        """计算决策的便利性"""
        time_required = option.get('time_required', 'medium')
        user_time_available = user_profile.get('time_available', 'limited')
        
        convenience_map = {
            'low': {'limited': 0.3, 'medium': 0.6, 'ample': 0.9},
            'medium': {'limited': 0.2, 'medium': 0.5, 'ample': 0.8},
            'high': {'limited': 0.1, 'medium': 0.3, 'ample': 0.7}
        }
        
        return convenience_map.get(time_required, {}).get(user_time_available, 0.5)

    def _calculate_irritation_risk(self, option: Dict, skin_sensitivity: str) -> float:
        """计算皮肤刺激风险"""
        products = option.get('products', [])
        risk = 0.0
        
        for product in products:
            # 根据产品成分计算刺激风险
            if 'fragrance' in product.get('ingredients', []):
                risk += 0.2
            if 'alcohol' in product.get('ingredients', []):
                risk += 0.3
            if 'retinol' in product.get('ingredients', []):
                risk += 0.25
        
        # 根据皮肤敏感度调整风险
        sensitivity_multiplier = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5
        }.get(skin_sensitivity, 1.0)
        
        return min(risk * sensitivity_multiplier, 1.0)  # 限制在0-1之间

    def _calculate_cost_risk(self, option: Dict, context: Dict) -> float:
        """计算成本风险"""
        products = option.get('products', [])
        total_cost = sum(p.get('cost', 0) for p in products)
        user_budget = context.get('user_profile', {}).get('budget', 'medium')
        
        # 基于预算的成本风险评估
        budget_thresholds = {
            'low': 50,
            'medium': 150,
            'high': 300
        }.get(user_budget, 150)
        
        if total_cost <= budget_thresholds * 0.5:
            return 0.0
        elif total_cost <= budget_thresholds:
            return 0.3
        elif total_cost <= budget_thresholds * 1.5:
            return 0.7
        else:
            return 1.0

    def _calculate_time_risk(self, option: Dict, context: Dict) -> float:
        """计算时间投入风险"""
        time_required = option.get('time_required', 'medium')
        user_time_available = context.get('user_profile', {}).get('time_available', 'limited')
        
        # 时间投入风险评估
        time_risk_map = {
            'low': {'limited': 0.1, 'medium': 0.0, 'ample': 0.0},
            'medium': {'limited': 0.6, 'medium': 0.2, 'ample': 0.0},
            'high': {'limited': 0.9, 'medium': 0.7, 'ample': 0.3}
        }
        
        return time_risk_map.get(time_required, {}).get(user_time_available, 0.5)

    def _get_risk_tolerance(self, user_profile: Dict) -> float:
        """获取用户的风险容忍度"""
        # 基于用户资料推断风险容忍度
        age = user_profile.get('age', 30)
        skin_sensitivity = user_profile.get('skin_sensitivity', 'medium')
        
        # 年龄越小通常越愿意冒险尝试新产品
        age_factor = 1.0 - (age - 18) / 100
        
        # 皮肤敏感度越高风险容忍度越低
        sensitivity_factor = {
            'low': 1.0,
            'medium': 0.7,
            'high': 0.4
        }.get(skin_sensitivity, 0.7)
        
        return min(max(age_factor * sensitivity_factor, 0.1), 0.9)  # 限制在0.1-0.9之间




class MentalFramework:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    async def infer(self, skin_params, context=None):
        """心理框架推理 - 修复版"""
        try:
            # 心理参数分析逻辑
            perceived_effectiveness = skin_params.get('perceived_effectiveness', 0.7)
            
            # 创建结果
            result = {
                'perceived_effectiveness': perceived_effectiveness,
                'user_satisfaction': 'high' if perceived_effectiveness > 0.8 else 'medium',
                'confidence_level': min(1.0, perceived_effectiveness * 1.1)
            }
            
            # 关键修复：调用 _calculate_certainty 方法
            result['certainty'] = self._calculate_certainty(result)
            result['framework_status'] = 'success'
            
            return result
            
        except Exception as e:
            logger.error(f"心理框架推理失败: {e}")
            return {
                'perceived_effectiveness': 0.5,
                'certainty': 0.1,
                'framework_status': 'error'
            }

    # 在 MentalFramework 类中添加
    def _calculate_certainty(self, mental_results):
        """计算心理框架的确定性"""
        try:
            # 根据心理参数计算
            perceived_effectiveness = mental_results.get('perceived_effectiveness', 0.5)
            
            # 简单计算
            certainty = perceived_effectiveness
            return max(0.1, min(1.0, certainty))
            
        except Exception as e:
            logger.error(f"心理框架计算确定性失败: {e}")
            return 0.1

    def _identify_cognitive_biases(self, context: Dict) -> List[str]:
        """识别认知偏差"""
        user_behavior = context.get('user_behavior', {})
        
        biases = []
        
        if user_behavior.get('product_switching_frequency', 0) > 4:
            biases.append('recency_bias')
        
        if user_behavior.get('brand_loyalty', 0) > 90:
            biases.append('confirmation_bias')
        
        return biases

    def _analyze_decision_making_style(self, context: Dict) -> str:
        """分析决策风格"""
        user_profile = context.get('user_profile', {})
        
        if user_profile.get('risk_tolerance', 'medium') == 'high':
            return 'risk_seeking'
        elif user_profile.get('risk_tolerance', 'medium') == 'low':
            return 'risk_averse'
        else:
            return 'balanced'

    def _assess_knowledge_structure(self, context: Dict) -> Dict:
        """评估知识结构"""
        user_profile = context.get('user_profile', {})
        
        return {
            'skincare_knowledge': user_profile.get('skincare_knowledge', 'basic'),
            'ingredient_awareness': user_profile.get('ingredient_knowledge', 'low'),
            'evidence_based': user_profile.get('evidence_based', 'low')
        }    




class EvolutionaryAlgorithmFramework:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    """完整的进化算法框架，用于皮肤健康产品推荐优化"""
    
    def __init__(self):
        # 进化算法参数
        self.population_size = 30  # 从20增加到30
        self.min_population_size = 8  # 从5增加到8
        self.generations = 15  # 从10增加到15
        self.mutation_rate = 0.18  # 从0.15增加到0.18
        self.crossover_rate = 0.75  # 从0.7增加到0.75
        self.elite_size = 3  # 从2增加到3
        self.immigrant_rate = 0.1  # 新增：移民率（引入新个体）
        
        # 产品组合参数
        self.max_products_per_solution = 6  # 从5增加到6
        self.min_products_per_solution = 3  # 从2增加到3
        
        # 新增：多样性增强参数
        self.diversity_weight = 0.3  # 产品多样性权重
        self.category_coverage_weight = 0.15  # 类别覆盖权重
        self.novelty_weight = 0.05  # 新颖性权重
        
        # 修改权重分配，提高目标匹配度的重要性
        self.fitness_weights = {
            'goal_match': 0.45,      # 从0.35提高到0.45
            'cost_efficiency': 0.20,  # 保持
            'product_diversity': 0.15, # 从0.25降低到0.15
            'category_coverage': 0.10, # 从0.15降低到0.10
            'user_preference': 0.10   # 保持
        }
        
        # 新增：产品类别映射（用于多样性优化）
        self.category_importance = {
            'cleanser': 1.0,
            'moisturizer': 1.0,
            'sunscreen': 1.2,  # 防晒产品更重要
            'serum': 1.1,
            'toner': 0.8,
            'mask': 0.7,
            'eye_cream': 0.9,
            'treatment': 1.1
        }
        
        # 缓存
        self.last_population = []
        self.best_solution_history = []
        self.generation_stats = []
        
        logger.info("进化算法框架初始化完成（优化版）")
    
    async def infer(self, skin_params: Optional[Dict] = None, context: Optional[Dict] = None) -> Dict:
        """进化算法推理 - 完整实现"""
        try:
            # 参数处理
            if context is None and skin_params is not None and isinstance(skin_params, dict):
                context = skin_params
                skin_params = {}
            
            if context is None:
                context = {}
            
            # 记录接收到的数据
            available_products = context.get('available_products', [])
            user_goals = context.get('user_goals', [])
            user_profile = context.get('user_profile', {})
            
            logger.info(f"进化算法框架接收到 {len(available_products)} 个产品, {len(user_goals)} 个用户目标")
            
            # 验证数据
            if len(available_products) < 2:
                logger.warning("产品数量不足，使用简化推理")
                return self._get_simplified_solution(context)
            
            # 生成初始种群
            population = self._initialize_population(context)
            
            if len(population) < self.min_population_size:
                logger.warning(f"种群初始化失败，大小不足: {len(population)}")
                return self._get_default_result(context)
            
            # 进化过程
            best_solution = None
            best_fitness = -1
            
            for generation in range(self.generations):
                # 计算适应度
                for individual in population:
                    individual['fitness'] = self._calculate_fitness(individual, context)
                
                # 选择最佳个体
                population.sort(key=lambda x: x['fitness'], reverse=True)
                current_best = population[0]
                
                if best_solution is None or current_best['fitness'] > best_fitness:
                    best_solution = copy.deepcopy(current_best)
                    best_fitness = current_best['fitness']
                
                # 记录统计信息
                self._record_generation_stats(population, generation)
                
                # 如果达到最大代数，停止
                if generation == self.generations - 1:
                    break
                
                # 创建新一代
                population = self._create_new_generation(population, context)
            
            # 计算确定性
            certainty = self._calculate_certainty(best_solution, population, context)
            
            # 构建最终结果
            result = {
                'population_analysis': self._analyze_population(population),
                'fitness_scores': self._calculate_all_fitness_scores(population, context),
                'best_solution': self._format_best_solution(best_solution, context),
                'evolution_summary': self._generate_evolution_summary(population, context),
                'certainty': certainty,
                'framework_status': 'success',
                'generations_completed': self.generations,
                'final_population_size': len(population)
            }
            
            # 保存状态
            self.last_population = population
            self.best_solution_history.append(best_solution)
            
            logger.info(f"进化算法完成: {len(population)} 个个体, 最佳适应度: {best_fitness:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"进化算法框架推理出错: {e}", exc_info=True)
            return self._get_default_result(context)
    

    
    def _enhance_diversity_mechanism(self, population: List[Dict], context: Dict) -> List[Dict]:
        """增强多样性机制 - 防止早熟收敛"""
        if len(population) < 2:
            return population
        
        # 计算种群多样性
        diversity_score = self._calculate_population_diversity(population)
        
        # 如果多样性过低，引入新个体
        if diversity_score < 0.3:
            logger.info(f"种群多样性低 ({diversity_score:.3f})，引入新个体")
            new_individuals = self._generate_diverse_individuals(context, 3)
            population.extend(new_individuals)
        
        # 移民策略：随机替换一些低适应度个体
        if random.random() < self.immigrant_rate and len(population) > self.min_population_size:
            immigrant_count = max(1, int(len(population) * 0.1))
            for _ in range(immigrant_count):
                if len(population) > self.min_population_size:
                    # 移除适应度最低的个体
                    population.sort(key=lambda x: x.get('fitness', 0))
                    population.pop(0)
                    
                    # 添加新移民个体
                    new_immigrant = self._generate_diverse_individuals(context, 1)[0]
                    new_immigrant['fitness'] = self._calculate_fitness(new_immigrant, context)
                    population.append(new_immigrant)
        
        return population
    
    def _calculate_population_diversity(self, population: List[Dict]) -> float:
        """计算种群多样性"""
        if len(population) < 2:
            return 0.0
        
        all_products = []
        for individual in population:
            product_ids = [p.get('id', '') for p in individual.get('products', [])]
            all_products.extend(product_ids)
        
        unique_products = set(all_products)
        total_occurrences = len(all_products)
        
        if total_occurrences == 0:
            return 0.0
        
        # 计算辛普森多样性指数
        product_counts = {}
        for product_id in all_products:
            product_counts[product_id] = product_counts.get(product_id, 0) + 1
        
        diversity_index = 0.0
        for count in product_counts.values():
            pi = count / total_occurrences
            diversity_index += pi * pi
        
        return 1 - diversity_index  # 多样性指数越高越好
    
    def _generate_diverse_individuals(self, context: Dict, count: int = 3) -> List[Dict]:
        """生成多样性个体 - 刻意选择不同的产品"""
        available_products = context.get('available_products', [])
        if not available_products or count <= 0:
            return []
        
        diverse_individuals = []
        
        # 按类别分组产品
        products_by_category = {}
        for product in available_products:
            category = product.get('category', 'unknown')
            if category not in products_by_category:
                products_by_category[category] = []
            products_by_category[category].append(product)
        
        for _ in range(count):
            # 随机选择不同类别的产品
            selected_categories = random.sample(
                list(products_by_category.keys()), 
                min(3, len(products_by_category))
            )
            
            selected_products = []
            for category in selected_categories:
                if products_by_category[category]:
                    product = random.choice(products_by_category[category])
                    selected_products.append(product)
            
            # 确保产品数量在合理范围内
            num_products = random.randint(
                self.min_products_per_solution,
                min(self.max_products_per_solution, len(selected_products))
            )
            
            if len(selected_products) > num_products:
                selected_products = random.sample(selected_products, num_products)
            
            # 创建个体
            individual = self._create_individual(selected_products, context)
            diverse_individuals.append(individual)
        
        return diverse_individuals



    def _initialize_population(self, context: Dict) -> List[Dict]:
        """初始化种群 - 完整实现"""
        try:
            available_products = context.get('available_products', [])
            user_goals = context.get('user_goals', [])
            
            if not available_products:
                return []
            
            population = []
            
            # 创建随机个体
            for _ in range(self.population_size):
                # 随机选择产品数量
                num_products = random.randint(
                    min(self.min_products_per_solution, len(available_products)),
                    min(self.max_products_per_solution, len(available_products))
                )
                
                # 随机选择产品（不重复）
                selected_products = random.sample(available_products, num_products)
                
                # 创建个体
                individual = self._create_individual(selected_products, context)
                population.append(individual)
            
            logger.info(f"成功初始化种群，大小: {len(population)}")
            return population
            
        except Exception as e:
            logger.error(f"种群初始化失败: {e}")
            return []
    
    def _create_individual(self, products: List[Dict], context: Dict) -> Dict:
        """创建个体 - 完整实现"""
        try:
            # 提取基因（产品特征）
            genes = []
            total_cost = 0
            all_benefits = set()
            
            for product in products:
                # 基本特征
                gene = {
                    'product_id': product.get('id', 'unknown'),
                    'category': product.get('category', 'unknown'),
                    'price': product.get('price', 0),
                    'benefits': product.get('benefits', []),
                    'ingredients': product.get('ingredients', [])
                }
                genes.append(gene)
                
                total_cost += product.get('price', 0)
                all_benefits.update(product.get('benefits', []))
            
            # 创建个体
            individual = {
                'id': f"ind_{random.randint(10000, 99999)}",
                'products': products,
                'genes': genes,
                'total_cost': total_cost,
                'unique_benefits': list(all_benefits),
                'created_at': time.time(),
                'age': 0,
                'fitness': 0.0,
                'selected_count': 0
            }
            
            return individual
            
        except Exception as e:
            logger.error(f"创建个体失败: {e}")
            return {
                'products': products,
                'genes': [],
                'total_cost': 0,
                'fitness': 0.1
            }
    

    
    def _calculate_fitness(self, individual: Dict, context: Dict) -> float:
        """计算个体适应度 - 优化版（增强目标匹配度计算）"""
        try:
            user_goals = context.get('user_goals', [])
            user_profile = context.get('user_profile', {})
            available_products = context.get('available_products', [])
            
            products = individual.get('products', [])
            total_cost = individual.get('total_cost', 0)
            unique_benefits = set(individual.get('unique_benefits', []))
            
            # 1. 增强目标匹配度计算
            goal_match_score = 0.0
            if user_goals and products:
                # 提取目标类型和优先级
                goal_info = {}
                for goal in user_goals:
                    goal_type = goal.get('type')
                    if goal_type:
                        goal_info[goal_type] = {
                            'priority': goal.get('priority', 1),
                            'coverage_count': 0,  # 覆盖该目标的产品数量
                            'coverage_strength': 0,  # 覆盖强度（基于产品类型）
                            'exclusive_coverage': False  # 是否有产品专门针对该目标
                        }
                
                if goal_info:
                    # 第一步：分析每个产品对目标的覆盖情况
                    for product in products:
                        product_benefits = set(product.get('benefits', []))
                        product_category = product.get('category', 'unknown')
                        
                        # 计算产品类别对目标的支持权重
                        category_support_weights = self._get_category_support_weights(product_category)
                        
                        # 检查产品覆盖了哪些目标
                        for goal_type in product_benefits.intersection(goal_info.keys()):
                            goal_info[goal_type]['coverage_count'] += 1
                            
                            # 增加覆盖强度（考虑产品类别的重要性）
                            strength_bonus = category_support_weights.get(goal_type, 0.3)
                            goal_info[goal_type]['coverage_strength'] += strength_bonus
                            
                            # 检查是否是专门针对该目标的产品
                            if self._is_dedicated_product(product, goal_type):
                                goal_info[goal_type]['exclusive_coverage'] = True
                    
                    # 第二步：计算每个目标的得分
                    goal_scores = []
                    total_priority = 0
                    
                    for goal_type, info in goal_info.items():
                        coverage_count = info['coverage_count']
                        coverage_strength = info['coverage_strength']
                        exclusive = info['exclusive_coverage']
                        priority = info['priority']
                        
                        total_priority += priority
                        
                        # 基础覆盖得分
                        if coverage_count == 0:
                            base_score = 0.0
                        elif coverage_count == 1:
                            base_score = 0.6
                        else:
                            base_score = 0.8 + min(0.2, (coverage_count - 1) * 0.1)
                        
                        # 覆盖强度加成
                        strength_bonus = min(0.3, coverage_strength * 0.1)
                        
                        # 专门产品加成
                        exclusive_bonus = 0.15 if exclusive else 0.0
                        
                        # 多样性加成：如果有不同类别的产品覆盖同一目标
                        diversity_bonus = 0.0
                        if coverage_count >= 2:
                            # 检查是否有不同类别的产品
                            product_categories = set()
                            for product in products:
                                if goal_type in product.get('benefits', []):
                                    product_categories.add(product.get('category', 'unknown'))
                            
                            if len(product_categories) > 1:
                                diversity_bonus = 0.1
                        
                        # 综合目标得分
                        goal_score = min(1.0, base_score + strength_bonus + exclusive_bonus + diversity_bonus)
                        
                        # 考虑目标优先级
                        weighted_score = goal_score * priority
                        goal_scores.append(weighted_score)
                    
                    # 第三步：计算总体目标匹配度
                    if total_priority > 0 and goal_scores:
                        weighted_sum = sum(goal_scores)
                        goal_match_score = weighted_sum / total_priority
                        
                        # 额外奖励：完全覆盖所有高优先级目标
                        high_priority_goals = [g for g in goal_info.values() if g.get('priority', 1) >= 3]
                        if high_priority_goals and all(g['coverage_count'] > 0 for g in high_priority_goals):
                            goal_match_score = min(1.0, goal_match_score + 0.1)
            
            # 2. 成本效益计算（优化）
            cost_efficiency_score = 1.0
            if total_cost > 0 and products:
                user_budget = user_profile.get('budget', 'medium')
                
                # 更细化的预算范围
                budget_ranges = {
                    'low': (0, 300),      # 低预算：0-300元
                    'medium': (200, 800),  # 中等预算：200-800元
                    'high': (500, 2000)    # 高预算：500-2000元
                }
                
                budget_min, budget_max = budget_ranges.get(user_budget, (200, 800))
                
                # 计算每个产品的平均价格
                avg_product_cost = total_cost / len(products)
                
                # 基于性价比的成本效益计算
                if total_cost < budget_min * 0.5:
                    # 成本过低，可能效果不足
                    ratio = total_cost / (budget_min * 0.5)
                    cost_efficiency_score = 0.4 + ratio * 0.3
                elif total_cost <= budget_min:
                    # 低于预算下限但合理
                    ratio = (total_cost - budget_min * 0.5) / (budget_min * 0.5)
                    cost_efficiency_score = 0.7 + ratio * 0.2
                elif total_cost <= budget_max:
                    # 在预算范围内，越接近上限分数越低
                    ratio = (total_cost - budget_min) / (budget_max - budget_min)
                    cost_efficiency_score = 0.9 - ratio * 0.2
                else:
                    # 超出预算，严格惩罚
                    overshoot = (total_cost - budget_max) / budget_max
                    cost_efficiency_score = max(0.1, 0.6 - overshoot * 2)
                
                # 考虑产品数量对性价比的影响
                if len(products) < 3:
                    cost_efficiency_score *= 0.9  # 产品太少，性价比可能不高
            
            # 3. 产品多样性（增强）
            diversity_score = 0.0
            category_diversity_score = 0.0
            benefit_diversity_score = 0.0
            
            if products:
                # 品类多样性
                categories = {p.get('category', 'unknown') for p in products}
                category_diversity_score = len(categories) / len(products)
                
                # 功效多样性
                all_benefits = []
                for product in products:
                    all_benefits.extend(product.get('benefits', []))
                
                unique_benefits = set(all_benefits)
                benefit_diversity_score = len(unique_benefits) / len(all_benefits) if all_benefits else 0
                
                # 品类重要性加权
                category_importance_score = 0.0
                for category in categories:
                    importance = self.category_importance.get(category, 0.5)
                    category_importance_score += importance
                
                if categories:
                    category_importance_score /= len(categories)
                
                # 综合多样性分数
                diversity_score = (
                    category_diversity_score * 0.4 +
                    benefit_diversity_score * 0.3 +
                    category_importance_score * 0.3
                )
            
            # 4. 品类覆盖度（增强）
            category_coverage_score = 0.0
            essential_categories = {'cleanser', 'moisturizer', 'sunscreen'}
            recommended_categories = {'toner', 'serum', 'mask'}
            
            if products:
                product_categories = {p.get('category', 'unknown') for p in products}
                
                # 必需品类覆盖
                covered_essentials = essential_categories.intersection(product_categories)
                essential_coverage = len(covered_essentials) / len(essential_categories)
                
                # 推荐品类覆盖
                covered_recommended = recommended_categories.intersection(product_categories)
                recommended_coverage = len(covered_recommended) / len(recommended_categories)
                
                # 综合品类覆盖度
                category_coverage_score = essential_coverage * 0.7 + recommended_coverage * 0.3
            
            # 5. 用户偏好匹配（增强）
            preference_score = 1.0
            skin_type = user_profile.get('skin_type', 'normal')
            sensitivity = user_profile.get('sensitivity', 'medium')
            
            if products:
                unsuitable_score = 0.0
                total_weight = 0.0
                
                for product in products:
                    product_weight = 1.0
                    
                    # 检查肤质匹配
                    product_skin_types = product.get('suitable_for', [])
                    if product_skin_types and skin_type not in product_skin_types:
                        unsuitable_score += 0.3 * product_weight
                    
                    # 检查敏感性成分
                    if sensitivity == 'high':
                        harsh_ingredients = ['alcohol', 'fragrance', 'essential_oils', 'parabens']
                        ingredients = [ing.lower() for ing in product.get('ingredients', [])]
                        
                        harsh_count = 0
                        for harsh in harsh_ingredients:
                            if any(harsh in ing for ing in ingredients):
                                harsh_count += 1
                        
                        if harsh_count > 0:
                            unsuitable_score += (harsh_count * 0.2) * product_weight
                    
                    # 检查产品是否过于复杂（对于敏感肌）
                    if sensitivity == 'high' and len(product.get('ingredients', [])) > 20:
                        unsuitable_score += 0.1 * product_weight
                    
                    total_weight += product_weight
                
                if total_weight > 0:
                    suitability_ratio = 1.0 - (unsuitable_score / total_weight)
                    preference_score = max(0.1, suitability_ratio)
            
            # 6. 协同效应分数（新增）
            synergy_score = self._calculate_synergy_score(products, goal_info if 'goal_info' in locals() else {})
            
            # 7. 加权总分（使用优化后的权重）
            total_fitness = (
                goal_match_score * self.fitness_weights['goal_match'] +
                cost_efficiency_score * self.fitness_weights['cost_efficiency'] +
                diversity_score * self.fitness_weights['product_diversity'] +
                category_coverage_score * self.fitness_weights['category_coverage'] +
                preference_score * self.fitness_weights['user_preference'] +
                synergy_score * 0.05  # 协同效应额外加分
            )
            
            # 确保在0-1范围内
            fitness = max(0.1, min(0.99, total_fitness))
            
            # 轻微随机扰动（增加探索性）
            fitness += random.uniform(-0.01, 0.01)
            fitness = max(0.1, min(0.99, fitness))
            
            return round(fitness, 4)
            
        except Exception as e:
            logger.error(f"计算适应度失败: {e}")
            return 0.1

    def _get_category_support_weights(self, category: str) -> Dict[str, float]:
        """获取产品类别对目标的支撑权重"""
        # 定义不同产品类别对护肤目标的支撑强度
        category_support = {
            'cleanser': {
                'hydration': 0.2,
                'anti-aging': 0.1,
                'brightening': 0.1,
                'acne_treatment': 0.3,
                'pore_minimizing': 0.3
            },
            'moisturizer': {
                'hydration': 0.4,
                'anti-aging': 0.3,
                'brightening': 0.2,
                'repair': 0.3,
                'soothing': 0.3
            },
            'sunscreen': {
                'protection': 0.5,
                'anti-aging': 0.4,
                'brightening': 0.2
            },
            'serum': {
                'anti-aging': 0.5,
                'brightening': 0.4,
                'hydration': 0.3,
                'firming': 0.4
            },
            'toner': {
                'hydration': 0.3,
                'brightening': 0.2,
                'balancing': 0.3,
                'exfoliating': 0.2
            },
            'mask': {
                'hydration': 0.3,
                'brightening': 0.3,
                'soothing': 0.3,
                'repair': 0.3
            },
            'eye_cream': {
                'anti-aging': 0.4,
                'hydration': 0.3,
                'brightening': 0.3
            },
            'treatment': {
                'acne_treatment': 0.5,
                'pore_minimizing': 0.4,
                'exfoliating': 0.4
            }
        }
        
        return category_support.get(category, {})

    def _is_dedicated_product(self, product: Dict, goal_type: str) -> bool:
        """判断产品是否专门针对某个目标"""
        # 获取产品的主要功效（通常是列在前面的功效）
        benefits = product.get('benefits', [])
        
        if not benefits:
            return False
        
        # 如果目标是产品的第一个或主要功效，认为是专门产品
        if benefits and benefits[0] == goal_type:
            return True
        
        # 如果产品只有少数几个功效且包含目标，也可以认为是专门产品
        if len(benefits) <= 2 and goal_type in benefits:
            return True
        
        # 根据产品名称判断
        product_name = product.get('name', '').lower()
        goal_keywords = {
            'hydration': ['hydrat', 'moist', 'water'],
            'anti-aging': ['anti-age', 'antiage', 'retinol', 'peptide'],
            'brightening': ['bright', 'whiten', 'vitamin c', 'vit c'],
            'acne_treatment': ['acne', 'blemish', 'salicylic'],
            'protection': ['sunscreen', 'spf', 'uv', 'sunblock']
        }
        
        if goal_type in goal_keywords:
            keywords = goal_keywords[goal_type]
            for keyword in keywords:
                if keyword in product_name:
                    return True
        
        return False

    def _calculate_synergy_score(self, products: List[Dict], goal_info: Dict) -> float:
        """计算产品协同效应分数"""
        if len(products) < 2:
            return 0.5  # 单一产品无法评估协同效应
        
        synergy_score = 0.5  # 基础分数
        
        # 检查是否有互补的产品组合
        complementary_pairs = 0
        for i in range(len(products)):
            for j in range(i+1, len(products)):
                if self._are_products_complementary(products[i], products[j]):
                    complementary_pairs += 1
        
        # 检查是否有重复功效（负面协同）
        duplicate_benefits = 0
        all_benefits = []
        for product in products:
            all_benefits.extend(product.get('benefits', []))
        
        from collections import Counter
        benefit_counts = Counter(all_benefits)
        for count in benefit_counts.values():
            if count > 2:  # 同一功效出现在超过2个产品中可能重复
                duplicate_benefits += count - 2
        
        # 计算协同分数
        max_pairs = len(products) * (len(products) - 1) / 2
        if max_pairs > 0:
            complement_ratio = complementary_pairs / max_pairs
            synergy_score += complement_ratio * 0.3
        
        # 减少重复功效的惩罚
        if len(all_benefits) > 0:
            duplicate_penalty = duplicate_benefits / len(all_benefits)
            synergy_score -= duplicate_penalty * 0.2
        
        return max(0.1, min(1.0, synergy_score))

    def _are_products_complementary(self, product1: Dict, product2: Dict) -> bool:
        """判断两个产品是否互补"""
        cat1 = product1.get('category', '')
        cat2 = product2.get('category', '')
        benefits1 = set(product1.get('benefits', []))
        benefits2 = set(product2.get('benefits', []))
        
        # 定义互补品类
        complementary_categories = [
            ('cleanser', 'moisturizer'),
            ('toner', 'serum'),
            ('serum', 'moisturizer'),
            ('treatment', 'moisturizer')
        ]
        
        # 检查品类互补
        if (cat1, cat2) in complementary_categories or (cat2, cat1) in complementary_categories:
            return True
        
        # 检查功效互补
        complementary_benefits = [
            ('exfoliating', 'hydration'),
            ('acne_treatment', 'soothing'),
            ('brightening', 'protection')
        ]
        
        for benefit1, benefit2 in complementary_benefits:
            if (benefit1 in benefits1 and benefit2 in benefits2) or \
            (benefit2 in benefits1 and benefit1 in benefits2):
                return True
        
        return False


    def _create_new_generation(self, population: List[Dict], context: Dict) -> List[Dict]:
        """创建新一代种群 - 完整实现"""
        try:
            new_population = []
            
            # 1. 精英保留
            population.sort(key=lambda x: x['fitness'], reverse=True)
            elites = population[:self.elite_size]
            for elite in elites:
                elite_copy = copy.deepcopy(elite)
                elite_copy['age'] += 1
                new_population.append(elite_copy)
            
            # 2. 锦标赛选择父代
            while len(new_population) < self.population_size:
                # 选择父代
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, context)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                
                # 变异
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1, context)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, context)
                
                # 更新年龄和ID
                child1['age'] = 0
                child1['id'] = f"ind_{random.randint(10000, 99999)}"
                child2['age'] = 0
                child2['id'] = f"ind_{random.randint(10000, 99999)}"
                
                # 计算适应度
                child1['fitness'] = self._calculate_fitness(child1, context)
                child2['fitness'] = self._calculate_fitness(child2, context)
                
                new_population.extend([child1, child2])
            
            # 截断到固定大小
            new_population = new_population[:self.population_size]
            
            logger.debug(f"创建新一代完成，大小: {len(new_population)}")
            return new_population
            
        except Exception as e:
            logger.error(f"创建新一代失败: {e}")
            return population
    
    def _tournament_selection(self, population: List[Dict], tournament_size: int = 3) -> Dict:
        """锦标赛选择"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        tournament.sort(key=lambda x: x['fitness'], reverse=True)
        return copy.deepcopy(tournament[0])
    
    def _crossover(self, parent1: Dict, parent2: Dict, context: Dict) -> Tuple[Dict, Dict]:
        """交叉操作 - 完整实现"""
        try:
            products1 = parent1.get('products', [])
            products2 = parent2.get('products', [])
            
            if len(products1) < 2 or len(products2) < 2:
                return copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # 单点交叉
            crossover_point = random.randint(1, min(len(products1), len(products2)) - 1)
            
            child1_products = products1[:crossover_point] + products2[crossover_point:]
            child2_products = products2[:crossover_point] + products1[crossover_point:]
            
            # 移除重复产品
            child1_products = self._remove_duplicate_products(child1_products)
            child2_products = self._remove_duplicate_products(child2_products)
            
            # 确保产品数量在合理范围内
            available_products = context.get('available_products', [])
            child1_products = self._adjust_product_count(child1_products, available_products)
            child2_products = self._adjust_product_count(child2_products, available_products)
            
            # 创建新个体
            child1 = self._create_individual(child1_products, context)
            child2 = self._create_individual(child2_products, context)
            
            return child1, child2
            
        except Exception as e:
            logger.error(f"交叉操作失败: {e}")
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    def _mutate(self, individual: Dict, context: Dict) -> Dict:
        """变异操作 - 完整实现"""
        try:
            products = individual.get('products', [])
            available_products = context.get('available_products', [])
            
            if not products or not available_products:
                return individual
            
            mutation_type = random.choice(['add', 'remove', 'replace', 'swap'])
            
            if mutation_type == 'add' and len(products) < self.max_products_per_solution:
                # 添加一个新产品
                new_product = random.choice(available_products)
                if new_product not in products:
                    products.append(new_product)
            
            elif mutation_type == 'remove' and len(products) > self.min_products_per_solution:
                # 移除一个产品
                products.pop(random.randint(0, len(products) - 1))
            
            elif mutation_type == 'replace':
                # 替换一个产品
                if products:
                    idx = random.randint(0, len(products) - 1)
                    new_product = random.choice(available_products)
                    products[idx] = new_product
            
            elif mutation_type == 'swap' and len(products) >= 2:
                # 交换两个产品的位置
                idx1, idx2 = random.sample(range(len(products)), 2)
                products[idx1], products[idx2] = products[idx2], products[idx1]
            
            # 移除重复并调整数量
            products = self._remove_duplicate_products(products)
            products = self._adjust_product_count(products, available_products)
            
            # 创建新个体
            mutated_individual = self._create_individual(products, context)
            
            return mutated_individual
            
        except Exception as e:
            logger.error(f"变异操作失败: {e}")
            return individual
    
    def _remove_duplicate_products(self, products: List[Dict]) -> List[Dict]:
        """移除重复产品"""
        seen_ids = set()
        unique_products = []
        
        for product in products:
            product_id = product.get('id')
            if product_id and product_id not in seen_ids:
                seen_ids.add(product_id)
                unique_products.append(product)
        
        return unique_products
    
    def _adjust_product_count(self, products: List[Dict], available_products: List[Dict]) -> List[Dict]:
        """调整产品数量到合理范围"""
        if len(products) < self.min_products_per_solution:
            # 添加产品
            needed = self.min_products_per_solution - len(products)
            available_ids = {p.get('id') for p in products}
            candidates = [p for p in available_products if p.get('id') not in available_ids]
            
            if candidates:
                products.extend(random.sample(candidates, min(needed, len(candidates))))
        
        elif len(products) > self.max_products_per_solution:
            # 移除产品
            products = random.sample(products, self.max_products_per_solution)
        
        return products
    
    def _analyze_population(self, population: List[Dict]) -> Dict:
        """分析种群 - 完整实现"""
        try:
            if not population:
                return {
                    'population_size': 0,
                    'average_fitness': 0.0,
                    'max_fitness': 0.0,
                    'min_fitness': 0.0,
                    'fitness_std': 0.0,
                    'diversity': 0.0,
                    'average_age': 0.0,
                    'average_products_per_individual': 0.0
                }
            
            fitnesses = [ind.get('fitness', 0) for ind in population]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            max_fitness = max(fitnesses)
            min_fitness = min(fitnesses)
            
            # 计算标准差
            if len(fitnesses) > 1:
                variance = sum((f - avg_fitness) ** 2 for f in fitnesses) / (len(fitnesses) - 1)
                fitness_std = math.sqrt(variance)
            else:
                fitness_std = 0.0
            
            # 计算多样性（基于产品类别）
            all_categories = set()
            for ind in population:
                for product in ind.get('products', []):
                    all_categories.add(product.get('category', 'unknown'))
            
            diversity = len(all_categories) / max(len(population), 1)
            
            # 平均年龄
            ages = [ind.get('age', 0) for ind in population]
            avg_age = sum(ages) / len(ages) if ages else 0
            
            # 平均产品数量
            product_counts = [len(ind.get('products', [])) for ind in population]
            avg_products = sum(product_counts) / len(product_counts) if product_counts else 0
            
            return {
                'population_size': len(population),
                'average_fitness': round(avg_fitness, 4),
                'max_fitness': round(max_fitness, 4),
                'min_fitness': round(min_fitness, 4),
                'fitness_std': round(fitness_std, 4),
                'diversity': round(diversity, 4),
                'average_age': round(avg_age, 2),
                'average_products_per_individual': round(avg_products, 2),
                'unique_categories': list(all_categories),
                'generation_stats': self.generation_stats[-5:] if self.generation_stats else []
            }
            
        except Exception as e:
            logger.error(f"分析种群失败: {e}")
            return {
                'population_size': len(population) if population else 0,
                'average_fitness': 0.1,
                'max_fitness': 0.1,
                'min_fitness': 0.1,
                'diversity': 0.0
            }
    
    def _calculate_all_fitness_scores(self, population: List[Dict], context: Dict) -> List[Dict]:
        """计算所有个体的适应度分数"""
        try:
            fitness_scores = []
            
            for individual in population:
                fitness = individual.get('fitness', 0)
                product_ids = [p.get('id', 'unknown') for p in individual.get('products', [])]
                
                fitness_scores.append({
                    'individual_id': individual.get('id', 'unknown'),
                    'fitness': round(fitness, 4),
                    'product_count': len(individual.get('products', [])),
                    'product_ids': product_ids,
                    'total_cost': individual.get('total_cost', 0),
                    'age': individual.get('age', 0)
                })
            
            # 按适应度排序
            fitness_scores.sort(key=lambda x: x['fitness'], reverse=True)
            
            return fitness_scores
            
        except Exception as e:
            logger.error(f"计算适应度分数失败: {e}")
            return []
    
    def _format_best_solution(self, best_solution: Dict, context: Dict) -> Dict:
        """格式化最佳解决方案"""
        try:
            products = best_solution.get('products', [])
            fitness = best_solution.get('fitness', 0)
            total_cost = best_solution.get('total_cost', 0)
            
            # 计算各种指标
            user_goals = context.get('user_goals', [])
            goal_types = {goal.get('type') for goal in user_goals if goal.get('type')}
            
            all_benefits = set()
            for product in products:
                all_benefits.update(product.get('benefits', []))
            
            matched_goals = goal_types.intersection(all_benefits)
            goal_match_ratio = len(matched_goals) / len(goal_types) if goal_types else 0
            
            # 产品类别分布
            categories = {}
            for product in products:
                category = product.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
            
            return {
                'products': products,
                'fitness': round(fitness, 4),
                'total_cost': total_cost,
                'product_count': len(products),
                'goal_match_ratio': round(goal_match_ratio, 4),
                'matched_goals': list(matched_goals),
                'category_distribution': categories,
                'unique_benefits': list(all_benefits),
                'recommendation_score': round(fitness * 100, 1),
                'individual_id': best_solution.get('id', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"格式化最佳解决方案失败: {e}")
            return {
                'products': best_solution.get('products', []),
                'fitness': best_solution.get('fitness', 0),
                'total_cost': 0,
                'product_count': 0
            }
    
    def _generate_evolution_summary(self, population: List[Dict], context: Dict) -> Dict:
        """生成进化摘要"""
        try:
            if not self.generation_stats:
                return {
                    'generations': 0,
                    'improvement_rate': 0.0,
                    'convergence': False,
                    'elapsed_time': 0.0,
                    'notes': '无进化历史'
                }
            
            # 计算改进率
            if len(self.generation_stats) >= 2:
                first_best = self.generation_stats[0]['best_fitness']
                last_best = self.generation_stats[-1]['best_fitness']
                improvement_rate = (last_best - first_best) / first_best if first_best > 0 else 0
            else:
                improvement_rate = 0.0
            
            # 检查收敛（最近3代改进很小）
            convergence = False
            if len(self.generation_stats) >= 3:
                recent_generations = self.generation_stats[-3:]
                fitness_values = [g['best_fitness'] for g in recent_generations]
                max_diff = max(fitness_values) - min(fitness_values)
                convergence = max_diff < 0.01  # 改进小于1%
            
            return {
                'generations': len(self.generation_stats),
                'improvement_rate': round(improvement_rate, 4),
                'convergence': convergence,
                'average_population_size': round(sum(g['population_size'] for g in self.generation_stats) / len(self.generation_stats)),
                'best_fitness_history': [g['best_fitness'] for g in self.generation_stats],
                'elapsed_time': sum(g.get('elapsed_time', 0) for g in self.generation_stats),
                'notes': f"完成 {len(self.generation_stats)} 代进化，{'已收敛' if convergence else '未收敛'}"
            }
            
        except Exception as e:
            logger.error(f"生成进化摘要失败: {e}")
            return {
                'generations': 0,
                'improvement_rate': 0.0,
                'convergence': False
            }
    
    def _calculate_certainty(self, best_solution: Dict, population: List[Dict], context: Optional[Dict] = None) -> float:
        """计算确定性 - 完整实现"""
        try:
            if best_solution is None:
                return 0.1
            
            fitness = best_solution.get('fitness', 0)
            
            # 基础确定性
            base_certainty = 0.3 + (fitness * 0.5)
            
            # 根据种群质量调整
            if population and len(population) > 0:
                avg_fitness = sum(ind.get('fitness', 0) for ind in population) / len(population)
                fitness_std = math.sqrt(sum((ind.get('fitness', 0) - avg_fitness) ** 2 for ind in population) / len(population))
                
                # 最佳个体明显优于平均水平
                if fitness_std > 0:
                    superiority = (fitness - avg_fitness) / fitness_std
                    base_certainty += min(0.3, superiority * 0.1)
            
            # 根据上下文调整
            if context:
                available_products = context.get('available_products', [])
                if available_products:
                    # 产品多样性
                    product_count = len(best_solution.get('products', []))
                    max_possible = min(self.max_products_per_solution, len(available_products))
                    if max_possible > 0:
                        coverage = product_count / max_possible
                        base_certainty += coverage * 0.1
            
            # 确保在合理范围内
            certainty = max(0.1, min(0.95, base_certainty))
            
            return round(certainty, 3)
            
        except Exception as e:
            logger.error(f"计算确定性失败: {e}")
            return 0.1
    
    def _record_generation_stats(self, population: List[Dict], generation: int):
        """记录代际统计信息"""
        try:
            if not population:
                return
            
            population.sort(key=lambda x: x['fitness'], reverse=True)
            best_fitness = population[0]['fitness']
            avg_fitness = sum(ind['fitness'] for ind in population) / len(population)
            
            stats = {
                'generation': generation,
                'best_fitness': round(best_fitness, 4),
                'average_fitness': round(avg_fitness, 4),
                'population_size': len(population),
                'elapsed_time': 0.0  # 实际实现中可以计时
            }
            
            self.generation_stats.append(stats)
            
        except Exception as e:
            logger.error(f"记录代际统计失败: {e}")
    
    def _get_simplified_solution(self, context: Dict) -> Dict:
        """当数据不足时的简化解决方案"""
        try:
            available_products = context.get('available_products', [])
            user_goals = context.get('user_goals', [])
            
            if not available_products:
                return self._get_default_result(context)
            
            # 选择前几个产品作为简化方案
            simplified_products = available_products[:min(3, len(available_products))]
            
            # 创建个体
            individual = self._create_individual(simplified_products, context)
            individual['fitness'] = self._calculate_fitness(individual, context)
            
            # 构建结果
            return {
                'population_analysis': {
                    'population_size': 1,
                    'average_fitness': individual['fitness'],
                    'max_fitness': individual['fitness'],
                    'min_fitness': individual['fitness'],
                    'diversity': 0.5
                },
                'fitness_scores': [{
                    'individual_id': individual['id'],
                    'fitness': individual['fitness'],
                    'product_count': len(simplified_products)
                }],
                'best_solution': self._format_best_solution(individual, context),
                'evolution_summary': {
                    'generations': 0,
                    'improvement_rate': 0.0,
                    'convergence': True,
                    'notes': '简化解决方案（数据不足）'
                },
                'certainty': 0.3,
                'framework_status': 'simplified'
            }
            
        except Exception as e:
            logger.error(f"生成简化解决方案失败: {e}")
            return self._get_default_result(context)
    
    def _get_default_result(self, context: Optional[Dict] = None) -> Dict:
        """获取默认结果"""
        available_products = context.get('available_products', []) if context else []
        
        return {
            'population_analysis': {
                'population_size': 0,
                'average_fitness': 0.1,
                'max_fitness': 0.1,
                'min_fitness': 0.1,
                'diversity': 0.0
            },
            'fitness_scores': [],
            'best_solution': {
                'products': available_products[:3],
                'fitness': 0.1,
                'total_cost': 0,
                'product_count': min(3, len(available_products))
            },
            'evolution_summary': {
                'generations': 0,
                'improvement_rate': 0.0,
                'convergence': False,
                'notes': '默认回退结果'
            },
            'certainty': 0.1,
            'framework_status': 'fallback',
            'error': '进化算法框架回退'
        }

 


class FatTailRiskFramework:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    async def infer(self, skin_params, context=None):
        """肥尾风险框架推理 - 修复版"""
        try:
            # 风险分析逻辑
            extreme_risks = ['紫外线伤害', '过敏反应', '环境污染']
            
            # 创建结果
            result = {
                'extreme_risks': extreme_risks,
                'risk_level': 'medium',
                'risk_probabilities': [0.05, 0.03, 0.02],
                'risk_mitigation': ['使用防晒', '选择温和产品', '注意清洁']
            }
            
            # 关键修复：调用 _calculate_certainty 方法
            result['certainty'] = self._calculate_certainty(result)
            result['framework_status'] = 'success'
            
            return result
            
        except Exception as e:
            logger.error(f"风险框架推理失败: {e}")
            return {
                'extreme_risks': [],
                'certainty': 0.1,
                'framework_status': 'error'
            }
    
    # 在 FatTailRiskFramework 类中添加
    def _calculate_certainty(self, risk_results):
        """计算肥尾风险框架的确定性"""
        try:
            # 根据风险参数计算
            extreme_risks = risk_results.get('extreme_risks', [])
            
            # 简单计算：风险越多，确定性越低
            risk_count = len(extreme_risks)
            if risk_count == 0:
                certainty = 0.8
            elif risk_count <= 2:
                certainty = 0.6
            else:
                certainty = 0.4
            
            return max(0.1, min(1.0, certainty))
            
        except Exception as e:
            logger.error(f"风险框架计算确定性失败: {e}")
            return 0.1

    def _identify_extreme_risks(self, context: Dict) -> List[Dict]:
        """识别极端风险"""
        risks = []
        
        # 识别皮肤相关的极端风险
        environmental_data = context.get('environmental_data', {})
        skin_analysis = context.get('skin_analysis', {})
        
        # 极端天气风险
        temperature = environmental_data.get('temperature', 25)
        if temperature > 40 or temperature < -10:
            risks.append({
                'type': 'extreme_temperature',
                'severity': 'high',
                'probability': 0.1,
                'impact': 'skin_damage'
            })
        
        # 皮肤过敏风险
        if skin_analysis.get('sensitivity', {}).get('score', 100) < 30:
            risks.append({
                'type': 'allergic_reaction',
                'severity': 'medium',
                'probability': 0.2,
                'impact': 'skin_irritation'
            })
        
        # 紫外线伤害风险
        uv_index = environmental_data.get('uv_index', 0)
        if uv_index > 11:
            risks.append({
                'type': 'extreme_uv_exposure',
                'severity': 'high',
                'probability': 0.3,
                'impact': 'sunburn_or_long_term_damage'
            })
        
        return risks

    def _define_risk_constraints(self, context: Dict) -> List[str]:
        """定义风险约束"""
        constraints = []
        
        # 基于风险的约束
        skin_analysis = context.get('skin_analysis', {})
        
        if skin_analysis.get('barrier', {}).get('score', 100) < 40:
            constraints.append('avoid_irritants')
        
        if skin_analysis.get('hydration', {}).get('score', 100) < 30:
            constraints.append('avoid_dehydrating_products')
        
        return constraints

    def _suggest_mitigation_strategies(self, context: Dict) -> List[Dict]:
        """建议缓解策略"""
        strategies = []
        risks = self._identify_extreme_risks(context)
        
        for risk in risks:
            if risk['severity'] == 'high':
                strategies.append({
                    'risk_type': risk['type'],
                    'strategy': self._get_mitigation_strategy(risk['type']),
                    'priority': 'urgent'
                })
            elif risk['severity'] == 'medium':
                strategies.append({
                    'risk_type': risk['type'],
                    'strategy': self._get_mitigation_strategy(risk['type']),
                    'priority': 'important'
                })
        
        return strategies

    def _get_mitigation_strategy(self, risk_type: str) -> str:
        """获取缓解策略"""
        strategy_map = {
            'extreme_temperature': '使用温度调节产品，避免长时间暴露',
            'allergic_reaction': '进行过敏测试，避免已知过敏原',
            'extreme_uv_exposure': '使用高SPF防晒霜，避免阳光直射时段'
        }
        
        return strategy_map.get(risk_type, '咨询专业皮肤科医生')


class EightFrameworkWorldModel:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self):
        # super().__init__()  # 移除super调用
        
        # ==================== 实例化九个框架 ====================
        # 原有的八个框架
        self.physics_framework = PhysicsFramework()  # 物理特性分析
        self.biology_framework = BiologyFramework()
        self.geometric_dynamics_framework = GeometricDynamicsFramework()
        # self.real2sim2real_framework = Real2Sim2RealSkinFramework()  # 暂时注释掉
        self.mental_framework = MentalFramework()
        self.decision_theory_framework = DecisionTheoryFramework()
        self.evolutionary_algorithm_framework = EvolutionaryAlgorithmFramework()
        self.fat_tail_risk_framework = FatTailRiskFramework()
        
        # [OK] 新增第九框架：深度学习框架
        self.deep_learning_framework = DeepLearningFramework()

        # 新增：测试数据生成器
        self.test_data_generator = EnhancedTestDataGenerator()
        
        # 新增：性能监控
        self.performance_monitor = {
            'execution_times': [],
            'solution_quality': [],
            'diversity_scores': []
        }
        
        # ==================== 更新集成权重 ====================
        # 现在有9个框架，重新分配权重（总和为1.0）
        self.integration_weights = {
            'physics': 0.12,              # 从0.15降低
            'biology': 0.15,              # 从0.2降低
            'geometric_dynamics': 0.12,   # 从0.15降低
            'real2sim2real': 0.0,         # 暂时设为0，因为被注释掉了
            'mental': 0.08,               # 从0.1降低
            'decision_theory': 0.08,      # 从0.1降低
            'evolutionary_algorithm': 0.08, # 从0.05增加
            'fat_tail_risk': 0.07,        # 从0.05增加
            'deep_learning': 0.30         # [OK] 新增，赋予较高权重
        }
        # 权重总和：0.12+0.15+0.12+0+0.08+0.08+0.08+0.07+0.30 = 1.00
        
        # ==================== 更新状态记忆 ====================
        self.state_memory = {
            'historical_states': [],      # 历史状态记录
            'trend_analysis': {},         # 趋势分析
            'pattern_recognition': {},    # 模式识别
            'anomaly_detection': {},      # 异常检测
            'deep_learning_insights': []  # [OK] 新增：深度学习洞察历史
        }
        
        # ==================== 更新框架间通信通道 ====================
        self.inter_framework_communication = {
            'physics': [],
            'biology': [],
            'geometric_dynamics': [],
            'mental': [],
            'decision_theory': [],
            'evolutionary_algorithm': [],
            'fat_tail_risk': [],
            'deep_learning': []  # [OK] 新增：深度学习通信通道
        }
        
        # ==================== 更新框架性能跟踪 ====================
        self.framework_performance = {
            'physics': {'success_count': 0, 'total_count': 0},
            'biology': {'success_count': 0, 'total_count': 0},
            'geometric_dynamics': {'success_count': 0, 'total_count': 0},
            'mental': {'success_count': 0, 'total_count': 0},
            'decision_theory': {'success_count': 0, 'total_count': 0},
            'evolutionary_algorithm': {'success_count': 0, 'total_count': 0},
            'fat_tail_risk': {'success_count': 0, 'total_count': 0},
            'deep_learning': {'success_count': 0, 'total_count': 0}  # [OK] 新增
        }
        
        logger.info("九框架世界模型初始化完成（包含深度学习框架）")
    
        self.db = None  # 添加数据库属性
    
    def set_database(self, db):
        """设置数据库"""
        self.db = db
    
    def enhance_context_with_products(self, context):
        """用产品数据增强上下文"""
        enhanced = context.copy()
        
        if hasattr(self, 'db') and self.db:
            # 获取所有产品
            products = self.db.get_all_products()
            
            # 添加产品数据到上下文
            enhanced['available_products'] = products
            enhanced['product_count'] = len(products)
            
            # 按类别分组
            products_by_category = {}
            for product in products:
                category = product.get('category', 'unknown')
                if category not in products_by_category:
                    products_by_category[category] = []
                products_by_category[category].append(product)
            
            enhanced['products_by_category'] = products_by_category
            
            # 按功效分组
            products_by_benefit = {}
            for product in products:
                benefits = product.get('benefits', [])
                for benefit in benefits:
                    if benefit not in products_by_benefit:
                        products_by_benefit[benefit] = []
                    products_by_benefit[benefit].append(product)
            
            enhanced['products_by_benefit'] = products_by_benefit
            
            print(f"[OK] 已增强上下文: {len(products)} 个产品，{len(products_by_category)} 个类别")
        else:
            enhanced['available_products'] = []
            enhanced['product_count'] = 0
            enhanced['products_by_category'] = {}
            enhanced['products_by_benefit'] = {}
            print("[WARN] 未设置数据库，无法增强产品上下文")
        
        return enhanced

    # 在推理前添加默认值处理
    def _ensure_state_fields(self, state: Dict) -> Dict:
        """确保状态包含所有必要字段"""
        # 基础皮肤指标
        base_fields = {
            'moisture_content': 0.5,
            'elasticity': 0.5,
            'collagen_level': 0.5,
            'oiliness': 0.5,
            'pigmentation': 0.5,
            'wrinkles': 0.5,
            'pores': 0.5,
            'sensitivity': 0.5,
            'certainty': 0.5,  # 添加 certainty 字段
        }
        
        # 合并现有状态和默认值
        for field, default in base_fields.items():
            if field not in state:
                state[field] = default
                logger.debug(f"为状态添加默认字段: {field} = {default}")
        
        return state

    def nine_framework_reasoning(self, state: Dict, context: Dict) -> Dict:
        """九框架推理（修复版本）- 同步版本"""
        try:
            # 1. 确保状态字段完整
            state = self._ensure_state_fields(state)
            
            # 2. 调用原有的推理逻辑（根据你的实际代码调整）
            # 这里需要替换为你的实际推理方法名
            
            # 示例：假设原始调用是这样的
            integrated_results = self.integrated_reasoning(state, context)
            
            # 3. 确保结果包含 certainty 字段
            if isinstance(integrated_results, dict):
                if 'certainty' not in integrated_results:
                    integrated_results['certainty'] = state.get('certainty', 0.5)
                elif integrated_results['certainty'] is None:
                    integrated_results['certainty'] = state.get('certainty', 0.5)
            else:
                # 如果结果不是字典，转换为字典
                integrated_results = {
                    'recommendations': integrated_results if isinstance(integrated_results, list) else [integrated_results],
                    'reasoning': '九框架推理结果',
                    'certainty': state.get('certainty', 0.5)
                }
            
            return integrated_results
            
        except Exception as e:
            logger.error(f"九框架推理失败: {str(e)}", exc_info=True)
            
            # 返回安全的默认结果
            return {
                'recommendations': ['温和洁面，保持皮肤清洁，使用基础保湿产品'],
                'reasoning': '由于系统推理失败，使用基础护肤建议',
                'certainty': 0.3,  # 低置信度
                'status': 'fallback',
                'error': str(e)[:100]  # 截断错误信息
            }

    # ==================== 增强的infer_async方法 ====================
    async def infer_async(self, state, action=None, context=None):
        """增强的异步推理方法 - 包含9个框架"""
        try:
            # 1. 输入验证
            self._validate_input(state, context)
            
            # 2. 增强上下文准备
            prepared_context = self._prepare_context_for_frameworks(state, context)
            
            # 3. 获取数字孪生状态
            digital_twin_state = self._get_default_digital_twin_state(state)
            
            # 4. 提取皮肤参数
            skin_params = self._extract_skin_parameters(digital_twin_state)
            
            # 5. [OK] 并行运行9个框架（新增深度学习框架）
            framework_tasks = [
                # 原有的7个框架
                asyncio.create_task(self._call_framework_safely(
                    self.physics_framework, skin_params, prepared_context
                )),
                asyncio.create_task(self._call_framework_safely(
                    self.biology_framework, skin_params, prepared_context
                )),
                asyncio.create_task(self._call_framework_safely(
                    self.geometric_dynamics_framework, skin_params, prepared_context
                )),
                asyncio.create_task(self._call_framework_safely(
                    self.mental_framework, skin_params, prepared_context
                )),
                asyncio.create_task(self._call_framework_safely(
                    self.decision_theory_framework, skin_params, prepared_context
                )),
                asyncio.create_task(self._call_framework_safely(
                    self.evolutionary_algorithm_framework, skin_params, prepared_context
                )),
                asyncio.create_task(self._call_framework_safely(
                    self.fat_tail_risk_framework, skin_params, prepared_context
                )),
                
                # [OK] 新增：深度学习框架（第9个框架）
                asyncio.create_task(self._call_framework_safely(
                    self.deep_learning_framework, skin_params, prepared_context
                ))
            ]
            
            # 6. 安全收集9个框架的结果（关键修复）
            framework_results = []
            for i, task in enumerate(framework_tasks):
                try:
                    result = await task
                    
                    # [OK] 关键修复：确保结果是字典类型
                    if not isinstance(result, dict):
                        logger.warning(f"框架 {i} 返回了非字典结果，类型: {type(result)}")
                        result = {
                            'framework_status': 'error',
                            'error': f'返回了非字典类型: {type(result)}',
                            'framework_name': self._get_framework_name_by_index(i)
                        }
                    
                    # [OK] 确保有certainty字段
                    if 'certainty' not in result or not isinstance(result['certainty'], (int, float)):
                        # 根据框架状态设置默认certainty
                        status = result.get('framework_status', 'unknown')
                        if status == 'success':
                            result['certainty'] = 0.8
                        elif status == 'simplified':
                            result['certainty'] = 0.6
                        elif status == 'error':
                            result['certainty'] = 0.1
                        elif status == 'baseline_mode':
                            result['certainty'] = 0.3
                        else:
                            result['certainty'] = 0.5
                    
                    framework_results.append(result)
                    
                    # [OK] 记录框架性能（保持原始逻辑）
                    framework_name = self._get_framework_name_by_index(i)
                    if result.get('framework_status') in ['success', 'simplified', 'baseline_mode']:
                        self.framework_performance[framework_name]['success_count'] += 1
                    self.framework_performance[framework_name]['total_count'] += 1
                        
                except Exception as e:
                    logger.error(f"框架 {i} 执行失败: {e}")
                    error_result = {
                        'framework_status': 'error',
                        'error': str(e),
                        'framework_name': self._get_framework_name_by_index(i),
                        'certainty': 0.1  # 错误状态下默认低certainty
                    }
                    framework_results.append(error_result)
                    
                    # [OK] 记录失败框架的性能（保持原始逻辑）
                    framework_name = self._get_framework_name_by_index(i)
                    self.framework_performance[framework_name]['total_count'] += 1
            
            # 7. [OK] 增强：为深度学习框架提供其他框架的结果
            if len(framework_results) >= 8:
                # 创建增强的上下文，包含前7个框架的结果
                enhanced_context = prepared_context.copy()
                enhanced_context['framework_results'] = framework_results[:7]  # 前7个框架的结果
                
                # 重新运行深度学习框架（使用增强上下文）
                try:
                    deep_learning_result = await self.deep_learning_framework.infer(
                        skin_params, enhanced_context
                    )
                    
                    # [OK] 确保深度学习框架结果有certainty键
                    if isinstance(deep_learning_result, dict):
                        if 'certainty' not in deep_learning_result or not isinstance(deep_learning_result['certainty'], (int, float)):
                            deep_learning_result['certainty'] = 0.8
                        framework_results[7] = deep_learning_result
                except Exception as e:
                    logger.error(f"深度学习框架增强推理失败: {e}")
                    # 确保即使增强推理失败，也有certainty字段
                    if 7 < len(framework_results):
                        framework_results[7] = self._ensure_framework_result_has_certainty(
                            framework_results[7], 'deep_learning'
                        )
            
            # 8. 实现框架间信息共享（更新包含深度学习）
            self._share_framework_results_with_deep_learning(framework_results)
            
            # 9. 执行跨框架推理（更新包含深度学习）
            cross_framework_results = self._cross_framework_inference_with_deep_learning(
                framework_results, digital_twin_state
            )
            
            # 10. 整合所有推理结果（更新包含深度学习）
            integrated_results = self._integrate_inferences_with_deep_learning(
                framework_results, cross_framework_results, digital_twin_state
            )

            # 11. 计算总体确定性
            certainty = self._calculate_certainty_with_deep_learning(framework_results, integrated_results)
            
            # 12. 更新状态记忆（包含深度学习洞察）
            self._update_state_memory_with_deep_learning(state, integrated_results, framework_results)
            
            # 13. 动态调整集成权重（基于性能）
            self._adjust_integration_weights_based_on_performance(framework_results)
            
            # 14. 构建最终结果
            final_result = {
                'inference': integrated_results,
                'certainty': certainty,  # 确保这个变量已计算
                'digital_twin_state': digital_twin_state,
                'framework_performance': self._get_framework_performance_summary(),
                'deep_learning_insights': framework_results[7].get('deep_learning_analysis', {}) 
                                        if len(framework_results) > 7 else {},
                'deep_learning_performance': self.deep_learning_framework.get_performance_summary(),
                'all_frameworks_success': all(
                    result.get('framework_status') in ['success', 'simplified', 'baseline_mode']
                    for result in framework_results 
                    if isinstance(result, dict)
                ),
                'total_frameworks': 9,  # [OK] 现在有9个框架
                'active_frameworks': [name for name, weight in self.integration_weights.items() if weight > 0],
                'integration_weights': self.integration_weights
            }
            
            # [OK] 关键修复：双重检查certainty是否存在
            if 'certainty' not in final_result or final_result['certainty'] is None:
                final_result['certainty'] = 0.5
            
            logger.info(f"九框架推理完成，总体certainty: {final_result['certainty']:.3f}")
            return final_result
            
        except Exception as e:
            logger.error(f"九框架推理失败: {e}")
            # [OK] 修复：返回完整的回退结果，确保包含所有必需字段
            return self._get_fallback_result(state, e)

    def _get_fallback_result(self, state, error=None):
        """获取回退结果 - 确保包含所有必需字段"""
        try:
            fallback_result = {
                'inference': {
                    'error': str(error) if error else 'Unknown error',
                    'fallback_mode': True,
                    'key_recommendations': ['系统暂时不可用，请稍后重试']
                },
                'certainty': 0.3,  # 回退结果的低certainty
                'digital_twin_state': self._get_default_digital_twin_state(state) if hasattr(self, '_get_default_digital_twin_state') else {},
                'framework_performance': self._get_framework_performance_summary(),
                'deep_learning_insights': {},
                'deep_learning_performance': {'total_recommendations': 0, 'success_rate': 0},
                'all_frameworks_success': False,
                'total_frameworks': 9,
                'active_frameworks': [],
                'integration_weights': getattr(self, 'integration_weights', {}),
                'fallback_reason': str(error) if error else 'unknown'
            }
            
            # 确保certainty存在且有效
            if 'certainty' not in fallback_result or not isinstance(fallback_result['certainty'], (int, float)):
                fallback_result['certainty'] = 0.3
                
            return fallback_result
            
        except Exception as fallback_error:
            logger.error(f"生成回退结果也失败了: {fallback_error}")
            return {
                'inference': {'error': str(error) if error else 'Unknown error', 'fallback_mode': True},
                'certainty': 0.3,
                'all_frameworks_success': False
            }

    def _ensure_framework_result_has_certainty(self, result, framework_name):
        """确保框架结果包含certainty字段"""
        if not isinstance(result, dict):
            return {
                'framework_status': 'error',
                'error': '非字典类型',
                'framework_name': framework_name,
                'certainty': 0.1
            }
        
        if 'certainty' not in result or not isinstance(result['certainty'], (int, float)):
            status = result.get('framework_status', 'unknown')
            if status == 'success':
                result['certainty'] = 0.8
            elif status == 'simplified':
                result['certainty'] = 0.6
            elif status == 'error':
                result['certainty'] = 0.1
            elif status == 'baseline_mode':
                result['certainty'] = 0.3
            else:
                result['certainty'] = 0.5
        
        return result

    def _calculate_certainty_with_deep_learning(self, framework_results: List[Dict], integrated_results: Dict) -> float:
        """计算总体确定性 - 安全版本"""
        try:
            # [OK] 安全地获取框架结果和集成结果
            if not framework_results:
                return 0.5  # 默认值
            
            # 1. 检查框架结果的有效性
            valid_frameworks = [r for r in framework_results if isinstance(r, dict)]
            if not valid_frameworks:
                return 0.5
            
            # 2. 收集所有框架的certainty值（带默认值）
            framework_certainties = []
            for result in valid_frameworks:
                if isinstance(result, dict):
                    # 尝试获取certainty
                    certainty = result.get('certainty')
                    
                    # 如果框架有明确的certainty值，使用它
                    if certainty is not None and isinstance(certainty, (int, float)):
                        framework_certainties.append(float(certainty))
                    else:
                        # 根据框架状态推断certainty
                        status = result.get('framework_status', 'unknown')
                        if status == 'success':
                            framework_certainties.append(0.8)
                        elif status in ['simplified', 'baseline_mode']:
                            framework_certainties.append(0.6)
                        elif status == 'error':
                            framework_certainties.append(0.3)
                        else:
                            framework_certainties.append(0.5)
            
            # 3. 如果至少有一个框架提供了certainty，使用平均值
            if framework_certainties:
                avg_certainty = sum(framework_certainties) / len(framework_certainties)
                
                # 根据框架成功率调整
                success_count = sum(1 for r in valid_frameworks 
                                if r.get('framework_status') in ['success', 'simplified', 'baseline_mode'])
                success_rate = success_count / len(valid_frameworks) if valid_frameworks else 0
                
                # 成功率越高，certainty越高
                if success_rate > 0.8:
                    adjusted_certainty = min(0.95, avg_certainty * 1.1)
                elif success_rate > 0.6:
                    adjusted_certainty = avg_certainty
                elif success_rate > 0.4:
                    adjusted_certainty = max(0.3, avg_certainty * 0.9)
                else:
                    adjusted_certainty = max(0.2, avg_certainty * 0.8)
                
                return round(adjusted_certainty, 3)
            
            # 4. 回退到默认值
            return 0.5
            
        except Exception as e:
            # 在出现任何错误时返回默认值
            logger.error(f"计算确定性失败，返回默认值0.5: {e}")
            return 0.5

    def _integrate_inferences_with_deep_learning(self, framework_results: List[Dict],
                                            cross_framework_results: Dict,
                                            digital_twin_state: Dict) -> Dict:
        """整合所有推理结果（包含深度学习）"""
        integrated = {}
        
        # ... 现有代码 ...
        
        # [OK] 关键修复：确保整合结果中有certainty字段
        if 'certainty' not in integrated:
            # 尝试从框架结果中计算
            if framework_results:
                certainties = []
                for result in framework_results:
                    if isinstance(result, dict):
                        certainty = result.get('certainty')
                        if isinstance(certainty, (int, float)):
                            certainties.append(certainty)
                
                if certainties:
                    integrated['certainty'] = sum(certainties) / len(certainties)
                else:
                    integrated['certainty'] = 0.5
            else:
                integrated['certainty'] = 0.5
        
        return integrated
        
    # ==================== 新增的辅助方法 ====================
    
    def _get_framework_name_by_index(self, index: int) -> str:
        """根据索引获取框架名称"""
        framework_names = [
            'physics', 'biology', 'geometric_dynamics', 
            'mental', 'decision_theory', 'evolutionary_algorithm', 
            'fat_tail_risk', 'deep_learning'
        ]
        return framework_names[index] if index < len(framework_names) else f'unknown_{index}'
    
    def _share_framework_results_with_deep_learning(self, framework_results: List[Dict]):
        """框架间结果共享（包含深度学习）"""
        # 原有的共享逻辑（如果有的话）
        if hasattr(super(), '_share_framework_results'):
            super()._share_framework_results(framework_results)
        
        # 为每个框架存储最近的结果
        for i, result in enumerate(framework_results):
            framework_name = self._get_framework_name_by_index(i)
            if framework_name in self.inter_framework_communication:
                self.inter_framework_communication[framework_name].append(result)
                
                # 限制存储数量
                if len(self.inter_framework_communication[framework_name]) > 10:
                    self.inter_framework_communication[framework_name] = \
                        self.inter_framework_communication[framework_name][-10:]
        
        # 特别为深度学习框架提供其他框架的关键洞察
        if len(framework_results) > 7:
            deep_learning_channel = self.inter_framework_communication.get('deep_learning', [])
            
            # 提取其他框架的关键洞察
            other_insights = []
            for i in range(7):  # 前7个框架
                if i < len(framework_results):
                    insight = self._extract_key_insight_for_deep_learning(
                        framework_results[i], self._get_framework_name_by_index(i)
                    )
                    if insight:
                        other_insights.append(insight)
            
            if other_insights:
                deep_learning_channel.append({
                    'type': 'other_framework_insights',
                    'timestamp': datetime.now().isoformat(),
                    'insights': other_insights
                })

    # 在 EightFrameworkWorldModel 类中添加测试方法
    async def test_deep_learning_integration(self) -> Dict:
        """测试深度学习框架集成"""
        test_data = {
            'skin_params': {
                'hydration': {'score': 65},
                'elasticity': {'score': 70},
                'barrier': {'score': 60}
            },
            'context': {
                'user_profile': {'age': 35, 'skin_type': 'normal'},
                'environmental_data': {'uv_index': 5, 'humidity': 60}
            }
        }
        
        try:
            # 测试深度学习框架
            result = await self.deep_learning_framework.infer(
                test_data['skin_params'], 
                test_data['context']
            )
            
            return {
                'integration_success': True,
                'deep_learning_status': result.get('framework_status'),
                'certainty': result.get('certainty'),
                'has_recommendations': 'deep_learning_analysis' in result,
                'message': '深度学习框架集成成功！'
            }
        except Exception as e:
            return {
                'integration_success': False,
                'error': str(e),
                'message': '深度学习框架集成失败！'
            }


    def _extract_key_insight_for_deep_learning(self, framework_result: Dict, framework_name: str) -> Optional[Dict]:
        """为深度学习提取框架的关键洞察"""
        if not isinstance(framework_result, dict):
            return None
        
        try:
            if framework_name == 'physics' and 'material_properties' in framework_result:
                return {
                    'source': framework_name,
                    'type': 'material_properties',
                    'count': len(framework_result['material_properties'])
                }
            elif framework_name == 'biology' and 'biological_state' in framework_result:
                return {
                    'source': framework_name,
                    'type': 'biological_state',
                    'data': framework_result['biological_state']
                }
            elif framework_name == 'evolutionary_algorithm' and 'best_solution' in framework_result:
                return {
                    'source': framework_name,
                    'type': 'optimized_solution',
                    'fitness': framework_result['best_solution'].get('fitness', 0)
                }
            elif framework_name == 'fat_tail_risk' and 'extreme_risks' in framework_result:
                return {
                    'source': framework_name,
                    'type': 'risk_assessment',
                    'risk_count': len(framework_result['extreme_risks'])
                }
        except Exception as e:
            logger.debug(f"提取{framework_name}框架洞察失败: {e}")
        
        return None
    
    def _cross_framework_inference_with_deep_learning(self, framework_results: List[Dict], 
                                                    digital_twin_state: Dict) -> Dict:
        """包含深度学习的跨框架推理"""
        cross_results = {}
        
        # 原有的跨框架推理（如果有的话）
        if hasattr(self, 'cross_framework_inference'):
            cross_results = self.cross_framework_inference(framework_results, digital_twin_state)
        
        # 添加深度学习特定的跨框架分析
        if len(framework_results) > 7:
            deep_learning_result = framework_results[7]
            
            # 深度学习与物理框架的集成
            if len(framework_results) > 0:
                cross_results['deep_physics_integration'] = self._integrate_deep_with_physics(
                    deep_learning_result, framework_results[0]
                )
            
            # 深度学习与生物框架的集成
            if len(framework_results) > 1:
                cross_results['deep_biology_integration'] = self._integrate_deep_with_biology(
                    deep_learning_result, framework_results[1]
                )
            
            # 深度学习与进化算法的集成
            if len(framework_results) > 5:
                cross_results['deep_evolutionary_integration'] = self._integrate_deep_with_evolutionary(
                    deep_learning_result, framework_results[5]
                )
            
            # 深度学习与风险框架的集成
            if len(framework_results) > 6:
                cross_results['deep_risk_integration'] = self._integrate_deep_with_risk(
                    deep_learning_result, framework_results[6]
                )
        
        return cross_results
    
    def _integrate_deep_with_physics(self, deep_result: Dict, physics_result: Dict) -> Dict:
        """深度学习与物理框架集成"""
        integration = {
            'integration_type': 'deep_physics',
            'status': 'active',
            'insights': []
        }
        
        try:
            # 提取物理特性
            if 'material_properties' in physics_result:
                material_props = physics_result['material_properties']
                integration['physics_insights'] = f"分析{len(material_props)}种物质特性"
            
            # 提取深度学习模式
            if 'deep_learning_analysis' in deep_result:
                dl_analysis = deep_result['deep_learning_analysis']
                if 'pattern_analysis' in dl_analysis:
                    patterns = dl_analysis['pattern_analysis']
                    if 'correlations' in patterns:
                        integration['dl_patterns'] = f"识别{len(patterns['correlations'])}种参数相关性"
                
                if 'intelligent_recommendations' in dl_analysis:
                    recs = dl_analysis['intelligent_recommendations']
                    priority_count = len(recs.get('priority_recommendations', []))
                    integration['recommendation_synergy'] = f"生成{priority_count}个优先级推荐"
        except Exception as e:
            integration['status'] = 'error'
            integration['error'] = str(e)
        
        return integration
    
    def _integrate_deep_with_biology(self, deep_result: Dict, biology_result: Dict) -> Dict:
        """深度学习与生物框架集成"""
        integration = {
            'integration_type': 'deep_biology',
            'status': 'active'
        }
        
        try:
            # 尝试提取生物状态
            if 'biological_state' in biology_result:
                bio_state = biology_result['biological_state']
                if isinstance(bio_state, dict) and 'health' in bio_state:
                    integration['biological_health'] = bio_state['health']
            else:
                integration['biological_health'] = 'unknown'
        except:
            integration['biological_health'] = 'unknown'
        
        return integration
    
    def _integrate_deep_with_evolutionary(self, deep_result: Dict, evolutionary_result: Dict) -> Dict:
        """深度学习与进化算法集成"""
        integration = {
            'integration_type': 'deep_evolutionary',
            'status': 'active',
            'potential_enhancements': []
        }
        
        try:
            # 检查进化算法结果
            if 'best_solution' in evolutionary_result:
                best_solution = evolutionary_result['best_solution']
                
                # 深度学习可以提供个性化调整
                if 'deep_learning_analysis' in deep_result:
                    dl_analysis = deep_result['deep_learning_analysis']
                    
                    # 如果深度学习识别出特定模式，可以调整进化算法
                    if 'pattern_analysis' in dl_analysis:
                        patterns = dl_analysis['pattern_analysis']
                        if 'anomalies' in patterns and patterns['anomalies']:
                            integration['potential_enhancements'].append(
                                '深度学习识别异常模式，可指导进化算法避开低效区域'
                            )
                    
                    # 如果深度学习有推荐，可以用于指导进化算法的初始种群
                    if 'intelligent_recommendations' in dl_analysis:
                        integration['potential_enhancements'].append(
                            '深度学习推荐可作为进化算法的优质初始解'
                        )
        except Exception as e:
            integration['status'] = 'error'
            integration['error'] = str(e)
        
        return integration
    
    def _integrate_deep_with_risk(self, deep_result: Dict, risk_result: Dict) -> Dict:
        """深度学习与风险框架集成"""
        integration = {
            'integration_type': 'deep_risk',
            'status': 'active',
            'synergies': []
        }
        
        try:
            # 提取风险信息
            if 'extreme_risks' in risk_result:
                risks = risk_result['extreme_risks']
                integration['risk_count'] = len(risks)
            
            # 深度学习的异常检测可以补充风险分析
            if 'deep_learning_analysis' in deep_result:
                dl_analysis = deep_result['deep_learning_analysis']
                if 'pattern_analysis' in dl_analysis:
                    patterns = dl_analysis['pattern_analysis']
                    if 'anomalies' in patterns and patterns['anomalies']:
                        integration['synergies'].append(
                            f"深度学习发现{len(patterns['anomalies'])}个异常，可用于增强风险预警"
                        )
        except Exception as e:
            integration['status'] = 'error'
            integration['error'] = str(e)
        
        return integration
    

    

        
    def _update_state_memory_with_deep_learning(self, state: Dict, integrated_results: Dict, 
                                              framework_results: List[Dict]):
        """更新状态记忆（包含深度学习）"""
        # 原有的状态记忆更新（如果有的话）
        if hasattr(self, '_update_state_memory'):
            self._update_state_memory(state, integrated_results)
        
        # 特别记录深度学习洞察
        if len(framework_results) > 7:
            deep_learning_result = framework_results[7]
            
            if 'deep_learning_analysis' in deep_learning_result:
                dl_insight = {
                    'timestamp': datetime.now().isoformat(),
                    'analysis': deep_learning_result['deep_learning_analysis'],
                    'certainty': deep_learning_result.get('certainty', 0.5)
                }
                
                self.state_memory['deep_learning_insights'].append(dl_insight)
                
                # 限制存储数量
                if len(self.state_memory['deep_learning_insights']) > 50:
                    self.state_memory['deep_learning_insights'] = self.state_memory['deep_learning_insights'][-50:]
    
    def _adjust_integration_weights_based_on_performance(self, framework_results: List[Dict]):
        """基于性能动态调整集成权重 - 修复版"""
        # 计算每个框架的成功率
        success_rates = {}
        
        # 遍历所有框架结果
        for i, result in enumerate(framework_results):
            framework_name = self._get_framework_name_by_index(i)
            
            # 计算本次调用的成功状态
            if isinstance(result, dict):
                is_success = result.get('framework_status') in ['success', 'simplified', 'baseline_mode']
            else:
                is_success = False
            
            # 更新该框架的成功率统计
            if framework_name not in success_rates:
                success_rates[framework_name] = {'successes': 0, 'total': 0}
            
            success_rates[framework_name]['total'] += 1
            if is_success:
                success_rates[framework_name]['successes'] += 1
        
        # 根据成功率调整权重
        total_weight_change = 0
        
        for framework_name, stats in success_rates.items():
            if stats['total'] > 0 and framework_name in self.integration_weights:
                success_rate = stats['successes'] / stats['total']
                
                # 如果成功率很高，增加权重
                if success_rate > 0.8:
                    increase = self.integration_weights[framework_name] * 0.1
                    self.integration_weights[framework_name] += increase
                    total_weight_change += increase
                # 如果成功率很低，减少权重
                elif success_rate < 0.3:
                    decrease = self.integration_weights[framework_name] * 0.1
                    self.integration_weights[framework_name] = max(0.01, self.integration_weights[framework_name] - decrease)
                    total_weight_change -= decrease
        
        # 如果总权重发生变化，重新归一化
        if abs(total_weight_change) > 0.001:
            self._normalize_integration_weights()
    
    def _normalize_integration_weights(self):
        """归一化集成权重，确保总和为1"""
        total_weight = sum(self.integration_weights.values())
        if total_weight > 0:
            for name in self.integration_weights:
                self.integration_weights[name] /= total_weight
    
    def _calculate_framework_contributions(self, framework_results: List[Dict]) -> Dict:
        """计算各框架对最终结果的贡献度"""
        contributions = {}
        
        for i, result in enumerate(framework_results):
            if i < len(framework_results):
                framework_name = self._get_framework_name_by_index(i)
                
                # 简化贡献度计算：基于成功状态和确定性
                if isinstance(result, dict):
                    certainty = result.get('certainty', 0.5)
                    status = result.get('framework_status', 'unknown')
                    
                    # 贡献度分数
                    if status in ['success', 'simplified', 'baseline_mode']:
                        contribution_score = certainty * 0.8 + 0.2
                    else:
                        contribution_score = certainty * 0.3
                else:
                    contribution_score = 0.1
                
                contributions[framework_name] = round(contribution_score, 3)
        
        return contributions
    
    def _get_framework_performance_summary(self) -> Dict:
        """获取框架性能摘要"""
        summary = {}
        for name, performance in self.framework_performance.items():
            if performance['total_count'] > 0:
                success_rate = performance['success_count'] / performance['total_count']
                summary[name] = {
                    'success_rate': round(success_rate, 3),
                    'success_count': performance['success_count'],
                    'total_count': performance['total_count'],
                    'weight': self.integration_weights.get(name, 0.0)
                }
        return summary
    
    def _record_performance_metrics(self, framework_results: List[Dict], final_result: Dict):
        """记录性能指标"""
        # 记录执行时间（简化）
        execution_time = 0.0  # 实际实现中可以计算
        
        # 记录解决方案质量
        solution_quality = final_result.get('certainty', 0.5)
        
        # 记录多样性分数（基于不同框架的贡献）
        framework_contributions = final_result.get('inference', {}).get('framework_contributions', {})
        diversity_score = len(framework_contributions) / 9 if framework_contributions else 0
        
        self.performance_monitor['execution_times'].append(execution_time)
        self.performance_monitor['solution_quality'].append(solution_quality)
        self.performance_monitor['diversity_scores'].append(diversity_score)
        
        # 限制记录数量
        for key in self.performance_monitor:
            if len(self.performance_monitor[key]) > 100:
                self.performance_monitor[key] = self.performance_monitor[key][-100:]
    

    
    def get_deep_learning_summary(self) -> Dict:
        """获取深度学习框架摘要"""
        if hasattr(self, 'deep_learning_framework'):
            return {
                'performance': self.deep_learning_framework.get_performance_summary(),
                'integration_weight': self.integration_weights.get('deep_learning', 0),
                'historical_insights_count': len(self.state_memory.get('deep_learning_insights', [])),
                'communication_channel_count': len(self.inter_framework_communication.get('deep_learning', []))
            }
        return {'error': '深度学习框架未初始化'}
    
    async def run_scalability_test(self, num_products_list=None, num_users=3):
        """运行扩展性测试"""
        if num_products_list is None:
            num_products_list = [10, 30, 50, 100]
        
        test_results = []
        
        for num_products in num_products_list:
            logger.info(f"开始测试 {num_products} 个产品的扩展性...")
            
            # 生成测试数据
            test_products = self.test_data_generator.generate_extensive_product_catalog(num_products)
            test_users = self.test_data_generator.generate_diverse_user_profiles(num_users)
            
            user_results = []
            
            for user_profile in test_users:
                start_time = time.time()
                
                # 准备测试上下文
                test_context = {
                    'available_products': test_products,
                    'user_profile': user_profile,
                    'user_goals': user_profile['user_goals'],
                    'skin_analysis': {
                        'moisture_content': random.uniform(0.3, 0.8),
                        'elasticity': random.uniform(0.4, 0.9),
                        'collagen_level': random.uniform(0.3, 0.8)
                    }
                }
                
                # 运行进化算法推理
                try:
                    evolutionary_result = await self.evolutionary_algorithm_framework.infer(
                        skin_params={}, context=test_context
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # 分析结果质量
                    quality_metrics = self._analyze_solution_quality(
                        evolutionary_result, test_context
                    )
                    
                    user_result = {
                        'user_id': user_profile['user_id'],
                        'execution_time': execution_time,
                        'num_products': num_products,
                        'best_fitness': evolutionary_result.get('best_solution', {}).get('fitness', 0),
                        'solution_size': len(evolutionary_result.get('best_solution', {}).get('products', [])),
                        'quality_metrics': quality_metrics,
                        'success': evolutionary_result.get('framework_status') == 'success'
                    }
                    
                    user_results.append(user_result)
                    
                    # 记录性能数据
                    self.performance_monitor['execution_times'].append({
                        'num_products': num_products,
                        'time': execution_time,
                        'user_id': user_profile['user_id']
                    })
                    
                except Exception as e:
                    logger.error(f"用户 {user_profile['user_id']} 测试失败: {e}")
                    user_results.append({
                        'user_id': user_profile['user_id'],
                        'error': str(e),
                        'success': False
                    })
            
            # 计算统计信息
            successful_results = [r for r in user_results if r.get('success', False)]
            if successful_results:
                avg_time = sum(r['execution_time'] for r in successful_results) / len(successful_results)
                avg_fitness = sum(r['best_fitness'] for r in successful_results) / len(successful_results)
                success_rate = len(successful_results) / len(user_results)
            else:
                avg_time = 0
                avg_fitness = 0
                success_rate = 0
            
            test_result = {
                'num_products': num_products,
                'num_users_tested': len(user_results),
                'success_rate': success_rate,
                'avg_execution_time': avg_time,
                'avg_best_fitness': avg_fitness,
                'user_results': user_results,
                'timestamp': datetime.now().isoformat()
            }
            
            test_results.append(test_result)
            logger.info(f"完成 {num_products} 个产品测试: 成功率={success_rate:.2f}, 平均时间={avg_time:.2f}s")
        
        return test_results
    
    def _analyze_solution_quality(self, evolutionary_result: Dict, context: Dict) -> Dict:
        """分析解决方案质量"""
        best_solution = evolutionary_result.get('best_solution', {})
        products = best_solution.get('products', [])
        
        if not products:
            return {
                'category_coverage': 0,
                'benefit_coverage': 0,
                'price_range_suitability': 0,
                'diversity_score': 0
            }
        
        # 1. 类别覆盖度
        categories = {p.get('category', 'unknown') for p in products}
        essential_categories = {'cleanser', 'moisturizer', 'sunscreen'}
        covered_essentials = essential_categories.intersection(categories)
        category_coverage = len(covered_essentials) / len(essential_categories)
        
        # 2. 功效覆盖度
        user_goals = context.get('user_goals', [])
        goal_types = {goal.get('type') for goal in user_goals}
        
        all_benefits = set()
        for product in products:
            all_benefits.update(product.get('benefits', []))
        
        matched_goals = goal_types.intersection(all_benefits)
        benefit_coverage = len(matched_goals) / len(goal_types) if goal_types else 0
        
        # 3. 价格适宜度
        user_budget = context.get('user_profile', {}).get('budget', 'medium')
        budget_ranges = {
            'low': (0, 100),
            'medium': (50, 300),
            'high': (200, 1000)
        }
        
        min_budget, max_budget = budget_ranges.get(user_budget, (50, 300))
        total_cost = sum(p.get('price', 0) for p in products)
        avg_cost = total_cost / len(products) if products else 0
        
        if avg_cost < min_budget * 0.7:
            price_suitability = 0.7
        elif avg_cost <= max_budget:
            price_suitability = 1.0
        else:
            overshoot = (avg_cost - max_budget) / max_budget
            price_suitability = max(0.1, 1.0 - overshoot)
        
        # 4. 多样性分数
        diversity_score = 0.0
        if len(products) >= 2:
            # 计算产品功效的多样性
            benefit_sets = [set(p.get('benefits', [])) for p in products]
            jaccard_scores = []
            
            for i in range(len(benefit_sets)):
                for j in range(i+1, len(benefit_sets)):
                    intersection = benefit_sets[i].intersection(benefit_sets[j])
                    union = benefit_sets[i].union(benefit_sets[j])
                    if union:
                        jaccard = len(intersection) / len(union)
                        jaccard_scores.append(1 - jaccard)  # 不相似度
        
            if jaccard_scores:
                diversity_score = sum(jaccard_scores) / len(jaccard_scores)
        
        return {
            'category_coverage': round(category_coverage, 3),
            'benefit_coverage': round(benefit_coverage, 3),
            'price_range_suitability': round(price_suitability, 3),
            'diversity_score': round(diversity_score, 3),
            'num_categories': len(categories),
            'num_unique_benefits': len(all_benefits)
        }
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        if not self.performance_monitor['execution_times']:
            return {'status': 'no_data'}
        
        execution_times = self.performance_monitor['execution_times']
        
        # 按产品数量分组
        times_by_product_count = {}
        for record in execution_times:
            count = record['num_products']
            if count not in times_by_product_count:
                times_by_product_count[count] = []
            times_by_product_count[count].append(record['time'])
        
        # 计算统计信息
        scalability_analysis = {}
        for count, times in times_by_product_count.items():
            if times:
                scalability_analysis[count] = {
                    'avg_time': round(sum(times) / len(times), 3),
                    'min_time': round(min(times), 3),
                    'max_time': round(max(times), 3),
                    'num_tests': len(times)
                }
        
        return {
            'scalability_analysis': scalability_analysis,
            'total_tests': len(execution_times),
            'recent_tests': execution_times[-10:] if len(execution_times) > 10 else execution_times
        }

    def _prepare_context_for_frameworks(self, state, context):
        """为所有框架准备统一的上下文 - 修复版"""
        if context is None:
            context = {}
        
        # 确保不覆盖已存在的产品数据
        if 'available_products' not in context:
            context['available_products'] = []
            logger.warning("上下文缺少 available_products，已添加空列表")
        else:
            # 确保产品数据是列表格式
            if not isinstance(context['available_products'], list):
                logger.warning(f"available_products 格式错误，期望列表，实际为 {type(context['available_products'])}")
                context['available_products'] = []
            elif context['available_products']:
                logger.info(f"上下文已有 {len(context['available_products'])} 个产品")
        
        # 确保用户目标存在
        if 'user_goals' not in context:
            context['user_goals'] = [
                {'type': 'moisturizing', 'priority': 1},
                {'type': 'anti-aging', 'priority': 2},
                {'type': 'protection', 'priority': 3}
            ]
        
        # 确保用户配置存在
        if 'user_profile' not in context:
            context['user_profile'] = {
                'skin_type': 'normal',
                'age': 30,
                'budget': 'medium',
                'sensitivity': 'medium'
            }
        
        # 确保皮肤分析数据存在
        if 'skin_analysis' not in context:
            # 尝试从state中提取
            if state and 'skin_analysis' in state:
                context['skin_analysis'] = state['skin_analysis']
            else:
                context['skin_analysis'] = {
                    'parameters': state.get('skin_parameters', {}),
                    'moisture_content': state.get('moisture_content', 0.5),
                    'elasticity': state.get('elasticity', 0.5),
                    'collagen_level': state.get('collagen_level', 0.5)
                }
        
        # 添加状态数据到上下文
        context['state'] = state.copy() if state else {}
        
        # 添加时间戳用于调试
        context['timestamp'] = time.time()
        
        return context

    def _call_framework_safely(self, framework, skin_params, context):
        """安全调用框架，处理不同的参数需求"""
        try:
            # 检查框架的 infer 方法签名
            import inspect
            
            # 获取框架的 infer 方法
            if hasattr(framework, 'infer'):
                infer_method = framework.infer
                
                # 检查方法参数数量
                sig = inspect.signature(infer_method)
                params = list(sig.parameters.keys())
                
                # 获取参数数量，排除 self
                num_params = len(params) - 1 if 'self' in params else len(params)
                
                if num_params >= 2:
                    # 如果框架接受至少两个参数（skin_params 和 context）
                    logger.debug(f"调用框架 {framework.__class__.__name__} 使用两个参数")
                    return infer_method(skin_params, context)
                elif num_params == 1:
                    # 如果框架只接受一个参数
                    logger.debug(f"调用框架 {framework.__class__.__name__} 使用一个参数")
                    return infer_method(skin_params)
                else:
                    # 无参数的情况（不太可能）
                    logger.warning(f"框架 {framework.__class__.__name__} 的 infer 方法不接受参数")
                    return infer_method()
                    
            else:
                logger.error(f"框架 {framework.__class__.__name__} 没有 infer 方法")
                return {}
                
        except Exception as e:
            logger.error(f"框架 {framework.__class__.__name__} 调用失败: {e}")
            # 返回默认结果
            return {
                'framework_status': 'error',
                'error': str(e),
                'certainty': 0.1
            }

    def _share_framework_results(self, framework_results):
        """实现框架间结果共享（修复版）"""
        framework_names = list(self.inter_framework_communication.keys())
        
        for i, framework_name in enumerate(framework_names):
            if i < len(framework_results) and framework_results[i]:
                # 存储框架结果
                self.inter_framework_communication[framework_name].append(framework_results[i])
                
                # 限制存储数量
                if len(self.inter_framework_communication[framework_name]) > 10:
                    self.inter_framework_communication[framework_name] = self.inter_framework_communication[framework_name][-10:]
        
        # 实现框架间数据交换
        self._exchange_framework_data()
    
    def _exchange_framework_data(self):
        """实现框架间数据交换"""
        # 物理框架结果可供生物和几何框架使用
        if self.inter_framework_communication['physics']:
            physics_data = self.inter_framework_communication['physics'][-1]
            
            # 物理数据传递给生物框架（如物质特性）
            if 'material_properties' in physics_data:
                self.inter_framework_communication['biology'].append({
                    'type': 'physics_to_biology',
                    'data': physics_data['material_properties']
                })
            
            # 物理数据传递给几何框架（如空间特性）
            if 'physical_constraints' in physics_data:
                self.inter_framework_communication['geometric_dynamics'].append({
                    'type': 'physics_to_geometric',
                    'data': physics_data['physical_constraints']
                })
        
        # 生物框架结果可供物理和风险框架使用
        if self.inter_framework_communication['biology']:
            biology_data = self.inter_framework_communication['biology'][-1]
            
            # 生物数据传递给风险框架
            if 'biological_state' in biology_data:
                self.inter_framework_communication['fat_tail_risk'].append({
                    'type': 'biology_to_risk',
                    'data': biology_data['biological_state']
                })



    
    def _update_performance_metrics(self, inference_results, actual_outcomes):
        """更新框架性能指标"""
        # 这里需要实现具体的性能评估逻辑
        pass

    def _detect_anomalies(self):
        """检测状态记忆中的异常模式 - 基础实现"""
        try:
            historical_states = self.state_memory.get('historical_states', [])
            if len(historical_states) < 2:
                return  # 数据不足，不进行检测
            
            # 获取最近几个状态
            recent_states = historical_states[-5:] if len(historical_states) >= 5 else historical_states
            
            anomalies = []
            
            # 1. 检查推理确定性是否骤降
            certainties = [state.get('key_metrics', {}).get('certainty', 0.5) for state in recent_states]
            if certainties and len(certainties) >= 3:
                avg_certainty = sum(certainties) / len(certainties)
                latest_certainty = certainties[-1]
                
                if latest_certainty < avg_certainty * 0.7:  # 下降超过30%
                    anomalies.append({
                        'type': 'certainty_drop',
                        'severity': 'medium',
                        'message': f'推理确定性从 {avg_certainty:.2f} 下降至 {latest_certainty:.2f}',
                        'timestamp': time.time()
                    })
            
            # 2. 检查皮肤参数是否急剧变化
            skin_params_history = []
            for state in recent_states:
                if 'skin_params' in state.get('state', {}):
                    skin_params_history.append(state['state']['skin_params'])
            
            if skin_params_history and len(skin_params_history) >= 2:
                # 简单检查最近两个状态的差异
                current_params = skin_params_history[-1]
                previous_params = skin_params_history[-2]
                
                # 计算关键参数的变化
                for param in ['hydration', 'elasticity', 'barrier']:
                    if param in current_params and param in previous_params:
                        current_val = current_params[param].get('score', 50)
                        prev_val = previous_params[param].get('score', 50)
                        change_percent = abs(current_val - prev_val) / max(prev_val, 1) * 100
                        
                        if change_percent > 20:  # 变化超过20%
                            anomalies.append({
                                'type': 'skin_param_rapid_change',
                                'parameter': param,
                                'change_percent': change_percent,
                                'severity': 'low' if change_percent < 40 else 'medium',
                                'message': f'{param} 参数变化 {change_percent:.1f}%',
                                'timestamp': time.time()
                            })
            
            # 如果有异常，记录到状态记忆中
            if anomalies:
                if 'anomalies' not in self.state_memory:
                    self.state_memory['anomalies'] = []
                
                self.state_memory['anomalies'].extend(anomalies)
                
                # 限制异常记录数量
                if len(self.state_memory['anomalies']) > 50:
                    self.state_memory['anomalies'] = self.state_memory['anomalies'][-50:]
                
                # 记录到调试日志
                if anomalies:
                    logger.debug(f"检测到 {len(anomalies)} 个异常")
                    
        except Exception as e:
            logger.error(f"检测异常时出错: {e}")
            # 忽略错误，不中断正常流程


    def _update_state_memory(self, current_state, inference_results):
        """更新状态记忆系统"""
        # 记录当前状态
        memory_entry = {
            'timestamp': time.time(),
            'state': current_state.copy(),
            'inference': inference_results.copy(),
            'key_metrics': self._extract_key_metrics(inference_results)
        }
        
        self.state_memory['historical_states'].append(memory_entry)
        
        # 保持最近100个状态记录
        if len(self.state_memory['historical_states']) > 100:
            self.state_memory['historical_states'].pop(0)
        
        # 更新趋势分析
        self._update_trend_analysis()
        
        # 检测异常模式
        self._detect_anomalies()
    
    def _extract_key_metrics(self, inference_results):
        """提取关键指标用于趋势分析"""
        metrics = {}
        
        # 从物理框架提取弹性指标
        if 'physics' in inference_results and 'elasticity' in inference_results['physics']:
            metrics['elasticity'] = inference_results['physics']['elasticity']
        
        # 从生物框架提取胶原蛋白水平
        if 'biology' in inference_results and 'collagen_level' in inference_results['biology']:
            metrics['collagen_level'] = inference_results['biology']['collagen_level']
        
        # 从心理框架提取感知效果
        if 'mental' in inference_results and 'perceived_effectiveness' in inference_results['mental']:
            metrics['perceived_effectiveness'] = inference_results['mental']['perceived_effectiveness']
        
        return metrics

    def _share_framework_results(self, framework_results):
        """实现框架间结果共享"""
        # 物理框架结果共享给生物框架
        if len(framework_results) > 0 and 'material_properties' in framework_results[0]:
            self.inter_framework_communication['physics_to_biology'].append(
                framework_results[0]['material_properties']
            )
        
        # 生物框架结果共享给几何框架
        if len(framework_results) > 1 and 'biological_state' in framework_results[1]:
            self.inter_framework_communication['biology_to_geometric'].append(
                framework_results[1]['biological_state']
            )
        
        # 心理框架结果共享给决策框架
        if len(framework_results) > 3 and 'perceived_effectiveness' in framework_results[3]:
            self.inter_framework_communication['mental_to_decision'].append(
                framework_results[3]['perceived_effectiveness']
            )
        
        # 风险框架结果共享给进化框架
        if len(framework_results) > 6 and 'extreme_risks' in framework_results[6]:
            self.inter_framework_communication['risk_to_evolutionary'].append(
                framework_results[6]['extreme_risks']
            )
    

    def _validate_input(self, state, context):
        """输入验证"""
        if not isinstance(state, dict):
            raise ValueError("state参数必须是字典类型")
        
        if context is not None and not isinstance(context, dict):
            raise ValueError("context参数必须是字典类型或None")
        
        # 检查必要的状态字段
        required_fields = ['moisture_content', 'elasticity', 'collagen_level']
        for field in required_fields:
            if field not in state:
                logger.warning(f"状态中缺少必要字段: {field}")

    def _get_default_digital_twin_state(self, state):
        """获取默认的数字孪生状态"""
        return {
            'simulation_status': 'fallback',
            'skin_parameters': state.copy(),
            'timestamp': time.time(),
            'certainty': 0.1
        }



    def _record_framework_success(self, framework_index):
        """记录框架成功运行"""
        framework_names = list(self.integration_weights.keys())
        if framework_index < len(framework_names):
            name = framework_names[framework_index]
            if name not in self.framework_performance:
                self.framework_performance[name] = {'success_count': 0, 'total_count': 0}
            self.framework_performance[name]['success_count'] += 1
            self.framework_performance[name]['total_count'] += 1

    def _record_framework_failure(self, framework_index, error_type):
        """记录框架运行失败"""
        framework_names = list(self.integration_weights.keys())
        if framework_index < len(framework_names):
            name = framework_names[framework_index]
            if name not in self.framework_performance:
                self.framework_performance[name] = {'success_count': 0, 'total_count': 0}
            self.framework_performance[name]['total_count'] += 1



    def _get_framework_performance_summary(self):
        """获取框架性能摘要"""
        summary = {}
        for name, performance in self.framework_performance.items():
            if performance['total_count'] > 0:
                success_rate = performance['success_count'] / performance['total_count']
                summary[name] = {
                    'success_rate': success_rate,
                    'success_count': performance['success_count'],
                    'total_count': performance['total_count']
                }
        return summary



    def _analyze_elasticity_collagen_correlation(self, elasticity, collagen_level):
        """分析弹性-胶原蛋白相关性"""
        correlation_score = min(1.0, elasticity * collagen_level * 2)
        return {
            'type': 'elasticity_collagen_correlation',
            'score': correlation_score,
            'recommendation': '考虑使用促进胶原蛋白生成的产品' if correlation_score < 0.6 else '保持当前的护理策略',
            'priority': 5
        }

    def _analyze_topology_moisture_relationship(self, topology, moisture):
        """分析拓扑-水分关系"""
        if topology == 'smooth' and moisture > 0.7:
            return {
                'type': 'topology_moisture_relationship',
                'status': 'optimal',
                'recommendation': '保持当前的保湿策略',
                'priority': 5
            }
        else:
            return {
                'type': 'topology_moisture_relationship',
                'status': 'suboptimal',
                'recommendation': '调整保湿策略以改善皮肤表面状态',
                'priority': 4
            }

    def _analyze_user_preference_alignment(self, effectiveness, strategy):
        """分析用户偏好-策略对齐"""
        alignment_score = effectiveness * 0.8 + 0.2  # 基础对齐分数
        return {
            'type': 'user_preference_strategy_alignment',
            'score': alignment_score,
            'recommendation': '微调当前策略以提高效果' if alignment_score < 0.7 else '当前策略与用户偏好高度匹配',
            'priority': 5
        }

    def _analyze_risk_adjusted_optimization(self, solution, risks):
        """分析风险调整优化"""
        risk_level = len(risks) * 0.1
        risk_status = '低风险' if risk_level < 0.3 else '中等风险' if risk_level < 0.6 else '高风险'
        return {
            'type': 'risk_adjusted_optimization',
            'risk_level': risk_level,
            'status': risk_status,
            'recommendation': '在采用优化方案时注意监控可能的风险',
            'priority': 4 if risk_level > 0.3 else 5
        }

    def _identify_key_recommendations(self, integrated_results):
        """识别关键推荐 - 修复版本"""
        recommendations = []
        
        # 安全访问框架结果
        def safe_get(framework_key, field_key):
            """安全获取框架结果中的字段"""
            if framework_key in integrated_results and isinstance(integrated_results[framework_key], dict):
                return integrated_results[framework_key].get(field_key)
            return None
        
        # 从生物学框架提取推荐
        biological_recommendations = safe_get('biology', 'biological_recommendations')
        if biological_recommendations:
            for rec in biological_recommendations:
                recommendations.append({
                    'source': 'biology',
                    'type': 'biological',
                    'content': rec,
                    'priority': 2
                })
        
        # 从物理学框架提取推荐
        physics_recommendations = safe_get('physics', 'physics_recommendations')
        if physics_recommendations:
            for rec in physics_recommendations:
                recommendations.append({
                    'source': 'physics',
                    'type': 'physics',
                    'content': rec,
                    'priority': 2
                })
        
        # 从决策理论框架提取推荐
        optimal_strategy = safe_get('decision_theory', 'optimal_strategy')
        if optimal_strategy:
            recommendations.append({
                'source': 'decision_theory',
                'type': 'strategy',
                'content': optimal_strategy,
                'priority': 3
            })
        
        # 从肥尾风险框架提取风险警告
        if 'key_risks' in integrated_results:
            for risk in integrated_results['key_risks']:
                recommendations.append({
                    'source': 'fat_tail_risk',
                    'type': 'risk_warning',
                    'content': risk,
                    'priority': 4
                })
        
        # 从跨框架分析提取综合推荐
        if 'cross_framework_analysis' in integrated_results:
            for key, value in integrated_results['cross_framework_analysis'].items():
                if isinstance(value, dict) and 'recommendation' in value:
                    recommendations.append({
                        'source': 'cross_framework',
                        'type': key,
                        'content': value['recommendation'],
                        'priority': 5
                    })
        
        # 按优先级排序
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations[:10]  # 返回前10个关键推荐


    def _optimize_recommendations_for_actionable_factors(self, recommendations, actionable_factors):
        """根据可改变因素优化推荐"""
        optimized = []
        for rec in recommendations:
            # 根据可改变因素调整推荐优先级
            adjusted_priority = rec['priority']
            if 'user_preference' in actionable_factors:
                adjusted_priority += 1  # 用户偏好相关的推荐优先级提高
            
            optimized_rec = rec.copy()
            optimized_rec['priority'] = min(5, adjusted_priority)  # 优先级上限为5
            optimized.append(optimized_rec)
        
        return optimized

    async def _call_framework_with_timeout(self, framework, skin_params, context, timeout=30):
        """带超时的框架调用，防止单个框架阻塞整个系统"""
        try:
            # 创建异步任务
            task = asyncio.create_task(
                self._call_framework_safely(framework, skin_params, context)
            )
            
            # 设置超时
            done, pending = await asyncio.wait([task], timeout=timeout)
            
            if task in done:
                return task.result()
            else:
                # 取消任务
                task.cancel()
                logger.warning(f"框架 {framework.__class__.__name__} 调用超时")
                
                # 返回超时结果
                return {
                    'framework_status': 'timeout',
                    'certainty': 0.1,
                    'error': f'框架调用超时（{timeout}秒）'
                }
                
        except asyncio.CancelledError:
            logger.warning(f"框架 {framework.__class__.__name__} 任务被取消")
            return {
                'framework_status': 'cancelled',
                'certainty': 0.1
            }
        except Exception as e:
            logger.error(f"框架 {framework.__class__.__name__} 调用异常: {e}")
            return {
                'framework_status': 'error',
                'certainty': 0.1,
                'error': str(e)
            }


    

    
    def _get_communication_summary(self):
        """获取框架间通信摘要 - 快速修复"""
        try:
            # 确保是字典类型
            if not isinstance(self.inter_framework_communication, dict):
                self.inter_framework_communication = {}
            
            communications = self.inter_framework_communication
            
            # 安全地处理
            if communications and isinstance(communications, dict):
                # 获取所有消息
                all_messages = []
                for messages in communications.values():
                    if isinstance(messages, list):
                        all_messages.extend(messages)
                
                return {
                    'total_messages': len(all_messages),
                    'latest_message': all_messages[-1] if all_messages else None,
                    'channels': list(communications.keys()),
                    'channel_count': len(communications)
                }
            else:
                return {
                    'total_messages': 0,
                    'latest_message': None,
                    'channels': [],
                    'channel_count': 0
                }
                
        except Exception as e:
            logger.error(f"获取通信摘要失败: {e}")
            return {
                'total_messages': 0,
                'latest_message': None,
                'channels': [],
                'channel_count': 0
            }
    
    def _extract_skin_parameters(self, digital_twin_state):
        """从数字孪生状态提取皮肤参数（增强版）"""
        if 'skin_parameters' in digital_twin_state:
            params = digital_twin_state['skin_parameters'].copy()
        else:
            params = digital_twin_state.copy()
        
        # 确保参数标准化
        standard_params = {
            'moisture_content': params.get('moisture_content', 0.5),
            'elasticity': params.get('elasticity', 0.5),
            'collagen_level': params.get('collagen_level', 0.5),
            'barrier_integrity': params.get('barrier_integrity', 0.5),
            'sebum_production': params.get('sebum_production', 0.5),
            'redness': params.get('redness', 0.5),
            'wrinkles': params.get('wrinkles', 0.5),
            'hydration': params.get('hydration', 0.5)
        }
        
        return standard_params
    
    def cross_framework_inference(self, framework_results, digital_twin_state):
        """跨框架推理（修复增强版）"""
        cross_results = {}
        
        # 安全获取框架结果
        physics_result = framework_results[0] if len(framework_results) > 0 else {}
        biology_result = framework_results[1] if len(framework_results) > 1 else {}
        geometric_result = framework_results[2] if len(framework_results) > 2 else {}
        mental_result = framework_results[3] if len(framework_results) > 3 else {}
        decision_result = framework_results[4] if len(framework_results) > 4 else {}
        evolutionary_result = framework_results[5] if len(framework_results) > 5 else {}
        risk_result = framework_results[6] if len(framework_results) > 6 else {}
        
        # 1. 物理-生物交叉分析：物质特性与生物状态的关系
        if physics_result and biology_result:
            material_props = physics_result.get('material_properties', {})
            biological_state = biology_result.get('biological_state', {})
            
            cross_results['physics_biology_integration'] = self._analyze_physics_biology_integration(
                material_props, biological_state
            )
        
        # 2. 生物-几何动力学交叉分析：生物状态与空间结构的关系
        if biology_result and geometric_result:
            cross_results['biology_geometric_integration'] = self._analyze_biology_geometric_integration(
                biology_result, geometric_result
            )
        
        # 3. 心理-决策交叉分析：心理状态与决策偏好的关系
        if mental_result and decision_result:
            cross_results['mental_decision_integration'] = self._analyze_mental_decision_integration(
                mental_result, decision_result
            )
        
        # 4. 进化算法-风险交叉分析：优化方案与风险评估的关系
        if evolutionary_result and risk_result:
            cross_results['evolutionary_risk_integration'] = self._analyze_evolutionary_risk_integration(
                evolutionary_result, risk_result
            )
        
        # 5. 多框架综合集成分析
        cross_results['comprehensive_integration'] = self._analyze_comprehensive_integration(
            framework_results, digital_twin_state
        )
        
        return cross_results
    
    def _analyze_physics_biology_integration(self, material_props, biological_state):
        """分析物理-生物集成关系"""
        analysis = {
            'integration_score': 0.0,
            'insights': [],
            'recommendations': []
        }
        
        # 分析物质特性与生物状态的匹配度
        if 'elasticity' in material_props and 'collagen_level' in biological_state:
            elasticity = material_props.get('elasticity', {}).get('score', 50)
            collagen = biological_state.get('collagen_level', {}).get('score', 50)
            
            # 计算匹配度
            match_score = 1.0 - abs(elasticity - collagen) / 100
            analysis['integration_score'] = match_score
            
            if match_score < 0.7:
                analysis['insights'].append('皮肤弹性与胶原蛋白水平不匹配')
                analysis['recommendations'].append('考虑使用促进胶原蛋白生成的产品')
            else:
                analysis['insights'].append('皮肤弹性与胶原蛋白水平匹配良好')
                analysis['recommendations'].append('保持当前的护理策略')
        
        return analysis
    
    def _analyze_biology_geometric_integration(self, biology_result, geometric_result):
        """分析生物-几何动力学集成关系"""
        analysis = {
            'integration_score': 0.0,
            'insights': [],
            'recommendations': []
        }
        
        # 分析生物状态与空间结构的匹配度
        biological_state = biology_result.get('biological_state', {})
        spatial_structure = geometric_result.get('spatial_structure', {})
        
        # 简化的匹配度计算
        analysis['integration_score'] = 0.75
        
        if spatial_structure.get('surface_topology', {}).get('roughness') == 'high':
            analysis['insights'].append('粗糙皮肤表面可能需要更强的屏障修复')
            analysis['recommendations'].append('使用含有神经酰胺的屏障修复产品')
        
        return analysis
    
    def _analyze_mental_decision_integration(self, mental_result, decision_result):
        """分析心理-决策集成关系"""
        analysis = {
            'alignment_score': 0.0,
            'insights': [],
            'recommendations': []
        }
        
        # 分析心理状态与决策策略的对齐度
        perceived_effectiveness = mental_result.get('perceived_effectiveness', 0.5)
        optimal_strategy = decision_result.get('optimal_strategy', {})
        
        # 计算对齐度
        alignment = perceived_effectiveness * 0.8 + 0.2  # 基础对齐分数
        analysis['alignment_score'] = alignment
        
        if alignment < 0.6:
            analysis['insights'].append('用户心理预期与最佳策略存在差距')
            analysis['recommendations'].append('调整策略以更好地满足用户期望')
        else:
            analysis['insights'].append('用户心理预期与最佳策略基本一致')
            analysis['recommendations'].append('继续执行当前策略')
        
        return analysis
    
    def _analyze_evolutionary_risk_integration(self, evolutionary_result, risk_result):
        """分析进化算法-风险集成关系"""
        analysis = {
            'risk_adjusted_score': 0.0,
            'insights': [],
            'recommendations': []
        }
        
        # 分析优化方案的风险调整
        optimized_solution = evolutionary_result.get('best_solution', {})
        extreme_risks = risk_result.get('extreme_risks', [])
        
        # 计算风险调整分数
        risk_factor = len(extreme_risks) * 0.1
        base_score = evolutionary_result.get('best_fitness', 0.5)
        
        analysis['risk_adjusted_score'] = base_score * (1 - risk_factor)
        
        if len(extreme_risks) > 0:
            analysis['insights'].append('优化方案需要考虑风险因素')
            analysis['recommendations'].append('在实施方案时注意风险监控')
        else:
            analysis['insights'].append('优化方案风险较低')
            analysis['recommendations'].append('可以安全地实施方案')
        
        return analysis
    
    def _analyze_comprehensive_integration(self, framework_results, digital_twin_state):
        """综合分析所有框架结果"""
        analysis = {
            'overall_insight': '',
            'key_findings': [],
            'priority_recommendations': []
        }
        
        # 收集各框架的关键发现
        key_findings = []
        
        for i, result in enumerate(framework_results):
            if result:
                if 'insights' in result:
                    key_findings.extend(result['insights'])
                elif 'key_recommendations' in result:
                    key_findings.append(f"框架{i}推荐: {result['key_recommendations'][:2]}")
        
        # 生成综合洞察
        if len(key_findings) >= 4:
            analysis['overall_insight'] = '多框架分析显示皮肤状态良好，建议持续当前护理'
        else:
            analysis['overall_insight'] = '多框架分析显示需要进一步优化护理策略'
        
        analysis['key_findings'] = key_findings[:5]  # 最多5个关键发现
        
        # 生成优先级推荐
        priority_recommendations = []
        
        # 从各框架提取推荐并按优先级排序
        for result in framework_results:
            if result and 'recommendations' in result:
                for rec in result['recommendations']:
                    priority = rec.get('priority', 3)
                    priority_recommendations.append({
                        'recommendation': rec.get('content', '未知'),
                        'priority': priority,
                        'source': result.get('framework_name', 'unknown')
                    })
        
        # 按优先级排序
        priority_recommendations.sort(key=lambda x: x['priority'], reverse=True)
        analysis['priority_recommendations'] = priority_recommendations[:3]  # 前3个优先级推荐
        
        return analysis
    
    def _integrate_inferences(self, framework_results, cross_framework_analysis, digital_twin_state):
        """整合各框架推理结果 - 修复版"""
        try:
            # 确保 framework_results 是列表
            if not isinstance(framework_results, list):
                logger.warning(f"framework_results 不是列表: {type(framework_results)}")
                framework_results = []
            
            # 框架名称映射
            framework_names = [
                'physics', 'biology', 'geometric_dynamics', 
                'mental', 'decision_theory', 
                'evolutionary_algorithm', 'fat_tail_risk'
            ]
            
            # 组织框架结果
            framework_results_dict = {}
            for i, result in enumerate(framework_results):
                if i < len(framework_names):
                    framework_name = framework_names[i]
                    
                    # 确保结果是字典
                    if isinstance(result, dict):
                        framework_results_dict[framework_name] = result
                    else:
                        logger.warning(f"框架 {framework_name} 的结果不是字典: {type(result)}")
                        framework_results_dict[framework_name] = {
                            'framework_status': 'error',
                            'error': f'结果类型错误: {type(result).__name__}',
                            'certainty': 0.1
                        }
            
            # 生成关键推荐
            key_recommendations = self._generate_key_recommendations(framework_results_dict, cross_framework_analysis)
            
            # 生成行动计划
            action_plan = self._generate_action_plan(key_recommendations, digital_twin_state)
            
            integrated = {
                'framework_results': framework_results_dict,
                'cross_framework_analysis': cross_framework_analysis,
                'digital_twin_state': digital_twin_state,
                'key_recommendations': key_recommendations,
                'action_plan': action_plan,
                'integration_status': 'success',
                'timestamp': time.time()
            }
            
            logger.info(f"整合完成，包含 {len(framework_results_dict)} 个框架结果")
            return integrated
            
        except Exception as e:
            logger.error(f"整合推理结果失败: {e}", exc_info=True)
            # 返回基本结构
            return {
                'framework_results': {},
                'cross_framework_analysis': cross_framework_analysis,
                'digital_twin_state': digital_twin_state,
                'key_recommendations': [],
                'action_plan': {},
                'integration_status': 'error',
                'error': str(e)
            }
    
    def _generate_key_recommendations(self, framework_results, cross_framework_results):
        """生成关键推荐"""
        recommendations = []
        
        # 从各框架提取推荐
        for i, result in enumerate(framework_results):
            if result and 'recommendations' in result:
                for rec in result['recommendations']:
                    if isinstance(rec, dict):
                        recommendations.append({
                            'content': rec.get('content', '未知'),
                            'priority': rec.get('priority', 3),
                            'source': f'framework_{i}'
                        })
        
        # 从跨框架分析提取推荐
        for key, analysis in cross_framework_results.items():
            if 'recommendations' in analysis:
                for rec in analysis['recommendations']:
                    recommendations.append({
                        'content': rec,
                        'priority': 4,  # 跨框架推荐优先级较高
                        'source': f'cross_framework_{key}'
                    })
        
        # 去重并按优先级排序
        unique_recommendations = []
        seen_contents = set()
        
        for rec in recommendations:
            if rec['content'] not in seen_contents:
                seen_contents.add(rec['content'])
                unique_recommendations.append(rec)
        
        # 按优先级排序
        unique_recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return unique_recommendations[:5]  # 返回前5个关键推荐
    
    def _generate_action_plan(self, key_recommendations, digital_twin_state):
        """生成行动计划"""
        action_plan = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_strategy': ''
        }
        
        # 根据推荐生成行动计划
        high_priority_recs = [rec for rec in key_recommendations if rec['priority'] >= 4]
        medium_priority_recs = [rec for rec in key_recommendations if 2 <= rec['priority'] < 4]
        
        # 立即行动（高优先级）
        for rec in high_priority_recs[:2]:  # 最多2个立即行动
            action_plan['immediate_actions'].append({
                'action': rec['content'],
                'priority': 'high',
                'expected_impact': 'significant'
            })
        
        # 短期目标（中优先级）
        for rec in medium_priority_recs[:3]:  # 最多3个短期目标
            action_plan['short_term_goals'].append({
                'goal': rec['content'],
                'timeline': '1-2 weeks',
                'success_indicators': ['improved_skin_parameters', 'increased_comfort']
            })
        
        # 长期策略
        skin_status = digital_twin_state.get('overall_status', 'normal')
        if skin_status == 'healthy':
            action_plan['long_term_strategy'] = '维持当前护理策略，定期监测皮肤状态'
        else:
            action_plan['long_term_strategy'] = '实施综合护理计划，逐步改善皮肤健康状况'
        
        return action_plan
    
    def _calculate_certainty(self, integrated_results: Dict) -> float:
        """计算总体确定性 - 简化版"""
        try:
            # 获取所有框架的贡献度
            framework_contributions = integrated_results.get('framework_contributions', {})
            
            if not framework_contributions:
                return 0.5
            
            # 计算平均贡献度
            avg_contribution = sum(framework_contributions.values()) / len(framework_contributions)
            
            # 基于贡献度计算确定性
            certainty = 0.3 + (avg_contribution * 0.5)
            
            # 确保在合理范围内
            return max(0.1, min(0.95, certainty))
        except Exception as e:
            logger.error(f"计算确定性失败: {e}")
            return 0.5
    

    
    def _extract_inference_summary(self, inference_results):
        """提取推理摘要"""
        summary = {
            'framework_count': len(inference_results.get('framework_results', {})),
            'cross_analysis_count': len(inference_results.get('cross_framework_analysis', {})),
            'recommendation_count': len(inference_results.get('key_recommendations', [])),
            'top_recommendation': None
        }
        
        key_recommendations = inference_results.get('key_recommendations', [])
        if key_recommendations:
            summary['top_recommendation'] = key_recommendations[0].get('content', '未知')
        
        return summary
    
    def _update_trend_analysis(self):
        """更新趋势分析"""
        historical_states = self.state_memory['historical_states']
        
        if len(historical_states) < 2:
            return
        
        # 分析关键指标趋势
        trends = {}
        
        # 分析确定性趋势
        certainties = [state['certainty'] for state in historical_states]
        if len(certainties) >= 3:
            # 计算移动平均
            recent_avg = sum(certainties[-3:]) / 3
            overall_avg = sum(certainties) / len(certainties)
            
            trends['certainty'] = {
                'recent_average': recent_avg,
                'overall_average': overall_avg,
                'trend': 'improving' if recent_avg > overall_avg else 'stable' if recent_avg == overall_avg else 'declining'
            }
        
        # 分析推荐类型趋势
        recommendation_types = {}
        for state in historical_states:
            for rec in state.get('key_recommendations', []):
                rec_type = rec.get('source', 'unknown')
                recommendation_types[rec_type] = recommendation_types.get(rec_type, 0) + 1
        
        if recommendation_types:
            trends['recommendation_distribution'] = recommendation_types
        
        self.state_memory['trend_analysis'] = trends
    
    def _adjust_integration_weights(self, inference_results, actual_outcomes=None):
        """动态调整集成权重 - 合并确定性和性能调整（优化版）"""
        
        # 1. 基于框架结果的确定性调整权重
        certainty_weights = {}
        framework_results = inference_results.get('framework_results', {})
        
        for framework_name, framework_result in framework_results.items():
            if framework_name in self.integration_weights:
                # 获取框架的确定性，如果没有则使用默认值
                certainty = framework_result.get('certainty', 0.5)
                # 根据确定性调整权重（0.1-0.3范围）
                certainty_weights[framework_name] = 0.1 + (certainty * 0.2)
        
        # 2. 基于框架历史性能调整权重
        performance_weights = {}
        for name in self.integration_weights:
            performance = self.framework_performance.get(name, {'success_count': 0, 'total_count': 0})
            
            if performance['total_count'] > 0:
                success_rate = performance['success_count'] / performance['total_count']
                # 根据成功率调整权重（0.05-0.25范围）
                performance_weights[name] = 0.05 + (success_rate * 0.2)
            else:
                # 对于没有历史数据的框架，使用默认权重
                performance_weights[name] = self.integration_weights[name]
        
        # 3. 综合两种调整策略（可调整权重）
        #    - 新框架：更依赖确定性（weight_certainty较高）
        #    - 成熟框架：更依赖历史性能（weight_performance较高）
        
        for framework_name in self.integration_weights:
            # 获取两种权重（如果不存在则使用当前权重）
            certainty_weight = certainty_weights.get(framework_name, self.integration_weights[framework_name])
            performance_weight = performance_weights.get(framework_name, self.integration_weights[framework_name])
            
            # 根据框架成熟度调整权重分配
            total_runs = self.framework_performance.get(framework_name, {}).get('total_count', 0)
            
            if total_runs < 10:  # 新框架（数据不足）
                # 更依赖确定性（70%）
                weight = certainty_weight * 0.7 + performance_weight * 0.3
            elif total_runs < 50:  # 发展中的框架
                # 平衡依赖（50%-50%）
                weight = (certainty_weight + performance_weight) / 2
            else:  # 成熟框架
                # 更依赖历史性能（70%）
                weight = certainty_weight * 0.3 + performance_weight * 0.7
            
            # 确保权重在合理范围内
            self.integration_weights[framework_name] = max(0.05, min(0.35, weight))
        
        # 4. 如果有实际结果，进行反馈学习
        if actual_outcomes:
            self._update_performance_metrics(inference_results, actual_outcomes)
        
        # 5. 归一化权重，确保总和为1
        self._normalize_weights()
        
        logger.debug(f"调整后的集成权重: {self.integration_weights}")
        
        return self.integration_weights

    def _normalize_weights(self):
        """归一化权重，确保总和为1"""
        total_weight = sum(self.integration_weights.values())
        if total_weight > 0:
            for name in self.integration_weights:
                self.integration_weights[name] /= total_weight

    def _update_performance_metrics(self, inference_results, actual_outcomes):
        """基于实际结果更新性能指标"""
        framework_results = inference_results.get('framework_results', {})
        
        for framework_name, framework_result in framework_results.items():
            if framework_name not in self.framework_performance:
                continue
            
            # 评估框架预测的准确性（简化版）
            # 实际应用中需要更复杂的准确性评估
            if 'key_predictions' in framework_result and 'actual_values' in actual_outcomes:
                # 计算预测误差
                predictions = framework_result['key_predictions']
                actuals = actual_outcomes['actual_values']
                
                # 简单误差计算（实际应用中需要更复杂）
                if isinstance(predictions, dict) and isinstance(actuals, dict):
                    errors = []
                    for key in predictions:
                        if key in actuals:
                            pred = predictions[key]
                            actual = actuals[key]
                            if isinstance(pred, (int, float)) and isinstance(actual, (int, float)):
                                error = abs(pred - actual) / (abs(actual) + 1e-6)
                                errors.append(error)
                    
                    if errors:
                        avg_error = sum(errors) / len(errors)
                        # 根据误差调整成功率
                        if avg_error < 0.1:  # 误差小于10%
                            self.framework_performance[framework_name]['success_count'] += 1
                        elif avg_error < 0.3:  # 误差小于30%
                            self.framework_performance[framework_name]['success_count'] += 0.5
                        else:
                            self.framework_performance[framework_name]['success_count'] += 0.2
                        
                        self.framework_performance[framework_name]['total_count'] += 1

    

    
    def _get_default_framework_result(self, framework_index):
        """获取框架的默认结果（增强版）"""
        default_results = [
            {
                'material_properties': {'elasticity': {'score': 50, 'unit': '无量纲'}},
                'physical_constraints': [],
                'certainty': 0.3,
                'recommendations': [{'content': '进行详细物理分析', 'priority': 3}]
            },
            {
                'biological_state': {'collagen_level': {'score': 50, 'interpretation': '中等'}},
                'homeostasis': {'balance': 'moderate'},
                'certainty': 0.3,
                'recommendations': [{'content': '进行详细生物分析', 'priority': 3}]
            },
            {
                'spatial_structure': {'surface_topology': {'roughness': 'medium'}},
                'symmetry_analysis': {'score': 0.5},
                'certainty': 0.3,
                'recommendations': [{'content': '进行详细几何分析', 'priority': 3}]
            },
            {
                'perceived_effectiveness': 0.5,
                'decision_making_style': 'balanced',
                'certainty': 0.3,
                'recommendations': [{'content': '考虑用户心理偏好', 'priority': 3}]
            },
            {
                'optimal_strategy': {'name': 'default_strategy', 'score': 0.5},
                'utility_analysis': {},
                'certainty': 0.3,
                'recommendations': [{'content': '基于更多信息优化决策', 'priority': 3}]
            },
            {
                'best_solution': {'products': [], 'routines': {}},
                'best_fitness': 0.5,
                'certainty': 0.3,
                'recommendations': [{'content': '基于更多数据优化方案', 'priority': 3}]
            },
            {
                'extreme_risks': [],
                'risk_constraints': [],
                'certainty': 0.3,
                'recommendations': [{'content': '进行详细风险评估', 'priority': 3}]
            }
        ]
        
        if framework_index < len(default_results):
            return default_results[framework_index]
        return {'certainty': 0.3, 'recommendations': []}
    
    def _record_framework_success(self, framework_index):
        """记录框架成功运行"""
        framework_names = list(self.framework_performance.keys())
        if framework_index < len(framework_names):
            name = framework_names[framework_index]
            self.framework_performance[name]['success_count'] += 1
            self.framework_performance[name]['total_count'] += 1
    
    def _record_framework_failure(self, framework_index, error_type):
        """记录框架运行失败"""
        framework_names = list(self.framework_performance.keys())
        if framework_index < len(framework_names):
            name = framework_names[framework_index]
            self.framework_performance[name]['total_count'] += 1
    
    def _get_framework_performance_summary(self):
        """获取框架性能摘要"""
        summary = {}
        for name, performance in self.framework_performance.items():
            if performance['total_count'] > 0:
                success_rate = performance['success_count'] / performance['total_count']
                summary[name] = {
                    'success_rate': success_rate,
                    'success_count': performance['success_count'],
                    'total_count': performance['total_count']
                }
            else:
                summary[name] = {
                    'success_rate': 0.0,
                    'success_count': 0,
                    'total_count': 0
                }
        return summary

    async def infer_with_actionable_factors(self, state, actionable_factors, context=None):
        """增强推理，结合数据修复和错误处理"""
        try:
            # 1. 数据验证和修复（保持旧版本的优点）
            if not isinstance(state, dict):
                raise ValueError("state参数必须是字典类型")
            
            # 修复水分含量范围
            if 'moisture_content' in state:
                moisture = state['moisture_content']
                if not isinstance(moisture, (int, float)):
                    try:
                        moisture = float(moisture)
                    except (ValueError, TypeError):
                        moisture = 50.0  # 默认值
                
                # 确保在合理范围内 (0-100)
                if moisture < 0 or moisture > 100:
                    moisture = max(0, min(100, moisture))
                    logger.warning(f"调整水分含量到合理范围: {moisture}")
                    state['moisture_content'] = moisture
                
                # 数据缩放
                state['moisture_content_scaled'] = moisture / 100.0
            
            # 2. 确保 actionable_factors 是字典（新版本的优点）
            if not isinstance(actionable_factors, dict):
                logger.warning(f"actionable_factors 应为字典类型，实际为: {type(actionable_factors)}")
                actionable_factors = {}
            
            # 3. 执行基本推理
            base_inference = await self.infer_async(state, None, context)
            
            # 验证推理结果
            if not isinstance(base_inference, dict):
                logger.error("基本推理返回非字典结果，使用默认结果")
                base_inference = self._get_fallback_result(state, "推理结果格式错误")
            
            # 4. 增强推理（使用正确的方法名）
            enhanced_inference = self._enhance_inference_with_actionable_factors(
                base_inference, actionable_factors, context
            )
            
            # 添加元数据
            enhanced_inference['enhancement_applied'] = True
            enhanced_inference['actionable_factors'] = actionable_factors
            
            return enhanced_inference
            
        except Exception as e:
            logger.error(f"增强推理失败: {str(e)}", exc_info=True)
            
            # 详细的错误信息
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': time.time()
            }
            
            # 返回带有错误信息的回退结果
            fallback = self._get_fallback_result(state, e)
            fallback['error_details'] = error_details
            fallback['fallback_mode'] = True
            
            return fallback





    async def _run_frameworks_safely(self, framework_tasks):
        """安全收集框架结果 - 改进版"""
        framework_results = []
        
        try:
            # 等待所有任务完成
            results = await asyncio.gather(*framework_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # 处理异常结果
                    logger.error(f"框架 {i} 执行异常: {result}")
                    framework_results.append(self._get_default_framework_result(i))
                elif result is None:
                    # 处理空结果
                    logger.warning(f"框架 {i} 返回空结果")
                    framework_results.append(self._get_default_framework_result(i))
                else:
                    # 正常结果
                    framework_results.append(result)
                    # 记录成功
                    self._record_framework_success(i)
                    
        except Exception as e:
            logger.error(f"收集框架结果时发生错误: {e}")
            # 返回默认结果
            for i in range(7):  # 7个框架
                framework_results.append(self._get_default_framework_result(i))
        
        return framework_results
    
    def _get_default_framework_result(self, framework_index):
        """获取框架的默认结果"""
        default_results = {
            0: {'material_properties': {}, 'elasticity': 0.5, 'certainty': 0.1},  # 物理
            1: {'biological_state': {}, 'collagen_level': 0.5, 'certainty': 0.1},  # 生物
            2: {'surface_topology': 'unknown', 'symmetry_analysis': {}, 'certainty': 0.1},  # 几何
            3: {'perceived_effectiveness': 0.5, 'certainty': 0.1},  # 心理
            4: {'optimal_strategy': 'default', 'certainty': 0.1},  # 决策
            5: {'optimized_solution': 'default', 'certainty': 0.1},  # 进化算法
            6: {'extreme_risks': [], 'certainty': 0.1}  # 肥尾风险
        }
        return default_results.get(framework_index, {'certainty': 0.1})


    
 

    
    def _share_framework_results(self, framework_results):
        """实现框架间结果共享"""
        framework_names = list(self.integration_weights.keys())
        for i, name in enumerate(framework_names):
            if i < len(framework_results):
                self.inter_framework_communication[name] = framework_results[i]
    
    def _enhance_inference_with_actionable_factors(self, basic_inference, actionable_factors, context):
        """结合可改变因素增强推理结果 - 修复版"""
        enhanced = basic_inference.copy()
        
        # 修复1：检查数据结构，处理两种情况
        # 情况1：basic_inference 是 {'inference': {...}} 结构
        # 情况2：basic_inference 直接是推理结果字典
        inference_data = None
        
        if 'inference' in enhanced:
            # 情况1：嵌套结构
            inference_data = enhanced['inference']
        else:
            # 情况2：直接就是推理结果
            inference_data = enhanced
        
        # 修复2：添加数据清理逻辑
        if 'key_recommendations' in inference_data and isinstance(inference_data['key_recommendations'], list):
            # 确保每个推荐项都是字典
            valid_recommendations = []
            for rec in inference_data['key_recommendations']:
                if isinstance(rec, dict):
                    valid_recommendations.append(rec)
                elif isinstance(rec, str):
                    # 如果是字符串，转换为字典
                    valid_recommendations.append({
                        'content': rec,
                        'priority': 3,  # 默认优先级
                        'source': 'converted_from_string'
                    })
            inference_data['key_recommendations'] = valid_recommendations
        
        # 修复3：设置可改变因素
        if 'actionable_factors' in inference_data:
            inference_data['actionable_factors'] = actionable_factors
        else:
            inference_data['actionable_factors'] = {}
        
        # 修复4：优化推荐（使用清理后的数据）
        if 'key_recommendations' in inference_data:
            inference_data['key_recommendations'] = self._optimize_recommendations_for_actionable_factors(
                inference_data['key_recommendations'], actionable_factors
            )
        
        # 修复5：更新结构
        if 'inference' in enhanced:
            enhanced['inference'] = inference_data
        else:
            enhanced = inference_data
        
        # 添加元数据
        enhanced['enhancement_applied'] = True
        enhanced['actionable_factors_processed'] = list(actionable_factors.keys()) if isinstance(actionable_factors, dict) else []
        
        return enhanced
    
    def _optimize_recommendations_for_actionable_factors(self, recommendations, actionable_factors):
        """为可改变因素优化推荐 - 修复版"""
        if not recommendations or not actionable_factors:
            return recommendations
        
        # 确保 recommendations 是列表
        if not isinstance(recommendations, list):
            logger.warning(f"推荐不是列表类型: {type(recommendations)}")
            return []
        
        optimized_recommendations = []
        
        for rec in recommendations:
            # 确保每个推荐项都是字典
            if not isinstance(rec, dict):
                logger.warning(f"跳过非字典推荐项: {type(rec)}")
                continue
            
            # 创建副本
            optimized_rec = rec.copy()
            
            # 应用可改变因素优化
            if isinstance(actionable_factors, dict):
                if actionable_factors.get('moisture_boost'):
                    # 如果是水分增强相关推荐，提高优先级
                    if any(keyword in rec.get('content', '').lower() for keyword in ['保湿', '水分', '补水']):
                        optimized_rec['priority'] = min(10, (rec.get('priority', 3) + 2))
                        optimized_rec['boosted'] = True
                
                if actionable_factors.get('protection_enhance'):
                    # 如果是保护增强相关推荐，提高优先级
                    if any(keyword in rec.get('content', '').lower() for keyword in ['防护', '防晒', '保护']):
                        optimized_rec['priority'] = min(10, (rec.get('priority', 3) + 1))
                        optimized_rec['protected'] = True
            
            optimized_recommendations.append(optimized_rec)
        
        # 按优先级排序
        try:
            optimized_recommendations.sort(key=lambda x: x.get('priority', 1), reverse=True)
        except AttributeError as e:
            logger.error(f"排序失败: {e}")
            # 如果排序失败，至少确保没有字符串
            filtered_recs = [r for r in optimized_recommendations if isinstance(r, dict)]
            return filtered_recs
        
        return optimized_recommendations
    

    
    def personalize_frameworks(self, user_profile):
        """超个性化定制：基于用户特征细化框架参数"""
        # 为每个框架应用个性化参数
        if hasattr(self.physics_framework, 'personalize'):
            self.physics_framework.personalize(user_profile)
        
        if hasattr(self.biology_framework, 'personalize'):
            self.biology_framework.personalize(user_profile)
        
        if hasattr(self.geometric_dynamics_framework, 'personalize'):
            self.geometric_dynamics_framework.personalize(user_profile)
        
        # 调整集成权重以适应用户特征
        if 'skin_type' in user_profile:
            if user_profile['skin_type'] == 'dry':
                self.integration_weights['biology'] += 0.05
                self.integration_weights['physics'] += 0.03
            elif user_profile['skin_type'] == 'oily':
                self.integration_weights['geometric_dynamics'] += 0.05
                self.integration_weights['real2sim2real'] += 0.03
        
        if 'age' in user_profile and user_profile['age'] > 50:
            self.integration_weights['biology'] += 0.05
            self.integration_weights['decision_theory'] += 0.02
    
    def generate_explanation(self, inference_results, explanation_level='medium'):
        """可解释性增强：生成推理解释"""
        explanation = {
            'level': explanation_level,
            'summary': '',
            'details': []
        }
        
        # 生成高级解释
        if explanation_level in ['high', 'medium']:
            explanation['summary'] = self._generate_summary_explanation(inference_results)
            explanation['details'] = self._generate_detailed_explanation(inference_results)
        
        # 生成可视化支持数据
        explanation['visualization_data'] = self._generate_visualization_data(inference_results)
        
        return explanation
    
    def _generate_summary_explanation(self, inference_results):
        """生成总结性解释"""
        # 基于关键推荐生成总结
        if 'key_recommendations' in inference_results:
            top_rec = inference_results['key_recommendations'][0]
            return f"基于多框架分析，我们建议您{top_rec['content']}。"
        
        return "我们已经分析了您的皮肤状况，并准备了个性化的护肤建议。"
    
    def _generate_detailed_explanation(self, inference_results):
        """生成详细解释"""
        details = []
        
        # 生物学解释
        if 'key_biological_state' in inference_results:
            details.append({
                'framework': 'biology',
                'explanation': f"从生物学角度分析，您的皮肤{inference_results['key_biological_state']}"
            })
        
        # 物理学解释
        if 'key_material_properties' in inference_results:
            details.append({
                'framework': 'physics',
                'explanation': f"从物理学角度分析，您的皮肤{inference_results['key_material_properties']}"
            })
        
        # 跨框架分析解释
        if 'cross_framework_analysis' in inference_results:
            for key, value in inference_results['cross_framework_analysis'].items():
                details.append({
                    'framework': 'cross_framework',
                    'explanation': f"跨框架分析显示{key}: {value}"
                })
        
        return details
    
    def _generate_visualization_data(self, inference_results):
        """生成可视化支持数据"""
        visualization_data = {}
        
        # 皮肤状态可视化数据
        if 'key_biological_state' in inference_results:
            visualization_data['biological_state'] = inference_results['key_biological_state']
        
        # 物质特性可视化数据
        if 'key_material_properties' in inference_results:
            visualization_data['material_properties'] = inference_results['key_material_properties']
        
        # 对称性分析可视化数据
        if 'key_symmetry_analysis' in inference_results:
            visualization_data['symmetry_analysis'] = inference_results['key_symmetry_analysis']
        
        return visualization_data
    
    def adapt_to_environment(self, environment_factors):
        """动态环境适应：根据环境因素调整护肤策略"""
        # 为每个框架应用环境适应性调整
        for framework in [
            self.physics_framework, 
            self.biology_framework, 
            self.real2sim2real_framework
        ]:
            if hasattr(framework, 'adapt_to_environment'):
                framework.adapt_to_environment(environment_factors)
        
        # 调整集成权重以适应环境
        if 'temperature' in environment_factors and environment_factors['temperature'] > 30:
            self.integration_weights['physics'] += 0.05  # 高温环境增加物理学框架权重
        
        if 'humidity' in environment_factors:
            if environment_factors['humidity'] < 30:
                self.integration_weights['biology'] += 0.05  # 干燥环境增加生物学框架权重
            elif environment_factors['humidity'] > 70:
                self.integration_weights['geometric_dynamics'] += 0.05  # 潮湿环境增加几何动力学框架权重
    
    def add_social_influence_framework(self, social_framework):
        """扩展创新框架：添加社交影响框架"""
        self.social_influence_framework = social_framework
        self.integration_weights['social_influence'] = 0.05
        
        # 重新归一化权重
        total_weight = sum(self.integration_weights.values())
        for name in self.integration_weights:
            self.integration_weights[name] /= total_weight
    

    def _analyze_topology_moisture_relationship(self, topology, moisture_distribution):
        """分析皮肤拓扑结构与水分分布的关系"""
        # 示例分析
        if 'roughness' in topology and topology['roughness'] > 0.6:
            return {
                'relationship': "皮肤粗糙区域水分蒸发更快",
                'recommendation': "在粗糙区域使用更滋润的产品"
            }
        
        return {
            'relationship': "皮肤拓扑结构与水分分布基本匹配",
            'recommendation': "保持当前的保湿策略"
        }
    
    def _analyze_user_preference_alignment(self, perceived_effectiveness, optimal_strategy):
        """分析用户偏好与最佳策略的一致性"""
        # 示例分析
        alignment_score = 0.7  # 模拟对齐分数
        
        if alignment_score > 0.8:
            return {
                'alignment': "高度一致",
                'recommendation': "继续当前的护肤策略"
            }
        elif alignment_score > 0.5:
            return {
                'alignment': "中等一致",
                'recommendation': "微调当前策略以提高效果"
            }
        else:
            return {
                'alignment': "不太一致",
                'recommendation': "考虑调整护肤策略以更好地满足您的需求"
            }
    
    def _analyze_risk_adjusted_optimization(self, optimized_solution, extreme_risks):
        """分析风险调整后的优化方案"""
        # 示例分析
        if len(extreme_risks) == 0:
            return {
                'risk_level': "低风险",
                'recommendation': "可以安全地采用优化方案"
            }
        
        return {
            'risk_level': "中等风险",
            'recommendation': "在采用优化方案时注意监控可能的风险"
        }
    
    # 数字孪生特定方法
    def run_digital_twin_simulation(self, initial_state, time_steps=100):
        """运行数字孪生模拟"""
        return self.real2sim2real_framework.run_simulation(initial_state, time_steps)
    
    def get_digital_twin_state(self):
        """获取当前数字孪生状态"""
        if hasattr(self.real2sim2real_framework, 'get_current_state'):
            return self.real2sim2real_framework.get_current_state()
        return None
    
    def compare_digital_twin_with_reality(self, real_state):
        """比较数字孪生与现实状态"""
        if hasattr(self.real2sim2real_framework, 'compare_with_reality'):
            return self.real2sim2real_framework.compare_with_reality(real_state)
        return None



class EnhancedTestDataGenerator:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    """增强测试数据生成器 - 用于测试系统扩展性"""
    
    @staticmethod
    def generate_extensive_product_catalog(num_products: int = 50) -> List[Dict]:
        """生成扩展的产品目录"""
        products = []
        
        # 产品类别分布
        categories = [
            ('cleanser', 15),  # 洁面
            ('moisturizer', 20),  # 保湿
            ('sunscreen', 10),  # 防晒
            ('serum', 15),  # 精华
            ('toner', 10),  # 爽肤水
            ('mask', 10),  # 面膜
            ('eye_cream', 10),  # 眼霜
            ('treatment', 10)  # 特殊护理
        ]
        
        # 产品功效
        benefits = [
            'hydration', 'moisturizing', 'anti-aging', 'brightening',
            'acne_treatment', 'pore_minimizing', 'soothing', 'repair',
            'firming', 'exfoliating', 'protection', 'balancing'
        ]
        
        # 适合肤质
        skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        
        product_id = 1000
        
        for category, percentage in categories:
            num_category_products = int(num_products * percentage / 100)
            
            for i in range(num_category_products):
                # 随机选择2-4个功效
                num_benefits = random.randint(2, 4)
                selected_benefits = random.sample(benefits, num_benefits)
                
                # 随机选择适合的肤质
                num_skin_types = random.randint(2, len(skin_types))
                suitable_for = random.sample(skin_types, num_skin_types)
                
                # 价格范围（根据不同类别）
                price_ranges = {
                    'cleanser': (30, 300),
                    'moisturizer': (50, 800),
                    'sunscreen': (60, 500),
                    'serum': (100, 1200),
                    'toner': (40, 400),
                    'mask': (20, 300),
                    'eye_cream': (80, 1000),
                    'treatment': (150, 1500)
                }
                
                min_price, max_price = price_ranges.get(category, (50, 500))
                price = random.randint(min_price, max_price)
                
                # 成分（示例）
                ingredients = EnhancedTestDataGenerator._generate_ingredients(category)
                
                product = {
                    'id': f"prod_{product_id}",
                    'name': f"{category.capitalize()} Product {i+1}",
                    'category': category,
                    'price': price,
                    'benefits': selected_benefits,
                    'suitable_for': suitable_for,
                    'ingredients': ingredients,
                    'brand': random.choice(['Brand A', 'Brand B', 'Brand C', 'Brand D', 'Brand E']),
                    'rating': round(random.uniform(3.5, 5.0), 1),
                    'popularity': random.randint(1, 100)
                }
                
                products.append(product)
                product_id += 1
        
        # 打乱顺序
        random.shuffle(products)
        return products
    
    @staticmethod
    def _generate_ingredients(category: str) -> List[str]:
        """根据类别生成成分"""
        base_ingredients = ['water', 'glycerin', 'hyaluronic_acid', 'niacinamide']
        
        category_specific = {
            'cleanser': ['surfactant', 'aloe_vera', 'chamomile'],
            'moisturizer': ['ceramides', 'squalane', 'peptides'],
            'sunscreen': ['zinc_oxide', 'titanium_dioxide', 'avobenzone'],
            'serum': ['vitamin_c', 'retinol', 'alpha_arbutin'],
            'toner': ['aha', 'bha', 'centella_asiatica'],
            'mask': ['clay', 'charcoal', 'honey'],
            'eye_cream': ['caffeine', 'vitamin_k', 'peptides'],
            'treatment': ['salicylic_acid', 'benzoyl_peroxide', 'azelaic_acid']
        }
        
        ingredients = base_ingredients.copy()
        if category in category_specific:
            # 随机选择1-3个类别特定成分
            specific = random.sample(category_specific[category], 
                                   random.randint(1, 3))
            ingredients.extend(specific)
        
        return ingredients
    
    @staticmethod
    def generate_diverse_user_profiles(num_profiles: int = 5) -> List[Dict]:
        """生成多样化的用户配置文件"""
        profiles = []
        
        for i in range(num_profiles):
            age = random.randint(18, 65)
            skin_type = random.choice(['normal', 'dry', 'oily', 'combination', 'sensitive'])
            
            # 生成不同的护肤目标
            all_goals = [
                {'type': 'hydration', 'priority': random.randint(1, 3)},
                {'type': 'anti-aging', 'priority': random.randint(1, 3)},
                {'type': 'brightening', 'priority': random.randint(1, 3)},
                {'type': 'acne_treatment', 'priority': random.randint(1, 3)},
                {'type': 'pore_minimizing', 'priority': random.randint(1, 3)},
                {'type': 'soothing', 'priority': random.randint(1, 3)}
            ]
            
            # 随机选择2-4个目标
            num_goals = random.randint(2, 4)
            user_goals = random.sample(all_goals, num_goals)
            
            # 预算水平
            budget_levels = ['low', 'medium', 'high']
            budget = random.choice(budget_levels)
            
            # 敏感性
            sensitivity_levels = ['low', 'medium', 'high']
            sensitivity = random.choice(sensitivity_levels)
            
            profile = {
                'user_id': f"test_user_{i+1}",
                'age': age,
                'skin_type': skin_type,
                'sensitivity': sensitivity,
                'budget': budget,
                'user_goals': user_goals,
                'existing_routine': random.choice(['minimal', 'basic', 'comprehensive']),
                'allergies': [] if random.random() > 0.2 else ['fragrance', 'essential_oils']
            }
            
            profiles.append(profile)
        
        return profiles


# 立即可用的实现方案
class DeepLearningFramework:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    """深度学习框架 - 皮肤护理智能推荐系统"""
    
    def __init__(self):
        # 模拟的神经网络参数
        self.model_weights = {
            'user_preferences': np.random.randn(10),
            'skin_parameters': np.random.randn(8),
            'product_features': np.random.randn(12),
            'environmental_factors': np.random.randn(4)
        }
        
        # 历史数据存储
        self.user_history = defaultdict(list)
        self.recommendation_history = []
        
        # 性能跟踪
        self.performance_metrics = {
            'total_recommendations': 0,
            'successful_recommendations': 0,
            'accuracy_history': [],
            'response_time': []
        }
        
        # 存储历史数据用于趋势预测
        self.historical_data = []
        self.trend_window = 10  # 趋势分析窗口大小
        
        logger.info("深度学习框架初始化完成")
    
    async def infer(self, skin_params: Dict, context: Optional[Dict] = None) -> Dict:
        """深度学习推理 - 生成个性化推荐"""
        try:
            start_time = datetime.now()
            
            # 1. 准备输入数据
            input_features = self._prepare_input_features(skin_params, context)
            
            # 2. 神经网络前向传播（简化版）
            predictions = self._neural_network_forward(input_features)
            
            # 3. 生成推荐
            recommendations = self._generate_recommendations(predictions, context)
            
            # 4. 分析模式和趋势
            pattern_analysis = self._analyze_patterns(skin_params, context)
            
            # 5. 预测趋势
            trend_predictions = self._predict_trends(skin_params, context)
            
            # 6. 计算置信度
            certainty = self._calculate_certainty(predictions, pattern_analysis)
            
            # 性能记录
            execution_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['response_time'].append(execution_time)
            self.performance_metrics['total_recommendations'] += 1
            self.performance_metrics['successful_recommendations'] += 1
            
            # 保存历史数据用于趋势分析
            self._save_historical_data(skin_params, predictions, context)
            
            # 构建结果
            result = {
                'deep_learning_analysis': {
                    'predictions': predictions,
                    'intelligent_recommendations': recommendations,
                    'pattern_analysis': pattern_analysis,
                    'trend_predictions': trend_predictions,
                    'feature_importance': self._calculate_feature_importance(input_features)
                },
                'certainty': certainty,
                'framework_status': 'success',
                'execution_time': execution_time,
                'model_version': '1.0.0'
            }
            
            # 记录历史
            self.recommendation_history.append({
                'timestamp': datetime.now().isoformat(),
                'input_features': input_features,
                'recommendations': recommendations,
                'certainty': certainty
            })
            
            logger.info(f"深度学习推理完成，生成 {len(recommendations.get('priority_recommendations', []))} 个推荐")
            return result
            
        except Exception as e:
            logger.error(f"深度学习推理失败: {e}")
            self.performance_metrics['total_recommendations'] += 1
            return {
                'deep_learning_analysis': {
                    'error': str(e),
                    'fallback_recommendations': self._get_fallback_recommendations()
                },
                'certainty': 0.1,
                'framework_status': 'error'
            }
    
    def _prepare_input_features(self, skin_params: Dict, context: Dict) -> Dict:
        """准备神经网络输入特征"""
        features = {}
        
        # 皮肤参数特征
        if skin_params:
            for key, value in skin_params.items():
                if isinstance(value, (int, float)):
                    features[f'skin_{key}'] = float(value)
                elif isinstance(value, dict) and 'score' in value:
                    features[f'skin_{key}'] = float(value['score']) / 100.0
                elif key in ['moisture_content', 'elasticity', 'collagen_level']:
                    # 处理特定皮肤参数
                    if isinstance(value, (int, float)):
                        features[f'skin_{key}'] = float(value)
        
        # 添加默认参数
        default_params = ['moisture_content', 'elasticity', 'collagen_level']
        for param in default_params:
            if f'skin_{param}' not in features:
                # 如果没有提供，使用随机值
                features[f'skin_{param}'] = random.uniform(0.3, 0.8)
        
        # 上下文特征
        if context:
            # 用户特征
            if 'user_profile' in context:
                user = context['user_profile']
                features['user_age'] = min(1.0, user.get('age', 30) / 100.0)
                features['user_skin_type'] = self._encode_skin_type(user.get('skin_type', 'normal'))
                features['user_budget'] = self._encode_budget(user.get('budget', 'medium'))
            
            # 环境特征
            if 'environmental_data' in context:
                env = context['environmental_data']
                features['env_uv'] = min(1.0, env.get('uv_index', 3) / 15.0)
                features['env_humidity'] = env.get('humidity', 60) / 100.0
                features['env_temperature'] = (env.get('temperature', 25) - 10) / 30.0
            
            # 产品特征（如果有）
            if 'available_products' in context:
                products = context['available_products']
                features['product_count'] = min(1.0, len(products) / 50.0)
        
        # 确保至少有一些特征
        if not features:
            features = {
                'skin_moisture': 0.5,
                'skin_elasticity': 0.5,
                'skin_collagen': 0.5,
                'user_age': 0.3,
                'env_uv': 0.2
            }
        
        return features
    
    def _neural_network_forward(self, features: Dict) -> Dict:
        """神经网络前向传播（简化版）"""
        # 将特征转换为向量
        feature_names = list(features.keys())
        feature_vector = np.array([features[name] for name in feature_names])
        
        # 模拟神经网络计算
        if len(feature_vector) > 0:
            # 简单的加权和计算（模拟神经网络）
            weights = np.random.randn(len(feature_vector))
            bias = np.random.randn()
            
            # 计算预测分数
            raw_score = np.dot(feature_vector, weights) + bias
            
            # 激活函数（sigmoid）
            prediction_score = 1 / (1 + np.exp(-raw_score))
            
            # 生成多维度预测
            predictions = {
                'overall_score': float(prediction_score),
                'hydration_need': float(features.get('skin_moisture', 0.5) * prediction_score),
                'protection_need': float(features.get('env_uv', 0.2) * 0.8 + prediction_score * 0.2),
                'repair_need': float(1 - features.get('skin_elasticity', 0.5) * prediction_score),
                'feature_contributions': dict(zip(feature_names, weights.tolist())),
                'confidence': min(0.95, max(0.1, prediction_score))
            }
        else:
            predictions = {
                'overall_score': 0.5,
                'hydration_need': 0.5,
                'protection_need': 0.5,
                'repair_need': 0.5,
                'confidence': 0.5
            }
        
        return predictions
    
    def _generate_recommendations(self, predictions: Dict, context: Dict) -> Dict:
        """基于预测生成智能推荐"""
        import logging
        logger = logging.getLogger(__name__)

        # 检查是否有 skin_analysis 参数
        skin_analysis = context.get('skin_analysis', {})
        logger.info(f"接收到的皮肤分析数据键值: {list(skin_analysis.keys())}")
        
        # 确保 skin_analysis 有必要的字段
        skin_analysis = self._ensure_analysis_fields(skin_analysis)
        logger.info(f"补全后的皮肤分析数据键值: {list(skin_analysis.keys())}")
        
        # 记录具体字段值
        logger.info(f"moisture_content: {skin_analysis.get('moisture_content', '未找到')}")
        logger.info(f"elasticity: {skin_analysis.get('elasticity', '未找到')}")
        logger.info(f"collagen_level: {skin_analysis.get('collagen_level', '未找到')}")


        
        # 从这里开始，skin_analysis 应该已经包含了所有必要字段
        # 所以下面这3行检查逻辑应该移到 _ensure_analysis_fields 方法中
        
        recommendations = {
            'priority_recommendations': [],
            'supplementary_recommendations': [],
            'avoid_recommendations': []
        }
        
        # 基于预测分数生成推荐
        overall_score = predictions.get('overall_score', 0.5)
        
        # 现在可以使用补全后的 skin_analysis 字段
        moisture_content = skin_analysis.get('moisture_content', 75)
        elasticity = skin_analysis.get('elasticity', 75)
        collagen_level = skin_analysis.get('collagen_level', 75)
        
        # 基于具体分析结果生成更精准的推荐
        if moisture_content < 70:
            recommendations['priority_recommendations'].append({
                'type': '保湿修复',
                'action': '使用含有神经酰胺、透明质酸的保湿产品',
                'reason': f'皮肤水分含量较低({moisture_content:.1f}/100)',
                'priority': 'high'
            })
        
        if collagen_level < 75:
            recommendations['priority_recommendations'].append({
                'type': '抗老紧致',
                'action': '添加视黄醇、胜肽或玻色因类产品',
                'reason': f'胶原蛋白水平偏低({collagen_level:.1f}/100)',
                'priority': 'medium'
            })
        
        # 原有的分数逻辑保持不变
        if overall_score < 0.3:
            # 高优先级需求
            if not recommendations['priority_recommendations']:  # 如果还没有添加推荐
                recommendations['priority_recommendations'].extend([
                    {
                        'type': '紧急修复',
                        'action': '使用屏障修复面霜，每日2次',
                        'reason': '皮肤屏障功能较弱',
                        'priority': 'high'
                    },
                    {
                        'type': '基础保湿',
                        'action': '使用含有神经酰胺的保湿产品',
                        'reason': '水分流失较快',
                        'priority': 'high'
                    }
                ])
        elif overall_score < 0.6:
            # 中等优先级
            if not recommendations['priority_recommendations']:
                recommendations['priority_recommendations'].append({
                    'type': '日常护理',
                    'action': '保持基础护肤流程（清洁-保湿-防晒）',
                    'reason': '皮肤状态稳定，需维持',
                    'priority': 'medium'
                })
                
                if predictions.get('protection_need', 0) > 0.7:
                    recommendations['supplementary_recommendations'].append({
                        'type': '防晒加强',
                        'action': '使用SPF30+防晒霜，每2小时补涂',
                        'reason': '环境紫外线较强'
                    })
        else:
            # 良好状态，预防性建议
            if not recommendations['priority_recommendations']:
                recommendations['priority_recommendations'].append({
                    'type': '预防维护',
                    'action': '继续当前护肤流程，定期检查皮肤状态',
                    'reason': '皮肤状态良好',
                    'priority': 'low'
                })
        
        # 根据上下文添加个性化推荐
        if context:
            user_profile = context.get('user_profile', {})
            skin_type = user_profile.get('skin_type', 'normal')
            
            if skin_type == 'dry':
                recommendations['supplementary_recommendations'].append({
                    'type': '干性皮肤护理',
                    'action': '使用油性成分较多的保湿产品',
                    'reason': '适合干性皮肤'
                })
            elif skin_type == 'oily':
                recommendations['supplementary_recommendations'].append({
                    'type': '油性皮肤护理',
                    'action': '使用水性保湿产品，避免油腻',
                    'reason': '适合油性皮肤'
                })
            
            # 根据年龄推荐
            age = user_profile.get('age', 30)
            if age > 40:
                recommendations['supplementary_recommendations'].append({
                    'type': '抗衰老',
                    'action': '考虑添加视黄醇或肽类产品',
                    'reason': '适合熟龄肌肤'
                })
        
        return recommendations
    
    def _ensure_analysis_fields(self, analysis_data):
        """确保分析数据包含必要字段"""
        if not analysis_data:
            return {}
        
        # 添加缺失字段
        if 'moisture_content' not in analysis_data:
            # 从 barrier.hydration 推导
            barrier_data = analysis_data.get('barrier', {})
            hydration = barrier_data.get('hydration', 0.5)
            analysis_data['moisture_content'] = min(100, max(0, hydration * 100))
        
        if 'elasticity' not in analysis_data:
            # 从 elasticity 模块获取
            elasticity_data = analysis_data.get('elasticity', {})
            analysis_data['elasticity'] = elasticity_data.get('score', 75)
        
        if 'collagen_level' not in analysis_data:
            # 从 collagen 模块获取
            collagen_data = analysis_data.get('collagen', {})
            analysis_data['collagen_level'] = collagen_data.get('score', 75)
        
        return analysis_data

    def _analyze_patterns(self, skin_params: Dict, context: Dict) -> Dict:
        """分析皮肤参数模式"""
        pattern_analysis = {
            'correlations': [],
            'anomalies': [],
            'trends': []
        }
        
        # 简单相关性分析
        if skin_params and isinstance(skin_params, dict):
            param_values = []
            param_names = []
            
            for key, value in skin_params.items():
                if isinstance(value, (int, float)):
                    param_values.append(float(value))
                    param_names.append(key)
                elif isinstance(value, dict) and 'score' in value:
                    param_values.append(float(value['score']))
                    param_names.append(key)
            
            if len(param_values) >= 2:
                # 计算简单的相关性
                avg_value = np.mean(param_values)
                pattern_analysis['average_score'] = float(avg_value)
                
                # 识别异常值
                std_value = np.std(param_values)
                for i, val in enumerate(param_values):
                    if abs(val - avg_value) > 1.5 * std_value:
                        pattern_analysis['anomalies'].append({
                            'parameter': param_names[i],
                            'value': val,
                            'deviation': float(abs(val - avg_value) / std_value)
                        })
        
        return pattern_analysis
    
    def _predict_trends(self, skin_params: Dict, context: Dict) -> Dict:
        """预测皮肤参数趋势"""
        trend_predictions = {
            'short_term': {},
            'long_term': {},
            'confidence': 0.7
        }
        
        # 使用历史数据预测趋势
        if self.historical_data and len(self.historical_data) >= 3:
            # 简单线性回归预测
            recent_data = self.historical_data[-min(5, len(self.historical_data)):]
            
            # 预测每个参数的趋势
            for param_name in ['moisture_content', 'elasticity', 'collagen_level']:
                param_values = []
                for data_point in recent_data:
                    if 'skin_params' in data_point and param_name in data_point['skin_params']:
                        value = data_point['skin_params'][param_name]
                        if isinstance(value, (int, float)):
                            param_values.append(float(value))
                
                if len(param_values) >= 2:
                    # 计算趋势（简单斜率）
                    x = list(range(len(param_values)))
                    slope = self._calculate_slope(x, param_values)
                    
                    trend_predictions['short_term'][param_name] = {
                        'current': param_values[-1] if param_values else 0.5,
                        'trend': 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable',
                        'slope': slope,
                        'prediction_7_days': min(1.0, max(0, param_values[-1] + slope * 7)) if param_values else 0.5
                    }
        
        # 如果没有足够历史数据，使用默认值
        if not trend_predictions['short_term']:
            for param_name in ['moisture_content', 'elasticity', 'collagen_level']:
                current_value = 0.5
                if skin_params and param_name in skin_params:
                    value = skin_params[param_name]
                    if isinstance(value, (int, float)):
                        current_value = float(value)
                
                trend_predictions['short_term'][param_name] = {
                    'current': current_value,
                    'trend': 'stable',
                    'slope': 0.0,
                    'prediction_7_days': current_value
                }
        
        return trend_predictions
    
    def _calculate_slope(self, x: List[float], y: List[float]) -> float:
        """计算线性回归斜率"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x_i * x_i for x_i in x)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = n * sum_x2 - sum_x * sum_x
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _save_historical_data(self, skin_params: Dict, predictions: Dict, context: Dict):
        """保存历史数据用于趋势分析"""
        data_point = {
            'timestamp': datetime.now().isoformat(),
            'skin_params': skin_params.copy() if skin_params else {},
            'predictions': predictions.copy(),
            'overall_score': predictions.get('overall_score', 0.5)
        }
        
        # 添加用户ID（如果有）
        if context and 'user_id' in context:
            data_point['user_id'] = context['user_id']
        
        self.historical_data.append(data_point)
        
        # 限制历史数据大小
        if len(self.historical_data) > 100:
            self.historical_data = self.historical_data[-100:]
    
    def _calculate_feature_importance(self, features: Dict) -> Dict:
        """计算特征重要性"""
        importance = {}
        
        for feature_name, value in features.items():
            # 简单的特征重要性计算（基于特征值的绝对值）
            importance[feature_name] = {
                'value': value,
                'importance': min(1.0, abs(value) * 2),
                'interpretation': self._interpret_feature(feature_name, value)
            }
        
        return importance
    
    def _calculate_certainty(self, predictions: Dict, pattern_analysis: Dict) -> float:
        """计算预测置信度"""
        certainty = 0.5  # 基础置信度
        
        # 基于预测分数调整
        overall_score = predictions.get('overall_score', 0.5)
        certainty += (overall_score - 0.5) * 0.3
        
        # 基于特征数量调整
        feature_count = len(predictions.get('feature_contributions', {}))
        if feature_count > 5:
            certainty += 0.1
        elif feature_count < 3:
            certainty -= 0.1
        
        # 基于模式分析调整
        if pattern_analysis.get('anomalies'):
            certainty -= len(pattern_analysis['anomalies']) * 0.05
        
        # 确保在合理范围内
        certainty = max(0.1, min(0.95, certainty))
        
        return round(certainty, 3)
    
    # 辅助方法
    def _encode_skin_type(self, skin_type: str) -> float:
        """编码皮肤类型为数值"""
        encoding = {
            'dry': 0.2,
            'normal': 0.5,
            'oily': 0.8,
            'combination': 0.6,
            'sensitive': 0.3
        }
        return encoding.get(skin_type.lower(), 0.5)
    
    def _encode_budget(self, budget: str) -> float:
        """编码预算为数值"""
        encoding = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.9
        }
        return encoding.get(budget.lower(), 0.6)
    
    def _interpret_feature(self, feature_name: str, value: float) -> str:
        """解释特征的含义"""
        if 'skin_' in feature_name:
            if value < 0.3:
                return '参数值较低，可能需要关注'
            elif value < 0.7:
                return '参数值正常'
            else:
                return '参数值良好'
        elif 'env_' in feature_name:
            return '环境影响因子'
        elif 'user_' in feature_name:
            return '用户特征'
        else:
            return '一般特征'
    
    def _get_fallback_recommendations(self) -> List[Dict]:
        """获取回退推荐"""
        return [{
            'type': '基础建议',
            'action': '保持日常清洁、保湿、防晒的基础护肤流程',
            'reason': '系统维护中，请稍后获取个性化推荐',
            'priority': 'medium'
        }]
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        total = self.performance_metrics['total_recommendations']
        successful = self.performance_metrics['successful_recommendations']
        
        avg_response_time = np.mean(self.performance_metrics['response_time']) if self.performance_metrics['response_time'] else 0
        
        return {
            'total_recommendations': total,
            'successful_recommendations': successful,
            'success_rate': successful / total if total > 0 else 0,
            'avg_response_time_seconds': round(avg_response_time, 3),
            'recent_performance': self.performance_metrics['response_time'][-5:] if self.performance_metrics['response_time'] else [],
            'history_size': len(self.recommendation_history),
            'historical_data_size': len(self.historical_data)
        }
    
    def clear_history(self, user_id: Optional[str] = None):
        """清除历史记录"""
        if user_id:
            # 清除特定用户的历史
            self.user_history[user_id] = []
        else:
            # 清除所有历史
            self.user_history.clear()
            self.recommendation_history.clear()
            self.historical_data.clear()
        
        logger.info("深度学习框架历史记录已清除")
class EnhancedDeepLearningModule:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    """阶段2：增强的深度学习模块（可逐步替换）"""
    
    def __init__(self):
        # 保持与第9框架的兼容性
        super().__init__()
        
        # 新增多模态融合能力
        self.fusion_module = MultiModalFusionModule()
        
    async def infer(self, skin_params, context=None):
        """增强推理：融合所有框架结果"""
        # 从世界模型获取所有框架结果
        all_framework_results = self._collect_all_framework_results(context)
        
        # 多模态融合
        fused_features = self.fusion_module(all_framework_results)
        
        # 深度学习分析
        analysis = self._advanced_analysis(fused_features)
        
        return analysis



class ProductDataManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    """产品数据管理器 - 集成到世界模型"""
    
    def __init__(self):
        self.recommender = ProductRecommender()
        self.product_db = self.recommender.db
        
    async def get_products_for_world_model(self, context: Dict) -> List[Dict]:
        """为世界模型获取产品数据"""
        try:
            # 确保有用户数据
            user_profile = context.get('user_profile', {})
            if not user_profile:
                user_profile = {'skin_type': 'normal', 'budget': 'medium'}
            
            # 获取适合进化算法的产品
            products = self.recommender.get_products_for_evolutionary(context)
            
            # 格式化为世界模型需要的格式
            formatted_products = []
            for product in products:
                formatted_product = {
                    'id': product['id'],
                    'name': product['name'],
                    'category': product['category'],
                    'price': product['price'],
                    'benefits': product['benefits'],
                    'ingredients': product['ingredients'],
                    'suitable_for': product['skin_types'],
                    'total_score': product['total_score']
                }
                formatted_products.append(formatted_product)
            
            return formatted_products
            
        except Exception as e:
            print(f"获取产品数据失败: {e}")
            return []
    
    def enhance_context_with_products(self, context: Dict) -> Dict:
        """用产品数据增强上下文"""
        enhanced_context = context.copy()
        
        # 如果上下文没有产品数据，添加产品数据
        if 'available_products' not in enhanced_context or not enhanced_context['available_products']:
            enhanced_context['available_products'] = self.get_products_for_world_model(enhanced_context)
        
        return enhanced_context
    
    def get_database_stats(self) -> Dict:
        """获取数据库统计信息"""
        all_products = self.product_db.get_all_products()
        
        # 按类别统计
        categories = {}
        for product in all_products:
            category = product['category']
            categories[category] = categories.get(category, 0) + 1
        
        # 按价格范围统计
        price_ranges = {}
        for product in all_products:
            price_range = product['price_range']
            price_ranges[price_range] = price_ranges.get(price_range, 0) + 1
        
        # 收集所有功效
        all_benefits = set()
        for product in all_products:
            all_benefits.update(product['benefits'])
        
        return {
            'total_products': len(all_products),
            'categories': categories,
            'price_ranges': price_ranges,
            'unique_benefits': len(all_benefits),
            'top_products': sorted(all_products, key=lambda x: x['total_score'], reverse=True)[:5]
        }