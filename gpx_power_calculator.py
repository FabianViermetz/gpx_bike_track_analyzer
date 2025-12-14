#!/usr/bin/env python3

import os
import re
import json
import math
import copy
import tempfile
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

import gpxpy
from geopy.distance import geodesic
from meteostat import Hourly, Point

import streamlit as st

import folium
import branca.colormap as cm
from streamlit_folium import st_folium

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.ndimage import median_filter

from datetime import datetime
from zoneinfo import ZoneInfo

try:
    from tzfpy import get_tz
except Exception:
    get_tz = None

FIX_TIME_ZONE = True
# === Global physical constants / limits (default values) ===
DEFAULT_HUMAN_BODY_EFFICIENCY, EFFICIENCY = 0.23, 0.23          # human mechanical efficiency (gross)
DEFAULT_ENERGY_BAR_SIZE_KCAL, ENERGY_BAR_SIZE_KCAL = 250., 250.          # human mechanical efficiency (gross)
DEFAULT_BASE_MET_DURING_ACTIVITY, BASE_MET_DURING_ACTIVITY = 1.2, 1.2  # ~light activity; tweak if you want  # [dimensionless MET]
DEFAULT_G, G = 9.81, 9.81                   # m/s²

DEFAULT_REFERENCE_CDA_VALUE, REFERENCE_CDA_VALUE = 0.324, 0.324 # cda for cycling [m^2]

DEFAULT_MAX_SPEED, MAX_SPEED = 120.0, 120.0          # km/h (for filters & histograms)
DEFAULT_MAX_GRADE, MAX_GRADE = 35.0, 35.0           # % absolute max grade for filters
DEFAULT_MAX_PWR, MAX_PWR = 1200.0, 1200.0           # W for filters & histograms

DEFAULT_TARGET_CADENCE = 85.   # RPM
DEFAULT_HYSTERESIS_REAR_RPM, HYSTERESIS_REAR_RPM = 15., 15.      # RPM hysteresis for rear shifts
DEFAULT_MIN_SHIFT_INTERVAL_REAR_SEC, MIN_SHIFT_INTERVAL_REAR_SEC = 15., 15.  # s min time between rear shifts

# Speeds used in filtering
DEFAULT_MIN_ACTIVE_SPEED, MIN_ACTIVE_SPEED = 1.0, 1.0     # km/h: above this counts as "active" riding
DEFAULT_MIN_VALID_SPEED, MIN_VALID_SPEED = MIN_ACTIVE_SPEED, MIN_ACTIVE_SPEED     # km/h: below this, samples are discarded

# Limit number of points used in map rendering (performance)
DEFAULT_MAX_MAP_POINTS, MAX_MAP_POINTS = 10000, 10000

# Default drivetrain (used only as fallback; actual gears from UI)
DEFAULT_FRONT_DERAILLEUR_TEETH, FRONT_DERAILLEUR_TEETH = "34, 50", "34, 50"
DEFAULT_FRONT_DERAILLEUR_TEETH, FRONT_DERAILLEUR_TEETH = "28, 46", "28, 46"
DEFAULT_REAR_DERAILLEUR_TEETH, REAR_DERAILLEUR_TEETH = "11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30, 34", "11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30, 34"
DEFAULT_REAR_DERAILLEUR_TEETH, REAR_DERAILLEUR_TEETH = "11,13,15,17,21,24,28,32,36,40", "11,13,15,17,21,24,28,32,36,40"
# Break combining thresholds
DEFAULT_COMBINE_BREAK_MAX_TIME_DIFF, COMBINE_BREAK_MAX_TIME_DIFF = 2.0, 2.0  # Minutes
DEFAULT_COMBINE_BREAK_MAX_DISTANCE, COMBINE_BREAK_MAX_DISTANCE = 100.0, 100.0  # Meters

DEFAULT_CLIMB_MIN_DISTANCE_KM, CLIMB_MIN_DISTANCE_KM = 1., 1.  # km
DEFAULT_CLIMB_MIN_ELEVATION_GAIN_M, CLIMB_MIN_ELEVATION_GAIN_M = 35., 35. # m
DEFAULT_CLIMB_END_AVERAGE_GRADE_PCT, CLIMB_END_AVERAGE_GRADE_PCT = 1., 1.  # %
DEFAULT_CLIMB_END_WINDOW_SIZE_M, CLIMB_END_WINDOW_SIZE_M = 20.0, 20.0  # meters
DEFAULT_MAX_DIST_BETWEEN_CLIMBS_M, MAX_DIST_BETWEEN_CLIMBS_M = 500., 500.  # meters

DEFAULT_BRAKE_DISTRIBUTION_FRONT, BRAKE_DISTRIBUTION_FRONT = 0.6, 0.6          # fraction of total braking power to one disk (assuming two disks)
DEFAULT_BRAKE_FRONT_DIAMETER_MM, BRAKE_FRONT_DIAMETER_MM = 160., 160.              # 160 mm rotor
DEFAULT_BRAKE_REAR_DIAMETER_MM, BRAKE_REAR_DIAMETER_MM = 160., 160.              # 160 mm rotor
DEFAULT_BRAKE_ADD_COOLING_FACTOR, BRAKE_ADD_COOLING_FACTOR = 0.3, 0.3             # empirical cooling factor -> 30% additional cooling area (much higher for icetech rotors)
DEFAULT_BRAKE_PERFORATION_FACTOR, BRAKE_PERFORATION_FACTOR = 0.3, 0.3          # fraction of rotor area that is holes (30% typical)

DEFAULT_SMOOTHING_WINDOW_SIZE_S, SMOOTHING_WINDOW_SIZE_S = 3., 3.  # meters

DEFAULT_REFERENCE_ROLLING_LOSS,REFERENCE_ROLLING_LOSS = 13.7, 13.7 # Continental Grand Prix TR 28; watts according to https://www.bicyclerollingresistance.com calculation based on 18mph, 94lbs
REFERENCE_LOAD_FOR_CRR_KG = 42.6376827 
REFERENCE_SPEED_FOR_CRR_MS = 8.04672

REFERENCE_HEIGHT_FOR_CDA = 1.80 # 1.80 as reference for the cda. so cda can be calculated using the body height

DEFAULT_AMBIENT_WIND_SPEED_MS, AMBIENT_WIND_SPEED_MS = 0., 0.
DEFAULT_AMBIENT_WIND_DIR_DEG, AMBIENT_WIND_DIR_DEG = 0., 0.
DEFAULT_AMBIENT_TEMP_C, AMBIENT_TEMP_C = 20., 20.
DEFAULT_AMBIENT_PRES_HPA, AMBIENT_PRES_HPA = 1013., 1013.
DEFAULT_AMBIENT_RHUM_PCT, AMBIENT_RHUM_PCT = 50., 50.
DEFAULT_AMBIENT_RHO, AMBIENT_RHO = 1.225, 1.225

SEGMENT_MEAN_MODE = False  # global/module-level toggle

details_lines = []

@dataclass
class RiderBikeConfig:
    body_mass: float          # kg
    body_height:float
    bike_mass: float          # kg
    extra_mass: float         # kg
    cda: float                # drag area [m²]
    crr: float                # rolling resistance [-]
    drivetrain_loss: float    # fraction, e.g. 0.04 = 4%
    wheel_circumference: float  # m
    front_der_teeth: list
    rear_der_teeth: list
    draft_effect: float       # dimensionless multiplier (<1 when drafting)
    draft_effect_dh: float       # dimensionless multiplier (<1 when drafting)
    draft_effect_uh: float       # dimensionless multiplier (<1 when drafting)

    @property
    def mass(self) -> float:
        return self.body_mass + self.bike_mass + self.extra_mass


# =====================================================================
# Analysis functions
# =====================================================================


def utc_to_local_from_gps(dt_utc: datetime, lat: float, lon: float, fallback_tz: str = "Europe/Berlin") -> datetime:
    """
    Convert timezone-aware UTC datetime to local time based on GPS position.
    If lookup fails, fall back to fallback_tz.
    """
    if dt_utc.tzinfo is None:
        # treat as UTC if naive
        from datetime import timezone
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)

    tzname = None
    if get_tz is not None:
        try:
            tzname = get_tz(lon, lat)  # note order: (lon, lat)
        except Exception:
            tzname = None

    try:
        print(dt_utc)
        return dt_utc.astimezone(ZoneInfo(tzname or fallback_tz))
    except Exception:
        print(dt_utc)
        return dt_utc

def print(str_):
    details_lines.append(str(str_))

def time_to_readable(t_days=0, t_hours=0, t_minutes=0, t_seconds=0, no_break=False):
    total_seconds = (
        t_seconds +
        t_minutes * 60 +
        t_hours * 3600 +
        t_days * 86400
    )
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    parts = []
    if days > 0:
        if no_break:
            parts.append(f"{int(days)}:")
        else:
            parts.append(f"{int(days)}d")
    if hours > 0 or days > 0:
        if no_break:
            parts.append(f"{int(hours):02d}:")
        else:
            parts.append(f"{int(hours)}h")
    if minutes > 0 or hours > 0 or days > 0:
        if no_break:
            parts.append(f"{int(minutes):02d}:")
        else:
            parts.append(f"{int(minutes)}m")
    if no_break:
        parts.append(f"{int(seconds):02d}")
    else:
        parts.append(f"{int(seconds)}s")
    if no_break and len(parts) > 1:
        return ''.join(parts)
    return '\u00A0'.join(parts)


def hampel_fast(x, window_size=5, n_sigmas=3.0):
    """
    Fast Hampel-like spike removal using SciPy's median_filter.
    - x: 1D array-like
    - window_size: half window (total window = 2*window_size+1)
    - n_sigmas: threshold in robust sigma units
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x

    # rolling median
    win = 2 * window_size + 1
    med = median_filter(x, size=win, mode="nearest")

    # residuals
    diff = x - med

    # rolling MAD (median of |x - med|)
    mad = median_filter(np.abs(diff), size=win, mode="nearest")

    # robust sigma estimate
    sigma = 1.4826 * mad

    # avoid division by zero: where sigma ~0, skip
    with np.errstate(divide="ignore", invalid="ignore"):
        outliers = np.abs(diff) > (n_sigmas * sigma)

    x_filt = x.copy()
    x_filt[outliers] = med[outliers]
    return x_filt

def moving_average(data, window_size):
    data = np.asarray(data, dtype=float)
    if data.size == 0 or window_size <= 1:
        return data.copy()

    # de-spike with fast Hampel
    data_despiked = hampel_fast(
        data,
        window_size=max(1, window_size // 2),
        n_sigmas=3.0,
    )

    n = data_despiked.size
    if n == 0:
        return data_despiked
    if window_size > n:
        window_size = n

    # centered moving average with edge padding instead of zeros
    kernel = np.ones(window_size, dtype=float) / window_size

    # choose padding so that the 'valid' convolution returns the original length
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left  # works for odd/even window_size

    padded = np.pad(
        data_despiked,
        (pad_left, pad_right),
        mode="edge",   # repeat boundary values instead of padding with 0
    )

    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed

def compute_disk_temperature(ambient_temp_c, speed_kph, headwind, brake_power_w, times_s, initial_disk_temp_c=25.0):

    BRAKE_FRONT_DIAMETER_M = BRAKE_FRONT_DIAMETER_MM/1000.               # 160 mm rotor
    BRAKE_REAR_DIAMETER_M = BRAKE_REAR_DIAMETER_MM/1000.               # 160 mm rotor
    disk_specific_heat_j_per_kg_per_K = 460  # J/(kg·K) for aluminum
    disk_density_kg_per_m3 = 7900         # kg/m³ for aluminum
    disk_thickness_m = 0.0018              # 1.8 mm thick
    disk_perforation_factor = 1. - BRAKE_PERFORATION_FACTOR            # fraction of solid area (not holes)
    disk_width_m = 0.030                  # 30 mm wide brake track
    disk_f_cooling_area_m2 = (math.pi * (BRAKE_FRONT_DIAMETER_M / 2)**2 - math.pi * (((BRAKE_FRONT_DIAMETER_M- 2*disk_width_m) / 2))**2) * (disk_perforation_factor + BRAKE_ADD_COOLING_FACTOR)
    disk_f_volume_m3 = disk_f_cooling_area_m2 * disk_thickness_m
    disk_f_mass_kg = disk_density_kg_per_m3 * disk_f_volume_m3
    disk_f_heat_capacity_J_per_K = disk_f_mass_kg * disk_specific_heat_j_per_kg_per_K
    disk_f_convection_W_per_K = lambda x: (6.7194*x + 5.8083) * disk_f_cooling_area_m2 / 0.160 * BRAKE_FRONT_DIAMETER_M * (4-3*disk_perforation_factor)  # from doi:10.3390/proceedings2060215

    disk_r_cooling_area_m2 = (math.pi * (BRAKE_REAR_DIAMETER_M / 2)**2 - math.pi * (((BRAKE_REAR_DIAMETER_M- 2*disk_width_m) / 2))**2) * (disk_perforation_factor + BRAKE_ADD_COOLING_FACTOR)
    disk_r_volume_m3 = disk_r_cooling_area_m2 * disk_thickness_m
    disk_r_mass_kg = disk_density_kg_per_m3 * disk_r_volume_m3
    disk_r_heat_capacity_J_per_K = disk_r_mass_kg * disk_specific_heat_j_per_kg_per_K
    disk_r_convection_W_per_K = lambda x: (6.7194*x + 5.8083) * disk_r_cooling_area_m2 / 0.160 * BRAKE_REAR_DIAMETER_M * (4-3*disk_perforation_factor)  # from doi:10.3390/proceedings2060215

    disk_f_temps_K = [initial_disk_temp_c + 273.15]
    disk_r_temps_K = [initial_disk_temp_c + 273.15]

    speed_kph = moving_average(speed_kph, window_size=10)
    headwind = moving_average(headwind, window_size=10)
    brake_power_w = moving_average(brake_power_w, window_size=10)
    for t_amb, v_kph, v_wind, p_brake_w, dt_s in zip(ambient_temp_c, speed_kph, headwind, brake_power_w, np.diff(times_s, prepend=times_s[0])):
        #front disk
        # p_brake_w = -215 /BRAKE_DISTRIBUTION_FRONT
        # v_kph = 48
        E_disk_J = disk_f_temps_K[-1] * disk_f_heat_capacity_J_per_K
        dT = disk_f_temps_K[-1] - (t_amb + 273.15)
        P_cool_W = disk_f_convection_W_per_K(v_kph/3.6) * dT
        E_disk_J += (-p_brake_w * (BRAKE_DISTRIBUTION_FRONT) - P_cool_W) * dt_s
        T_new_K = E_disk_J / disk_f_heat_capacity_J_per_K
        disk_f_temps_K.append(T_new_K)

        #rear disk
        E_disk_J = disk_r_temps_K[-1] * disk_r_heat_capacity_J_per_K
        dT = disk_r_temps_K[-1] - (t_amb + 273.15)
        P_cool_W = disk_r_convection_W_per_K(v_kph/3.6) * dT
        E_disk_J += (-p_brake_w * (1-BRAKE_DISTRIBUTION_FRONT) - P_cool_W) * dt_s
        T_new_K = E_disk_J / disk_r_heat_capacity_J_per_K
        disk_r_temps_K.append(T_new_K)

    disk_f_temps_C = [t - 273.15 for t in disk_f_temps_K[1:]]
    disk_r_temps_C = [t - 273.15 for t in disk_r_temps_K[1:]]
    return disk_f_temps_C, disk_r_temps_C    


def compute_grade_array(elevations, distances_km):
    grades = []
    for i in range(0, len(elevations)):
        dx = (distances_km[i]) * 1000.0
        dz = elevations[i] - elevations[i - 1]
        grade = dz / dx if dx > 0 else 0.0
        grades.append(grade)
    return grades


def compute_bearing(lat1, lon1, lat2, lon2):
    # all in degrees
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def air_density_from_meteostat(temp_c, pres_hpa, rh_percent):
    """
    Compute air density rho [kg/m³] from Meteostat weather data.
    temp_c: Temperature in °C (array-like)
    pres_hpa: Pressure in hPa (array-like)
    rh_percent: Relative humidity in % (array-like)
    """
    temp_c = np.asarray(temp_c, dtype=float)
    pres_hpa = np.asarray(pres_hpa, dtype=float)
    rh_percent = np.asarray(rh_percent, dtype=float)

    Rd = 287.05      # J/(kg·K)
    Rv = 461.495     # J/(kg·K)

    T = temp_c + 273.15          # K
    p = pres_hpa * 100.0         # hPa -> Pa
    phi = rh_percent / 100.0     # 0..1

    # Saturation vapor pressure (Tetens)
    p_sat = 610.94 * np.exp((17.625 * temp_c) / (temp_c + 243.04))

    p_v = phi * p_sat            # partial pressure water vapor
    p_d = p - p_v                # partial pressure dry air

    rho = (p_d / (Rd * T)) + (p_v / (Rv * T))
    return rho


def get_meteostat_series(latitudes, longitudes, times, use_wind_data=True):
    """
    Returns per-time arrays for the given GPX times:
    dict with keys: 'wspd_ms', 'wdir_deg', 'temp_c', 'pres_hpa', 'rhum_pct', 'rho'
    All arrays have len(times).
    Weather is taken from a single Meteostat point (midpoint of track) but time-aligned.
    """
    idx = pd.to_datetime(times)

    if not use_wind_data:
        print("Wind data disabled, using defaults & standard density.")
        n = len(times)
        return {
            "wspd_ms": np.full(n, AMBIENT_WIND_SPEED_MS),
            "wdir_deg": np.full(n, AMBIENT_WIND_DIR_DEG),
            "temp_c":  np.full(n, AMBIENT_TEMP_C),
            "pres_hpa": np.full(n, AMBIENT_PRES_HPA),
            "rhum_pct": np.full(n, AMBIENT_RHUM_PCT),
            "rho":     np.full(n, AMBIENT_RHO),
        }

    mid_lat = float(np.mean(latitudes))
    mid_lon = float(np.mean(longitudes))

    point = Point(mid_lat, mid_lon)
    start = min(times).replace(tzinfo=None) - timedelta(hours=1)
    end   = max(times).replace(tzinfo=None) + timedelta(hours=1)

    print(f"Fetching Meteostat weather data for {mid_lat:.4f}, {mid_lon:.4f} from {start} to {end}...")

    try:
        data = Hourly(point, start, end)
        df = data.fetch()

        required = ['temp', 'pres', 'rhum', 'wspd', 'wdir']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Meteostat data missing column '{col}'")

        for col in required:
            df[col] = df[col].ffill().bfill()

        df['rho'] = air_density_from_meteostat(df['temp'], df['pres'], df['rhum'])
        df.index = pd.to_datetime(df.index)

        wspd_ms = []
        wdir_deg = []
        temp_c = []
        pres_hpa = []
        rhum_pct = []
        rho = []

        for t in idx:
            if t < df.index[0] or t > df.index[-1]:
                # outside available data range -> fallback
                wspd_ms.append(0.0)
                wdir_deg.append(0.0)
                temp_c.append(15.0)
                pres_hpa.append(1013.0)
                rhum_pct.append(50.0)
                rho.append(1.225)
            else:
                row = df.loc[:t].iloc[-1]
                wspd_ms.append(row['wspd'] / 3.6)  # km/h -> m/s
                wdir_deg.append(row['wdir'])
                temp_c.append(row['temp'])
                pres_hpa.append(row['pres'])
                rhum_pct.append(row['rhum'])
                rho.append(row['rho'])

        return {
            "wspd_ms": np.array(wspd_ms),
            "wdir_deg": np.array(wdir_deg),
            "temp_c":  np.array(temp_c),
            "pres_hpa": np.array(pres_hpa),
            "rhum_pct": np.array(rhum_pct),
            "rho":     np.array(rho),
        }

    except Exception as e:
        print(f"Warning: failed to fetch Meteostat data ({e}). Using defaults.")
        n = len(times)
        return {
            "wspd_ms": np.zeros(n),
            "wdir_deg": np.zeros(n),
            "temp_c":  np.full(n, 20.0),
            "pres_hpa": np.full(n, 1013.0),
            "rhum_pct": np.full(n, 50.0),
            "rho":     np.full(n, 1.225),
        }


def build_gear_list(cfg):
    gear_list = []
    for f in cfg.front_der_teeth:
        for r in cfg.rear_der_teeth:
            gear_list.append({
                'front': f,
                'rear': r,
                'ratio': f / r,
                'name': f"{f}x{r}"
            })
    gear_list.sort(key=lambda x: x['ratio'], reverse=True)
    gear_by_combo = {(g['front'], g['rear']): i for i, g in enumerate(gear_list)}
    return gear_list, gear_by_combo


def find_best_rear_gear(front_teeth, wheel_rpm, target_cadence, rear_teeth_list, gear_by_combo):
    best_rear = None
    best_error = float('inf')
    for rear in rear_teeth_list:
        if (front_teeth, rear) in gear_by_combo:
            cadence = wheel_rpm * (rear / front_teeth)
            error = abs(cadence - target_cadence)
            if error < best_error:
                best_error = error
                best_rear = rear
    return best_rear


def manage_front_gear(current_front, current_rear, wheel_rpm, front_teeth_list, rear_teeth_list):
    rear_idx = rear_teeth_list.index(current_rear)
    at_easy_end = rear_idx >= len(rear_teeth_list) - 1
    at_hard_end = rear_idx <= 0

    if at_easy_end:
        front_idx = front_teeth_list.index(current_front)
        if front_idx > 0:
            return front_teeth_list[front_idx - 1]
    elif at_hard_end:
        front_idx = front_teeth_list.index(current_front)
        if front_idx < len(front_teeth_list) - 1:
            return front_teeth_list[front_idx + 1]
    return current_front


def compute_power(speed, acc, grade, wind_speed, cfg: RiderBikeConfig, rho_air, ride_type):
    """
    speed: m/s, wind_speed: m/s, grade: dz/dx, angles in degrees, rho_air: kg/m³.
    """

    if speed < MIN_VALID_SPEED / 3.6:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    slope_angle = math.atan(grade)
    v_rel = speed + wind_speed

    if ride_type == "climb":
        local_draft_factor = cfg.draft_effect_uh
    elif ride_type == "downhill":
        local_draft_factor = cfg.draft_effect_dh
    else:
        local_draft_factor = cfg.draft_effect


    Fg = cfg.mass * G * math.sin(slope_angle)
    Fr = cfg.mass * G * cfg.crr * math.cos(slope_angle)
    Fd = 0.5 * rho_air * cfg.cda * v_rel**2 * local_draft_factor
    Fa = cfg.mass * acc
    
    P_g = (Fg) * speed
    P_r = (Fr) * speed
    P_d = (Fd) * speed
    P_a = (Fa) * speed
    p = (P_g + P_r + P_d + P_a) / (1.0 - cfg.drivetrain_loss) if (P_g + P_r + P_d + P_a) > 0 else (P_g + P_r + P_d + P_a) / (1.0 - 0.01) # 1% in hubs only full loss only when pedaling
    P_pt = p - (P_g + P_r + P_d + P_a)
    return p, P_g, P_r, P_d, P_a, P_pt


# --------------------------------------------------------------------
# Global switch: which hydration model to use by default
# --------------------------------------------------------------------
HYDRATION_MODEL_MODE = "iso"   # "iso" or "sport"

# --------------------------------------------------------------------
# Parameter sets (tweak as you like)
# --------------------------------------------------------------------

HYDRATION_PARAMS_ISO = {
    # Metabolic / body
    "efficiency": EFFICIENCY,         # mech -> metabolic
    "basal_W": 80.0,            # basic heat production [W]
    "body_height_m": 1.80,      # default height for DuBois area

    # ISO-style limits (DIN EN ISO 7933)
    "swmax_W_m2_acclim": 500.0,  # acclimatized
    "swmax_W_m2_nonacclim": 400.0,
    "dmax_frac_with_water": 0.05,  # 5 % of body mass
    "dmax_frac_no_water": 0.03,    # 3 %

    # Shape of the empirical sweating relation
    # base_lph = a + b * max(M - Met_ref, 0)
    "a_lph": 0.20,              # base sweating at low workload
    "b_lph_per_Wm2": 0.0025,    # slope vs. M [W/m²]
    "Met_ref_Wm2": 80.0,        # reference metabolic rate

    # Temperature factor: 1 + k_T * max(T - T_ref, 0)
    "T_ref_C": 20.0,
    "k_T": 0.05,

    # Humidity factor: 1 + k_RH * max(RH - RH_ref, 0)
    "RH_ref_pct": 40.0,
    "k_RH": 0.01,

    # Air velocity factor: 1 - k_air * max(v_air - v_ref, 0)
    # (higher air speed -> bessere Kühlung -> weniger benötigter Schweiß)
    "v_ref_ms": 0.3,
    "k_air": 0.07,
    "air_factor_min": 0.5,
    "air_factor_max": 1.2,

    # Assume acclimatized worker by default
    "acclim": True,
}

HYDRATION_PARAMS_SPORT = {
    # Start from ISO params and loosen some constraints for Sport
    **HYDRATION_PARAMS_ISO,

    # Sportler können typischerweise mehr schwitzen
    "swmax_W_m2_acclim": 700.0,     # erlaubt höhere Spitzen
    "swmax_W_m2_nonacclim": 550.0,

    # Etwas aggressivere Steigung vs. M
    "a_lph": 0.25,
    "b_lph_per_Wm2": 0.0030,

    # Temperatur- und Feuchte-Sensitivität etwas höher
    "k_T": 0.06,
    "k_RH": 0.012,
}

# --- PHS / ISO 7933 helpers -----------------------------------------------
def _aggregate_to_minutes(times, **series):
    """
    Robustly aggregate arbitrary timestamped data to 1-minute bins.

    - Works even if data is sparse or has long breaks.
    - Ensures a complete sequence t_min = 0..N.
    - Missing-minute samples are filled sensibly:
        * Environmental variables: last known value
        * Metabolic/work: filled with 0 (rest)
    """

    if len(times) == 0:
        out = {"t_min": np.array([], dtype=float)}
        out.update({k: np.array([], dtype=float) for k in series.keys()})
        return out

    # Convert timestamps to seconds since start
    t0 = times[0].timestamp()
    tsec = np.array([t.timestamp() - t0 for t in times], dtype=float)

    # Integer minute index
    minute_idx = np.floor_divide(tsec, 60).astype(int)

    n_min = int(minute_idx.max()) + 1  # covers entire ride duration

    # Initialize output
    out = {"t_min": np.arange(n_min, dtype=float)}

    for name, arr in series.items():
        arr = np.asarray(arr, dtype=float)
        aggregated = np.full(n_min, np.nan, dtype=float)

        for m in range(n_min):
            mask = (minute_idx == m)
            if np.any(mask):
                aggregated[m] = float(np.nanmean(arr[mask]))

        # === FILLING RULES ========

        if name in ("Met_Wm2", "Work_Wm2"):
            # Minutes without riding → at rest → 0
            aggregated = np.nan_to_num(aggregated, nan=0.0)

        elif name in ("Ta_C", "RH_pct", "Va_ms"):
            # Environmental variables → forward-fill them
            valid = np.isfinite(aggregated)
            if not valid.any():
                # If NEVER valid, just fill zeros to avoid NaNs
                aggregated[:] = 0.0
            else:
                # forward fill
                for i in range(1, n_min):
                    if not np.isfinite(aggregated[i]):
                        aggregated[i] = aggregated[i-1]

                # if first minute is NaN, back-fill
                if not np.isfinite(aggregated[0]):
                    # take first finite value
                    first_val = aggregated[np.isfinite(aggregated)][0]
                    aggregated[0] = first_val

        else:
            # All other arrays → forward fill then backfill
            valid = np.isfinite(aggregated)
            if valid.any():
                for i in range(1, n_min):
                    if not np.isfinite(aggregated[i]):
                        aggregated[i] = aggregated[i-1]
                if not np.isfinite(aggregated[0]):
                    aggregated[0] = aggregated[np.isfinite(aggregated)][0]
            else:
                aggregated[:] = 0.0

        out[name] = aggregated

    return out

@dataclass
class PhsClothingParams:
    Icl_clo: float = 0.5      # clothing insulation [clo]
    imst: float = 0.38        # static moisture permeability index
    Ap: float = 0.0           # fraction of body covered by reflective clothing
    Fr: float = 0.42          # reflection coeff of special materials
    Ecl: float = 0.97         # emissivity of clothed surface
    posture: int = 1          # 1 standing, 2 sitting, 3 crouching


def _phs_person_params(body_mass_kg: float, height_m: float, accl: bool, can_drink_freely: bool):
    """
    Derive ISO 7933 person parameters: Adu, aux, SWmax, wmax, Dmax.
    """
    Adu = _body_surface_area_du_bois(body_mass_kg, height_m)  # m²
    # aux: heat for 1°C increase of body per m² (ISO 7933 Annex E)
    aux = 3490.0 * body_mass_kg / Adu

    # max sweat rate (W/m²) and max wettedness
    SWmax = 400.0
    wmax = 0.85
    if accl:
        SWmax = 500.0
        wmax = 1.0

    # max water loss Dmax (g)
    if can_drink_freely:
        Dmax = 0.05 * body_mass_kg * 1000.0
    else:
        Dmax = 0.03 * body_mass_kg * 1000.0

    return Adu, aux, SWmax, wmax, Dmax


def _phs_compute(
    t_min,
    Ta_C,
    RH_pct,
    Va_ms,
    Met_Wm2,
    Work_Wm2,
    body_mass_kg: float,
    height_m: float,
    accl: bool = True,
    can_drink_freely: bool = True,
    clothing: PhsClothingParams | None = None,
):
    """
    Core PHS loop following ISO 7933 Annex A/E.

    All series are 1-D np.arrays on 1-min grid (same length).
    Returns dict of time series (len n_min).
    """
    if clothing is None:
        clothing = PhsClothingParams()

    Ta = np.asarray(Ta_C, dtype=float)
    RH = np.asarray(RH_pct, dtype=float)
    Va = np.asarray(Va_ms, dtype=float)
    Met = np.asarray(Met_Wm2, dtype=float)
    Work = np.asarray(Work_Wm2, dtype=float)
    n = len(t_min)

    Adu, aux, SWmax, wmax, Dmax = _phs_person_params(body_mass_kg, height_m, accl, can_drink_freely)

    # ---- clothing & geometry (mostly constant over time here) ----
    Icl = clothing.Icl_clo
    imst = clothing.imst
    Ap = clothing.Ap
    Fr = clothing.Fr
    Ecl = clothing.Ecl
    posture = clothing.posture

    # effective radiating area Ardu
    if posture == 1:
        Ardu = 0.77
    elif posture == 2:
        Ardu = 0.70
    else:
        Ardu = 0.67

    # static clothing insulation
    Iclst = Icl * 0.155
    fcl = 1.0 + 0.28 * Icl
    Iast = 0.111
    Itotst = Iclst + Iast / fcl

    # we ignore explicit walking speed input; we will derive Walksp from Met as in ISO
    # dynamic clothing insulation etc. depends on Var (relative air speed) and Walksp per minute

    # constants for exponential averaging (ISO)
    ConstTeq = math.exp(-1.0 / 10.0)  # core temp vs Met
    ConstTsk = math.exp(-1.0 / 3.0)   # skin temp
    ConstSW  = math.exp(-1.0 / 10.0)  # sweat rate

    # ---- storage for outputs ----
    Tsk_arr  = np.zeros(n)
    Tcr_arr  = np.zeros(n)
    Tre_arr  = np.zeros(n)
    Tskeq_arr = np.zeros(n)
    Tskeqcl_arr = np.zeros(n)
    Tskeqnu_arr = np.zeros(n)

    Psk_arr  = np.zeros(n)
    Tcl_arr  = np.zeros(n)
    Conv_arr = np.zeros(n)
    Rad_arr  = np.zeros(n)
    Cres_arr = np.zeros(n)
    Eres_arr = np.zeros(n)
    Emax_arr = np.zeros(n)
    Ereq_arr = np.zeros(n)
    wreq_arr = np.zeros(n)

    SWreq_arr = np.zeros(n)
    SWp_arr   = np.zeros(n)
    Ep_arr    = np.zeros(n)
    storage_arr = np.zeros(n)
    SWtotg_arr  = np.zeros(n)

    Dlimloss_arr = np.full(n, np.nan)
    DlimTcr_arr  = np.full(n, np.nan)

    # ---- initial conditions (ISO) ----
    Tre = 36.8
    Tcr = 36.8
    Tsk = 34.1
    Tcreq = 36.8
    TskTcrwg = 0.3
    SWp = 0.0
    SWtot = 0.0
    Dlimtcr = 999.0
    Dlimloss = 999.0

    # we approximate mean radiant temperature as air temperature
    Tr = Ta.copy()

    for i in range(n):
        Ta_i = Ta[i]
        Tr_i = Tr[i]
        RH_i = RH[i]
        Va_i = max(Va[i], 0.0)
        Met_i = Met[i]
        Work_i = Work[i]

        # partial water vapour pressure Pa (kPa) (ISO formula)
        Pa_i = 0.6105 * math.exp(17.27 * Ta_i / (Ta_i + 237.3)) * (RH_i / 100.0)

        # --- dynamic clothing influence (wind & movement) ---
        # walking speed derived from Met (ISO); capped at 0.7 m/s
        Walksp = 0.0052 * (Met_i - 58.0)
        if Walksp < 0.0:
            Walksp = 0.0
        if Walksp > 0.7:
            Walksp = 0.7

        # relative air velocity Var (we keep ISO default: Var = Va if speed undefined)
        Var = Va_i

        Vaux = min(Var, 3.0)
        Waux = min(Walksp, 1.5)

        CORcl = 1.044 * math.exp((0.066 * Vaux - 0.398) * Vaux + (0.094 * Waux - 0.378) * Waux)
        CORcl = min(CORcl, 1.0)
        CORia = math.exp((0.047 * Vaux - 0.472) * Vaux + (0.117 * Waux - 0.342) * Waux)
        CORia = min(CORia, 1.0)

        CORtot = CORcl
        if Icl <= 0.6:
            CORtot = ((0.6 - Icl) * CORia + Icl * CORcl) / 0.6

        Itotdyn = Itotst * CORtot
        Iadyn = CORia * Iast
        Icldyn = Itotdyn - Iadyn / fcl

        CORe = (2.6 * CORtot - 6.5) * CORtot + 4.9
        imdyn = imst * CORe
        if imdyn > 0.9:
            imdyn = 0.9
        Rtdyn = Itotdyn / imdyn / 16.7  # evaporative resistance

        # save previous values for this minute
        Tre0 = Tre
        Tcr0 = Tcr
        Tsk0 = Tsk
        Tcreq0 = Tcreq
        TskTcrwg0 = TskTcrwg

        # --- equilibrium core temp associated with Met ---
        Tcreqm = 0.0036 * Met_i + 36.6
        Tcreq = Tcreq0 * ConstTeq + Tcreqm * (1.0 - ConstTeq)

        # heat storage due to change in core-equilibrium temp (no sweat yet)
        dStoreq = aux / 60.0 * (Tcreq - Tcreq0) * (1.0 - TskTcrwg0)

        # --- skin temp prediction (equilibrium & exponential averaging) ---
        # clothed model
        Tskeqcl = (
            12.165
            + 0.02017 * Ta_i
            + 0.04361 * Tr_i
            + 0.19354 * Pa_i
            - 0.25315 * Va_i
        )
        Tskeqcl += 0.005346 * Met_i + 0.51274 * Tre

        # nude model
        Tskeqnu = (
            7.191
            + 0.064 * Ta_i
            + 0.061 * Tr_i
            + 0.198 * Pa_i
            - 0.348 * Va_i
        )
        Tskeqnu += 0.616 * Tre

        if Icl >= 0.6:
            Tskeq = Tskeqcl
        elif Icl <= 0.2:
            Tskeq = Tskeqnu
        else:
            # interpolate 0.2 < clo < 0.6
            Tskeq = Tskeqnu + 2.5 * (Tskeqcl - Tskeqnu) * (Icl - 0.2)

        if i == 0:
            Tsk = Tskeq  # first minute
        else:
            Tsk = Tsk0 * ConstTsk + Tskeq * (1.0 - ConstTsk)

        # saturated vapour pressure at skin surface (kPa)
        Psk = 0.6105 * math.exp(17.27 * Tsk / (Tsk + 237.3))

        # --- clothing surface temperature Tcl (iterative) ---
        Z = 3.5 + 5.2 * Var
        if Var > 1.0:
            Z = 8.7 * Var**0.6

        auxR = 5.67e-8 * Ardu
        Eclr = (1.0 - Ap) * Ecl + Ap * (1.0 - Fr)

        Tcl = Tr_i + 0.1  # initial guess


        Tcl = Tr_i + 0.1  # initial guess
        for _ in range(50):
            # convection coefficient Hc
            Hc = 2.38 * abs(Tcl - Ta_i) ** 0.25
            if Z > Hc:
                Hc = Z

            # radiation coefficient HR
            if abs(Tcl - Tr_i) < 1e-6:
                HR = 4.0 * Eclr * auxR * (Tcl + 273.0) ** 3
            else:
                HR = (
                    Eclr
                    * auxR
                    * ((Tcl + 273.0) ** 4 - (Tr_i + 273.0) ** 4)
                    / (Tcl - Tr_i)
                )

            Tcl1 = (
                (fcl * (Hc * Ta_i + HR * Tr_i) + Tsk / Icldyn)
                / (fcl * (Hc + HR) + 1.0 / Icldyn)
            )

            # guard against NaN/inf
            if not np.isfinite(Tcl1):
                Tcl1 = Tcl
                break

            if abs(Tcl1 - Tcl) <= 1e-3:
                Tcl = Tcl1
                break

            Tcl = 0.5 * (Tcl + Tcl1)
        else:
            # did not converge in 50 iters; best effort
            # Tcl is whatever last value we had
            pass



        # --- heat exchanges ---
        Texp = 28.56 + 0.115 * Ta_i + 0.641 * Pa_i
        Cres = 0.001516 * Met_i * (Texp - Ta_i)
        Eres = 0.00127 * Met_i * (59.34 + 0.53 * Ta_i - 11.63 * Pa_i)

        Conv = fcl * Hc * (Tcl - Ta_i)
        Rad = fcl * HR * (Tcl - Tr_i)
        Emax = (Psk - Pa_i) / Rtdyn if Rtdyn > 0 else 0.0

        # required evaporation rate for thermal balance
        Ereq = Met_i - dStoreq - Work_i - Cres - Eres - Conv - Rad

        # required wettedness
        wreq = Ereq / Emax if Emax > 0 else 0.0

        # --- required sweat rate SWreq (with all special cases) ---
        if Ereq <= 0:
            Ereq = 0.0
            SWreq = 0.0
        elif Emax <= 0:
            Emax = 0.0
            SWreq = SWmax
        elif wreq >= 1.7:
            wreq = 1.7
            SWreq = SWmax
        else:
            Eveff = 1.0 - 0.5 * (wreq ** 2)
            if wreq > 1.0:
                Eveff = 0.5 * (2.0 - wreq) ** 2
            SWreq = Ereq / Eveff
            if SWreq > SWmax:
                SWreq = SWmax

        # predicted sweat rate (exponential averaging)
        SWp = SWp * ConstSW + SWreq * (1.0 - ConstSW)

        # predicted evaporation Ep
        if SWp <= 0:
            SWp = 0.0
            Ep = 0.0
        else:
            k = Emax / SWp if SWp > 0 else 0.0
            wp = 1.0
            if k >= 0.5:
                wp = -k + math.sqrt(k * k + 2.0)
            if wp > wmax:
                wp = wmax
            Ep = wp * Emax

        # total heat storage this minute
        dStorage = Ereq - Ep + dStoreq

        # --- core temperature from heat storage (iterative) ---
        Tcr1 = Tcr0
        for _ in range(50):
            # skin-core weighting
            TskTcrwg = 0.3 - 0.09 * (Tcr1 - 36.8)
            if TskTcrwg > 0.3:
                TskTcrwg = 0.3
            if TskTcrwg < 0.1:
                TskTcrwg = 0.1

            Tcr_tmp = dStorage / (aux / 60.0) + Tsk0 * TskTcrwg0 / 2.0 - Tsk * TskTcrwg / 2.0
            Tcr_tmp = (Tcr_tmp + Tcr0 * (1.0 - TskTcrwg0 / 2.0)) / (1.0 - TskTcrwg / 2.0)

            # guard against NaN/inf
            if not np.isfinite(Tcr_tmp):
                Tcr_tmp = Tcr1
                break

            if abs(Tcr_tmp - Tcr1) <= 1e-3:
                Tcr = Tcr_tmp
                break

            Tcr1 = 0.5 * (Tcr1 + Tcr_tmp)
        else:
            # no convergence; just take the last iterate
            Tcr = Tcr1






        # rectal temp
        Tre = Tre0 + (2.0 * Tcr - 1.962 * Tre0 - 1.31) / 9.0

        # cumulative water loss
        SWtot = SWtot + SWp + Eres     # W/m²
        # SWtotg: total water loss in grams
        SWtotg = SWtot * 2.67 * Adu / 1.8 / 60.0

        # duration limits
        if Dlimloss == 999.0 and SWtotg >= Dmax:
            Dlimloss = t_min[i]  # min since start
        if Dlimtcr == 999.0 and Tre >= 38.0:
            Dlimtcr = t_min[i]

        # ---- store in arrays ----
        Tsk_arr[i] = Tsk
        Tcr_arr[i] = Tcr
        Tre_arr[i] = Tre
        Tskeq_arr[i] = Tskeq
        Tskeqcl_arr[i] = Tskeqcl
        Tskeqnu_arr[i] = Tskeqnu

        Psk_arr[i] = Psk
        Tcl_arr[i] = Tcl
        Conv_arr[i] = Conv
        Rad_arr[i] = Rad
        Cres_arr[i] = Cres
        Eres_arr[i] = Eres
        Emax_arr[i] = Emax
        Ereq_arr[i] = Ereq
        wreq_arr[i] = wreq

        SWreq_arr[i] = SWreq
        SWp_arr[i] = SWp
        Ep_arr[i] = Ep
        storage_arr[i] = dStorage
        SWtotg_arr[i] = SWtotg
        Dlimloss_arr[i] = Dlimloss
        DlimTcr_arr[i] = Dlimtcr

    return {
        "t_min": t_min,
        "Tsk_C": Tsk_arr,
        "Tcr_C": Tcr_arr,
        "Tre_C": Tre_arr,
        "Tskeq_C": Tskeq_arr,
        "Tskeq_clothed_C": Tskeqcl_arr,
        "Tskeq_nude_C": Tskeqnu_arr,
        "Psk_kPa": Psk_arr,
        "Tcl_C": Tcl_arr,
        "Conv_Wm2": Conv_arr,
        "Rad_Wm2": Rad_arr,
        "Cres_Wm2": Cres_arr,
        "Eres_Wm2": Eres_arr,
        "Emax_Wm2": Emax_arr,
        "Ereq_Wm2": Ereq_arr,
        "wreq": wreq_arr,
        "SWreq_Wm2": SWreq_arr,
        "SWp_Wm2": SWp_arr,
        "Ep_Wm2": Ep_arr,
        "storage_Wm2": storage_arr,
        "SWtot_g": SWtotg_arr,
        "Dlimloss_min": Dlimloss_arr,
        "DlimTcr_min": DlimTcr_arr,
        "Adu_m2": Adu,
    }


def _body_surface_area_du_bois(weight_kg: float, height_m: float) -> float:
    """
    DuBois-Fläche (m²), wie in vielen Normen (ISO 8996 etc.).
    """
    return 0.202 * (weight_kg ** 0.425) * (height_m ** 0.725)


def _metabolic_rate_W_per_m2(
    avg_power_w: float,
    body_mass_kg: float,
    params: dict,
) -> tuple[float, float]:
    """
    Aus mechanischer Leistung und Grundumsatz eine metabolische Rate pro Fläche ableiten.

    Returns:
        (Met_W_m2, ADu_m2)
    """
    height_m = params.get("body_height_m", 1.80)
    eff = params.get("efficiency", 0.23)
    basal_W = params.get("basal_W", 80.0)

    adu = _body_surface_area_du_bois(body_mass_kg, height_m)
    total_metabolic_W = max(avg_power_w, 0.0) / eff + basal_W
    Met = total_metabolic_W / max(adu, 1e-6)
    return Met, adu


def compute_phs_timeseries_from_filtered(filtered, cfg, mode: str = HYDRATION_MODEL_MODE, can_drink_freely: bool = True):
    """
    Build ISO 7933 PHS time series from already-filtered ride data.

    Uses:
      filtered["filtered_times"]
      filtered["filtered_temp"] (°C)
      filtered["filtered_rhum"] (%)
      filtered["filtered_speeds_kph"]
      filtered["filtered_head_winds"] (m/s, signed)
      filtered["filtered_powers"] (W)

    cfg: RiderBikeConfig (must have cfg.body_mass; height is taken from hydration params).
    """
    times = np.asarray(filtered["filtered_times"], dtype=object)
    temp_C = np.asarray(filtered["filtered_temp"], dtype=float)
    rhum_pct = np.asarray(filtered["filtered_rhum"], dtype=float)
    spd_kph = np.asarray(filtered["filtered_speeds_kph"], dtype=float)
    head_ms = np.asarray(filtered["filtered_head_winds"], dtype=float)
    pow_W = np.asarray(filtered["filtered_powers"], dtype=float)

    # relative air speed (simplified): riding speed + headwind component
    v_rel_ms = np.maximum(0.0, spd_kph / 3.6 + head_ms)

    # person parameters
    params = HYDRATION_PARAMS_ISO if mode == "iso" else HYDRATION_PARAMS_SPORT
    height_m = params.get("body_height_m", 1.80)
    eff = params.get("efficiency", EFFICIENCY)
    basal_W = params.get("basal_W", 80.0)

    Adu = _body_surface_area_du_bois(cfg.body_mass, height_m)

    # per-sample metabolic and mechanical rates per m²
    P_pos = np.clip(pow_W, 0.0, None)
    metabolic_W = P_pos / eff + basal_W
    Met_Wm2 = metabolic_W / Adu
    Work_Wm2 = P_pos / Adu

    # aggregate to 1-min bins
    agg = _aggregate_to_minutes(
        times,
        Ta_C=temp_C,
        RH_pct=rhum_pct,
        Va_ms=v_rel_ms,
        Met_Wm2=Met_Wm2,
        Work_Wm2=Work_Wm2,
    )

    accl = params.get("acclim", True)

    phs = _phs_compute(
        t_min=agg["t_min"],
        Ta_C=agg["Ta_C"],
        RH_pct=agg["RH_pct"],
        Va_ms=agg["Va_ms"],
        Met_Wm2=agg["Met_Wm2"],
        Work_Wm2=agg["Work_Wm2"],
        body_mass_kg=cfg.body_mass,
        height_m=height_m,
        accl=accl,
        can_drink_freely=can_drink_freely,
    )

    # derive a sweat *volume* rate [L/h] from SWp_Wm2
    # 1 W/m² -> 1.47 g/(m²·h); Adri ~ Adu m² -> grams/h -> L/h
    SWp_L_per_h = phs["SWp_Wm2"] * 1.47 * phs["Adu_m2"] / 1000.0

    phs["SWp_L_per_h"] = SWp_L_per_h
    return phs

def _resample_phs_to_filtered(phs, filtered_times):
    """
    Resample 1-min PHS outputs onto per-sample filtered_times.

    phs: dict returned by _phs_compute / compute_phs_timeseries_from_filtered
    filtered_times: np.ndarray of datetime objects (same as filtered["filtered_times"])

    Returns a dict with the same keys as phs, but resampled
    (only for scalar time series like Tcr, Tsk, etc.).
    """
    ft = np.asarray(filtered_times, dtype=object)
    if ft.size == 0 or phs["t_min"].size == 0:
        return {}

    t0 = ft[0].timestamp()
    tsec_filtered = np.array([t.timestamp() for t in ft], dtype=float)
    tsec_phs = t0 + phs["t_min"] * 60.0  # 1 min steps from same start

    def _interp_series(key):
        arr = np.asarray(phs[key], dtype=float)
        if arr.size < 2:
            # nothing to interpolate properly; just repeat or fill
            return np.full_like(tsec_filtered, arr[0] if arr.size == 1 else np.nan, dtype=float)
        return np.interp(tsec_filtered, tsec_phs, arr)

    keys_to_resample = [
        "Tsk_C", "Tcr_C", "Tre_C",
        "SWp_Wm2", "SWp_L_per_h",
        "Ereq_Wm2", "Ep_Wm2", "Emax_Wm2",
        "Conv_Wm2", "Rad_Wm2", "Cres_Wm2", "Eres_Wm2",
        "storage_Wm2",
    ]

    out = {}
    for k in keys_to_resample:
        if k in phs:
            out[k + "_aligned"] = _interp_series(k)

    return out


def estimate_water_loss_rate_lph(
    temp_c: float,
    rh_percent: float,
    avg_speed_ms: float,
    avg_headwind_ms: float,
    avg_power_w: float,
    body_mass_kg: float,
    mode: str = HYDRATION_MODEL_MODE,
    can_drink_freely: bool = True,
) -> float:
    """
    ISO- oder Sport-Variante der Schweißrate [L/h].

    mode: "iso" -> streng an ISO-Grenzen (SWmax, Dmax) angelehnt
          "sport" -> sportangepasste Parameter
    """
    params = HYDRATION_PARAMS_ISO if mode == "iso" else HYDRATION_PARAMS_SPORT

    # 1) Metabolische Rate
    Met, adu = _metabolic_rate_W_per_m2(avg_power_w, body_mass_kg, params)

    # 2) Baseline vs. M (einfaches lineares Modell)
    a_lph = params["a_lph"]
    b_lph = params["b_lph_per_Wm2"]
    Met_ref = params["Met_ref_Wm2"]
    Met_eff = max(Met - Met_ref, 0.0)
    base_lph = a_lph + b_lph * Met_eff

    # 3) Umweltfaktoren
    T_ref = params["T_ref_C"]
    k_T = params["k_T"]
    temp_factor = 1.0 + k_T * max(temp_c - T_ref, 0.0)

    RH_ref = params["RH_ref_pct"]
    k_RH = params["k_RH"]
    rh_percent = np.clip(rh_percent, 0.0, 100.0)
    hum_factor = 1.0 + k_RH * max(rh_percent - RH_ref, 0.0)

    v_ref = params["v_ref_ms"]
    k_air = params["k_air"]
    v_air = max(avg_speed_ms + avg_headwind_ms, 0.0)
    air_factor = 1.0 - k_air * max(v_air - v_ref, 0.0)
    air_factor = np.clip(
        air_factor,
        params["air_factor_min"],
        params["air_factor_max"],
    )

    water_lph = base_lph * temp_factor * hum_factor * air_factor

    # 4) SWmax aus DIN: maximale Schweißrate
    acclim = params.get("acclim", True)
    SWmax_Wm2 = params["swmax_W_m2_acclim"] if acclim else params["swmax_W_m2_nonacclim"]

    # 1 W/m² -> 1.47 g/(m²·h); total g/h -> L/h
    grams_per_hour = SWmax_Wm2 * 1.47 * adu
    SWmax_lph = grams_per_hour / 1000.0

    water_lph = float(np.clip(water_lph, 0.0, SWmax_lph))

    # optional: Dmax (max. totaler Verlust) kannst du später mit der Zeitreihe checken

    return water_lph


def max_allowed_water_loss_l(
    body_mass_kg: float,
    mode: str = HYDRATION_MODEL_MODE,
    can_drink_freely: bool = True,
) -> float:
    """
    Dmax gemäß DIN:
    - 5 % Körpermasse mit Wasser
    - 3 % ohne Wasser
    (Sport-Modell nutzt gleiche Logik, aber du kannst die Fraktionen in den Parametern ändern.)
    """
    params = HYDRATION_PARAMS_ISO if mode == "iso" else HYDRATION_PARAMS_SPORT
    frac = params["dmax_frac_with_water"] if can_drink_freely else params["dmax_frac_no_water"]
    return body_mass_kg * frac  # [kg ≈ L]


def compute_water_loss_timeseries(
    filtered_temp_c,
    filtered_rhum_pct,
    filtered_speeds_kph,
    filtered_headwind_ms,
    filtered_powers_w,
    filtered_times,
    body_mass_kg,
    mode: str = HYDRATION_MODEL_MODE,
    can_drink_freely: bool = True,
):
    """
    Zeitreihe der Schweißrate [L/h] und kumulativen Wasserverluste [L].

    Rückgabe:
        water_rate_L_per_h : np.ndarray (len N)
        cum_water_loss_L   : np.ndarray (len N)
        dmax_L             : float (Grenze nach DIN / Parametern)
    """
    n = len(filtered_times)
    if n == 0:
        return np.array([]), np.array([]), 0.0

    temp_c = np.asarray(filtered_temp_c, dtype=float)
    rh = np.asarray(filtered_rhum_pct, dtype=float)
    spd_kph = np.asarray(filtered_speeds_kph, dtype=float)
    headwind = np.asarray(filtered_headwind_ms, dtype=float)
    power = np.asarray(np.clip(filtered_powers_w,0.0,None), dtype=float)
    spd_ms = spd_kph / 3.6

    # dt [s] pro Sample
    tsec = np.array([t.timestamp() for t in filtered_times], dtype=float)
    if n > 1:
        dt = np.diff(tsec)
        dt0 = float(np.median(dt)) if dt.size > 0 else 1.0
        dt = np.concatenate(([dt0], dt))
    else:
        dt = np.array([1.0])

    water_rate = np.empty(n, dtype=float)

    for i in range(n):
        if dt[i] > 30.:
            pwr = 0
        else:
            pwr = power[i]
        water_rate[i] = estimate_water_loss_rate_lph(
            temp_c=temp_c[i],
            rh_percent=rh[i],
            avg_speed_ms=spd_ms[i],
            avg_headwind_ms=headwind[i],
            avg_power_w=pwr,
            body_mass_kg=body_mass_kg,
            mode=mode,
            can_drink_freely=can_drink_freely,
        )

    # [L/h] -> [L] pro Step
    step_loss_L = water_rate * dt / 3600.0
    cum_water_L = np.cumsum(step_loss_L)

    dmax_L = max_allowed_water_loss_l(body_mass_kg, mode=mode, can_drink_freely=can_drink_freely)

    # Optional: harte Begrenzung
    # cum_water_L = np.minimum(cum_water_L, dmax_L)

    return water_rate, cum_water_L, dmax_L


def _extract_point_sensors(point):
    """
    Try to extract heart rate [bpm], cadence [rpm], power [W] and temperature [°C]
    from typical Garmin / GPX extension tags.
    Returns (hr, cad, power, temp) where each can be float or np.nan if missing.
    """
    hr = np.nan
    cad = np.nan
    pwr = np.nan
    temp = np.nan

    exts = getattr(point, "extensions", None)
    if not exts:
        return hr, cad, pwr, temp

    for ext in exts:
        try:
            for child in ext.iter():
                if child.text is None:
                    continue
                txt = child.text.strip()
                if not txt:
                    continue

                tag = child.tag
                if '}' in tag:
                    tag = tag.split('}', 1)[1]
                tag = tag.lower()

                if tag in ("hr", "gpxtpx:hr"):
                    try:
                        hr = float(txt)
                    except ValueError:
                        pass
                elif tag in ("cad", "gpxtpx:cad"):
                    try:
                        cad = float(txt)
                    except ValueError:
                        pass
                elif tag in ("power", "gpxtpx:power", "gpxdata:power"):
                    try:
                        pwr = float(txt)
                    except ValueError:
                        pass
                elif tag in ("atemp", "temp", "gpxtpx:atemp", "gpxdata:temp"):
                    try:
                        temp = float(txt)
                    except ValueError:
                        pass
        except Exception:
            continue

    return hr, cad, pwr, temp


def _parse_and_interpolate_gpx(gpx_path, progress_cb=None):
    """Parse GPX, extract sensors, resample to 1 Hz, and build distance arrays."""
    def _report(pct, text):
        if progress_cb is not None:
            progress_cb(int(pct), text)

    _report(0, "Parsing GPX file...")
    print(f"\n=== Analyzing file: {gpx_path} ===")

    with open(gpx_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    lats, lons, raw_alts, dists, times, speeds = [], [], [], [], [], []
    hr_raw, cad_raw, pwr_raw, temp_dev_raw = [], [], [], []
    prev_point = None
    prev_time = None

    # --- NEW: timezone-fix helpers/state (fixed) ---
    times_utc_used = []          # authoritative timeline for computations
    times_local_display = []     # only for UI / printing (optional)

    track_tzinfo = None          # determine once for whole track
    prev_time_utc = None
    # ----------------------------------------------

    for track in gpx.tracks:
        for segment in track.segments:
            for i, point in enumerate(segment.points):
                if point.time is None:
                    continue

                lats.append(point.latitude)
                lons.append(point.longitude)
                raw_alts.append(point.elevation)

                # ---------- time handling (robust) ----------
                dt = point.time

                # normalize to aware UTC
                if dt.tzinfo is None:
                    dt_utc = dt.replace(tzinfo=timezone.utc)
                else:
                    dt_utc = dt.astimezone(timezone.utc)

                # enforce strictly increasing UTC (protect speed/resampling)
                if prev_time_utc is not None and dt_utc <= prev_time_utc:
                    dt_utc = prev_time_utc + timedelta(seconds=1)

                prev_time_utc = dt_utc
                times_utc_used.append(dt_utc)

                # what you store in `times` should be UTC for all calculations
                times.append(dt_utc)

                # optional: compute local time for display only
                if FIX_TIME_ZONE:
                    try:
                        if track_tzinfo is None:
                            # determine tz once from first valid point/time
                            dt_local_first = utc_to_local_from_gps(
                                dt_utc, float(point.latitude), float(point.longitude)
                            )
                            track_tzinfo = dt_local_first.tzinfo

                        if track_tzinfo is not None:
                            times_local_display.append(dt_utc.astimezone(track_tzinfo))
                        else:
                            times_local_display.append(dt_utc)
                    except Exception:
                        times_local_display.append(dt_utc)
                # --------------------------------------------

                hr_v, cad_v, pwr_v, temp_v = _extract_point_sensors(point)
                hr_raw.append(hr_v)
                cad_raw.append(cad_v)
                pwr_raw.append(pwr_v)
                temp_dev_raw.append(temp_v)

                if prev_point:
                    dist = geodesic(
                        (prev_point.latitude, prev_point.longitude),
                        (point.latitude, point.longitude)
                    ).meters

                    # IMPORTANT: always use UTC for dt
                    time_diff = (times[-1] - prev_time).total_seconds()
                    if time_diff <= 0:
                        time_diff = 1.0  # safety; should be rare due to monotonic clamp

                    speed = dist / time_diff
                    speeds.append(speed * 3.6)
                    dists.append(dist / 1000.0)
                else:
                    speeds.append(0.0)
                    dists.append(0.0)

                prev_point = point
                prev_time = times[-1]  # UTC


    cut_beginning = 0
    cut_end = 0
    # store both for later use
    # times = times_local_display if FIX_TIME_ZONE else None
    lats = lats[cut_beginning:cut_end-1]
    lons = lons[cut_beginning:cut_end-1]
    raw_alts = raw_alts[cut_beginning:cut_end-1]
    times = times[cut_beginning:cut_end-1]
    times_local_display = times_local_display[cut_beginning:cut_end-1]
    hr_raw = hr_raw[cut_beginning:cut_end-1]
    cad_raw = cad_raw[cut_beginning:cut_end-1]
    pwr_raw = pwr_raw[cut_beginning:cut_end-1]
    temp_dev_raw = temp_dev_raw[cut_beginning:cut_end-1]
    speeds = speeds[cut_beginning:cut_end-1]
    dists = dists[cut_beginning:cut_end-1]
    


    if len(times) == 0:
        raise ValueError("No time data found in GPX file.")

    times = [t.replace(tzinfo=None) for t in times]
    start_time = times[0]
    end_time = times[-1]
    _report(15, "Interpolating & smoothing...")

    if len(times) > 1:
        total_seconds = int((end_time - start_time).total_seconds())
        new_times = [start_time + timedelta(seconds=i) for i in range(total_seconds + 1)]
        timestamps = np.array([t.timestamp() for t in times])
        new_timestamps = np.array([t.timestamp() for t in new_times])

        lats = np.interp(new_timestamps, timestamps, lats)
        lons = np.interp(new_timestamps, timestamps, lons)
        raw_alts = np.interp(new_timestamps, timestamps, raw_alts)
        speeds = np.interp(new_timestamps, timestamps, speeds)

        def _interp_sensor(raw_list):
            arr = np.asarray(raw_list, dtype=float)
            mask = np.isfinite(arr)
            if np.count_nonzero(mask) < 2:
                return np.full_like(new_timestamps, np.nan, dtype=float)
            return np.interp(new_timestamps, timestamps[mask], arr[mask])

        hr_series = _interp_sensor(hr_raw)
        cad_series = _interp_sensor(cad_raw)
        pow_series = _interp_sensor(pwr_raw)
        temp_dev_series = _interp_sensor(temp_dev_raw)

        times = new_times

        total_dists = []
        total_dist = 0.0
        for d in dists:
            total_dist += d
            total_dists.append(total_dist)
        total_dists = np.interp(new_timestamps, timestamps, total_dists)
        dists = []
        last_dist = total_dists[0]
        for d in total_dists:
            dists.append(d - last_dist)
            last_dist = d
    else:
        hr_series = np.array(hr_raw, dtype=float)
        cad_series = np.array(cad_raw, dtype=float)
        pow_series = np.array(pwr_raw, dtype=float)
        temp_dev_series = np.array(temp_dev_raw, dtype=float)
        total_dists = np.array(dists, dtype=float)

    # timestep in minutes
    timestep_min = np.median(np.diff([t.timestamp() for t in times])) / 60.0

    return {
        "gpx_path": gpx_path,
        "lats": np.array(lats),
        "lons": np.array(lons),
        "raw_alts": np.array(raw_alts),
        "dists": np.array(dists),
        "total_dists": np.array(total_dists),
        "times": np.array(times, dtype=object),
        "speeds_kph": np.array(speeds),
        "hr_series": np.array(hr_series),
        "cad_series": np.array(cad_series),
        "pow_series": np.array(pow_series),
        "temp_dev_series": np.array(temp_dev_series),
        "start_time": start_time,
        "end_time": end_time,
        "timestep_min": timestep_min,
        "times_local_display": times_local_display,
    }


def _smooth_and_grades(core):
    """Apply smoothing and compute grades.
    
    core:
        "gpx_path": gpx_path,
        "lats": np.array(lats),
        "lons": np.array(lons),
        "raw_alts": np.array(raw_alts),
        "dists": np.array(dists),
        "total_dists": np.array(total_dists),
        "times": np.array(times, dtype=object),
        "speeds_kph": np.array(speeds),
        "hr_series": np.array(hr_series),
        "cad_series": np.array(cad_series),
        "pow_series": np.array(pow_series),
        "temp_dev_series": np.array(temp_dev_series),
        "start_time": start_time,
        "end_time": end_time,
        "timestep_min": timestep_min,
    
    """

    
    speeds_kph = core["speeds_kph"]
    raw_alts = core["raw_alts"]
    dists = core["dists"]
    total_dists = np.cumsum(dists)

    smoothed_total_dists = moving_average(total_dists, window_size=SMOOTHING_WINDOW_SIZE_S)
    smoothed_speeds_kph = moving_average(speeds_kph, window_size=SMOOTHING_WINDOW_SIZE_S)
    smoothed_alts = moving_average(raw_alts, window_size=SMOOTHING_WINDOW_SIZE_S)
    smoothed_speeds_ms = smoothed_speeds_kph / 3.6

    grades = compute_grade_array(smoothed_alts, dists)
    smoothed_grades = moving_average(grades, window_size=SMOOTHING_WINDOW_SIZE_S)
    grades_array = np.array(smoothed_grades)

    return {
        "smoothed_speeds_kph": smoothed_speeds_kph,
        "smoothed_speeds_ms": smoothed_speeds_ms,
        "smoothed_alts": smoothed_alts,
        "grades": grades,
        "dists": dists,
        "smoothed_grades": smoothed_grades,
        "grades_array": grades_array,
        "total_dists": smoothed_total_dists,
        "times": core["times"]
    }


def _weather_and_air_density(core, use_wind_data, progress_cb=None):
    """Fetch Meteostat weather along route and compute air density."""
    def _report(pct, text):
        if progress_cb is not None:
            progress_cb(int(pct), text)

    _report(35, "Fetching weather data...")

    lats = core["lats"]
    lons = core["lons"]
    times = core["times"]

    weather = get_meteostat_series(lats, lons, times, use_wind_data=use_wind_data)
    return weather


def _headwind_and_power(core, smooth, ride_type_series, weather, cfg, progress_cb=None):
    """Compute headwind, acceleration, power, smoothed power, and brake power."""
    def _report(pct, text):
        if progress_cb is not None:
            progress_cb(int(pct), text)

    _report(55, "Computing headwind & power...")

    lats = core["lats"]
    lons = core["lons"]
    times = core["times"]

    smoothed_speeds_ms = smooth["smoothed_speeds_ms"]
    smoothed_speeds_kph = smooth["smoothed_speeds_kph"]
    smoothed_grades = smooth["smoothed_grades"]

    wind_speeds_ms = weather["wspd_ms"]
    wind_dirs_deg = weather["wdir_deg"]
    rho_series = weather["rho"]

    powers = []
    powers_details = {"slope": [], "roll": [], "drag": [], "acc": [], "pt": []}
    headwinds = []
    accelerations = []
    bearing = 0.0

    for i in range(1, len(smoothed_speeds_ms)):
        lat1, lon1 = lats[i - 1], lons[i - 1]
        lat2, lon2 = lats[i], lons[i]
        bearing = compute_bearing(lat1, lon1, lat2, lon2)

        wind_speed = wind_speeds_ms[i]
        wind_dir = wind_dirs_deg[i]

        theta = math.radians((wind_dir - bearing) % 360)
        headwind = wind_speed * math.cos(theta)
        headwinds.append(headwind)

        dv = smoothed_speeds_ms[i] - smoothed_speeds_ms[i - 1]
        dt = times[i].timestamp() - times[i - 1].timestamp()
        acceleration = dv / dt if dt > 0 else 0.0
        accelerations.append(acceleration)

    if len(headwinds) == 0:
        headwinds = [0.0]
    headwinds = [headwinds[0]] + headwinds
    accelerations = [0] + accelerations

    smoothed_headwinds = moving_average(headwinds, window_size=SMOOTHING_WINDOW_SIZE_S)

    for i in range(1, len(smoothed_speeds_ms)):
        p, p_g, p_r, p_d, p_a, p_pt = compute_power(
            smoothed_speeds_ms[i],
            accelerations[i],
            smoothed_grades[i],
            smoothed_headwinds[i],
            cfg,
            rho_series[i],
            ride_type_series[i]
        )
        powers.append(p)
        powers_details["slope"].append(p_g)
        powers_details["roll"].append(p_r)
        powers_details["drag"].append(p_d)
        powers_details["acc"].append(p_a)
        powers_details["pt"].append(p_pt)

    if len(powers) == 0:
        powers = [0.0]
        powers_details["slope"] = [0.0]
        powers_details["roll"] = [0.0]
        powers_details["drag"] = [0.0]
        powers_details["acc"] = [0.0]
        powers_details["pt"] = [0.0]

    powers = [powers[0]] + powers
    for k in powers_details:
        powers_details[k] = [powers_details[k][0]] + powers_details[k]

    smoothed_powers = moving_average(powers, window_size=SMOOTHING_WINDOW_SIZE_S)

    smoothed_powers_details = {
        "slope": moving_average(powers_details["slope"], window_size=SMOOTHING_WINDOW_SIZE_S),
        "roll": moving_average(powers_details["roll"], window_size=SMOOTHING_WINDOW_SIZE_S),
        "drag": moving_average(powers_details["drag"], window_size=SMOOTHING_WINDOW_SIZE_S),
        "acc": moving_average(powers_details["acc"], window_size=SMOOTHING_WINDOW_SIZE_S),
        "pt": moving_average(powers_details["pt"], window_size=SMOOTHING_WINDOW_SIZE_S),
    }

    smoothed_decel_powers = np.clip(smoothed_powers_details["acc"], None, 0.0)
    smoothed_brake_powers = np.clip(smoothed_powers, None, 0.0)

    return {
        "headwinds": np.array(headwinds),
        "accelerations": np.array(accelerations),
        "smoothed_headwinds": np.array(smoothed_headwinds),
        "powers": np.array(powers),
        "smoothed_powers": np.array(smoothed_powers),
        "powers_details": powers_details,
        "smoothed_powers_details": smoothed_powers_details,
        "smoothed_decel_powers": np.array(smoothed_decel_powers),
        "smoothed_brake_powers": np.array(smoothed_brake_powers),
    }


def _filter_and_disk_temp(core, smooth, ride_type_series, power_data, weather, progress_cb=None):
    """Apply validity filters, compute cumulative distance, brake temps, and filtered arrays."""
    def _report(pct, text):
        if progress_cb is not None:
            progress_cb(int(pct), text)

    _report(70, "Filtering, breaks & stats...")

    smoothed_speeds_kph = smooth["smoothed_speeds_kph"]
    smoothed_alts = smooth["smoothed_alts"]
    grades = smooth["grades"]
    times = core["times"]
    dists = core["dists"]
    lats = core["lats"]
    lons = core["lons"]
    raw_alts = core["raw_alts"]
    speeds = core["speeds_kph"]
    total_dists_raw = core["total_dists"]

    smoothed_powers = power_data["smoothed_powers"]
    smoothed_powers_details = power_data["smoothed_powers_details"]
    smoothed_brake_powers = power_data["smoothed_brake_powers"]
    smoothed_decel_powers = power_data["smoothed_decel_powers"]
    smoothed_headwinds = power_data["smoothed_headwinds"]
    accelerations = power_data["accelerations"]

    temp_series = weather["temp_c"]
    rho_series = weather["rho"]
    pres_series = weather["pres_hpa"]
    rhum_series = weather["rhum_pct"]
    wind_series = weather["wspd_ms"]
    wind_dir_series = weather["wdir_deg"]

    grades_array = np.array(moving_average(grades, window_size=SMOOTHING_WINDOW_SIZE_S))

    valid = (
        (smoothed_speeds_kph < MAX_SPEED) &
        (np.abs(grades_array * 100.0) < MAX_GRADE) &
        (smoothed_powers > -MAX_PWR) & (smoothed_powers < MAX_PWR)
    )
    active = (smoothed_speeds_kph > MIN_ACTIVE_SPEED)

    print(f"Valid entries (pre min-speed): {np.count_nonzero(valid)} / {len(valid)} "
          f"({np.count_nonzero(valid)/len(valid):.1%})")
    print(f"Active entries (> {MIN_ACTIVE_SPEED:.1f} km/h): {np.count_nonzero(active)} / {len(active)} "
          f"({np.count_nonzero(active)/len(active):.1%})")

    valid = (
        (smoothed_speeds_kph > MIN_VALID_SPEED) &
        (smoothed_speeds_kph < MAX_SPEED) &
        (np.abs(grades_array * 100.0) < MAX_GRADE) &
        (smoothed_powers > -MAX_PWR) & (smoothed_powers < MAX_PWR)
    )

    total_dists = []
    total_dist = 0.0
    for d, v in zip(dists, valid):
        if v:
            total_dist += d
        total_dists.append(total_dist)
    total_dists = np.array(total_dists)

    _report(80, "Computing brake temps...")
    temp_front_disk_c, temp_rear_disk_c = compute_disk_temperature(
        temp_series,
        smoothed_speeds_kph,
        smoothed_headwinds,
        smoothed_brake_powers,
        np.array([t.timestamp() for t in times]),
        initial_disk_temp_c=temp_series[0]
    )

    idx = valid
    total_distance_km = total_dists[idx][-1] if np.any(idx) else 0.0
    elevation_gain_m = float(np.sum(np.clip(np.diff(smoothed_alts), 0, None))) if len(smoothed_alts) > 1 else 0.0

    return {
        "valid": idx,
        "total_dists": total_dists,
        "filtered_head_winds": np.array(smoothed_headwinds)[idx],
        "filtered_lats": lats[idx],
        "filtered_lons": lons[idx],
        "filtered_dists": total_dists[idx],
        "filtered_times": times[idx],
        "filtered_deltadists": dists[idx],
        "filtered_raw_alts": raw_alts[idx],
        "filtered_raw_speeds": speeds[idx],
        "filtered_accelerations": accelerations[idx],
        "filtered_smoothed_alts": smoothed_alts[idx],
        "filtered_speeds_kph": smoothed_speeds_kph[idx],
        "filtered_powers": smoothed_powers[idx],
        "filtered_pwr_details": {
            "slope": smoothed_powers_details["slope"][idx],
            "roll": smoothed_powers_details["roll"][idx],
            "drag": smoothed_powers_details["drag"][idx],
            "acc": smoothed_powers_details["acc"][idx],
            "pt": smoothed_powers_details["pt"][idx],
        },
        "filtered_brake_powers": smoothed_brake_powers[idx],
        "filtered_decel_powers": smoothed_decel_powers[idx],
        "filtered_temp_front_disk_c": np.array(temp_front_disk_c)[idx],
        "filtered_temp_rear_disk_c": np.array(temp_rear_disk_c)[idx],
        "filtered_grades": grades_array[idx] * 100.0,
        "filtered_rho": np.array(rho_series)[idx],
        "filtered_temp": np.array(temp_series)[idx],
        "filtered_pres": np.array(pres_series)[idx],
        "filtered_rhum": np.array(rhum_series)[idx],
        "filtered_wind_dirs": np.array(wind_dir_series)[idx],
        "filtered_global_winds": np.array(wind_series)[idx],
        "total_distance_km": total_distance_km,
        "elevation_gain_m": elevation_gain_m,
        "filtered_ride_type": ride_type_series[idx]
    }


def _filter_device_sensors(core, valid_mask):
    """Filter interpolated device sensor series with same mask."""
    if "hr_series" in core:
        return {
            "filtered_hr_bpm": core["hr_series"][valid_mask],
            "filtered_cad_meas_rpm": core["cad_series"][valid_mask],
            "filtered_power_meas_w": core["pow_series"][valid_mask],
            "filtered_temp_device_c": core["temp_dev_series"][valid_mask],
        }
    else:
        return {
            "filtered_hr_bpm": np.array([]),
            "filtered_cad_meas_rpm": np.array([]),
            "filtered_power_meas_w": np.array([]),
            "filtered_temp_device_c": np.array([]),
        }


def _compute_kcal_timeseries(filtered_times, filtered_powers, timestep_min, cfg):
    """Compute cumulative kcal from active power and base metabolism."""
    ft = np.array(filtered_times)
    fp = np.array(filtered_powers, dtype=float)

    if len(ft) > 1:
        tsec = np.array([t.timestamp() for t in ft], dtype=float)
        dt = np.diff(tsec)
        dt0 = float(np.median(dt)) if dt.size > 0 else timestep_min * 60.0
        dt = np.concatenate(([dt0], dt))
    else:
        tsec = np.array([ft[0].timestamp()]) if len(ft) == 1 else np.array([0.0])
        dt = np.array([timestep_min * 60.0])

    P_pos = np.clip(fp, 0.0, None)
    mech_E_J = P_pos * np.clip(dt, None, 2)
    metab_E_J = mech_E_J / EFFICIENCY
    kcal_active = metab_E_J / 4184.0

    base_rate_kcal_per_s = BASE_MET_DURING_ACTIVITY * cfg.body_mass / 3600.0
    kcal_base = base_rate_kcal_per_s * dt

    cum_kcal_active = np.cumsum(kcal_active)
    cum_kcal_base = np.cumsum(kcal_base)
    cum_kcal_total = cum_kcal_active + cum_kcal_base
    rel_time_s = tsec - tsec[0]

    return {
        "cum_kcal_active": cum_kcal_active,
        "cum_kcal_base": cum_kcal_base,
        "cum_kcal_total": cum_kcal_total,
        "rel_time_s": rel_time_s,
        "dt": dt,
        "tsec": tsec,
    }


def _detect_breaks(smoothed_speeds_kph, times, total_dists, timestep_min, break_threshold_min):
    """Detect short and long breaks based on speed threshold."""
    is_stopped = smoothed_speeds_kph <= MIN_ACTIVE_SPEED
    breaks = []
    n = len(is_stopped)
    i = 0
    MIN_BREAK_DURATION_MIN = 0.5

    while i < n:
        if not is_stopped[i]:
            i += 1
            continue
        start_idx = i
        while i < n and is_stopped[i]:
            i += 1
        end_idx = i - 1
        duration_min = (end_idx - start_idx + 1) * timestep_min
        if duration_min < MIN_BREAK_DURATION_MIN:
            continue
        btype = "long" if duration_min >= break_threshold_min else "short"
        start_time_break = times[start_idx]
        end_time_break = times[end_idx]
        lat_center = None
        lon_center = None
        # lat,lon not needed directly here; caller can add if wanted
        breaks.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time": start_time_break,
            "end_time": end_time_break,
            "duration_min": duration_min,
            "type": btype,
            "lat": lat_center,
            "lon": lon_center,
            "time_min": (start_idx * timestep_min),
            "distance_km": float(total_dists[start_idx]) if start_idx < len(total_dists) else 0.0
        })

    return breaks


def _augment_break_positions(breaks, lats, lons):
    """Add lat/lon centers for breaks based on indices."""
    for b in breaks:
        si = b["start_idx"]
        ei = b["end_idx"] + 1
        if si < len(lats) and ei <= len(lats):
            b["lat"] = float(np.mean(lats[si:ei]))
            b["lon"] = float(np.mean(lons[si:ei]))

def _detect_directional_segments(
    smoothed_alts,
    deltadists,
    times,
    dists,
    direction,
    min_distance_km,
    min_elev_gain_m,
    end_window_size_m,
    end_avg_grade_pct,
    max_gap_m,
):
    """
    Generic slope detector.

    direction = +1 → climbs (uphill)
    direction = -1 → downhills (downhill, but returned with positive gain & grade)

    Returns a list of segments with:
        start_idx, end_idx, distance_km, elevation_gain_m (>0),
        average_grade_pct (>0), start_time (min), end_time (min),
        start_distance_km, end_distance_km
    """
    segments = []
    n = len(smoothed_alts)
    if n == 0:
        return segments

    # Work on "height" h so that increasing h always means "uphill"
    h = direction * np.asarray(smoothed_alts, dtype=float)
    dists_step = np.asarray(deltadists, dtype=float)
    times = np.asarray(times, dtype=object)
    dists_cum = np.asarray(dists, dtype=float)

    start_idx = None

    for i in range(n):
        if start_idx is None:
            start_idx = i
            continue

        end_idx = i
        dist_km = float(np.sum(dists_step[start_idx:end_idx + 1]))
        if dist_km < min_distance_km:
            continue

        elev_gain_h = float(h[end_idx] - h[start_idx])
        if elev_gain_h >= min_elev_gain_m:
            # refine end using sliding window, like your original logic
            for j in range(end_idx + 1, n):
                window_distance_m = 0.0
                window_start_idx = j
                while window_distance_m < end_window_size_m and window_start_idx > 0:
                    window_distance_m += dists_step[window_start_idx] * 1000.0
                    window_start_idx -= 1
                window_elev_gain_h = float(h[j] - h[window_start_idx])
                window_dist_km = float(np.sum(dists_step[window_start_idx:j + 1]))
                window_grade = (
                    window_elev_gain_h / (window_dist_km * 1000.0)
                    if window_dist_km > 0 else 0.0
                )
                if window_grade < end_avg_grade_pct / 100.0:
                    end_idx = j
                    break

            elev_gain_h = float(h[end_idx] - h[start_idx])
            dist_km = float(np.sum(dists_step[start_idx:end_idx]))
            avg_grade_pct = (
                (elev_gain_h / (dist_km * 1000.0)) * 100.0
                if dist_km > 0 else 0.0
            )

            segments.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "distance_km": dist_km,
                "elevation_gain_m": elev_gain_h,           # always positive magnitude
                "average_grade_pct": avg_grade_pct,        # always positive magnitude
                "start_time": times[start_idx].timestamp() / 60.0 - times[0].timestamp() / 60.0,
                "end_time": times[end_idx].timestamp() / 60.0 - times[0].timestamp() / 60.0,
                "start_distance_km": dists_cum[start_idx],
                "end_distance_km": dists_cum[end_idx],
            })

        # move start forward
        start_idx = end_idx

    return segments


def _combine_adjacent_segments(segments, max_gap_m):
    """
    Fuse segments that are closer than max_gap_m in distance.
    Works for both climbs and downhills.
    """
    if not segments:
        return []

    combined = []
    i = 0
    while i < len(segments):
        current = segments[i]
        j = i + 1
        while j < len(segments):
            dist_between_m = (
                segments[j]["start_distance_km"] * 1000.0
                - current["end_distance_km"] * 1000.0
            )
            if dist_between_m <= max_gap_m:
                # merge j into current
                new_start_idx = current["start_idx"]
                new_end_idx = segments[j]["end_idx"]
                new_start_dist = current["start_distance_km"]
                new_end_dist = segments[j]["end_distance_km"]
                new_dist_km = new_end_dist - new_start_dist

                new_elev_gain = (
                    current["elevation_gain_m"] + segments[j]["elevation_gain_m"]
                )
                new_avg_grade_pct = (
                    (new_elev_gain / (new_dist_km * 1000.0)) * 100.0
                    if new_dist_km > 0 else 0.0
                )

                current = {
                    "start_idx": new_start_idx,
                    "end_idx": new_end_idx,
                    "distance_km": new_dist_km,
                    "elevation_gain_m": new_elev_gain,
                    "average_grade_pct": new_avg_grade_pct,
                    "start_time": current["start_time"],
                    "end_time": segments[j]["end_time"],
                    "start_distance_km": new_start_dist,
                    "end_distance_km": new_end_dist,
                }
                j += 1
            else:
                break
        combined.append(current)
        i = j

    return combined

def _detect_climbs_and_downhills(smoothed_alts, deltadists, times, dists):
    """
    Detect climbs and downhills using a unified slope detection.

    `smooth` is the dict from _smooth_... with:
        "smoothed_speeds_kph": smoothed_speeds_kph,
        "smoothed_speeds_ms": smoothed_speeds_ms,
        "smoothed_alts": smoothed_alts,
        "grades": grades,
        "smoothed_grades": smoothed_grades,
        "grades_array": grades_array,
    """


    # --- Climbs (uphill: direction = +1) ---
    climbs = _detect_directional_segments(
        smoothed_alts=smoothed_alts,
        deltadists=deltadists,
        times=times,
        dists=dists,
        direction=+1,
        min_distance_km=CLIMB_MIN_DISTANCE_KM,
        min_elev_gain_m=CLIMB_MIN_ELEVATION_GAIN_M,
        end_window_size_m=CLIMB_END_WINDOW_SIZE_M,
        end_avg_grade_pct=CLIMB_END_AVERAGE_GRADE_PCT,
        max_gap_m=MAX_DIST_BETWEEN_CLIMBS_M,
    )
    combined_climbs = _combine_adjacent_segments(climbs, MAX_DIST_BETWEEN_CLIMBS_M)

    # --- Downhills (downhill: direction = -1, treat drop as positive gain) ---
    downhills = _detect_directional_segments(
        smoothed_alts=smoothed_alts,
        deltadists=deltadists,
        times=times,
        dists=dists,
        direction=-1,  # <- key difference
        min_distance_km=CLIMB_MIN_DISTANCE_KM,          # same thresholds as before
        min_elev_gain_m=CLIMB_MIN_ELEVATION_GAIN_M,
        end_window_size_m=CLIMB_END_WINDOW_SIZE_M,
        end_avg_grade_pct=CLIMB_END_AVERAGE_GRADE_PCT,
        max_gap_m=MAX_DIST_BETWEEN_CLIMBS_M,
    )
    combined_downhills = _combine_adjacent_segments(downhills, MAX_DIST_BETWEEN_CLIMBS_M)

    ride_type_series = _create_ride_type_timeseries(len(smoothed_alts), combined_climbs, combined_downhills)
    return climbs, combined_climbs, downhills, combined_downhills, ride_type_series


def _create_ride_type_timeseries(n_samples, combined_climbs, combined_downhills):
    """
    Create a categorical ride-type series of length n_samples.

    Output: ride_type (np.ndarray of dtype=object)
        Each entry is one of:
            "level"
            "climb"
            "downhill"

    Priority:
        - downhill overrides climb (in case of overlap, although ideally impossible)
    """

    # start with everything marked as "level"
    ride_type = np.full(n_samples, "level", dtype=object)

    # mark climbs
    for seg in combined_climbs:
        i0 = int(seg.get("start_idx", 0))
        i1 = int(seg.get("end_idx", i0))
        # clamp safely
        i0 = max(i0, 0)
        i1 = min(i1, n_samples - 1)
        if i1 >= i0:
            ride_type[i0:i1 + 1] = "climb"

    # mark downhills (override where present)
    for seg in combined_downhills:
        i0 = int(seg.get("start_idx", 0))
        i1 = int(seg.get("end_idx", i0))
        i0 = max(i0, 0)
        i1 = min(i1, n_samples - 1)
        if i1 >= i0:
            ride_type[i0:i1 + 1] = "downhill"

    return ride_type


def _augment_climb_metrics(climb_list, filtered_dists, filtered_powers, timestep_min):
    """Add duration, average power, VAM to each climb."""
    d = np.asarray(filtered_dists, dtype=float)
    p = np.asarray(filtered_powers, dtype=float)

    for c in climb_list:
        d0 = float(c.get("start_distance_km", 0.0))
        d1 = float(c.get("end_distance_km", d0))
        if d1 < d0:
            d0, d1 = d1, d0

        eps = 1e-6
        mask = (d >= d0 - eps) & (d <= d1 + eps)
        n_steps = int(np.count_nonzero(mask))
        duration_min = n_steps * timestep_min

        if n_steps > 0:
            p_seg = p[mask]
            p_seg = p_seg[np.isfinite(p_seg)]
            if p_seg.size > 0:
                p_pos = p_seg[p_seg > 0]
                if p_pos.size > 0:
                    avg_p = float(np.mean(p_pos))
                else:
                    avg_p = float(np.mean(p_seg))
            else:
                avg_p = 0.0
        else:
            avg_p = 0.0

        elev_gain = float(c.get("elevation_gain_m", 0.0))
        hours = duration_min / 60.0
        vam = elev_gain / hours if hours > 0 else 0.0

        c["duration_min"] = duration_min
        c["avg_power_w"] = avg_p
        c["vam_m_per_h"] = vam


def _build_5km_segments(filtered, timestep_min):
    """Build 5 km segments from filtered arrays and compute metrics."""
    filtered_dists = filtered["filtered_dists"]
    filtered_speeds_kph = filtered["filtered_speeds_kph"]
    filtered_powers = filtered["filtered_powers"]
    filtered_pwr_details = filtered["filtered_pwr_details"]
    filtered_grades = filtered["filtered_grades"]
    filtered_smoothed_alts = filtered["filtered_smoothed_alts"]

    segments_5km = []
    if len(filtered_dists) <= 1:
        return segments_5km

    seg_length_km = 5.0
    total_dist_km_seg = filtered_dists[-1]
    seg_start_km = 0.0
    seg_index = 1
    fd = np.array(filtered_dists, dtype=float)
    fs = np.array(filtered_speeds_kph, dtype=float)
    fp = np.array(filtered_powers, dtype=float)
    fp_g = np.array(filtered_pwr_details["slope"], dtype=float)
    fp_r = np.array(filtered_pwr_details["roll"], dtype=float)
    fp_d = np.array(filtered_pwr_details["drag"], dtype=float)
    fp_a = np.array(filtered_pwr_details["acc"], dtype=float)
    fp_pt = np.array(filtered_pwr_details["pt"], dtype=float)
    fg = np.array(filtered_grades, dtype=float)
    fa = np.array(filtered_smoothed_alts, dtype=float)

    while seg_start_km < total_dist_km_seg - 1e-6:
        seg_end_km = min(seg_start_km + seg_length_km, total_dist_km_seg)
        if seg_end_km < total_dist_km_seg:
            mask = (fd >= seg_start_km) & (fd < seg_end_km)
        else:
            mask = (fd >= seg_start_km) & (fd <= seg_end_km + 1e-9)
        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            seg_start_km = seg_end_km
            seg_index += 1
            continue

        dist_km = seg_end_km - seg_start_km
        duration_min = idxs.size * timestep_min
        sp_seg = fs[idxs]
        pw_seg = fp[idxs]
        pwg_seg = fp_g[idxs]
        pwr_seg = fp_r[idxs]
        pwd_seg = fp_d[idxs]
        pwa_seg = fp_a[idxs]
        pwpt_seg = fp_pt[idxs]
        gr_seg = fg[idxs]

        i0, i1 = idxs[0], idxs[-1]
        elev_gain_m_seg = 0.0
        if i1 > i0:
            elev_diff = np.diff(fa[i0:i1 + 1])
            elev_gain_m_seg = float(np.sum(np.clip(elev_diff, 0, None)))

        avg_spd = float(np.mean(sp_seg)) if sp_seg.size > 0 else 0.0

        if np.any(pw_seg > 0):
            avg_pwr = float(np.mean(np.clip(pw_seg, 0, None)))
            avg_pwr_g = float(np.mean(pwg_seg[:]))
            avg_pwr_r = float(np.mean(pwr_seg[:]))
            avg_pwr_d = float(np.mean(pwd_seg[:]))
            avg_pwr_ac = float(np.mean(np.clip(pwa_seg[:], 0, None)))
            avg_pwr_dc = float(np.mean(np.clip(pwa_seg[:], None, 0.0)))
            avg_pwr_b = float(np.mean(-1.*np.clip(pw_seg[:], None, 0.0)))
            avg_pwr_pt = float(np.mean(pwpt_seg[:]))
        else:
            avg_pwr = float(np.mean(pw_seg)) if pw_seg.size > 0 else 0.0
            avg_pwr_g = float(np.mean(pwg_seg)) if pwg_seg.size > 0 else 0.0
            avg_pwr_r = float(np.mean(pwr_seg)) if pwr_seg.size > 0 else 0.0
            avg_pwr_d = float(np.mean(pwd_seg)) if pwd_seg.size > 0 else 0.0
            avg_pwr_ac = float(np.mean(np.clip(pwa_seg[:], 0.0, None))) if pwa_seg.size > 0 else 0.0
            avg_pwr_dc = float(np.mean(np.clip(pwa_seg[:], None, 0.0))) if pwa_seg.size > 0 else 0.0
            avg_pwr_b = float(np.mean(-1.*np.clip(pw_seg[:], None, 0.0))) if pw_seg.size > 0 else 0.0
            avg_pwr_pt = float(np.mean(pwpt_seg[:])) if pwa_seg.size > 0 else 0.0

        avg_grd = float(np.mean(gr_seg)) if gr_seg.size > 0 else 0.0

        segments_5km.append({
            "index": seg_index,
            "start_km": seg_start_km,
            "end_km": seg_end_km,
            "distance_km": dist_km,
            "duration_min": duration_min,
            "avg_speed_kph": avg_spd,
            "avg_power_w": avg_pwr,
            "avg_pwr_g_w": avg_pwr_g,
            "avg_pwr_r_w": avg_pwr_r,
            "avg_pwr_d_w": avg_pwr_d,
            "avg_pwr_a_w": avg_pwr_ac,
            "avg_pwr_dc_w": avg_pwr_dc,
            "avg_pwr_b_w": avg_pwr_b,
            "avg_pwr_pt_w": avg_pwr_pt,
            "avg_grade_pct": avg_grd,
            "elevation_gain_m": elev_gain_m_seg,
        })

        seg_start_km = seg_end_km
        seg_index += 1

    return segments_5km


def _gear_simulation(filtered, cfg, target_cadence, exclude_downhill_gears):
    """Simulate gear selection, shifts, and cadence stats."""
    filtered_speeds_kph = filtered["filtered_speeds_kph"]
    filtered_powers = filtered["filtered_powers"]
    filtered_dists = filtered["filtered_dists"]

    gear_list, gear_by_combo = build_gear_list(cfg)

    current_front = cfg.front_der_teeth[-1]
    current_rear_idx = 0
    current_rear = cfg.rear_der_teeth[current_rear_idx]
    current_gear_idx = gear_by_combo[(current_front, current_rear)]

    selected_gear_idx = []
    selected_cadences = []
    last_shift_time_rear = -9999
    last_shift_time_front = -9999

    for i_filtered, sp_kph in enumerate(filtered_speeds_kph):
        sp_ms = sp_kph / 3.6
        wheel_rpm = (sp_ms / cfg.wheel_circumference) * 60.0

        current_gear = gear_list[current_gear_idx]
        current_cadence = wheel_rpm * (current_gear['rear'] / current_gear['front'])
        cadence_error = current_cadence - target_cadence

        time_since_rear_shift = i_filtered - last_shift_time_rear

        should_shift_rear = (
            (time_since_rear_shift >= MIN_SHIFT_INTERVAL_REAR_SEC) or
            (abs(cadence_error) >= HYSTERESIS_REAR_RPM)
        )

        if should_shift_rear:
            best_rear = find_best_rear_gear(
                current_front, wheel_rpm, target_cadence,
                cfg.rear_der_teeth, gear_by_combo
            )
            if best_rear and best_rear != current_rear:
                current_rear = best_rear
                current_gear_idx = gear_by_combo[(current_front, current_rear)]
                last_shift_time_rear = i_filtered

        new_front = manage_front_gear(
            current_front, current_rear, wheel_rpm,
            cfg.front_der_teeth, cfg.rear_der_teeth
        )

        if new_front != current_front:
            current_front = new_front
            best_rear = find_best_rear_gear(
                current_front, wheel_rpm, target_cadence,
                cfg.rear_der_teeth, gear_by_combo
            )
            if best_rear and (current_front, best_rear) in gear_by_combo:
                current_rear = best_rear
                current_gear_idx = gear_by_combo[(current_front, current_rear)]
            last_shift_time_front = i_filtered
            last_shift_time_rear = i_filtered

        current_gear = gear_list[current_gear_idx]
        final_cadence = wheel_rpm * (current_gear['rear'] / current_gear['front'])
        selected_gear_idx.append(current_gear_idx)
        selected_cadences.append(final_cadence)

    selected_gear_names = [gear_list[i]['name'] for i in selected_gear_idx]
    selected_front_teeth = [gear_list[i]['front'] for i in selected_gear_idx]
    selected_rear_teeth = [gear_list[i]['rear'] for i in selected_gear_idx]

    powers_arr = np.array(filtered_powers)

    if exclude_downhill_gears:
        active_mask_gears = powers_arr > 0
    else:
        active_mask_gears = np.ones(len(powers_arr), dtype=bool)

    active_indices = [i for i in range(len(selected_gear_idx)) if active_mask_gears[i]]
    active_selected_gear_names = [selected_gear_names[i] for i in active_indices]
    active_selected_front_teeth = [selected_front_teeth[i] for i in active_indices]
    active_selected_rear_teeth = [selected_rear_teeth[i] for i in active_indices]
    active_selected_cadences = [selected_cadences[i] for i in active_indices]

    front_shifts = sum(
        1 for i in range(1, len(active_selected_front_teeth))
        if active_selected_front_teeth[i] != active_selected_front_teeth[i - 1]
    )
    rear_shifts = sum(
        1 for i in range(1, len(active_selected_rear_teeth))
        if active_selected_rear_teeth[i] != active_selected_rear_teeth[i - 1]
    )

    if len(active_selected_cadences) > 0:
        cadence_arr = np.array(active_selected_cadences)
        cadence_mean = float(np.mean(cadence_arr))
        cadence_std = float(np.std(cadence_arr))
        index_max_cadence_within_active = int(np.argmax(cadence_arr))
        index_max_cadence = active_indices[index_max_cadence_within_active]
        dist_max_cadence = float(filtered_dists[index_max_cadence])
        gear_combo_max_cadence = selected_gear_names[index_max_cadence]
        speed_max_cadence = float(filtered["filtered_speeds_kph"][index_max_cadence])
    else:
        cadence_arr = np.array([])
        cadence_mean = 0.0
        cadence_std = 0.0
        dist_max_cadence = 0.0
        gear_combo_max_cadence = None
        speed_max_cadence = 0.0

    return {
        "gear_list": gear_list,
        "selected_front_teeth": selected_front_teeth,
        "selected_rear_teeth": selected_rear_teeth,
        "selected_gear_names": selected_gear_names,
        "selected_cadences": selected_cadences,
        "active_selected_gear_names": active_selected_gear_names,
        "active_selected_front_teeth": active_selected_front_teeth,
        "active_selected_rear_teeth": active_selected_rear_teeth,
        "active_selected_cadences": active_selected_cadences,
        "front_shifts": front_shifts,
        "rear_shifts": rear_shifts,
        "cadence_arr": cadence_arr,
        "cadence_mean": cadence_mean,
        "cadence_std": cadence_std,
        "dist_max_cadence": dist_max_cadence,
        "gear_combo_max_cadence": gear_combo_max_cadence,
        "speed_max_cadence": speed_max_cadence,
    }



def analyze_gpx_file(
    gpx_path: str,
    cfg: RiderBikeConfig,
    target_cadence: int = DEFAULT_TARGET_CADENCE,
    use_wind_data: bool = True,
    exclude_downhill_gears: bool = True,
    break_threshold_min: float = 10.0,
    progress_cb=None
):
    """
    Main analysis: orchestrates all steps and returns a big result dict.
    """

    def _report(pct, text):
        if progress_cb is not None:
            progress_cb(int(pct), text)

    # 1) Parse + interpolate core time series
    core = _parse_and_interpolate_gpx(gpx_path, progress_cb=_report)
    local_start_time = core["times_local_display"][0]
    local_end_time = core["times_local_display"][-1]
    start_time = core["start_time"]
    end_time = core["end_time"]
    timestep_min = core["timestep_min"]

    # 2) Smoothing & grades
    smooth = _smooth_and_grades(core)

    # 3) Weather / air density
    weather = _weather_and_air_density(core, use_wind_data, progress_cb=_report)

    # 9) Climbs & downhills
    _report(40, "Find climbs...")
    
    climbs, combined_climbs, downhills, combined_downhills, ride_type_series = _detect_climbs_and_downhills(smooth["smoothed_alts"], smooth["dists"], smooth["times"], smooth["total_dists"])

    # 4) Headwind & power components
    power_data = _headwind_and_power(core, smooth, ride_type_series, weather, cfg, progress_cb=_report)

    # 5) Filter & brake temps
    filtered = _filter_and_disk_temp(core, smooth, ride_type_series, power_data, weather, progress_cb=_report)

    # 6) Device-based sensors filtered
    filtered_sensors = _filter_device_sensors(core, filtered["valid"])

    # 7) Kcal time series
    kcal_data = _compute_kcal_timeseries(
        filtered["filtered_times"],
        filtered["filtered_powers"],
        timestep_min,
        cfg
    )

    #7.1 Water loss / sweating
    water_rate_L_per_h, cum_water_loss_L, dmax_L = compute_water_loss_timeseries(
        filtered_temp_c=filtered["filtered_temp"],
        filtered_rhum_pct=filtered["filtered_rhum"],
        filtered_speeds_kph=filtered["filtered_speeds_kph"],
        filtered_headwind_ms=filtered["filtered_head_winds"],
        filtered_powers_w=filtered["filtered_powers"],
        filtered_times=filtered["filtered_times"],
        body_mass_kg=cfg.body_mass,
        mode=HYDRATION_MODEL_MODE,
        can_drink_freely=True,
    )

    total_water_loss_L = float(cum_water_loss_L[-1]) if cum_water_loss_L.size > 0 else 0.0

    #7.2 ISO 7933 / PHS model (thermophysiology)
    phs = compute_phs_timeseries_from_filtered(
        filtered,
        cfg,
        mode=HYDRATION_MODEL_MODE,
        can_drink_freely=True,
    )
    phs_aligned = _resample_phs_to_filtered(phs, filtered["filtered_times"])

    # --- Build distance-per-minute lookup (for plotting PHS over distance) ---

    # filtered timing information
    f_times = np.array([t.timestamp() for t in filtered["filtered_times"]], dtype=float)
    f_dist = np.array(filtered["filtered_dists"], dtype=float)

    if f_times.size > 1:
        t0 = f_times[0]

        # phs times in seconds since start
        phs_seconds = phs["t_min"] * 60.0
        phs_times_abs = t0 + phs_seconds

        # distance at each PHS minute timestamp
        phs_dist_km = np.interp(phs_times_abs, f_times, f_dist)
    else:
        # fallback if only one point exists
        phs_dist_km = np.zeros_like(phs["t_min"])


    # 8) Breaks
    breaks = _detect_breaks(
        smooth["smoothed_speeds_kph"],
        core["times"],
        filtered["total_dists"],
        timestep_min,
        break_threshold_min,
    )

    _augment_break_positions(breaks, core["lats"], core["lons"])
    short_breaks = [b for b in breaks if b["type"] == "short"]
    long_breaks = [b for b in breaks if b["type"] == "long"]
    short_break_total_min = float(sum(b["duration_min"] for b in short_breaks))
    long_break_total_min = float(sum(b["duration_min"] for b in long_breaks))

    climbs, combined_climbs, downhills, combined_downhills, _ = _detect_climbs_and_downhills(filtered["filtered_smoothed_alts"], filtered["filtered_deltadists"], filtered["filtered_times"], filtered["filtered_dists"])

    _augment_climb_metrics(climbs, filtered["filtered_dists"], filtered["filtered_powers"], timestep_min)
    _augment_climb_metrics(combined_climbs, filtered["filtered_dists"], filtered["filtered_powers"], timestep_min)

    # climb summary metrics
    max_vam = 0.0
    avg_climb_power = 0.0
    if combined_climbs:
        vams = [c.get("vam_m_per_h", 0.0) for c in combined_climbs]
        max_vam = float(max(vams)) if vams else 0.0
        weighted_list = []
        for c in combined_climbs:
            p = c.get("avg_power_w", 0.0)
            w = c.get("duration_min", 0.0)
            if w > 0:
                weighted_list.append((p, w))
        if weighted_list:
            num = sum(p * w for p, w in weighted_list)
            den = sum(w for _, w in weighted_list)
            avg_climb_power = float(num / den) if den > 0 else 0.0

    # 10) Segments 5 km
    _report(87, "Build 5 km segments...")
    segments_5km = _build_5km_segments(filtered, timestep_min)
    best_seg_power_5km = 0.0
    if segments_5km:
        vals_seg = [s["avg_power_w"] for s in segments_5km if not math.isnan(s.get("avg_power_w", 0.0))]
        if vals_seg:
            best_seg_power_5km = float(max(vals_seg))

    # 11) Gear simulation
    _report(90, "Simulating gear shifts...")
    gear_sim = _gear_simulation(filtered, cfg, target_cadence, exclude_downhill_gears)

    # 12) Summary stats & console printouts
    powers_arr = np.array(filtered["filtered_powers"])
    active_mask = powers_arr > 0
    total_duration_min = len(powers_arr) * timestep_min
    active_duration_min = np.count_nonzero(active_mask) * timestep_min
    pause_duration_min = total_duration_min - active_duration_min

    active_powers = powers_arr[active_mask]
    avg_power_active = float(np.mean(active_powers)) if len(active_powers) > 0 else 0.0

    work_joules = float(np.sum(active_powers) * timestep_min * 60.0)
    kcal_burned = work_joules / (EFFICIENCY * 4184.0) if EFFICIENCY > 0 else 0.0
    met_active = (kcal_burned / cfg.body_mass / (active_duration_min / 60.0)
                  if active_duration_min > 0 else 0.0)
    avg_power_with_freewheeling = np.mean(np.clip(powers_arr, 0, None))

    print("\n=== Ride Summary ===")
    print(f"Total duration:      {total_duration_min:.1f} min")
    print(f"Active duration:     {active_duration_min:.1f} min")
    print(f"Pause duration:      {pause_duration_min:.1f} min")
    print(f"Average pedaling power:       {avg_power_active:.1f} W (non-negative only)")
    print(f"Average positive power:       {avg_power_with_freewheeling:.1f} W (including freewheeling only)")
    print(f"Calories burned:     {kcal_burned:.0f} kcal (active)")
    print(f"Estimated MET:       {met_active:.1f} (active)")

    print("\n=== Gear Shifts ===")
    print(f"Front derailleur shifts: {gear_sim['front_shifts']}")
    print(f"Rear derailleur shifts:  {gear_sim['rear_shifts']}")
    print(f"Total shifts:            {gear_sim['front_shifts'] + gear_sim['rear_shifts']}")

    print("\n=== Cadence Statistics ===")
    print(f"Target cadence:    {target_cadence} RPM")
    print(f"Mean cadence:      {gear_sim['cadence_mean']:.1f} RPM")
    print(f"Std dev cadence:   {gear_sim['cadence_std']:.1f} RPM")
    if len(gear_sim['cadence_arr']) > 0:
        print(f"Min cadence:       {np.min(gear_sim['cadence_arr']):.1f} RPM")
        print(f"Max cadence:       {np.max(gear_sim['cadence_arr']):.1f} RPM")
    else:
        print("No cadence data in active periods.")

    print("\n=== Break Statistics ===")
    print(f"Short breaks (< {break_threshold_min:.1f} min): {len(short_breaks)} x, "
          f"total {short_break_total_min:.1f} min")
    print(f"Long breaks (>= {break_threshold_min:.1f} min): {len(long_breaks)} x, "
          f"total {long_break_total_min:.1f} min")

    print("\n=== Climb Statistics ===")
    print(f"Detected climbs: {len(climbs)}")
    for i, climb in enumerate(climbs):
        print(f" Climb {i+1}: {climb['distance_km']:.2f} km, "
              f"{climb['elevation_gain_m']:.0f} m gain, "
              f"avg grade {climb['average_grade_pct']:.1f} %, "
              f"VAM {climb.get('vam_m_per_h', 0.0):.0f} m/h, "
              f"P_avg {climb.get('avg_power_w', 0.0):.0f} W, "
              f"from {time_to_readable(t_minutes=climb['start_time'])} to {time_to_readable(t_minutes=climb['end_time'])}, "
              f"distance {climb['start_distance_km']:.2f} km to {climb['end_distance_km']:.2f} km")
    print("\n=== Climb Combination ===")
    for i, climb in enumerate(combined_climbs):
        print(f" Combined Climb {i+1}: {climb['distance_km']:.2f} km, "
              f"{climb['elevation_gain_m']:.0f} m gain, "
              f"avg grade {climb['average_grade_pct']:.1f} %, "
              f"VAM {climb.get('vam_m_per_h', 0.0):.0f} m/h, "
              f"P_avg {climb.get('avg_power_w', 0.0):.0f} W, "
              f"from {time_to_readable(t_minutes=climb['start_time'])} to {time_to_readable(t_minutes=climb['end_time'])}, "
              f"distance {climb['start_distance_km']:.2f} km to {climb['end_distance_km']:.2f} km")

    _report(100, "Done.")

    # Device sensor stats (optional)
    if filtered_sensors["filtered_hr_bpm"].size > 0 and np.any(np.isfinite(filtered_sensors["filtered_hr_bpm"])):
        avg_hr = float(np.nanmean(filtered_sensors["filtered_hr_bpm"]))
        max_hr = float(np.nanmax(filtered_sensors["filtered_hr_bpm"]))
        print(f"\n=== Heart Rate (device) ===")
        print(f"Avg HR: {avg_hr:.0f} bpm, Max HR: {max_hr:.0f} bpm")

    if filtered_sensors["filtered_power_meas_w"].size > 0 and np.any(np.isfinite(filtered_sensors["filtered_power_meas_w"])):
        avg_pow_meas = float(np.nanmean(np.clip(filtered_sensors["filtered_power_meas_w"], 0, None)))
        print(f"\n=== Measured Power (powermeter) ===")
        print(f"Avg measured pedaling power: {avg_pow_meas:.1f} W (non-negative samples)")

    # Final result dict (same keys as before, just filled from substeps)
    result = {
        "local_start_time": local_start_time,
        "local_end_time": local_end_time,
        "gpx_path": gpx_path,
        "timestep_min": timestep_min,
        "filtered_times": filtered["filtered_times"],
        "filtered_dists": filtered["filtered_dists"],
        "filtered_lats": filtered["filtered_lats"],
        "filtered_lons": filtered["filtered_lons"],
        "filtered_raw_alts": filtered["filtered_raw_alts"],
        "filtered_smoothed_alts": filtered["filtered_smoothed_alts"],
        "filtered_raw_speeds": filtered["filtered_raw_speeds"],
        "filtered_speeds_kph": filtered["filtered_speeds_kph"],
        "filtered_powers": filtered["filtered_powers"],
        "filtered_pwr_details": filtered["filtered_pwr_details"],
        "filtered_brake_powers": filtered["filtered_brake_powers"],
        "filtered_decel_powers": filtered["filtered_decel_powers"],
        "filtered_temp_front_disk_c": filtered["filtered_temp_front_disk_c"],
        "filtered_temp_rear_disk_c": filtered["filtered_temp_rear_disk_c"],
        "filtered_grades": filtered["filtered_grades"],
        "filtered_head_winds": filtered["filtered_head_winds"],
        "filtered_global_winds": filtered["filtered_global_winds"],
        "filtered_wind_dirs": filtered["filtered_wind_dirs"],
        "filtered_rho": filtered["filtered_rho"],
        "filtered_temp": filtered["filtered_temp"],
        "filtered_pres": filtered["filtered_pres"],
        "filtered_rhum": filtered["filtered_rhum"],
        "filtered_ride_type": filtered["filtered_ride_type"],
        # "wind_series_full": weather["wspd_ms"],
        # "wind_dir_series_full": weather["wdir_deg"],
        # "rho_series_full": weather["rho"],
        # "temp_series_full": weather["temp_c"],
        # "pres_series_full": weather["pres_hpa"],
        # "rhum_series_full": weather["rhum_pct"],
        # device channels
        **filtered_sensors,
        # gears
        "selected_front_teeth": gear_sim["selected_front_teeth"],
        "selected_rear_teeth": gear_sim["selected_rear_teeth"],
        "selected_gear_names": gear_sim["selected_gear_names"],
        "selected_cadences": gear_sim["selected_cadences"],
        "active_selected_gear_names": gear_sim["active_selected_gear_names"],
        "active_selected_cadences": gear_sim["active_selected_cadences"],
        "front_shifts": gear_sim["front_shifts"],
        "rear_shifts": gear_sim["rear_shifts"],
        "cadence_arr": gear_sim["cadence_arr"],
        "cadence_mean": gear_sim["cadence_mean"],
        "cadence_std": gear_sim["cadence_std"],
        "dist_max_cadence": gear_sim["dist_max_cadence"],
        "gear_combo_max_cadence": gear_sim["gear_combo_max_cadence"],
        "speed_max_cadence": gear_sim["speed_max_cadence"],
        "gear_list": gear_sim["gear_list"],
        # durations & energy
        "total_duration_min": total_duration_min,
        "active_duration_min": active_duration_min,
        "pause_duration_min": pause_duration_min,
        "avg_power_active": avg_power_active,
        "avg_power_with_freewheeling": avg_power_with_freewheeling,
        "kcal_burned": kcal_burned,
        "met_active": met_active,
        # breaks
        "breaks": breaks,
        "short_break_count": len(short_breaks),
        "short_break_total_min": short_break_total_min,
        "long_break_count": len(long_breaks),
        "long_break_total_min": long_break_total_min,
        "long_break_threshold_min": break_threshold_min,
        # climbs & segments
        "climbs": climbs,
        "downhills": downhills,
        "combined_climbs": combined_climbs,
        "combined_downhills": combined_downhills,
        "segments_5km": segments_5km,
        "max_climb_vam_m_per_h": max_vam,
        "avg_climb_power_w": avg_climb_power,
        "best_segment_5km_power_w": best_seg_power_5km,
        # metadata
        "start_time": start_time,
        "end_time": end_time,
        "total_distance_km": filtered["total_distance_km"],
        "elevation_gain_m": filtered["elevation_gain_m"],
        "cfg": cfg,
        "target_cadence": target_cadence,
        "use_wind_data": use_wind_data,
        # kcal timeseries
        "cum_kcal_active": kcal_data["cum_kcal_active"],
        "cum_kcal_base": kcal_data["cum_kcal_base"],
        "cum_kcal_total": kcal_data["cum_kcal_total"],
        "rel_time_s": kcal_data["rel_time_s"],
        # hydration / water loss (simple model)
        "water_model_mode": HYDRATION_MODEL_MODE,
        "water_rate_L_per_h": water_rate_L_per_h,
        "cum_water_loss_L": cum_water_loss_L,
        "total_water_loss_L": total_water_loss_L,
        "water_loss_limit_L": dmax_L,        
        # PHS / ISO 7933 thermophysiological model (native 1-min resolution)
        "phs_t_min": phs["t_min"],
        "phs_Tsk_C": phs["Tsk_C"],
        "phs_Tcr_C": phs["Tcr_C"],
        "phs_Tre_C": phs["Tre_C"],
        "phs_SWp_Wm2": phs["SWp_Wm2"],
        "phs_SWp_L_per_h": phs["SWp_L_per_h"],
        "phs_Ereq_Wm2": phs["Ereq_Wm2"],
        "phs_Ep_Wm2": phs["Ep_Wm2"],
        "phs_Emax_Wm2": phs["Emax_Wm2"],
        "phs_Conv_Wm2": phs["Conv_Wm2"],
        "phs_Rad_Wm2": phs["Rad_Wm2"],
        "phs_Cres_Wm2": phs["Cres_Wm2"],
        "phs_Eres_Wm2": phs["Eres_Wm2"],
        "phs_storage_Wm2": phs["storage_Wm2"],
        "phs_SWtot_g": phs["SWtot_g"],
        "phs_Dlimloss_min": phs["Dlimloss_min"],
        "phs_DlimTcr_min": phs["DlimTcr_min"],
        # PHS aligned to filtered samples (same length as filtered_dists)
        # "phs_aligned_Tsk_C": phs_aligned["Tsk_C_aligned"],
        # "phs_aligned_Tcr_C": phs_aligned["Tcr_C_aligned"],
        # "phs_aligned_Tre_C": phs_aligned["Tre_C_aligned"],
        # "phs_aligned_SWp_Wm2": phs_aligned["SWp_Wm2_aligned"],
        # "phs_aligned_SWp_L_per_h": phs_aligned["SWp_L_per_h_aligned"],
        # "phs_aligned_Ereq_Wm2": phs_aligned["Ereq_Wm2_aligned"],
        # "phs_aligned_Ep_Wm2": phs_aligned["Ep_Wm2_aligned"],
        # "phs_aligned_Emax_Wm2": phs_aligned["Emax_Wm2_aligned"],
        # "phs_aligned_Conv_Wm2": phs_aligned["Conv_Wm2_aligned"],
        # "phs_aligned_Rad_Wm2": phs_aligned["Rad_Wm2_aligned"],
        # "phs_aligned_Cres_Wm2": phs_aligned["Cres_Wm2_aligned"],
        # "phs_aligned_Eres_Wm2": phs_aligned["Eres_Wm2_aligned"],
        # "phs_aligned_storage_Wm2": phs_aligned["storage_Wm2_aligned"],
        "phs_dist_km": phs_dist_km,
        "phs": phs,

    }
    return result


def fun_energy_tooltip(kcal):
    """
    Returns a fun tooltip expressing kcal as equivalent foods and scaled activities.
    Non-food items dynamically scale their unit (e.g. "throwing 1200 snowballs")
    instead of "12 × throwing 100 snowballs".
    """

    # -----------------------------
    #  Realistic food kcal values
    # -----------------------------

    eatings = [
        "eating",
        "inhaling",
        "enjoying",
        "consuming",
        "devouring",
        "desintegrating"
    ]

    foods = [
        ("Haribo gummy bears", 6),             # kcal per piece
        ("Haribo gummy worms", 35),
        ("chocolate bars", 260),
        ("energy gels", 90),
        ("energy bars", 220),
        ("bananas", 105),
        ("apples", 85),
        ("bell peppers", 30),
        ("slices of pizza", 285),
        ("croissants", 260),
        ("donuts", 220),
        ("espressi", 3),
        ("liters of ketchup", 1100),
    ]

    # -----------------------------
    #  Fun non-food equivalents
    #  format: ("label_base", base_unit_count, kcal_per_unit)
    # -----------------------------
    actions = [
        ("throwing","snowballs", 100, 60),       # 60 kcal per 100 snowballs
        ("doing", "pushups", 100, 40),            # 40 kcal per 100 pushups
        ("lifting a bike", "meters", 1, .2),              # 5 kcal per lift
        ("charging", "smartphones", 1, 20),       # 20 kcal per charge
        ("boiling", "liters of water", 1, 80),              # 80 kcal per liter boiled
        ("petting dogs for", "minutes", 10, 20),              # 20 kcal per 10 min
        ("arguing", "hours on the internet", 1, 90),    # 90 kcal per hour (fun)
    ]

    random.shuffle(eatings)
    random.shuffle(foods)
    random.shuffle(actions)

    kcal = float(kcal)
    if kcal <= 0:
        return "No calories burned. Maybe the bike carried YOU? 😄"

    lines = []

    # -----------------------------
    #  Food equivalents
    # -----------------------------
    for (name, k), act in zip(foods[:3],eatings[:3]):
        n = kcal / k
        lines.append(f"{act} {n:.1f} {name}")

    # -----------------------------
    #  Action equivalents (scaled)
    # -----------------------------
    for label1,label2, base_count, k in actions[:2]:
        # total number of units of the base action
        units = kcal / k

        lines.append(f"{label1} {(units*base_count):.1f} {label2}")

    # -----------------------------
    #  Closing messages
    # -----------------------------
    closers = [
        "That's a respectable snack debt.",
        "Time to refuel! 🚴‍♂️🔥",
        "Your mitochondria salute you.",
        "Definitely earned a treat after this one.",
        "A heroic metabolic achievement.",
    ]
    end = random.choice(closers)

    return "Burned energy equivalents:\n" + "\n".join(lines) + "\n\n" + end

# =============================================================================
# Small “log” helper (Streamlit-friendly)
# =============================================================================
def log(msg: str):
    st.session_state.setdefault("log_lines", [])
    st.session_state["log_lines"].append(str(msg))


def print(str_):
    details_lines.append(str(str_))

def render_log():
    lines = st.session_state.get("log_lines", [])
    if not lines:
        st.info("Log is empty.")
        return
    st.code("\n".join(lines[-400:]), language="text")



# =============================================================================
# Config helpers
# =============================================================================
def build_config_from_ui(state) -> "RiderBikeConfig":
    
    global BRAKE_FRONT_DIAMETER_MM, BRAKE_REAR_DIAMETER_MM
    global BRAKE_ADD_COOLING_FACTOR, BRAKE_PERFORATION_FACTOR, BRAKE_DISTRIBUTION_FRONT
    global EFFICIENCY, MAX_SPEED, MAX_GRADE, MAX_PWR, BASE_MET_DURING_ACTIVITY
    global HYSTERESIS_REAR_RPM, MIN_SHIFT_INTERVAL_REAR_SEC
    global MAX_MAP_POINTS, MIN_VALID_SPEED, MIN_ACTIVE_SPEED
    global COMBINE_BREAK_MAX_TIME_DIFF, COMBINE_BREAK_MAX_DISTANCE
    global G
    global CLIMB_MIN_DISTANCE_KM, CLIMB_MIN_ELEVATION_GAIN_M, CLIMB_END_AVERAGE_GRADE_PCT, CLIMB_END_WINDOW_SIZE_M, MAX_DIST_BETWEEN_CLIMBS_M
    global SMOOTHING_WINDOW_SIZE_S
    global REFERENCE_CDA_VALUE
    global AMBIENT_TEMP_C, AMBIENT_RHO, AMBIENT_PRES_HPA, AMBIENT_RHUM_PCT, AMBIENT_WIND_DIR_DEG, AMBIENT_WIND_SPEED_MS

    EFFICIENCY = float(state["EFFICIENCY"])
    BASE_MET_DURING_ACTIVITY = float(state["BASE_MET_DURING_ACTIVITY"])

    MAX_SPEED = float(state["MAX_SPEED"])
    MAX_GRADE = float(state["MAX_GRADE"])
    MAX_PWR = float(state["MAX_PWR"])
    MIN_ACTIVE_SPEED = float(state["MIN_ACTIVE_SPEED"])
    MIN_VALID_SPEED = float(state["MIN_VALID_SPEED"])

    HYSTERESIS_REAR_RPM = int(state["HYSTERESIS_REAR_RPM"])
    MIN_SHIFT_INTERVAL_REAR_SEC = int(state["MIN_SHIFT_INTERVAL_REAR_SEC"])
    MAX_MAP_POINTS = int(state["MAX_MAP_POINTS"])
    COMBINE_BREAK_MAX_TIME_DIFF = int(state["COMBINE_BREAK_MAX_TIME_DIFF"])
    COMBINE_BREAK_MAX_DISTANCE = int(state["COMBINE_BREAK_MAX_DISTANCE"])
    G = float(state["GRAVITY"])
    CLIMB_MIN_DISTANCE_KM = float(state["CLIMB_MIN_DISTANCE_KM"])
    CLIMB_MIN_ELEVATION_GAIN_M = int(state["CLIMB_MIN_ELEVATION_GAIN_M"])
    CLIMB_END_AVERAGE_GRADE_PCT = float(state["CLIMB_END_AVERAGE_GRADE_PCT"])
    CLIMB_END_WINDOW_SIZE_M = int(state["CLIMB_END_WINDOW_SIZE_M"])
    MAX_DIST_BETWEEN_CLIMBS_M = int(state["MAX_DIST_BETWEEN_CLIMBS_M"])
    SMOOTHING_WINDOW_SIZE_S = int(state["SMOOTHING_WINDOW_SIZE_S"])
    BRAKE_FRONT_DIAMETER_MM = float(state["BRAKE_FRONT_DIAMETER_MM"])
    BRAKE_REAR_DIAMETER_MM = float(state["BRAKE_REAR_DIAMETER_MM"])
    BRAKE_ADD_COOLING_FACTOR = float(state["BRAKE_ADD_COOLING_FACTOR"])
    BRAKE_PERFORATION_FACTOR = float(state["BRAKE_PERFORATION_FACTOR"])
    BRAKE_DISTRIBUTION_FRONT = float(state["BRAKE_DISTRIBUTION_FRONT"])
    REFERENCE_CDA_VALUE = float(state["REFERENCE_CDA_VALUE"])

    AMBIENT_TEMP_C = float(state["AMBIENT_TEMP_C"])
    AMBIENT_RHO = float(state["AMBIENT_RHO"])
    AMBIENT_PRES_HPA = float(state["AMBIENT_PRES_HPA"])
    AMBIENT_RHUM_PCT = float(state["AMBIENT_RHUM_PCT"])
    AMBIENT_WIND_DIR_DEG = float(state["AMBIENT_WIND_DIR_DEG"])
    AMBIENT_WIND_SPEED_MS = float(state["AMBIENT_WIND_SPEED_MS"])

    # Keep your original scaling logic for CdA with height
    # cda_scaled = REFERENCE_CDA_VALUE * (1 - 0.5*(1-(height/REFERENCE_HEIGHT_FOR_CDA)))
    height = float(state["body_height"])
    CDA_VALUE = float(REFERENCE_CDA_VALUE) * (1.0 - 0.5 * (1.0 - (height / REFERENCE_HEIGHT_FOR_CDA)))
    calculated_crr = float(state["REFERENCE_ROLLING_LOSS"]) / (REFERENCE_LOAD_FOR_CRR_KG * 9.81 * REFERENCE_SPEED_FOR_CRR_MS)


    front_teeth = [int(x.strip()) for x in str(state["front_teeth"]).split(",") if x.strip()]
    rear_teeth  = [int(x.strip()) for x in str(state["rear_teeth"]).split(",") if x.strip()]

    return RiderBikeConfig(
        body_mass=float(state["body_mass"]),
        body_height=height,
        bike_mass=float(state["bike_mass"]),
        extra_mass=float(state["extra_mass"]),
        cda=float(CDA_VALUE),
        crr=calculated_crr,
        drivetrain_loss=float(state["drivetrain_loss"]),
        wheel_circumference=float(state["wheel_circ"]),
        front_der_teeth=front_teeth,
        rear_der_teeth=rear_teeth,
        draft_effect=float(state["draft_effect"]),
        draft_effect_dh=float(state["draft_effect_dh"]),
        draft_effect_uh=float(state["draft_effect_uh"]),
    )


def _arr(x, dtype=float):
    if x is None:
        return None
    a = np.asarray(x)
    try:
        return a.astype(dtype)
    except Exception:
        return np.asarray(a, dtype=object)

def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return 0

def _clip_same_len(*arrays):
    """Return arrays clipped to the minimum common length (ignores None)."""
    arrs = [a for a in arrays if a is not None]
    if not arrs:
        return arrays
    n = min(len(a) for a in arrs)
    out = []
    for a in arrays:
        out.append(a[:n] if a is not None else None)
    return tuple(out)

def segment_mean_by_distance(dists_km, values, spacing_km):
    d = np.asarray(dists_km, dtype=float)
    v = np.asarray(values, dtype=float)
    n = min(d.size, v.size)
    if n == 0 or spacing_km <= 0:
        return v.copy()
    d = d[:n]; v = v[:n]
    total_dist = d[-1]
    if total_dist <= 0:
        return v.copy()

    edges = np.arange(0.0, total_dist + spacing_km, spacing_km)
    if edges[-1] < total_dist - 1e-9:
        edges = np.append(edges, total_dist)

    out = v.copy()
    for i in range(len(edges) - 1):
        x0, x1 = edges[i], edges[i + 1]
        mask = (d >= x0) & (d < x1) if i < len(edges) - 2 else (d >= x0) & (d <= x1 + 1e-9)
        if np.any(mask):
            out[mask] = np.nanmean(v[mask])
    return out

def _add_ride_type_shading(fig, d_km, ride_type, climb_color="red", downhill_color="green", opacity=0.12):
    """Adds vrect shading for climb/downhill segments (like your Qt axvspan)."""
    if ride_type is None:
        return
    ride_type = np.asarray(ride_type, dtype=object)
    if len(ride_type) != len(d_km) or len(d_km) < 2:
        return

    def add_segments(target, color):
        in_seg = False
        s = 0
        for i in range(len(ride_type)):
            is_t = (ride_type[i] == target)
            if is_t and not in_seg:
                in_seg = True
                s = i
            elif (not is_t) and in_seg:
                e = i - 1
                fig.add_vrect(x0=float(d_km[s]), x1=float(d_km[e]), fillcolor=color, opacity=opacity, line_width=0)
                in_seg = False
        if in_seg:
            fig.add_vrect(x0=float(d_km[s]), x1=float(d_km[-1]), fillcolor=color, opacity=opacity, line_width=0)

    add_segments("climb", climb_color)
    add_segments("downhill", downhill_color)

# =============================================================================
# Plot builders (matplotlib figs)
# =============================================================================
def plot_profiles(result, highlight_climbs=False):
    d   = _arr(result.get("filtered_dists"))
    alt = _arr(result.get("filtered_smoothed_alts"))
    grd = _arr(result.get("filtered_grades"))
    spd = _arr(result.get("filtered_speeds_kph"))
    pwr = _arr(result.get("filtered_powers"))
    wind= _arr(result.get("filtered_head_winds"))
    rho = _arr(result.get("filtered_rho"))
    ride_type = result.get("filtered_ride_type", None)

    d, alt, grd, spd, pwr, wind, rho = _clip_same_len(d, alt, grd, spd, pwr, wind, rho)
    if d is None or len(d) == 0:
        return go.Figure().add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": True}]]
    )

    fig.add_trace(go.Scatter(x=d, y=alt, name="Elevation [m]"), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=d, y=grd, name="Grade [%]"), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=d, y=spd, name="Speed [km/h]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=d, y=pwr, name="Power [W]"), row=3, col=1)

    fig.add_trace(go.Scatter(x=d, y=wind, name="Headwind [m/s]"), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=d, y=rho, name="ρ [kg/m³]", line=dict(dash="dash")), row=4, col=1, secondary_y=True)

    fig.update_yaxes(title_text="Elev. [m]", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Grade [%]", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Speed [km/h]", row=2, col=1)
    fig.update_yaxes(title_text="Power [W]", row=3, col=1)
    fig.update_yaxes(title_text="Headwind [m/s]", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="ρ [kg/m³]", row=4, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Distance [km]", row=4, col=1)

    fig.update_layout(height=850, margin=dict(l=30, r=20, t=40, b=30), legend=dict(orientation="h"))

    if highlight_climbs:
        _add_ride_type_shading(fig, d, ride_type)

    return fig

def plot_distributions(result, max_speed=120.0, max_pwr=1500.0):
    spd = _arr(result.get("filtered_speeds_kph"))
    pwr = _arr(result.get("filtered_powers"))
    grd = _arr(result.get("filtered_grades"))
    cadence_arr = _arr(result.get("active_selected_cadences", []))
    gear_list = result.get("gear_list", [])
    gear_usage_names = result.get("active_selected_gear_names", [])
    timestep_min = float(result.get("timestep_min", 1.0))

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=("Power Distribution", "Speed Distribution", "Grade Distribution",
                        "Cadence Distribution (active)", "", "Gear Usage (F x R)"),
        specs=[[{}, {}, {}],
               [{}, {"colspan": 2}, None]],
        horizontal_spacing=0.08, vertical_spacing=0.12
    )

    # Power hist (time-weighted)
    if pwr is not None and len(pwr) > 0:
        bins = np.linspace(-max_pwr, max_pwr, 50)
        counts, edges = np.histogram(pwr, bins=bins)
        minutes = counts * timestep_min
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = (edges[1:] - edges[:-1])
        fig.add_trace(go.Bar(x=centers, y=minutes, width=widths, name="Power"), row=1, col=1)
    fig.update_xaxes(title_text="Power [W]", row=1, col=1)
    fig.update_yaxes(title_text="Time [min]", row=1, col=1)

    # Speed hist
    if spd is not None and len(spd) > 0:
        bins = np.linspace(0, max_speed, 50)
        counts, edges = np.histogram(spd, bins=bins)
        minutes = counts * timestep_min
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = (edges[1:] - edges[:-1])
        fig.add_trace(go.Bar(x=centers, y=minutes, width=widths, name="Speed"), row=1, col=2)
    fig.update_xaxes(title_text="Speed [km/h]", row=1, col=2)
    fig.update_yaxes(title_text="Time [min]", row=1, col=2)

    # Grade hist
    if grd is not None and len(grd) > 0:
        bins = np.linspace(-15, 15, 50)
        counts, edges = np.histogram(grd, bins=bins)
        minutes = counts * timestep_min
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = (edges[1:] - edges[:-1])
        fig.add_trace(go.Bar(x=centers, y=minutes, width=widths, name="Grade"), row=1, col=3)
    fig.update_xaxes(title_text="Grade [%]", row=1, col=3)
    fig.update_yaxes(title_text="Time [min]", row=1, col=3)

    # Cadence hist
    if cadence_arr is not None and len(cadence_arr) > 0:
        bins = np.arange(0, 160, 2)
        counts, edges = np.histogram(cadence_arr, bins=bins)
        minutes = counts * timestep_min
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = (edges[1:] - edges[:-1])
        fig.add_trace(go.Bar(x=centers, y=minutes, width=widths, name="Cadence"), row=2, col=1)
    fig.update_xaxes(title_text="Cadence [RPM]", row=2, col=1)
    fig.update_yaxes(title_text="Time [min]", row=2, col=1)

    # Gear usage
    if gear_list and gear_usage_names:
        gear_labels = [g["name"] for g in gear_list]
        gear_counts = np.array([gear_usage_names.count(lbl) for lbl in gear_labels], dtype=float) * timestep_min
        fig.add_trace(go.Bar(x=gear_labels, y=gear_counts, name="Gear usage"), row=2, col=2)
        fig.update_xaxes(tickangle=90, row=2, col=2)
        fig.update_yaxes(title_text="Time [min]", row=2, col=2)

    fig.update_layout(height=750, margin=dict(l=30, r=20, t=50, b=30), showlegend=False)
    return fig


def plot_segments(result):
    segs = result.get("segments_5km", [])
    if not segs:
        return go.Figure().add_annotation(
            text="No 5 km segments available", x=0.5, y=0.5, showarrow=False
        )

    labels = [f"{s.get('start_km',0):.0f}-{s.get('end_km',0):.0f} km" for s in segs]
    x0 = np.arange(len(labels), dtype=float)

    # signed component powers (IMPORTANT: keep original signs)
    p_a  = np.array([s.get("avg_pwr_a_w",  0.0) for s in segs], float)
    p_dc = np.array([s.get("avg_pwr_dc_w", 0.0) for s in segs], float)
    p_pt = np.array([s.get("avg_pwr_pt_w", 0.0) for s in segs], float)
    p_r  = np.array([s.get("avg_pwr_r_w",  0.0) for s in segs], float)
    p_g  = np.array([s.get("avg_pwr_g_w",  0.0) for s in segs], float)
    p_d  = np.array([s.get("avg_pwr_d_w",  0.0) for s in segs], float)
    p_b  = np.array([s.get("avg_pwr_b_w",  0.0) for s in segs], float)

    p_tot = np.array([s.get("avg_power_w", 0.0) for s in segs], float)

    components = [
        ("Acceleration", p_a),
        ("Deceleration", p_dc),
        ("Powertrain",   p_pt),
        ("Rolling",      p_r),
        ("Slope",        p_g),
        ("Drag",         p_d),
        ("Braking",      p_b),
    ]

    # visual parameters
    base_width = 0.90
    shrink = 0.10
    min_width = 0.18

    # right edge of the bar group (constant!)
    right_edge = x0 + 0.5 * base_width

    fig = go.Figure()

    # running signed stack
    base = np.zeros(len(x0), dtype=float)

    for i, (name, y) in enumerate(components):
        y = np.asarray(y, dtype=float)
        width = max(min_width, base_width - i * shrink)

        # RIGHT-ALIGNED centers
        x = right_edge - 0.5 * width

        fig.add_trace(go.Bar(
            x=x,
            y=y,
            base=base,
            width=width,
            name=name,
            opacity=0.9,
            hovertemplate=(
                f"{name}<br>"
                "%{customdata}<br>"
                "Δ = %{y:.1f} W<br>"
                "Start = %{base:.1f} W"
                "<extra></extra>"
            ),
            customdata=labels,
        ))

        # advance running sum (THIS is the stacking rule)
        base = base + y

    # total power reference
    fig.add_trace(go.Scatter(
        x=x0,
        y=p_tot,
        name="Avg. pedaling power",
        mode="lines+markers",
        hovertemplate="%{x}<br>%{y:.0f} W<extra></extra>",
    ))

    fig.update_layout(
        barmode="overlay",  # stacking is manual via base=
        title="5 km segments – Power Components",
        xaxis_title="5 km segments",
        yaxis_title="Average power [W]",
        height=520,
        margin=dict(l=30, r=20, t=60, b=120),
        legend=dict(orientation="h"),
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=x0,
        ticktext=labels,
        tickangle=45
    )

    return fig

def plot_human(result):
    d = _arr(result.get("filtered_dists"))
    if d is None or len(d) == 0:
        return go.Figure().add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)

    hr = _arr(result.get("filtered_hr_bpm", np.zeros_like(d)))
    kcal_total = _arr(result.get("cum_kcal_total", np.zeros_like(d)))
    cum_water_loss_L = _arr(result.get("cum_water_loss_L", np.zeros_like(d)))

    # Clip the base series to same length
    d, hr, kcal_total, cum_water_loss_L = _clip_same_len(d, hr, kcal_total, cum_water_loss_L)
    if d is None or len(d) == 0:
        return go.Figure().add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)

    # Decide which PHS mode we’re in (match your original logic)
    has_raw_phs = ("phs" in result) and ("phs_dist_km" in result) and isinstance(result.get("phs"), dict)
    phs = result.get("phs", {}) if has_raw_phs else {}

    # Build 4 rows, with secondary y on row2 and row4 (like your twinx usage)
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=False, vertical_spacing=0.055,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
        ],
        subplot_titles=("Heart rate", "Energy + water", "PHS temperatures", "PHS sweat + heat fluxes")
    )

    # ----------------
    # Row 1: HR vs d
    # ----------------
    fig.add_trace(go.Scatter(x=d, y=hr, name="Heartrate", mode="lines"), row=1, col=1)
    fig.update_yaxes(title_text="HR [1/min]", row=1, col=1)

    # -----------------------------------
    # Row 2: kcal vs d + water (secondary)
    # -----------------------------------
    fig.add_trace(go.Scatter(x=d, y=kcal_total, name="Cum Energy (total)", mode="lines"),
                  row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=d, y=cum_water_loss_L, name="Cum Water Loss", mode="lines",
                             line=dict(dash="dash")),
                  row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Energy [kcal]", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Water Loss [L]", row=2, col=1, secondary_y=True)

    # ----------------------------
    # Row 3 + 4: PHS detailed branch
    # ----------------------------
    if has_raw_phs:
        pdist = _arr(result.get("phs_dist_km"))
        # Pull raw PHS arrays (as in your matplotlib debug branch)
        Tcr = _arr(phs.get("Tcr_C"))
        Tsk = _arr(phs.get("Tsk_C"))
        SW  = _arr(phs.get("SWp_L_per_h"))
        Ep  = _arr(phs.get("Ep_Wm2"))
        Conv= _arr(phs.get("Conv_Wm2"))
        Rad = _arr(phs.get("Rad_Wm2"))

        # Clip everything to pdist length (robust)
        def clip_to_x(x, y):
            if x is None or y is None:
                return None, None
            n = min(len(x), len(y))
            if n <= 0:
                return None, None
            return np.asarray(x[:n], float), np.asarray(y[:n], float)

        # Row 3: temps vs pdist
        xT, yT = clip_to_x(pdist, Tcr)
        if xT is not None:
            fig.add_trace(go.Scatter(x=xT, y=yT, name="Core T (PHS)", mode="lines"), row=3, col=1)
        xS, yS = clip_to_x(pdist, Tsk)
        if xS is not None:
            fig.add_trace(go.Scatter(x=xS, y=yS, name="Skin T (PHS)", mode="lines"), row=3, col=1)

        fig.update_xaxes(title_text="Distance [km] (PHS grid)", row=3, col=1)
        fig.update_yaxes(title_text="Temp [°C]", row=3, col=1)

        # Row 4: sweat left, heat flux right vs pdist
        xSW, ySW = clip_to_x(pdist, SW)
        if xSW is not None:
            fig.add_trace(go.Scatter(x=xSW, y=ySW, name="Sweat [L/h] (PHS)", mode="lines"),
                          row=4, col=1, secondary_y=False)

        xEp, yEp = clip_to_x(pdist, Ep)
        if xEp is not None:
            fig.add_trace(go.Scatter(x=xEp, y=yEp, name="Evap. loss [W/m²] (PHS)", mode="lines"),
                          row=4, col=1, secondary_y=True)

        # Conv+Rad
        if Conv is not None and Rad is not None and pdist is not None:
            xC, yC = clip_to_x(pdist, Conv)
            xR, yR = clip_to_x(pdist, Rad)
            if xC is not None and xR is not None:
                n = min(len(yC), len(yR), len(xC), len(xR))
                dry = np.asarray(yC[:n], float) + np.asarray(yR[:n], float)
                fig.add_trace(go.Scatter(x=np.asarray(xC[:n], float), y=dry,
                                         name="Dry heat (conv+rad) [W/m²] (PHS)",
                                         mode="lines", line=dict(dash="dot")),
                              row=4, col=1, secondary_y=True)

        fig.update_xaxes(title_text="Distance [km] (PHS grid)", row=4, col=1)
        fig.update_yaxes(title_text="Sweat [L/h]", row=4, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Heat flux [W/m²]", row=4, col=1, secondary_y=True)

    else:
        # ----------------------------
        # Fallback: aligned arrays on d
        # ----------------------------
        Tcr = _arr(result.get("phs_aligned_Tcr_C"))
        Tsk = _arr(result.get("phs_aligned_Tsk_C"))
        SW  = _arr(result.get("phs_aligned_SWp_L_per_h"))
        Ep  = _arr(result.get("phs_aligned_Ep_Wm2"))
        Conv= _arr(result.get("phs_aligned_Conv_Wm2"))
        Rad = _arr(result.get("phs_aligned_Rad_Wm2"))

        # Clip each to d
        def clip_to_d(y):
            if y is None:
                return None
            n = min(len(d), len(y))
            if n <= 0:
                return None
            return np.asarray(y[:n], float)

        Tcr = clip_to_d(Tcr)
        Tsk = clip_to_d(Tsk)
        SW  = clip_to_d(SW)
        Ep  = clip_to_d(Ep)
        Conv= clip_to_d(Conv)
        Rad = clip_to_d(Rad)

        if Tcr is not None:
            fig.add_trace(go.Scatter(x=d[:len(Tcr)], y=Tcr, name="Core T (aligned)", mode="lines"), row=3, col=1)
        if Tsk is not None:
            fig.add_trace(go.Scatter(x=d[:len(Tsk)], y=Tsk, name="Skin T (aligned)", mode="lines"), row=3, col=1)

        fig.update_xaxes(title_text="Distance [km]", row=3, col=1)
        fig.update_yaxes(title_text="Temp [°C]", row=3, col=1)

        if SW is not None:
            fig.add_trace(go.Scatter(x=d[:len(SW)], y=SW, name="Sweat [L/h] (aligned)", mode="lines"),
                          row=4, col=1, secondary_y=False)

        if Ep is not None:
            fig.add_trace(go.Scatter(x=d[:len(Ep)], y=Ep, name="Evap. loss [W/m²] (aligned)", mode="lines"),
                          row=4, col=1, secondary_y=True)

        if Conv is not None and Rad is not None:
            n = min(len(Conv), len(Rad), len(d))
            dry = np.asarray(Conv[:n], float) + np.asarray(Rad[:n], float)
            fig.add_trace(go.Scatter(x=d[:n], y=dry, name="Dry heat (conv+rad) [W/m²] (aligned)",
                                     mode="lines", line=dict(dash="dot")),
                          row=4, col=1, secondary_y=True)

        fig.update_xaxes(title_text="Distance [km]", row=4, col=1)
        fig.update_yaxes(title_text="Sweat [L/h]", row=4, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Heat flux [W/m²]", row=4, col=1, secondary_y=True)

        # Optional: if literally nothing was plotted in row 3/4, show a hint
        if (Tcr is None and Tsk is None and SW is None and Ep is None and Conv is None and Rad is None):
            fig.add_annotation(
                text="No PHS data found (missing phs+phs_dist_km and missing phs_aligned_*).",
                x=0.5, y=0.08, xref="paper", yref="paper", showarrow=False
            )

    # Cosmetics
    fig.update_layout(
        height=900,
        margin=dict(l=30, r=20, t=50, b=30),
        legend=dict(orientation="h"),
    )

    return fig

def plot_environment(result):
    d = _arr(result.get("filtered_dists"))
    if d is None or len(d) == 0:
        return go.Figure().add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)

    ws   = _arr(result.get("filtered_global_winds"))
    wd   = _arr(result.get("filtered_wind_dirs"))
    rho  = _arr(result.get("filtered_rho"))
    temp = _arr(result.get("filtered_temp"))
    pres = _arr(result.get("filtered_pres"))
    rhum = _arr(result.get("filtered_rhum"))

    d, ws, wd, rho, temp, pres, rhum = _clip_same_len(d, ws, wd, rho, temp, pres, rhum)

    fig = make_subplots(
        rows=3, cols=2, shared_xaxes=True, vertical_spacing=0.08, horizontal_spacing=0.08,
        subplot_titles=("Windspeed", "Wind direction", "Air density", "Air temperature", "Air pressure", "Relative humidity")
    )

    fig.add_trace(go.Scatter(x=d, y=ws,  name="Wind [m/s]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=wd,  name="Dir [°]"),    row=1, col=2)
    fig.add_trace(go.Scatter(x=d, y=rho, name="ρ"),          row=2, col=1)
    fig.add_trace(go.Scatter(x=d, y=temp,name="Temp"),       row=2, col=2)
    fig.add_trace(go.Scatter(x=d, y=pres,name="Pressure"),   row=3, col=1)
    fig.add_trace(go.Scatter(x=d, y=rhum,name="RH"),         row=3, col=2)

    fig.update_yaxes(title_text="m/s", row=1, col=1)
    fig.update_yaxes(title_text="°",   row=1, col=2)
    fig.update_yaxes(title_text="kg/m³", row=2, col=1)
    fig.update_yaxes(title_text="°C",  row=2, col=2)
    fig.update_yaxes(title_text="hPa", row=3, col=1)
    fig.update_yaxes(title_text="%",   row=3, col=2)

    fig.update_xaxes(title_text="Distance [km]", row=3, col=1)
    fig.update_xaxes(title_text="Distance [km]", row=3, col=2)

    fig.update_layout(height=750, margin=dict(l=30, r=20, t=50, b=30), showlegend=False)
    return fig


# =============================================================================
# Map builder (Folium)
# =============================================================================
def build_map(result, mode: str, marker_spacing_km: float, long_break_thr_min: float):
    lats_full = np.asarray(result["filtered_lats"], dtype=float)
    lons_full = np.asarray(result["filtered_lons"], dtype=float)
    d_full = np.asarray(result["filtered_dists"], dtype=float)
    timestamps = np.asarray(result["filtered_times"])
    total_break_duration = result["long_break_total_min"] + result["short_break_total_min"]
    if lats_full.size == 0 or lons_full.size == 0:
        return None, None, None

    # metric selection
    if mode == "Speed [km/h]":
        values_full = np.asarray(result["filtered_speeds_kph"], dtype=float)
        caption = "Speed [km/h]"
    elif mode == "Power [W]":
        values_full = np.asarray(result["filtered_powers"], dtype=float)
        caption = "Power [W]"
    elif mode == "Grade [%]":
        values_full = np.asarray(result["filtered_grades"], dtype=float)
        caption = "Grade [%]"
    elif mode == "Headwind [m/s]":
        values_full = np.asarray(result["filtered_head_winds"], dtype=float)
        caption = "Headwind [m/s]"
    elif mode == "Air density [kg/m³]":
        values_full = np.asarray(result["filtered_rho"], dtype=float)
        caption = "Air density [kg/m³]"
    elif mode == "Cadence [RPM]":
        values_full = np.asarray(result["selected_cadences"], dtype=float)
        caption = "Cadence [RPM]"
    elif mode.startswith("Climbs"):
        values_full = np.asarray(result["filtered_grades"], dtype=float)
        caption = "Grade [%]"
    elif mode == "None (single color)":
        values_full = None
        caption = ""
    else:
        values_full = np.asarray(result["filtered_speeds_kph"], dtype=float)
        caption = "Speed [km/h]"

    # segment-mean mode (same-length)
    if SEGMENT_MEAN_MODE and values_full is not None and d_full.size > 1 and marker_spacing_km > 0:
        values_full = segment_mean_by_distance(d_full, values_full, float(marker_spacing_km))

    # subsample for performance (drawing only)
    n = len(lats_full)
    idx = np.linspace(0, n - 1, MAX_MAP_POINTS, dtype=int) if n > MAX_MAP_POINTS else np.arange(n, dtype=int)

    lats = lats_full[idx]
    lons = lons_full[idx]
    vals = np.asarray(values_full, dtype=float)[idx] if values_full is not None else None

    center_lat = float(np.mean(lats_full))
    center_lon = float(np.mean(lons_full))
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

    positions = list(zip(lats.tolist(), lons.tolist()))

    # climb/downhill masks if needed
    is_climb_full = np.zeros_like(d_full, dtype=bool)
    combined_climbs = result.get("combined_climbs", result.get("climbs", []))
    for c in (combined_climbs or []):
        x0 = c.get("start_distance_km", 0.0)
        x1 = c.get("end_distance_km", 0.0)
        is_climb_full |= (d_full >= x0) & (d_full <= x1)
    is_climb = is_climb_full[idx] if len(is_climb_full) == len(d_full) else np.zeros_like(idx, dtype=bool)

    is_downhill_full = np.zeros_like(d_full, dtype=bool)
    combined_downhills = result.get("combined_downhills", result.get("downhills", []))
    for c in (combined_downhills or []):
        x0 = c.get("start_distance_km", 0.0)
        x1 = c.get("end_distance_km", 0.0)
        is_downhill_full |= (d_full >= x0) & (d_full <= x1)
    is_downhill = is_downhill_full[idx] if len(is_downhill_full) == len(d_full) else np.zeros_like(idx, dtype=bool)

    if mode.startswith("Climbs") and ((combined_climbs or []) or (combined_downhills or [])):
        for i in range(len(positions) - 1):
            in_climb = bool(is_climb[i] or is_climb[i + 1])
            in_down = bool(is_downhill[i] or is_downhill[i + 1])
            color = "red" if in_climb else ("green" if in_down else "blue")
            folium.PolyLine([positions[i], positions[i + 1]], color=color, weight=4, opacity=0.9).add_to(m)
    else:
        if vals is None or vals.size < 2 or np.nanmin(vals) == np.nanmax(vals):
            folium.PolyLine(positions, color="blue", weight=3, opacity=0.9).add_to(m)
        else:
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            colormap = cm.linear.viridis.scale(vmin, vmax)
            if caption:
                colormap.caption = caption
            colormap.add_to(m)
            val_list = vals.tolist()
            for i in range(len(positions) - 1):
                folium.PolyLine([positions[i], positions[i + 1]], color=colormap(val_list[i]),
                                weight=4, opacity=0.9).add_to(m)

    # Start/end markers
    full_positions = list(zip(lats_full.tolist(), lons_full.tolist()))
    folium.Marker(full_positions[0], popup="Start").add_to(m)
    folium.Marker(full_positions[-1],
                  popup=(
                      f"End<br>"
                      f"Total\u00A0Duration:\u00A0{time_to_readable(t_seconds=timestamps[-1].timestamp() - timestamps[0].timestamp())}<br>"
                      f"Ride\u00A0Duration:\u00A0{time_to_readable(t_minutes=(timestamps[-1].timestamp()/60. - timestamps[0].timestamp()/60.)-total_break_duration)}<br>"
                      f"Break\u00A0Duration:\u00A0{time_to_readable(t_minutes=total_break_duration)}<br>"
                         )
                  ).add_to(m)

    # Distance markers
    if marker_spacing_km > 0 and d_full.size > 0:
        next_mark = float(marker_spacing_km)
        for i, (d_km, lat, lon) in enumerate(zip(d_full, lats_full, lons_full)):
            if d_km + 1e-9 >= next_mark:
                t_minutes = timestamps[i].timestamp()/60. - timestamps[0].timestamp()/60.
                folium.CircleMarker(
                    location=[float(lat), float(lon)],
                    radius=3,
                    color="red",
                    fill=True,
                    fillOpacity=0.9,
                    popup=(
                        f"Distance:\u00A0{next_mark:.1f}\u00A0km<br>"
                        f"Time:\u00A0{time_to_readable(t_minutes=t_minutes)}"
                    )
                ).add_to(m)
                next_mark += float(marker_spacing_km)

    # combine break markers close to each other (time + distance)
    combined_breaks = []
    breaks = result.get("breaks", [])
    if breaks:
        breaks_sorted = sorted(breaks, key=lambda b: b["time_min"])
        current = breaks_sorted[0].copy()
        current["break_count"] = 1  # number of merged breaks

        for b in breaks_sorted[1:]:
            # time difference between end of current cluster and start of next break
            dt_min = b["time_min"] - (current["time_min"] + current["duration_min"])
            try:
                dist_m = geodesic(
                    (current["lat"], current["lon"]),
                    (b["lat"], b["lon"])
                ).meters
            except ValueError:
                # if we get bad coordinates for any reason, don't merge
                dist_m = COMBINE_BREAK_MAX_DISTANCE + 1.0

            if dt_min <= COMBINE_BREAK_MAX_TIME_DIFF and dist_m <= COMBINE_BREAK_MAX_DISTANCE:
                # merge b into current cluster with weighted averages
                n = current["break_count"]

                # total duration is just sum
                current["duration_min"] += b["duration_min"]

                # distance: average over breaks
                current["distance_km"] = (current["distance_km"] * n + b["distance_km"]) / (n + 1)

                # lat/lon: average over breaks
                current["lat"] = (current["lat"] * n + b["lat"]) / (n + 1)
                current["lon"] = (current["lon"] * n + b["lon"]) / (n + 1)

                current["break_count"] = n + 1

                # upgrade to long break if combined duration exceeds threshold
                if current["duration_min"] >= long_break_thr_min:
                    current["type"] = "long"
            else:
                combined_breaks.append(current)
                current = b.copy()
                current["break_count"] = 1

        combined_breaks.append(current)
    else:
        combined_breaks = []

    # Markers for breaks
    for b in combined_breaks:
        lat_b = b["lat"]
        lon_b = b["lon"]
        dur = b["duration_min"]
        btype = b["type"]
        btime = b["time_min"]
        bdist = b["distance_km"]
        bcount = b.get("break_count", 1)
        if btype == "long":
            color = "darkgreen"
            radius = 6
            label = "Long break"
        else:
            color = "orange"
            radius = 4
            label = "Short break"
        folium.CircleMarker(
            location=[lat_b, lon_b],
            radius=radius,
            color=color,
            fill=True,
            fillOpacity=0.9,
            popup=f"{label}:<br>"
                f"Time:\u00A0{time_to_readable(t_minutes=btime)}<br>"
                f"Duration:\u00A0{time_to_readable(t_minutes=dur)}<br>"
                f"Distance:\u00A0{bdist:.2f}\u00A0km<br>"
                f"{f'Combined\u00A0Breaks:\u00A0{bcount}' if bcount > 1 else ''}"
        ).add_to(m)

    return m, values_full, caption

def plot_cmap_metric(result, y_label, values_full, cursor_idx=None, spacing_km=5.0, segment_mean_mode=True):
    d = _arr(result.get("filtered_dists"))
    if values_full is None or d is None or len(d) == 0:
        return go.Figure().add_annotation(text="No metric to plot", x=0.5, y=0.5, showarrow=False)

    vals = _arr(values_full)
    d, vals = _clip_same_len(d, vals)
    if len(d) == 0:
        return go.Figure().add_annotation(text="No metric to plot", x=0.5, y=0.5, showarrow=False)

    v_plot = vals
    if segment_mean_mode and spacing_km and len(d) > 1:
        v_plot = segment_mean_by_distance(d, vals, float(spacing_km))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d, y=v_plot, name=y_label))
    fig.update_layout(height=260, margin=dict(l=30, r=20, t=20, b=30), showlegend=False)
    fig.update_xaxes(title_text="Distance [km]")
    fig.update_yaxes(title_text=y_label)

    if cursor_idx is not None and 0 <= int(cursor_idx) < len(d):
        x_sel = float(d[int(cursor_idx)])
        fig.add_vline(x=x_sel, line_dash="dash")

    return fig

def plot_sankey(result, min_kj=0.5):
    segs = result.get("segments_5km", [])
    if not segs:
        return go.Figure().add_annotation(text="No 5 km segments available", x=0.5, y=0.5, showarrow=False)

    p_g  = np.array([s.get("avg_pwr_g_w",  0.0) for s in segs], float)
    p_r  = np.array([s.get("avg_pwr_r_w",  0.0) for s in segs], float)
    p_d  = np.array([s.get("avg_pwr_d_w",  0.0) for s in segs], float)
    p_a  = np.array([s.get("avg_pwr_a_w",  0.0) for s in segs], float)
    p_dc = np.array([s.get("avg_pwr_dc_w", 0.0) for s in segs], float)
    p_b  = np.array([s.get("avg_pwr_b_w",  0.0) for s in segs], float)
    p_pt = np.array([s.get("avg_pwr_pt_w", 0.0) for s in segs], float)
    p_ped= np.array([s.get("avg_power_w",  0.0) for s in segs], float)

    dur_min = np.array([s.get("duration_min", 0.0) for s in segs], float)
    dur_sec = dur_min * 60.0
    valid = np.isfinite(dur_sec) & (dur_sec > 0)
    if not np.any(valid):
        return go.Figure().add_annotation(text="No valid segment durations for Sankey", x=0.5, y=0.5, showarrow=False)

    dur_sec = dur_sec[valid]
    def v(a): return np.where(np.isfinite(a), a, 0.0)[valid]

    p_g, p_r, p_d, p_a, p_dc, p_b, p_pt, p_ped = map(v, [p_g,p_r,p_d,p_a,p_dc,p_b,p_pt,p_ped])

    def E_pos_kj(p):
        p = np.clip(p, 0.0, None)
        return float(np.sum(p * dur_sec) / 1000.0)

    def E_neg_kj(p):
        p = np.clip(p, None, 0.0)
        return float(np.sum(-p * dur_sec) / 1000.0)

    E_ped_in = E_pos_kj(p_ped)
    if E_ped_in <= 0:
        return go.Figure().add_annotation(text="No positive pedaling energy for Sankey", x=0.5, y=0.5, showarrow=False)

    inflows = [("Pedaling", E_ped_in)]
    outflows = []

    comps = [
        ("Acceleration", p_a),
        ("Deceleration", p_dc),
        ("Powertrain", p_pt),
        ("Rolling", p_r),
        ("Slope", p_g),
        ("Drag", p_d),
        ("Braking", p_b),
    ]

    for name, p in comps:
        ep = E_pos_kj(p)
        en = E_neg_kj(p)
        if ep >= min_kj:
            outflows.append((name + "+", ep))
        if en >= min_kj:
            inflows.append((name + "-", en))

    inflows.sort(key=lambda x: x[1], reverse=True)
    outflows.sort(key=lambda x: x[1], reverse=True)

    sum_in = sum(v for _, v in inflows)
    sum_out = sum(v for _, v in outflows)
    residual = sum_in - sum_out
    if abs(residual) > min_kj:
        if residual > 0:
            outflows.append(("Residual loss", residual))
        else:
            inflows.append(("Residual gain", abs(residual)))

    # Sankey needs explicit links.
    # We'll make a single central node "Energy balance".
    center = "Energy"
    nodes = [center] + [n for n,_ in inflows] + [n for n,_ in outflows]

    idx = {name:i for i,name in enumerate(nodes)}
    source = []
    target = []
    value = []

    for name, v_in in inflows:
        source.append(idx[name])
        target.append(idx[center])
        value.append(v_in)

    for name, v_out in outflows:
        source.append(idx[center])
        target.append(idx[name])
        value.append(v_out)

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=nodes, pad=15, thickness=15),
        link=dict(source=source, target=target, value=value)
    )])

    fig.update_layout(
        title="Energy flows by sign over whole ride (5 km segments aggregated)",
        height=520,
        margin=dict(l=30, r=20, t=60, b=30)
    )
    return fig

def plot_mechanical(result):
    d = _arr(result.get("filtered_dists"))
    if d is None or len(d) == 0:
        return go.Figure().add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)

    front_teeth = _arr(result.get("selected_front_teeth", None))
    rear_teeth  = _arr(result.get("selected_rear_teeth", None))
    tf = _arr(result.get("filtered_temp_front_disk_c", None))
    tr = _arr(result.get("filtered_temp_rear_disk_c", None))

    # clip each series to d
    def clip_to_d(y):
        if y is None:
            return None
        n = min(len(d), len(y))
        return y[:n]

    ft = clip_to_d(front_teeth)
    rt = clip_to_d(rear_teeth)
    tf = clip_to_d(tf)
    tr = clip_to_d(tr)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Gear Selection vs Distance", "Brake Disk Temperatures vs Distance"))

    if ft is not None:
        fig.add_trace(go.Scatter(x=d[:len(ft)], y=ft, name="Front teeth"), row=1, col=1)
    if rt is not None:
        fig.add_trace(go.Scatter(x=d[:len(rt)], y=rt, name="Rear teeth"), row=1, col=1)

    if tf is not None:
        fig.add_trace(go.Scatter(x=d[:len(tf)], y=tf, name="Front disk temp [°C]"), row=2, col=1)
    if tr is not None:
        fig.add_trace(go.Scatter(x=d[:len(tr)], y=tr, name="Rear disk temp [°C]"), row=2, col=1)

    fig.update_yaxes(title_text="Teeth", row=1, col=1)
    fig.update_yaxes(title_text="Temp [°C]", row=2, col=1)
    fig.update_xaxes(title_text="Distance [km]", row=2, col=1)

    fig.update_layout(height=650, margin=dict(l=30, r=20, t=60, b=30), legend=dict(orientation="h"))
    return fig


# =============================================================================
# Streamlit App
# =============================================================================
st.set_page_config(page_title="GPX Bike Tour Analyzer", layout="wide")

# =============================================================================
# Run-control state (prevents re-analysis on widget changes)
# =============================================================================
st.session_state.setdefault("results", {})         # key -> result dict
st.session_state.setdefault("details", {})         # key -> details dict
st.session_state.setdefault("order", [])           # upload order
st.session_state.setdefault("selected_key", None)  # current selection

# New: store uploaded bytes so you can re-analyze later without re-upload
st.session_state.setdefault("uploads_bytes", {})   # key -> bytes
st.session_state.setdefault("uploads_name", {})    # key -> original name

# New: "button edge" flag
st.session_state.setdefault("do_analyze_now", False)

def request_analyze():
    st.session_state.do_analyze_now = True


st.title("GPX Bike Tour Analyzer")

# Initialize session stores
st.session_state.setdefault("results", {})         # key -> result dict
st.session_state.setdefault("details", {})         # key -> details dict
st.session_state.setdefault("order", [])           # upload order
st.session_state.setdefault("selected_key", None)  # current selection

# ---------------- Sidebar settings ----------------
with st.sidebar:
    st.header("Settings")

    # Basic settings
    with st.expander("Rider and Behavior", expanded=True):
        st.session_state["body_mass"] = st.number_input(
            "Rider mass [kg]",
            min_value=30.0,
            max_value=150.0,
            value=70.0,
            step=0.5,
            help="Total rider mass.\n\nDefault: 70.0 kg",
        )
        st.session_state["body_height"] = st.number_input(
            "Rider height [m]",
            min_value=1.0,
            max_value=2.5,
            value=1.80,
            step=0.01,
            help="Rider height used for CdA scaling (if applicable).\n\nDefault: 1.80 m",
        )
        st.session_state["draft_effect"] = st.number_input(
            "Level draft factor [-]",
            min_value=0.1,
            max_value=1.5,
            value=1.0,
            step=0.05,
            help=f"Aerodynamic drafting multiplier on flat terrain (1.0 = no draft, 0.8 = 20 % reduced air drag).\n\nDefault: 1.0 [-]",
        )
        st.session_state["draft_effect_dh"] = st.number_input(
            "Downhill draft factor [-]",
            min_value=0.1,
            max_value=1.5,
            value=1.0,
            step=0.05,
            help="Drafting downhill?! Who does this?\n\nDefault: 1.0 [-]",
        )
        st.session_state["draft_effect_uh"] = st.number_input(
            "Uphill draft factor [-]",
            min_value=0.1,
            max_value=1.5,
            value=1.0,
            step=0.05,
            help="Drafting uphill? consider pro career!\n\nDefault: 1.0 [-]",
        )
        st.session_state["target_cadence"] = st.number_input(
            "Target cadence [RPM]",
            min_value=40,
            max_value=140,
            value=85,
            step=1,
            help="Cadence the rider tries to maintain during active pedaling.\n\nDefault: 85 RPM",
        )

    with st.expander("Bike", expanded=True):
        st.session_state["front_teeth"] = st.text_input(
            "Front teeth (comma-separated)",
            value="34,50",
            help='Chainring tooth counts, comma-separated. Example: "34,50".\n\nDefault: 34,50',
        )
        st.session_state["rear_teeth"] = st.text_input(
            "Rear teeth (comma-separated)",
            value="11,12,13,14,15,17,19,21,24,28,32",
            help='Cassette tooth counts, comma-separated. Example: "11,12,13,...".\n\nDefault: 11,12,13,14,15,17,19,21,24,28,32',
        )
        st.session_state["bike_mass"] = st.number_input(
            "Bike mass [kg]",
            min_value=3.0,
            max_value=30.0,
            value=9.0,
            step=0.5,
            help="Bike mass without rider.\n\nDefault: 9.0 kg",
        )
        st.session_state["extra_mass"] = st.number_input(
            "Extra mass [kg]",
            min_value=0.0,
            max_value=30.0,
            value=3.0,
            step=0.5,
            help="Extra carried mass (bags, bottles, tools, etc.).\n\nDefault: 3.0 kg",
        )
        st.session_state["BRAKE_FRONT_DIAMETER_MM"] = st.number_input(
            "Front brake dia [mm]",
            min_value=100.0,
            max_value=300.0,
            value=float(DEFAULT_BRAKE_FRONT_DIAMETER_MM),
            step=1.0,
            help=(
                "Front rotor diameter used by the brake temperature/energy model (if enabled).\n\n"
                f"Default: {DEFAULT_BRAKE_FRONT_DIAMETER_MM:.0f} mm"
            ),
        )
        st.session_state["BRAKE_REAR_DIAMETER_MM"] = st.number_input(
            "Rear brake dia [mm]",
            min_value=80.0,
            max_value=300.0,
            value=float(DEFAULT_BRAKE_REAR_DIAMETER_MM),
            step=1.0,
            help=(
                "Rear rotor diameter used by the brake temperature/energy model (if enabled).\n\n"
                f"Default: {DEFAULT_BRAKE_REAR_DIAMETER_MM:.0f} mm"
            ),
        )
        st.session_state["REFERENCE_ROLLING_LOSS"] = st.number_input(
            "Single Wheel Rolling loss [W]",
            min_value=0.0,
            max_value=40.0,
            value=float(DEFAULT_REFERENCE_ROLLING_LOSS),
            step=0.01,
            format="%.3f",
            help=(
                "Constant rolling loss per wheel at the reference condition (used by your rolling model).\n\n"
                f"Default: {DEFAULT_REFERENCE_ROLLING_LOSS:.3f} W"
            ),
        )

    with st.expander("Environment", expanded=False):
        # --- Ambient / environment overrides (manual) ---
        st.session_state["use_wind"] = st.checkbox(
            "Use weather data (Meteostat)",
            value=True,
            help="If enabled, use Meteostat weather. If disabled, use the ambient overrides below.\n\nDefault: off",
        )

        st.session_state["AMBIENT_WIND_SPEED_MS"] = st.number_input(
            "Ambient wind speed [m/s]",
            min_value=0.0,
            max_value=40.0,
            value=float(AMBIENT_WIND_SPEED_MS),
            step=0.1,
            format="%.1f",
            help=(
                "Background wind speed magnitude. Used if weather data is disabled or as a constant override.\n\n"
                f"Default: {DEFAULT_AMBIENT_WIND_SPEED_MS:.1f} m/s"
            ),
        )

        st.session_state["AMBIENT_WIND_DIR_DEG"] = st.number_input(
            "Ambient wind direction [°]",
            min_value=0.0,
            max_value=360.0,
            value=float(AMBIENT_WIND_DIR_DEG),
            step=1.0,
            format="%.0f",
            help=(
                "Wind direction in degrees. Ensure consistency with your model "
                "(meteorological 'from' vs mathematical 'to').\n\n"
                f"Default: {DEFAULT_AMBIENT_WIND_DIR_DEG:.0f} °"
            ),
        )

        st.session_state["AMBIENT_TEMP_C"] = st.number_input(
            "Ambient temperature [°C]",
            min_value=-30.0,
            max_value=50.0,
            value=float(AMBIENT_TEMP_C),
            step=0.5,
            format="%.1f",
            help=(
                "Air temperature. Affects air density and human thermoregulation (PHS).\n\n"
                f"Default: {DEFAULT_AMBIENT_TEMP_C:.1f} °C"
            ),
        )

        st.session_state["AMBIENT_PRES_HPA"] = st.number_input(
            "Ambient pressure [hPa]",
            min_value=800.0,
            max_value=1100.0,
            value=float(AMBIENT_PRES_HPA),
            step=1.0,
            format="%.0f",
            help=(
                "Barometric air pressure. Sea level is ~1013 hPa; decreases with altitude.\n\n"
                f"Default: {DEFAULT_AMBIENT_PRES_HPA:.0f} hPa"
            ),
        )

        st.session_state["AMBIENT_RHUM_PCT"] = st.number_input(
            "Ambient relative humidity [%]",
            min_value=0.0,
            max_value=100.0,
            value=float(AMBIENT_RHUM_PCT),
            step=1.0,
            format="%.0f",
            help=(
                "Relative humidity of the air. Strongly affects evaporative cooling and sweat efficiency.\n\n"
                f"Default: {DEFAULT_AMBIENT_RHUM_PCT:.0f} %"
            ),
        )

        st.session_state["AMBIENT_RHO"] = st.number_input(
            "Ambient air density ρ [kg/m³]",
            min_value=0.80,
            max_value=1.35,
            value=float(AMBIENT_RHO),
            step=0.005,
            format="%.3f",
            help=(
                "Air density override. If set manually, ensure consistency with pressure and temperature.\n\n"
                f"Default: {DEFAULT_AMBIENT_RHO:.3f} kg/m³"
            ),
        )

        st.session_state["GRAVITY"] = st.number_input(
            "Gravity [m/s²]",
            min_value=9.0,
            max_value=10.0,
            value=DEFAULT_G,
            step=0.0001,
            format="%.4f",
            help=f"Gravitational acceleration.\n\nDefault: {DEFAULT_G:.3f} m/s²",
        )

    with st.expander("Analysis", expanded=False):
        st.session_state["MAX_SPEED"] = st.number_input(
            "Max speed [km/h]",
            min_value=10.0,
            max_value=200.0,
            value=float(DEFAULT_MAX_SPEED),
            step=5.0,
            help=f"Upper bound for plotting/histograms and some sanity clipping.\n\nDefault: {DEFAULT_MAX_SPEED:.1f} km/h",
        )
        st.session_state["MAX_GRADE"] = st.number_input(
            "Max grade [%]",
            min_value=1.0,
            max_value=100.0,
            value=float(DEFAULT_MAX_GRADE),
            step=1.0,
            help=f"Upper bound for plotting/histograms and some sanity clipping.\n\nDefault: {DEFAULT_MAX_GRADE:.0f} %",
        )
        st.session_state["MAX_PWR"] = st.number_input(
            "Max power [W]",
            min_value=100.0,
            max_value=3000.0,
            value=float(DEFAULT_MAX_PWR),
            step=50.0,
            help=f"Upper bound for plotting/histograms and some sanity clipping.\n\nDefault: {DEFAULT_MAX_PWR:.0f} W",
        )
        st.session_state["MIN_ACTIVE_SPEED"] = st.number_input(
            "Min active speed [km/h]",
            min_value=0.0,
            max_value=20.0,
            value=float(DEFAULT_MIN_ACTIVE_SPEED),
            step=0.1,
            help=f"Speed threshold to consider the rider 'active' (vs standing/break).\n\nDefault: {DEFAULT_MIN_ACTIVE_SPEED:.1f} km/h",
        )
        st.session_state["MIN_VALID_SPEED"] = st.number_input(
            "Min valid speed [km/h]",
            min_value=0.0,
            max_value=20.0,
            value=float(DEFAULT_MIN_VALID_SPEED),
            step=0.1,
            help=f"Lower speed cutoff for validity checks (e.g., GPS jitter / near-zero noise).\n\nDefault: {DEFAULT_MIN_VALID_SPEED:.1f} km/h",
        )
        st.session_state["COMBINE_BREAK_MAX_TIME_DIFF"] = st.number_input(
            "Combine Breaks within [min]",
            min_value=1.0,
            max_value=60.0,
            value=float(DEFAULT_COMBINE_BREAK_MAX_TIME_DIFF),
            step=1.0,
            help=f"Merge breaks if their timestamps are within this time difference.\n\nDefault: {DEFAULT_COMBINE_BREAK_MAX_TIME_DIFF:.0f} min",
        )
        st.session_state["COMBINE_BREAK_MAX_DISTANCE"] = st.number_input(
            "Combine Breaks within [m]",
            min_value=1.0,
            max_value=1000.0,
            value=float(DEFAULT_COMBINE_BREAK_MAX_DISTANCE),
            step=1.0,
            help=f"Merge breaks if their locations are within this distance.\n\nDefault: {DEFAULT_COMBINE_BREAK_MAX_DISTANCE:.0f} m",
        )
        st.session_state["CLIMB_MIN_DISTANCE_KM"] = st.number_input(
            "Min climb length [km]",
            min_value=0.1,
            max_value=20.0,
            value=float(DEFAULT_CLIMB_MIN_DISTANCE_KM),
            step=0.1,
            help=f"Minimum distance for a climb segment to be detected.\n\nDefault: {DEFAULT_CLIMB_MIN_DISTANCE_KM:.1f} km",
        )
        st.session_state["CLIMB_MIN_ELEVATION_GAIN_M"] = st.number_input(
            "Min alt gain in climb [m]",
            min_value=0.1,
            max_value=2000.0,
            value=float(DEFAULT_CLIMB_MIN_ELEVATION_GAIN_M),
            step=0.1,
            help=f"Minimum elevation gain for a segment to qualify as a climb.\n\nDefault: {DEFAULT_CLIMB_MIN_ELEVATION_GAIN_M:.1f} m",
        )
        st.session_state["CLIMB_END_AVERAGE_GRADE_PCT"] = st.number_input(
            "Max grade at climb end [%]",
            min_value=0.0,
            max_value=float(MAX_GRADE),
            value=float(DEFAULT_CLIMB_END_AVERAGE_GRADE_PCT),
            step=0.1,
            help=f"Climb end criterion: if average grade falls below this, climb may end.\n\nDefault: {DEFAULT_CLIMB_END_AVERAGE_GRADE_PCT:.1f} %",
        )
        st.session_state["CLIMB_END_WINDOW_SIZE_M"] = st.number_input(
            "Climb end detection window [m]",
            min_value=1.0,
            max_value=1000.0,
            value=float(DEFAULT_CLIMB_END_WINDOW_SIZE_M),
            step=1.0,
            help=f"Window length used to evaluate the climb end criterion.\n\nDefault: {DEFAULT_CLIMB_END_WINDOW_SIZE_M:.0f} m",
        )
        st.session_state["MAX_DIST_BETWEEN_CLIMBS_M"] = st.number_input(
            "Combine Climbs within [m]",
            min_value=1.0,
            max_value=1000.0,
            value=float(DEFAULT_MAX_DIST_BETWEEN_CLIMBS_M),
            step=1.0,
            help=f"Merge climbs separated by less than this distance.\n\nDefault: {DEFAULT_MAX_DIST_BETWEEN_CLIMBS_M:.0f} m",
        )

    with st.expander("Simulation", expanded=False):
        st.session_state["HYSTERESIS_REAR_RPM"] = st.number_input(
            "Allowable Cadence Error",
            min_value=1.0,
            max_value=60.0,
            value=float(DEFAULT_HYSTERESIS_REAR_RPM),
            step=1.0,
            help=f"Cadence tolerance band (hysteresis) used in gear selection/shift logic.\n\nDefault: {DEFAULT_HYSTERESIS_REAR_RPM:.0f} RPM",
        )
        st.session_state["MIN_SHIFT_INTERVAL_REAR_SEC"] = st.number_input(
            "Max. time between shifts",
            min_value=1.0,
            max_value=10000.0,
            value=float(DEFAULT_MIN_SHIFT_INTERVAL_REAR_SEC),
            step=1.0,
            help=f"Minimum time between rear shifts to avoid rapid oscillation.\n\nDefault: {DEFAULT_MIN_SHIFT_INTERVAL_REAR_SEC:.0f} s",
        )

    with st.expander("Human Body Model", expanded=False):
        st.session_state["EFFICIENCY"] = st.number_input(
            "Human Body Efficiency [-]",
            min_value=0.05,
            max_value=0.4,
            value=float(DEFAULT_HUMAN_BODY_EFFICIENCY),
            step=0.01,
            format="%.3f",
            help=f"Gross mechanical efficiency (mechanical power / metabolic power).\n\nDefault: {DEFAULT_HUMAN_BODY_EFFICIENCY:.3f} [-]",
        )
        st.session_state["BASE_MET_DURING_ACTIVITY"] = st.number_input(
            "Base MET [-]",
            min_value=1.0,
            max_value=2.0,
            value=float(DEFAULT_BASE_MET_DURING_ACTIVITY),
            step=0.05,
            format="%.2f",
            help=f"Baseline MET during activity (used in the metabolic model).\n\nDefault: {DEFAULT_BASE_MET_DURING_ACTIVITY:.2f} [-]",
        )

    with st.expander("Advanced settings", expanded=False):
        st.session_state["long_break_threshold_min"] = st.number_input(
            "Long break threshold [min]",
            min_value=1.0,
            max_value=180.0,
            value=10.0,
            step=0.5,
            help="Breaks longer than this are classified as 'long'.\n\nDefault: 10.0 min",
        )

        st.session_state["wheel_circ"] = st.number_input(
            "Wheel circumference [m]",
            min_value=1.5,
            max_value=2.8,
            value=2.096,
            step=0.001,
            format="%.3f",
            help="Wheel circumference used to convert cadence ↔ speed for gearing.\n\nDefault: 2.096 m",
        )
        st.session_state["drivetrain_loss"] = st.number_input(
            "Drivetrain loss [-]",
            min_value=0.0,
            max_value=0.2,
            value=0.03,
            step=0.005,
            format="%.3f",
            help="Fractional drivetrain loss (0.03 = 3%).\n\nDefault: 0.030 [-]",
        )

        st.session_state["REFERENCE_CDA_VALUE"] = st.number_input(
            "Reference CdA [m²]",
            min_value=0.1,
            max_value=1.5,
            value=float(DEFAULT_REFERENCE_CDA_VALUE),
            step=0.01,
            format="%.7f",
            help=f"Reference CdA value used for aerodynamic drag.\n\nDefault: {DEFAULT_REFERENCE_CDA_VALUE:.7f} m²",
        )

        st.session_state["MAX_MAP_POINTS"] = st.number_input(
            "Max map points",
            min_value=100,
            max_value=200000,
            value=int(DEFAULT_MAX_MAP_POINTS),
            step=1000,
            help=f"Subsample limit for map drawing to keep Folium responsive.\n\nDefault: {DEFAULT_MAX_MAP_POINTS:d}",
        )
        st.session_state["SMOOTHING_WINDOW_SIZE_S"] = st.number_input(
            "Smoothing window [s]",
            min_value=1.0,
            max_value=20.0,
            value=float(DEFAULT_SMOOTHING_WINDOW_SIZE_S),
            step=1.0,
            help=f"Smoothing window length for speed/grade/altitude filtering.\n\nDefault: {DEFAULT_SMOOTHING_WINDOW_SIZE_S:.1f} s",
        )

        st.session_state["BRAKE_ADD_COOLING_FACTOR"] = st.number_input(
            "Brake cooling factor [-]",
            min_value=0.0,
            max_value=5.0,
            value=float(DEFAULT_BRAKE_ADD_COOLING_FACTOR),
            step=0.025,
            format="%.3f",
            help=f"Additional cooling multiplier for the brake thermal model.\n\nDefault: {DEFAULT_BRAKE_ADD_COOLING_FACTOR:.3f} [-]",
        )
        st.session_state["BRAKE_PERFORATION_FACTOR"] = st.number_input(
            "Brake perforation factor [-]",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_BRAKE_PERFORATION_FACTOR),
            step=0.025,
            format="%.3f",
            help=f"Perforation/ventilation factor for rotor cooling.\n\nDefault: {DEFAULT_BRAKE_PERFORATION_FACTOR:.3f} [-]",
        )
        st.session_state["BRAKE_DISTRIBUTION_FRONT"] = st.number_input(
            "Front brake power distribution [-]",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_BRAKE_DISTRIBUTION_FRONT),
            step=0.025,
            format="%.3f",
            help=f"Fraction of braking power assumed on the front brake.\n\nDefault: {DEFAULT_BRAKE_DISTRIBUTION_FRONT:.3f} [-]",
        )



# ---------------- Upload + analyze ----------------
col_u1, col_u2 = st.columns([2, 3], vertical_alignment="top")

with col_u1:
    st.subheader("Upload GPX")
    uploads = st.file_uploader("Select GPX files", type=["gpx"], accept_multiple_files=True)

    # Persist uploads into session_state (so they survive reruns)
    if uploads:
        for up in uploads:
            key = up.name
            if key not in st.session_state["uploads_bytes"]:
                st.session_state["uploads_bytes"][key] = up.getvalue()
                st.session_state["uploads_name"][key] = up.name
                if key not in st.session_state["order"]:
                    st.session_state["order"].append(key)
                if st.session_state["selected_key"] is None:
                    st.session_state["selected_key"] = key

    keys = list(st.session_state["order"])
    if not keys:
        st.info("Upload at least one GPX.")
    else:
        # Choose what to analyze (default: all)
        analyze_keys = st.multiselect(
            "Files to analyze",
            options=keys,
            default=keys,
        )

        # IMPORTANT: button sets a flag, doesn't run analysis directly
        st.button("Analyze selected files", type="primary", on_click=request_analyze)

        # Only run heavy analysis when flag is set
        if st.session_state.do_analyze_now:
            st.session_state.do_analyze_now = False
            details_lines = []

            cfg = build_config_from_ui(st.session_state)

            prog = st.progress(0.0, text="Analyzing...")
            n_total = max(1, len(analyze_keys))
            n_done = 0

            for key in analyze_keys:
                gpx_bytes = st.session_state["uploads_bytes"].get(key, None)
                if not gpx_bytes:
                    log(f"Skipping (no bytes found): {key}")
                    continue

                # write to temp file because analyzer expects a path
                suffix = "_" + re.sub(r"[^a-zA-Z0-9_.-]+", "_", key)
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                    tf.write(gpx_bytes)
                    tmp_path = tf.name

                try:
                    log(f"Analyzing: {key}")
                    result = analyze_gpx_file(
                        tmp_path,
                        cfg=cfg,
                        target_cadence=int(st.session_state["target_cadence"]),
                        use_wind_data=bool(st.session_state["use_wind"]),
                        break_threshold_min=float(st.session_state["long_break_threshold_min"]),
                        progress_cb=None,
                    )
                    if result is not None:
                        st.session_state["results"][key] = result
                        st.session_state["details"][key] = details_lines
                        details_lines = []
                        st.session_state["selected_key"] = key
                        log(f"Done: {key}")
                    else:
                        log(f"Analyzer returned None: {key}")

                except Exception as e:
                    log(f"ERROR analyzing {key}: {e}")

                finally:
                    n_done += 1
                    prog.progress(n_done / n_total, text=f"Analyzing... {n_done}/{n_total}")

            prog.empty()
            st.success("Analysis finished.")

with col_u2:
    st.subheader("Loaded rides")

    keys = st.session_state["order"]
    if not keys:
        st.info("No uploads yet.")
    else:
        # Show selection among uploaded files (not only analyzed)
        sel = st.selectbox(
            "Select ride",
            options=keys,
            index=keys.index(st.session_state["selected_key"]) if st.session_state["selected_key"] in keys else 0
        )
        st.session_state["selected_key"] = sel

        # Build stats only for analyzed rides
        analyzed_keys = [k for k in keys if k in st.session_state["results"]]
        if not analyzed_keys:
            st.warning("No analyzed rides yet. Click 'Analyze selected files'.")
        else:
            rows = []
            for k in analyzed_keys:
                r = st.session_state["results"][k]
                path = r.get("gpx_path", "")
                start = r.get("start_time")
                date_str = start.strftime("%Y-%m-%d") if start else "-"
                rows.append({
                    "File": os.path.basename(path) if path else k,
                    "Date": date_str,
                    "Distance [km]": float(r.get("total_distance_km", 0.0)),
                    "Elev gain [m]": float(r.get("elevation_gain_m", 0.0)),
                    "Duration [min]": float(r.get("total_duration_min", 0.0)),
                    "Avg power [W]": float(r.get("avg_power_with_freewheeling", 0.0)),
                    "Max VAM [m/h]": float(r.get("max_climb_vam_m_per_h", 0.0)),
                    "Avg climb P [W]": float(r.get("avg_climb_power_w", 0.0)),
                    "Best 5 km P [W]": float(r.get("best_segment_5km_power_w", 0.0)),
                })
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# ---------------- Main content for selected ride ----------------
key = st.session_state.get("selected_key", None)
if key and key in st.session_state["results"]:
    result = st.session_state["results"][key]
    details = st.session_state["details"][key]
    # top “Track Info” block
    st.subheader("Track Info")
    c1, c2, c3, c4 = st.columns(4)
    start = result.get("start_time")
    end = result.get("end_time")

    dist = float(result.get("total_distance_km", 0.0))
    elev = float(result.get("elevation_gain_m", 0.0))
    total_dur = float(result.get("total_duration_min", 0.0))
    active_dur = float(result.get("active_duration_min", 0.0))

    kcal_cum = float(np.asarray(result.get("cum_kcal_active", [0]))[-1])
    kcal_base = float(np.asarray(result.get("cum_kcal_base", [0]))[-1])
    kcal_total = float(np.asarray(result.get("cum_kcal_total", [0]))[-1])

    start_local = result.get("local_start_time")
    end_local = result.get("local_end_time")

    with c1:
        st.metric("Date", start_local.strftime("%Y-%m-%d") if start else "-")
        st.metric("Start", start_local.strftime("%H:%M:%S") if start else "-")
        st.metric("End", end_local.strftime("%H:%M:%S") if end else "-")
    with c2:
        st.metric("Distance", f"{dist:.1f} km")
        st.metric("Duration (moving)", f"{time_to_readable(t_minutes=total_dur)}")
        st.metric("Duration (pedaling)", f"{time_to_readable(t_minutes=active_dur)}")
    with c3:
        st.metric("Elevation gain", f"{elev:.0f} m")
        st.metric("Avg power (in motion)", f"{float(result.get('avg_power_with_freewheeling',0)):.0f} W")
        st.metric("Avg power (pedaling)", f"{float(result.get('avg_power_active',0)):.0f} W")
        #st.metric("Avg power (total)", f"{float(result.get('avg_power',0)):.0f} W")
    with c4:
        st.metric("Calories (activity)", f"{kcal_cum:.0f} kcal")
        st.metric("Calories (base)", f"{kcal_base:.0f} kcal")
        st.metric("Calories (total)", f"{kcal_total:.0f} kcal")
        # If you want the fun tooltip: put it in an expander/text
        with st.expander("Fun equivalents 🍝", expanded=False):
            try:
                st.markdown(fun_energy_tooltip(kcal_total))
            except Exception:
                st.write("(fun tooltip unavailable - missing helper)")

    # tabs like Qt
    tabs = st.tabs(["Profiles", "Segments", "Distributions", "Energy Balance", "Mechanical", "Human Body", "Environment", "Map", "Details", "Log"])

    cfg = build_config_from_ui(st.session_state)

    with tabs[0]:
        highlight = st.checkbox("Highlight climbs/downhills", value=False)
        fig = plot_profiles(result, highlight_climbs=highlight)
        st.plotly_chart(fig, width='stretch')

    with tabs[1]:
        fig = plot_segments(result)
        st.plotly_chart(fig, width='stretch')

    with tabs[2]:
        fig = plot_distributions(result, max_speed=MAX_SPEED, max_pwr=MAX_PWR)
        st.plotly_chart(fig, width='stretch')

    with tabs[3]:
        fig = plot_sankey(result)
        st.plotly_chart(fig, width='stretch')

    with tabs[4]:
        fig = plot_mechanical(result)
        st.plotly_chart(fig, width='stretch')

    with tabs[5]:
        fig = plot_human(result)
        st.plotly_chart(fig, width='stretch')

    with tabs[6]:
        fig = plot_environment(result)
        st.plotly_chart(fig, width='stretch')

    with tabs[7]:
        c1_map, c2_map, c3_map = st.columns(3)

        with c1_map:
            mode = st.selectbox(
                "Color by",
                [
                    "Speed [km/h]",
                    "Power [W]",
                    "Grade [%]",
                    "Headwind [m/s]",
                    "Air density [kg/m³]",
                    "Cadence [RPM]",
                    "Climbs (segments)",
                    "None (single color)",
                ],
                index=0
            )

        with c2_map:
            st.session_state["marker_spacing_km"] = st.number_input(
                "Map markers every [km]",
                min_value=0.5,
                max_value=1000.0,
                value=5.0,
                step=0.5,
                help="Distance between red distance markers on the map.\n\nDefault: 5.0 km",
            )

        with c3_map:
            SEGMENT_MEAN_MODE = st.checkbox(
                "Display mean values between markers",
                value=SEGMENT_MEAN_MODE,
                #help="If enabled, use Meteostat weather. If disabled, use the ambient overrides below.\n\nDefault: off",
            )

        m, values_full, caption = build_map(
            result,
            mode=mode,
            marker_spacing_km=float(st.session_state["marker_spacing_km"]),
            long_break_thr_min=float(st.session_state["long_break_threshold_min"]),
        )

        if m is None:
            st.warning("No GPS data to display.")
        else:
            # distance slider replacing hover-sync
            d_full = np.asarray(result["filtered_dists"], dtype=float)
            # render map
            html = m.get_root().render()
            st.components.v1.html(html, height=850, scrolling=True)

            # metric plot under map
            fig = plot_cmap_metric(
                result,
                y_label=caption if caption else mode,
                values_full=values_full,
                cursor_idx=None,
                spacing_km=float(st.session_state["marker_spacing_km"]),
                segment_mean_mode=SEGMENT_MEAN_MODE
            )
            st.plotly_chart(fig, width='stretch')

    with tabs[8]:
        # “Details” (similar to your stats_details)
        st.subheader("Details")
        # lines = []
        # path = result.get("gpx_path", "")
        # fname = os.path.basename(path) if path else key
        # lines.append(f"File: {fname}")
        # if start:
        #     lines.append(f"Date: {start.strftime('%Y-%m-%d')}  Start: {start.strftime('%H:%M:%S')}")
        # lines.append(f"Distance: {dist:.1f} km, Elevation gain: {elev:.0f} m, Duration: {total_dur:.1f} min")
        # lines.append(f"Avg active power: {float(result.get('avg_power_active',0)):.0f} W")

        # lines.append("\nClimbs (combined):")
        # combined_climbs = result.get("combined_climbs", [])
        # if not combined_climbs:
        #     lines.append("  (none)")
        # else:
        #     for i, c in enumerate(combined_climbs, start=1):
        #         lines.append(
        #             f"  #{i}: {c.get('distance_km',0):.2f} km, {c.get('elevation_gain_m',0):.0f} m, "
        #             f"{c.get('average_grade_pct',0):.1f} %, dur {c.get('duration_min',0):.1f} min, "
        #             f"VAM {c.get('vam_m_per_h',0):.0f} m/h, P_avg {c.get('avg_power_w',0):.0f} W"
        #         )
        # st.code("\n".join(details_lines), language="text")
        st.code("\n".join(details), language="text")
       
        

    with tabs[9]:
        st.subheader("Log")
        render_log()

else:
    st.info("Select a file and click 'Analyze selected files' first.")
