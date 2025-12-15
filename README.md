GPX Power & Ride Analysis Tool

Overview

This repository contains a single-file Python application for advanced analysis of cycling GPX files.
The focus is on physics-based power estimation, energetic accounting, and detailed ride statistics,
exposed through an interactive Streamlit-based web interface.

The tool is intended for technically interested cyclists, engineers, and researchers who prefer
transparent, model-driven calculations over black-box metrics.

All logic is deliberately contained in one Python file to keep assumptions, models, and numerical
methods easy to inspect, modify, and extend.


Repository Structure

- gpx_power_calculator.py : main application containing UI, models, and plotting
- requirements.txt       : Python dependencies (primarily for Streamlit)
- README.txt / README.md  : documentation


Installation

1) Create a virtual environment (recommended)

Linux / macOS:
```shell
python -m venv venv
source venv/bin/activate
```

Windows (PowerShell):
```shell
python -m venv venv
venv\Scripts\Activate.ps1
```

2) Install dependencies

```shell
pip install -r requirements.txt
```


Running the Application

Start the Streamlit app with:

```shell
streamlit run gpx_power_calculator.py
```

Streamlit will print a local URL (typically http://localhost:8501) which can be opened in a browser.


User Input Parameters

The Streamlit interface exposes a number of user-defined parameters that directly affect the
physical power and energy model. While naming may change, the parameters fall into the following
conceptual groups.

Rider and Bike Parameters

- Total mass [kg]
  Combined mass of rider, bike, and equipment. This parameter affects gravitational, rolling,
  and inertial forces.

- Mechanical efficiency [-]
  Ratio between mechanical output power and metabolic input power. Used to convert mechanical
  work into an estimate of energetic or caloric cost.


Aerodynamic Parameters

- Drag area (CdA) [m^2]
  Lumped aerodynamic parameter combining frontal area and drag coefficient. This is typically
  the dominant loss term at higher speeds.

- Air density [kg/m^3]
  Either assumed constant or derived from ambient conditions. Influences aerodynamic drag
  quadratically.

- Wind speed and direction
  Used to compute the relative air velocity between rider and surrounding air, allowing
  headwind and tailwind effects to be modeled explicitly.


Rolling Resistance and Drivetrain Losses

- Rolling resistance coefficient (Crr) [-]
  Surface- and tire-dependent loss term, particularly relevant at low speeds and on rough terrain.

- Drivetrain efficiency [-]
  Optional factor accounting for losses in chain, bearings, and transmission components.


Numerical and Signal-Processing Parameters

- Smoothing and filtering parameters
  Applied to elevation, speed, or gradient signals to reduce noise amplification when computing
  derivatives.

- Resampling or interpolation settings
  Used to ensure numerically stable differentiation and integration of GPX-derived signals.


Physical Model Description

Kinematic Reconstruction

From the GPX data, the following quantities are reconstructed as functions of time or distance:

- Distance along the track
- Speed
- Acceleration
- Road gradient (slope)

Special care is taken to avoid numerical artifacts caused by noisy elevation data, irregular
sampling, or non-monotonic timestamps.


Force Model

The longitudinal force acting on the rider-bike system is modeled as the sum of several
physically motivated contributions:

Total force equals gravitational force plus aerodynamic drag plus rolling resistance plus
inertial force.

Gravitational Force

Proportional to total mass, gravitational acceleration, and the sine of the road slope.
This term dominates during sustained climbing and is highly sensitive to elevation accuracy.

Aerodynamic Drag

Proportional to air density, drag area, and the square of the relative air speed. This term
dominates at higher velocities and is strongly affected by wind conditions.

Rolling Resistance

Proportional to total mass, gravitational acceleration, rolling resistance coefficient, and
the cosine of the road slope. This term is weakly slope-dependent and surface-specific.

Inertial Force

Proportional to total mass and longitudinal acceleration. This term becomes relevant during
accelerations, attacks, and stop-and-go riding.


Power Calculation

Instantaneous mechanical power demand is computed as the product of total longitudinal force
and forward velocity. Negative power values, typically occurring during downhill coasting,
are handled explicitly and may be visualized or excluded depending on context.


Energy and Metabolic Cost

Mechanical work is obtained by integrating power over time. Using the specified mechanical
efficiency, an estimate of metabolic energy expenditure is derived. Results may be expressed
in joules, kilojoules, or kilocalories.


Terrain Classification

The track is segmented into uphill, downhill, and flat or mixed sections based on gradient
thresholds. This enables terrain-specific statistics and clearer visualization of power and
energy contributions.


Outputs and Visualizations

The application provides interactive plots and tables summarizing speed, elevation, power,
energy expenditure, and time or distance spent in different terrain categories. All
visualizations are generated directly from the underlying physical model.


Intended Use and Limitations

This tool is not intended as a replacement for calibrated power meters. Accuracy depends
strongly on GPX data quality, elevation noise, wind assumptions, and chosen model parameters.
It is best suited for comparative analysis, sensitivity studies, education, and research.


Requirements

- Python 3.9 or newer recommended
- See requirements.txt for the exact dependency list


License

Provided as-is for research, educational, and personal use. No guarantees are given regarding
correctness or fitness for a particular purpose.


Author

Fabian Viermetz

readme created by chatgpt. It is not comprehensive.
