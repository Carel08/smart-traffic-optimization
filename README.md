# Smart City Traffic Light Optimization 🚦

## Overview

This project builds a simplified smart traffic management system that compares fixed-timing traffic lights against adaptive ML + optimization-based traffic control.

The system is designed to:

- Simulate traffic across 4 to 10 intersections
- Generate realistic traffic demand using time of day, weather, and events
- Predict near-term congestion using machine learning
- Optimize signal timing to reduce average wait time
- Demonstrate scenarios such as rain, accidents, pedestrians, and emergency vehicle priority

## Setup
## Setup

This project was developed using a Conda environment.

### Environment

* Conda version: 23.7.4
* Python version: 3.11

Create the Conda environment:

```bash
conda create -n trafficopt python=3.11 -y
```

Activate the environment:

```bash
conda activate trafficopt
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify installation:

```bash
python -c "import numpy, pandas, sklearn, scipy, streamlit, plotly, networkx; print('All dependencies installed')"
```

## Run CLI

```bash
python main.py
```

Example:

```bash
python main.py --num-intersections 10 --scenario accident --controller adaptive
```

## Run Streamlit Demo

```bash
streamlit run streamlit_app.py
```

