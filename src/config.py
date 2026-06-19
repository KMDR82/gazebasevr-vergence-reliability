"""Shared constants for the vergence-reliability pipeline.

All values are taken directly from the analysis notebook so that the package
reproduces the manuscript numbers exactly.
"""

# Acquisition
FS = 250.0                      # sampling rate (Hz)

# Cleaning / filtering
VMAX_DPS = 60.0                 # |velocity| spike threshold for blink masking (deg/s)
LOWPASS_FC = 10.0               # Butterworth low-pass cutoff (Hz)
BUTTER_ORDER = 2                # Butterworth order

# Segmentation
ZT_EPS = 1e-6                   # minimum |Δ depth| treated as a stimulus transition
PRE_S = 0.30                    # pre-transition baseline window for demanded step (s)
SETTLE_S = 0.20                 # offset into a segment before reading the new level (s)
MIN_FINITE_FRAC = 0.60          # keep a segment only if >60% of samples are finite
MIN_SEGMENTS = 3                # minimum segments per direction to form a template

# Quality-control filter (per recording)
QC = dict(valid_frac=0.80, corr=0.50, n_conv=6, n_div=6)

# Descriptive biomarkers
ONSET_FRAC = 0.10               # latency threshold = 10% of response amplitude
SETTLE_FRAC = 0.90              # settling time = first crossing of 90% amplitude
MOVE_WINDOW_S = 1.20            # window for the peak-velocity search (s)

# First-order model bounds: [t0, tau, A]
M1_P0 = [0.15, 0.20, None]      # A0 filled at fit time from the steady level
M1_BOUNDS = ([0.0, 0.02, 0.05], [1.0, 2.0, 10.0])

# Second-order model bounds: [t0, wn, zeta, A]
M2_P0 = [0.15, 8.0, 0.70, None]
M2_BOUNDS = ([0.0, 1.0, 0.05, 0.05], [1.0, 50.0, 0.999, 10.0])

# Reliability
ICC_TYPE = "ICC(2,1)"           # two-way random, absolute agreement, single rater
N_BOOTSTRAP = 2000
TARGET_ICC = 0.75               # for the Spearman-Brown "sessions needed" calc

# Multiple testing
FDR_ALPHA = 0.05

DESC_FEATS = ["conv_latency_ms", "div_latency_ms", "conv_gain", "div_gain",
              "conv_peakvel", "div_peakvel", "conv_settle_ms", "div_settle_ms"]
