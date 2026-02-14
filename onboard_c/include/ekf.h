/**
 * @file ekf.h
 * @brief Extended Kalman Filter for position smoothing (4-state).
 *
 * State: [lat, lon, vlat, vlon]
 * Constant-velocity motion model with Mahalanobis gating.
 */
#ifndef EKF_H
#define EKF_H

#include "vps_types.h"

/** EKF configuration. */
typedef struct {
    double process_noise;      /* Q diagonal (default 1e-10) */
    double measurement_noise;  /* R base (default 1e-8) */
    double gate_threshold;     /* Mahalanobis gate (default 5.0) */
    double max_gap_s;          /* Reset after this gap (default 30.0) */
} vps_ekf_config_t;

/** EKF state. */
typedef struct {
    double x[4];      /* state: [lat, lon, vlat, vlon] */
    double P[4][4];   /* covariance matrix */
    double last_t;    /* last update timestamp */
    bool initialized;
    double last_gate; /* last Mahalanobis distance */
} vps_ekf_state_t;

/** Initialize EKF with default config. */
vps_ekf_config_t vps_ekf_default_config(void);

/** Initialize EKF state (uninitialized). */
void vps_ekf_init(vps_ekf_state_t *state);

/** Reset EKF to uninitialized. */
void vps_ekf_reset(vps_ekf_state_t *state);

/**
 * Update EKF with a measurement.
 * @return true if measurement was accepted (passed gate)
 */
bool vps_ekf_update(vps_ekf_state_t *state, const vps_ekf_config_t *cfg,
                    vps_geopoint_t measurement, double hdop, double t);

/** Predict position at time t (without measurement). */
vps_geopoint_t vps_ekf_predict(const vps_ekf_state_t *state, double t);

/** Get current velocity estimate. */
vps_velocity_t vps_ekf_velocity(const vps_ekf_state_t *state);

/** Get current speed in m/s. */
double vps_ekf_speed(const vps_ekf_state_t *state);

/** Get current position estimate. */
vps_geopoint_t vps_ekf_position(const vps_ekf_state_t *state);

#endif /* EKF_H */
