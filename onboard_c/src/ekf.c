/**
 * @file ekf.c
 * @brief Extended Kalman Filter (4-state constant velocity).
 *
 * All matrix operations are inline 4x4 — no BLAS dependency.
 */
#include "ekf.h"
#include <math.h>
#include <string.h>

/* --- 4x4 matrix helpers --- */

static void mat4_zero(double m[4][4]) {
    memset(m, 0, sizeof(double) * 16);
}

static void mat4_eye(double m[4][4]) {
    mat4_zero(m);
    m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0;
}

static void mat4_add(double c[4][4], const double a[4][4], const double b[4][4]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            c[i][j] = a[i][j] + b[i][j];
}

static void mat4_mul(double c[4][4], const double a[4][4], const double b[4][4]) {
    double tmp[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            tmp[i][j] = 0;
            for (int k = 0; k < 4; k++)
                tmp[i][j] += a[i][k] * b[k][j];
        }
    memcpy(c, tmp, sizeof(double) * 16);
}

static void mat4_transpose(double out[4][4], const double in[4][4]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            out[i][j] = in[j][i];
}

/* --- EKF implementation --- */

vps_ekf_config_t vps_ekf_default_config(void) {
    vps_ekf_config_t cfg;
    cfg.process_noise = 1e-10;
    cfg.measurement_noise = 1e-8;
    cfg.gate_threshold = 5.0;
    cfg.max_gap_s = 30.0;
    return cfg;
}

void vps_ekf_init(vps_ekf_state_t *state) {
    memset(state, 0, sizeof(*state));
    state->initialized = false;
    state->last_gate = 0.0;
}

void vps_ekf_reset(vps_ekf_state_t *state) {
    vps_ekf_init(state);
}

static void build_F(double F[4][4], double dt) {
    mat4_eye(F);
    F[0][2] = dt;  /* lat += vlat * dt */
    F[1][3] = dt;  /* lon += vlon * dt */
}

static void build_Q(double Q[4][4], double q, double dt) {
    mat4_zero(Q);
    double dt2 = dt * dt;
    double dt3 = dt2 * dt / 2.0;
    double dt4 = dt2 * dt2 / 4.0;
    Q[0][0] = q * dt4;  Q[0][2] = q * dt3;
    Q[1][1] = q * dt4;  Q[1][3] = q * dt3;
    Q[2][0] = q * dt3;  Q[2][2] = q * dt2;
    Q[3][1] = q * dt3;  Q[3][3] = q * dt2;
}

bool vps_ekf_update(vps_ekf_state_t *state, const vps_ekf_config_t *cfg,
                    vps_geopoint_t measurement, double hdop, double t) {
    if (!state->initialized) {
        /* First measurement — initialize */
        state->x[0] = measurement.lat;
        state->x[1] = measurement.lon;
        state->x[2] = 0.0;  /* vlat */
        state->x[3] = 0.0;  /* vlon */
        mat4_eye(state->P);
        for (int i = 0; i < 4; i++) state->P[i][i] = 1e-6;
        state->last_t = t;
        state->initialized = true;
        state->last_gate = 0.0;
        return true;
    }

    double dt = t - state->last_t;
    if (dt < 0) return false;

    /* Reset on long gap */
    if (dt > cfg->max_gap_s) {
        vps_ekf_reset(state);
        return vps_ekf_update(state, cfg, measurement, hdop, t);
    }

    /* --- Predict --- */
    double F[4][4], Ft[4][4], Q[4][4];
    build_F(F, dt);
    mat4_transpose(Ft, F);
    build_Q(Q, cfg->process_noise, dt);

    /* x_pred = F * x */
    double x_pred[4];
    for (int i = 0; i < 4; i++) {
        x_pred[i] = 0;
        for (int j = 0; j < 4; j++)
            x_pred[i] += F[i][j] * state->x[j];
    }

    /* P_pred = F * P * F' + Q */
    double FP[4][4], P_pred[4][4];
    mat4_mul(FP, F, state->P);
    mat4_mul(P_pred, FP, Ft);
    mat4_add(P_pred, P_pred, Q);

    /* --- Measurement (H = [1,0,0,0; 0,1,0,0]) --- */
    double z[2] = {measurement.lat, measurement.lon};
    double y[2] = {z[0] - x_pred[0], z[1] - x_pred[1]};  /* innovation */

    double R = cfg->measurement_noise * hdop * hdop;

    /* S = H * P_pred * H' + R  (2x2 since H picks first 2 states) */
    double S[2][2];
    S[0][0] = P_pred[0][0] + R;
    S[0][1] = P_pred[0][1];
    S[1][0] = P_pred[1][0];
    S[1][1] = P_pred[1][1] + R;

    /* Invert 2x2 */
    double det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    if (fabs(det) < 1e-30) return false;
    double Si[2][2];
    Si[0][0] =  S[1][1] / det;
    Si[0][1] = -S[0][1] / det;
    Si[1][0] = -S[1][0] / det;
    Si[1][1] =  S[0][0] / det;

    /* Mahalanobis distance: d² = y' * S⁻¹ * y */
    double d2 = y[0] * (Si[0][0] * y[0] + Si[0][1] * y[1])
              + y[1] * (Si[1][0] * y[0] + Si[1][1] * y[1]);
    state->last_gate = sqrt(fabs(d2));

    if (state->last_gate > cfg->gate_threshold) {
        /* Outlier — reject but still advance time */
        memcpy(state->x, x_pred, sizeof(x_pred));
        memcpy(state->P, P_pred, sizeof(P_pred));
        state->last_t = t;
        return false;
    }

    /* Kalman gain: K = P_pred * H' * S⁻¹  (4x2) */
    double K[4][2];
    for (int i = 0; i < 4; i++) {
        K[i][0] = P_pred[i][0] * Si[0][0] + P_pred[i][1] * Si[1][0];
        K[i][1] = P_pred[i][0] * Si[0][1] + P_pred[i][1] * Si[1][1];
    }

    /* x = x_pred + K * y */
    for (int i = 0; i < 4; i++) {
        state->x[i] = x_pred[i] + K[i][0] * y[0] + K[i][1] * y[1];
    }

    /* P = (I - K*H) * P_pred */
    double KH[4][4];
    mat4_zero(KH);
    for (int i = 0; i < 4; i++) {
        KH[i][0] = K[i][0];
        KH[i][1] = K[i][1];
    }
    double IKH[4][4];
    mat4_eye(IKH);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            IKH[i][j] -= KH[i][j];
    mat4_mul(state->P, IKH, P_pred);

    state->last_t = t;
    return true;
}

vps_geopoint_t vps_ekf_predict(const vps_ekf_state_t *state, double t) {
    vps_geopoint_t p = {0.0, 0.0};
    if (!state->initialized) return p;
    double dt = t - state->last_t;
    p.lat = state->x[0] + state->x[2] * dt;
    p.lon = state->x[1] + state->x[3] * dt;
    return p;
}

vps_velocity_t vps_ekf_velocity(const vps_ekf_state_t *state) {
    vps_velocity_t v = {0.0, 0.0};
    if (state->initialized) {
        v.vn = state->x[2];
        v.ve = state->x[3];
    }
    return v;
}

double vps_ekf_speed(const vps_ekf_state_t *state) {
    if (!state->initialized) return 0.0;
    double vn = state->x[2];
    double ve = state->x[3];
    /* Convert degree/s to approximate m/s */
    double vn_ms = vn * 111320.0;
    double ve_ms = ve * 111320.0 * cos(state->x[0] * 3.14159265358979 / 180.0);
    return sqrt(vn_ms * vn_ms + ve_ms * ve_ms);
}

vps_geopoint_t vps_ekf_position(const vps_ekf_state_t *state) {
    vps_geopoint_t p = {0.0, 0.0};
    if (state->initialized) {
        p.lat = state->x[0];
        p.lon = state->x[1];
    }
    return p;
}
