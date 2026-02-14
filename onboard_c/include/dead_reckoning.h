/**
 * @file dead_reckoning.h
 * @brief Constant-velocity dead reckoning fallback.
 */
#ifndef DEAD_RECKONING_H
#define DEAD_RECKONING_H

#include "vps_types.h"

typedef struct {
    vps_geopoint_t ref_pos;
    double vn_mps;
    double ve_mps;
    double ref_hdop;
    double ref_t;
    double hdop_growth_rate;
    double max_extrap_s;
    bool has_reference;
} vps_dr_state_t;

/** Initialize dead reckoning state. */
void vps_dr_init(vps_dr_state_t *dr, double max_extrap_s, double hdop_growth_rate);

/** Update reference position and velocity. */
void vps_dr_update_ref(vps_dr_state_t *dr, vps_geopoint_t pos,
                       double vn, double ve, double hdop, double t);

/**
 * Extrapolate position at time t.
 * @param pos_out output position
 * @param hdop_out output HDOP (degraded)
 * @return true if valid extrapolation
 */
bool vps_dr_extrapolate(const vps_dr_state_t *dr, double t,
                        vps_geopoint_t *pos_out, double *hdop_out);

#endif /* DEAD_RECKONING_H */
