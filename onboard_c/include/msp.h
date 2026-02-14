/**
 * @file msp.h
 * @brief MSP (MultiWii Serial Protocol) GPS injection.
 */
#ifndef MSP_H
#define MSP_H

#include "vps_types.h"
#include <stddef.h>

#define MSP_CMD_SET_RAW_GPS 201
#define MSP_HEADER_SIZE 5   /* $M< + len + cmd */
#define MSP_GPS_PAYLOAD 18
#define MSP_GPS_FRAME_SIZE (MSP_HEADER_SIZE + MSP_GPS_PAYLOAD + 1) /* +checksum */

/** MSP GPS data. */
typedef struct {
    uint8_t  fix_type;       /* 0=no fix, 2=2D, 3=3D */
    uint8_t  num_sat;
    int32_t  lat;            /* degrees * 1e7 */
    int32_t  lon;            /* degrees * 1e7 */
    int16_t  altitude_m;
    uint16_t speed_cms;      /* cm/s */
    uint16_t heading_deg10;  /* degrees * 10 */
    uint16_t hdop;           /* HDOP * 100 */
} vps_msp_gps_t;

/** Build MSP GPS data from position. */
vps_msp_gps_t vps_msp_from_position(vps_geopoint_t pos, double speed_mps,
                                    double heading_deg, double hdop,
                                    bool has_fix);

/**
 * Encode MSP_SET_RAW_GPS frame.
 * @param out buffer (must be >= MSP_GPS_FRAME_SIZE = 24 bytes)
 * @param gps GPS data to encode
 * @return frame size (always 24)
 */
int vps_msp_encode(uint8_t *out, const vps_msp_gps_t *gps);

/** Compute MSP checksum (XOR of len + cmd + payload). */
uint8_t vps_msp_checksum(const uint8_t *data, size_t len);

#endif /* MSP_H */
