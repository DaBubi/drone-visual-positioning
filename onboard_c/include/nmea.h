/**
 * @file nmea.h
 * @brief NMEA sentence generation (GGA/RMC).
 */
#ifndef NMEA_H
#define NMEA_H

#include "vps_types.h"
#include <stddef.h>

/** Compute NMEA checksum (XOR of chars between $ and *). */
uint8_t vps_nmea_checksum(const char *sentence);

/**
 * Format a $GPGGA sentence.
 * @param buf output buffer (must be >= 128 bytes)
 * @param pos GPS position
 * @param fix_quality 0=no fix, 1=GPS fix
 * @param hdop horizontal DOP
 * @param altitude altitude in meters
 * @return number of bytes written (excluding null terminator)
 */
int vps_format_gga(char *buf, size_t buflen,
                   vps_geopoint_t pos, int fix_quality,
                   double hdop, double altitude);

/**
 * Format a $GPRMC sentence.
 * @param buf output buffer (must be >= 128 bytes)
 * @param pos GPS position
 * @param active true if fix is valid
 * @param speed_knots ground speed in knots
 * @param heading_deg heading in degrees (0=N, clockwise)
 * @return number of bytes written
 */
int vps_format_rmc(char *buf, size_t buflen,
                   vps_geopoint_t pos, bool active,
                   double speed_knots, double heading_deg);

#endif /* NMEA_H */
