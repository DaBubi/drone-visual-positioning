# Drone Visual Positioning System

## Overview
Estimate GPS coordinates from a drone camera pointed straight down (nadir) by
matching the image against georeferenced satellite imagery.

## Architecture
```
drone image → tile retrieval → feature matching → homography → GPS
```

1. **Tile Retrieval**: NetVLAD descriptors + FAISS index to find candidate satellite tiles
2. **Feature Matching**: SuperPoint + LightGlue for point correspondences
3. **Homography**: RANSAC to compute drone→satellite geometric transform
4. **GPS Extraction**: Map pixel coordinates through tile math to lat/lon
5. **Dashboard**: Web UI showing position estimates, confidence, matched tiles

## Tech Stack
- Python 3.13+, PyTorch, OpenCV, Kornia
- SuperPoint + LightGlue for matching
- FAISS for vector similarity search
- Flask for dashboard
- pygeotile / pyproj for coordinate math

## Key Parameters
- Edge threshold for match confidence: inlier ratio > 30%
- Tile zoom levels: 15 (coarse), 17 (fine), 19 (precise)
- Tile source: Mapbox Satellite API

## Conventions
- Type hints everywhere, Pydantic models for data
- Async where possible (tile downloads)
- Small testable functions
- See ALGORITHMS.md for detailed algorithm descriptions
