# Experimental Data

This directory contains code to generate the input data for the monitor.

Each subdirectory represents a data source. `setup.sh` fetches and sets the data source up (for example, fetching its Git repository and building it). The inference scripts run inference, and save the predictions along with the input data to `predictions/`.

