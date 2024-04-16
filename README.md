# Financial Data Analysis Tool

## Description
This application reads minute resolution stock price data, processes the data to calculate various metrics, and uses these metrics to predict IV using a HARQ model.

## Features
- Load and parse stock price data from CSV files.
- Calculate intraday returns, variance, and quarticity.
- Group data by days to process daily, weekly, and monthly statistics.
- Fit a HARQ model to the data using nlopt optimization library.
- Predict implied volatility and compare with actual values using Mean Squared Error.

## Dependencies
- C++11 or higher
- [nlopt](https://nlopt.readthedocs.io/en/latest/) - Nonlinear optimization library

## Installation
1. Ensure you have a C++ compiler that supports C++11.
2. Install nlopt:
   - On Ubuntu: `sudo apt-get install libnlopt-dev`
   - On other platforms, follow the instructions from the [nlopt documentation](https://nlopt.readthedocs.io/en/latest/NLopt_Installation/).

## Usage
1. Prepare your CSV file containing the stock price data. The expected format is:
