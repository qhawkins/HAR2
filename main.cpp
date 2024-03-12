#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <Dense>
#include <nlopt.h>
#include <nlopt.hpp>

std::vector<double> loadPrices(const std::string& filename) {
    std::vector<double> Price;
    std::ifstream file(filename); // Adjust the path if necessary
    std::string line, cell;

    getline(file, line); // Assuming the first line is headers
    // Read prices from CSV
    while (std::getline(file, line)) {
        std::stringstream ss(line);

        // Iterate through each column in the CSV row
        for (int i = 0; i < 6; ++i) {
            std::getline(ss, cell, ',');
            if (i == 4) { // The 6th column (0-indexed, so i == 5)
                try {
                    Price.push_back(std::stod(cell)); // Convert string to double and store
                } catch (...) {
                    // Handle conversion error if necessary, for simplicity just continue
                    continue;
                }
            }
        }
    }
    file.close();

    return Price;
}

// Function to parse a timestamp string and convert it to a tm structure
std::tm parseTimestamp(const std::string& timestamp) {
    std::tm datetime = {};
    std::stringstream ss(timestamp);

    // Assuming the format is YYYY-MM-DD HH:MM:SS
    char dateSeparator, timeSeparator;
    int year, month, day, hour, minute, second;

    ss >> year >> dateSeparator >> month >> dateSeparator >> day
       >> hour >> timeSeparator >> minute >> timeSeparator >> second;

    datetime.tm_year = year - 1900; // tm_year is years since 1900
    datetime.tm_mon = month - 1;    // tm_mon is 0-11
    datetime.tm_mday = day;
    datetime.tm_hour = hour;
    datetime.tm_min = minute;
    datetime.tm_sec = second;

    // Adjust based on your needs, e.g., setting tm_isdst to 0 or 1
    datetime.tm_isdst = -1; // Let the system determine DST

    return datetime;
}

std::vector<std::tm> loadTimestamps(const std::string& filename) {
    std::vector<std::tm> Timestamps;
    std::ifstream file(filename); // Adjust the path if necessary
    std::string line, cell;

    getline(file, line); // Assuming the first line is headers
    // Read prices from CSV
    while (std::getline(file, line)) {
        std::stringstream ss(line);

        // Iterate through each column in the CSV row
        for (int i = 0; i < 6; ++i) {
            std::getline(ss, cell, ',');
            if (i == 0) { // The 6th column (0-indexed, so i == 5)
                try {
                    Timestamps.push_back(parseTimestamp(cell)); // Convert string to double and store
                } catch (...) {
                    // Handle conversion error if necessary, for simplicity just continue
                    continue;
                }
            }
        }
    }
    file.close();

    return Timestamps;
}

// Function to calculate intraday returns
std::vector<double> calculateIntradayReturns(const std::vector<double>& prices) {
    std::vector<double> returns;
    //std::cout << "intraday prices size: " << prices.size() << std::endl;
    for (size_t i = 0; i < prices.size()-1; ++i) {
        //std::cout << i << std::endl;
        double return_i = (prices[i+1] / prices[i])-1;
        //std::cout << "return_i: " << return_i << std::endl;
        returns.push_back(std::pow(return_i, 2));
    }
    //exit(10384);
    return returns;
}

//calculate variance and quarticity
std::vector<double> calculateVariance(const std::vector<double>& prices) {
    if (prices.size() < 2) {
        // Not enough data points to calculate quarticity
        exit(1029384);
    }

    int n = prices.size();

    std::vector<double> intradayReturns = calculateIntradayReturns(prices);

    //realized variance calculation
    std::vector<double> squaredReturns(intradayReturns.size());
    std::vector<double> quarticReturns(intradayReturns.size());
    std::transform(intradayReturns.begin(), intradayReturns.end(), squaredReturns.begin(),
                   [](double x) { return x * x; });
    std::transform(intradayReturns.begin(), intradayReturns.end(), quarticReturns.begin(),
                   [](double x) { return std::pow(x, 4); });
    
    double realizedVariance = std::accumulate(squaredReturns.begin(), squaredReturns.end(), 0.0);
    double realizedQuarticity = std::accumulate(quarticReturns.begin(), quarticReturns.end(), 0.0);
    realizedQuarticity = realizedQuarticity*(n/3);
    //std::cout << "realizedVariance: " << realizedVariance << std::endl;
    return {realizedVariance, realizedQuarticity};
}

std::vector<int> dayGroupsIdx(std::vector<std::tm> timestamps){
    std::vector<int> dayGroupsIdx(1, 0);
    int day = timestamps[0].tm_mday;
    for (size_t i = 0; i < timestamps.size(); ++i) {
        if (timestamps[i].tm_mday != day){
            dayGroupsIdx.push_back(i);
            day = timestamps[i].tm_mday;
        }
    }
    return dayGroupsIdx;
}

struct HarqModelData {
    std::vector<double> rv_d; // Daily realized variance
    std::vector<double> rv_w; // Weekly realized variance
    std::vector<double> rv_m; // Monthly realized variance
    std::vector<double> rq_d;  // Daily realized quarticity
    std::vector<double> rv;   // Actual realized variance values for the objective function calculation
};

double objectiveFunction(unsigned n, const double* x, double* grad, void* f_data) {
    HarqModelData *harqData = reinterpret_cast<HarqModelData *>(f_data);
    double sumOfSquaredResiduals = 0.0;

    for (size_t i = 0; i < harqData->rv.size(); ++i) {
        double prediction = x[0] // β0
                             + (x[1] + x[4] * harqData->rq_d[i]) * harqData->rv_d[i] // (β1 + β1Q * RQ_t_d-1) * RV_t_d-1
                             + x[2] * harqData->rv_w[i] // β2 * RV_t_w-1
                             + x[3] * harqData->rv_m[i]; // β3 * RV_t_m-1

        double residual = harqData->rv[i] - prediction;
        sumOfSquaredResiduals += residual * residual;

        if (grad) {
            grad[0] += -2 * residual; // dS/dβ0
            grad[1] += -2 * residual * harqData->rv_d[i]; // dS/dβ1
            grad[2] += -2 * residual * harqData->rv_w[i]; // dS/dβ2, similar for others
            grad[3] += -2 * residual * harqData->rv_m[i]; // dS/dβ3
            grad[4] += -2 * residual * harqData->rq_d[i] * harqData->rv_d[i]; // dS/dβ1Q
        }

    }
    //std::cout << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << " " << x[4] << " " << sumOfSquaredResiduals << std::endl;

    return sumOfSquaredResiduals;
}

std::vector<double> calcDWMMetrics(std::vector<std::vector<double>>& prices){
    double dayQuarticity;
    double dayVariance;
    
    double actualVariance;

    double quarticity;
    double variance;

    double accWeekQuarticity = 0.0;
    double accWeekVariance = 0.0;

    double accMonthQuarticity = 0.0;
    double accMonthVariance = 0.0;
    //std::cout << "prices size: " << prices.size() << std::endl
    dayVariance = calculateVariance(prices[21])[0];
    dayQuarticity = calculateVariance(prices[21])[1];
    //std::cout << "actual variance" << std::endl;
    actualVariance = calculateVariance(prices[22])[0];
    //std::cout << "actualVariance: " << actualVariance << std::endl;
    for (int i = 17; i < 22; ++i){
        quarticity = calculateVariance(prices[i])[1];
        variance = calculateVariance(prices[i])[0];

        accWeekQuarticity += quarticity/5;
        accWeekVariance += variance/5;
    }

    for (int i = 0; i < 22; ++i){
        quarticity = calculateVariance(prices[i])[1];
        variance = calculateVariance(prices[i])[0];

        accMonthQuarticity += quarticity/22;
        accMonthVariance += variance/22;
    }

    return {actualVariance, dayQuarticity, dayVariance, accWeekQuarticity, accWeekVariance, accMonthQuarticity, accMonthVariance};

}

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}

