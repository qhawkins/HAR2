#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>
#include <iomanip>
#include <ctime>
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
        double return_i = (std::log(prices[i+1])-std::log(prices[i]));
        //double log_return = std::log(return_i);
        //std::cout << "return_i: " << return_i << std::endl;
        returns.push_back(return_i);
        //std::cout << "log_return: " << log_return << std::endl;
    }
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
                   [](double x) { return std::pow(x, 2); });
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
    std::vector<double> rq_w;  // Weekly realized quarticity
    std::vector<double> rq_m;  // Monthly realized quarticity
    std::vector<double> rv;   // Actual realized variance values for the objective function calculation
};

double objectiveFunction(unsigned n, const double* x, double* grad, void* f_data) {
    HarqModelData* harqData = reinterpret_cast<HarqModelData*>(f_data);
    double sumOfSquaredResiduals = 0.0;
    double k = 4.685; // Tuning constant for Tukey's bisquare estimator
    
    for (size_t i = 0; i < harqData->rv.size(); ++i) {
        double fi = harqData->rv[i] - x[0] -
                    (x[1] * harqData->rv_d[i]) -
                    (x[2] * harqData->rv_w[i]) -
                    (x[3] * harqData->rv_m[i]);
        
        double absResidual = std::abs(fi);
        
        if (absResidual <= k) {
            double weight = std::pow(1 - std::pow(absResidual / k, 2), 2);
            sumOfSquaredResiduals += weight * std::pow(fi, 2);
        } else {
            sumOfSquaredResiduals += std::pow(k, 2) / 6.0;
        }
    }
    
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
    //std::cout << std::accumulate(prices[22].begin(), prices[22].end(), 0.0) << std::endl;
    //exit(100);
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

    return {actualVariance, dayVariance, accWeekVariance, accMonthVariance, dayQuarticity, accWeekQuarticity, accMonthQuarticity};

}

std::vector<std::vector<double>> accumulateDWMMetrics(std::vector<std::vector<std::vector<double>>>& prices, int endDay, int optimHorizon, std::vector<int>& dayIdxs){
    std::vector<std::vector<double>> metrics;
    std::vector<std::vector<double>> selPrices;
    int startIdx, endIdx;

    for (int i = 0; i < optimHorizon; ++i ){
        //startIdx = dayIdxs[i];
        //endIdx = dayIdxs[i+1];
        selPrices = prices[i];
        //std::cout << "selPrices size: " << selPrices.size() << std::endl;
        metrics.push_back(calcDWMMetrics(selPrices));
    }

    //metrics=calcDWMMetrics(selPrices));

    return metrics;
}

double calcHARQIv(std::vector<double>& inputs){
    double beta0 = inputs[0];
    double beta1 = inputs[1];
    double beta2 = inputs[2];
    double beta3 = inputs[3];
    
    double dVariance = inputs[4];
    double wVariance = inputs[5];
    double mVariance = inputs[6];

    double u = inputs[7];


    double harq = beta0 + (beta1 * dVariance) + 
                    (beta2 * wVariance) + 
                    (beta3 * mVariance);
    
    
    //double transformed_harq = std::exp(harq)-1;

    double iv = std::pow(harq, .5)*std::pow(252, .5)*100;
    //std::cout << "harq: " << harq << " transformed_harq: " << transformed_harq << " iv: " << iv << "\n";
    //std::cout << "beta0: " << beta0 << " beta1: " << beta1 << " beta1q: " << beta1q << " beta2: " << beta2 << " beta3: " << beta3 << " u: " << u << "\n";

    return iv;
}

std::vector<std::vector<double>> parseDays(std::vector<double>& prices, std::vector<int> dayGroups, int day){
    std::vector<std::vector<double>> intradayGrouped;
    std::vector<double> dayPrices;
    //std::cout << "parse days day: " << day << std::endl;
    for (int i = day-22; i <= day; ++i){
        dayPrices = std::vector<double>(prices.begin()+dayGroups[i], prices.begin()+dayGroups[i+1]);
        //std::cout << "day prices at i: " << i << " size: " << dayPrices.size() << " and value of " << std::accumulate(dayPrices.begin(), dayPrices.end(), 0.0) << std::endl;
        intradayGrouped.push_back(dayPrices);
    }
    return intradayGrouped;    
}

std::pair<double, double> calculateMeanStdDev(const std::vector<double>& data) {
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();

    double squaredSum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double stdDev = std::sqrt((squaredSum / data.size()) - (mean * mean));

    return std::make_pair(mean, stdDev);
}

std::vector<double> trainHarq(std::vector<double>& prices, std::vector<int>& dayIdxs, int optim_horizon, int day){
    std::vector<std::vector<std::vector<double>>> intradayGrouped;
    for (int i = day-optim_horizon; i < day; ++i){
        //std::cout << "i: " << i << std::endl;
        intradayGrouped.push_back(parseDays(prices, dayIdxs, i));    
    }
    //exit(1028382);
    std::cout << intradayGrouped.size() << std::endl;
    //std::cout << "intradayGrouped size: " << intradayGrouped.size() << std::endl;
    
    std::vector<std::vector<double>> metrics = accumulateDWMMetrics(intradayGrouped, day, optim_horizon, dayIdxs);

    std::vector<double> selMetrics;

    std::vector<double> dRV;
    std::vector<double> wRV;
    std::vector<double> mRV;
    std::vector<double> dQ;
    std::vector<double> wQ;
    std::vector<double> mQ;
    std::vector<double> rv;

    //actualVariance, dayVariance, accWeekVariance, accMonthVariance, dayQuarticity, accWeekQuarticity, accMonthQuarticity

    for (size_t i  = 0; i < metrics.size(); ++i){
        selMetrics = metrics[i];
        rv.push_back(selMetrics[0]);
        dRV.push_back(selMetrics[1]);
        wRV.push_back(selMetrics[2]);
        mRV.push_back(selMetrics[3]);
        dQ.push_back(selMetrics[4]);
        wQ.push_back(selMetrics[5]);
        mQ.push_back(selMetrics[6]);
        }

    HarqModelData harqData {
        .rv_d = dRV,
        .rv_w = wRV,
        .rv_m = mRV,
        .rq_d = dQ,
        .rq_w = wQ,
        .rq_m = mQ,
        .rv = rv
    };

    double dVariance = selMetrics[1];
    double wVariance = selMetrics[2];
    double mVariance = selMetrics[3];

    std::vector<double> betas(4);
    betas[0] = 0;
    betas[1] = 0.1;
    betas[2] = 0.1;
    betas[3] = 1.1e-10;
    //betas[4] = -.3;
    std::vector<double> lb(4);
    lb[0] = -1;
    lb[1] = 0.05;
    lb[2] = 0.05;
    lb[3] = 1e-10;

    std::vector<double> ub(4);
    ub[0] = 1;
    ub[1] = 1;
    ub[2] = 1;
    ub[3] = 1;


    nlopt::algorithm alg = nlopt::LN_NELDERMEAD;

    nlopt::opt optimizer = nlopt::opt(alg, 4);

    optimizer.set_min_objective(objectiveFunction, &harqData);

    optimizer.set_lower_bounds(lb);
    optimizer.set_upper_bounds(ub);
    //optimizer.set_xtol_rel(1e-2);
    optimizer.set_maxeval(10000000);
    optimizer.set_stopval(1e-20);
    //optimizer.set_ftol_rel(1e-20);
    //optimizer.set_xtol_abs(1e-3);
    
    double minf; /* the minimum objective value, upon return */

    try{
        nlopt::result result = optimizer.optimize(betas, minf);
        
    }
    catch(std::exception &e){
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }

    double beta0 = betas[0];
    double beta1 = betas[1];
    double beta2 = betas[2];
    double beta3 = betas[3];

    double u = minf;

    std::cout << "beta0: " << beta0 << " beta1: " << beta1 << " beta2: " << beta2 << " beta3: " 
    << beta3 << " u: " << u << std::endl;


    return {beta0, beta1, beta2, beta3, dVariance, wVariance, mVariance, u};

}

double trueIv(std::vector<double>& prices, std::vector<int>& dayIdxs, int day){
    std::vector<double> dayPrices = std::vector<double>(prices.begin()+dayIdxs[day], prices.begin()+dayIdxs[day+1]);
    double realizedVariance = calculateVariance(dayPrices)[0];
    double iv = std::pow(realizedVariance, .5)*std::pow(252, .5)*100;
    
    return iv;
}

double calcMSE(double calcIv, double trueIv){
    double mse = std::pow((calcIv-trueIv), 2);
    return mse;
}

int main() {
    std::vector<std::tm> timestamps = loadTimestamps("AAPL_1min_firstratedata.csv");

    std::vector<double> prices = loadPrices("AAPL_1min_firstratedata.csv");
    
    std::vector<int> dayGroups = dayGroupsIdx(timestamps);

    int trainHorizon = 66*2;

    std::cout << "Day groups size: " << dayGroups.size() << std::endl;
    
    std::vector<double> inputs = std::vector<double>(14, 0);
    
    std::vector<double> ivMSE;

    for (int i  = trainHorizon+22; i < dayGroups.size()-1; ++i){
        inputs = trainHarq(prices, dayGroups, trainHorizon, i);
        double iv = calcHARQIv(inputs);
        double true_iv = trueIv(prices, dayGroups, i);
        double mse = calcMSE(iv, true_iv);

        ivMSE.push_back(mse);

        std::cout << "Predicted Annualized IV for day " << i << " is " << iv << 
                "%, True IV is: " << true_iv << "%, MSE between actual and predicted is: " << mse << std::endl;
        }
    std::cout << "Mean MSE: " << std::accumulate(ivMSE.begin(), ivMSE.end(), 0.0)/ivMSE.size() << std::endl;

    return 0;
}

