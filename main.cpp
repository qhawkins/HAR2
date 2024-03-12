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


int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}

