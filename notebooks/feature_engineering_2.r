install.packages("languageserver", lib = "C:/Users/rshaw/Desktop/EC Utbildning - Data Science/Kurs 9 - Project/Project/ds23_projektkurs/predictive-maintenance/notebooks") # nolint
install.packages("zoo")

library(zoo)
library(dplyr)
library(lubridate)
library(tidyverse)

telemetry <- read.csv("data/raw/PdM_telemetry.csv")
errors <- read.csv("data/raw/PdM_errors.csv")
maintenance <- read.csv("data/raw/PdM_maint.csv")
failures <- read.csv("data/raw/PdM_failures.csv")
machines <- read.csv("data/raw/PdM_Machines.csv")

dim(telemetry)
head(telemetry)
str(telemetry)
summary(telemetry)

str(errors)
str(failures)
str(maintenance)
str(machines)

sum(is.na(telemetry))
sum(is.na(errors))
sum(is.na(maintenance))
sum(is.na(failures))
sum(is.na(machines))
colnames(telemetry)

class(errors$datetime)
class(maintenance$datetime)
class(failures$datetime)
class(telemetry$datetime)

# Convert datetime columns to POSIXct format
errors$datetime <- ymd_hms(errors$datetime)
maintenance$datetime <- ymd_hms(maintenance$datetime)
failures$datetime <- ymd_hms(failures$datetime)
telemetry$datetime <- ymd_hms(telemetry$datetime)

class(errors$datetime)
class(maintenance$datetime)
class(failures$datetime)
class(telemetry$datetime)

str(telemetry$machineID)
str(errors$machineID)
str(errors$datetime)


merged_data <- left_join(telemetry, errors, by = c("machineID", "datetime"))
merged_data <- left_join(merged_data, failures, by = c("machineID", "datetime"))
merged_data <- left_join(merged_data, maintenance, by = c("machineID", "datetime"), relationship = "many-to-many") # nolint # We are aware that there are valid many-to-many relationships likely 
# because some machines had multiple maintenance actions at the same time, and we thus want to keep all records. #nolint
merged_data <- left_join(merged_data, machines, by = "machineID")
head(merged_data)

# NA values: These NAs appear as there is no matching record in the joined datasets (errors, failures, or maintenance) for the specific machineID and datetime. #nolint
# A machine may not have experienced an error, failure, or maintenance at every time point recorded in telemetry, so those columns remain NA for those timestamps. #nolint
# Below, we assign default messages to replace NA values.

merged_data <- merged_data %>%
  mutate(errorID = ifelse(is.na(errorID), "No Error", errorID),
         failure = ifelse(is.na(failure), "No Failure", failure),
         comp = ifelse(is.na(comp), "No Maintenance", comp))

View(merged_data %>% slice(1:50))

# Create target variables to predict time until next failure.
merged_data <- merged_data %>%
  mutate(datetime = as.POSIXct(datetime))

# Sort data by machineID and datetime
merged_data <- merged_data %>%
  arrange(machineID, datetime)

# Calculate time until the next failure in hours
merged_data <- merged_data %>%
  group_by(machineID) %>%
  mutate(time_until_next_failure = as.numeric(difftime(lead(datetime, default = last(datetime)), datetime, units = "hours"))) %>% # nolint
  ungroup()

# Feature Engingeering
# 1. Create rolling averages for telemetry data - rolling average for pressure
# 2. Error count over last 24 hours
# 3. Time since last failure: How long ago was the last failure event?
# 4. Time since last error: How long ago was the last error event?

#1. Pressure rolling average
merged_data <- merged_data %>%
  group_by(machineID) %>%
  arrange(datetime) %>%
  mutate(pressure_rolling_avg = rollmean(pressure, k = 24, fill = NA, align = "right")) %>% # nolint
  ungroup()

#2. Error count last 24hr
merged_data <- merged_data %>%
  group_by(machineID) %>%
  mutate(error_count_24h = rollapplyr(!is.na(errorID), width = 24, FUN = sum, fill = 0, align = "right")) # nolint

#3. Time since last failure
merged_data <- merged_data %>%
  group_by(machineID) %>%
  mutate(time_since_last_failure = as.numeric(difftime(datetime, lag(datetime[failure != ""], default = first(datetime)), units = "hours"))) %>% # nolint
  ungroup()

#4. Time since last error
# Calculate time since the last error in hours
merged_data <- merged_data %>%
  group_by(machineID) %>%
  mutate(time_since_last_error = as.numeric(difftime(datetime, lag(datetime[!is.na(errorID)], default = first(datetime)), units = "hours"))) %>% # nolint
  ungroup()

str(merged_data)

# Fit a linear regression model predicting failure
linear_model <- lm(time_until_next_failure ~ volt + rotate + pressure_rolling_avg + vibration + error_count_24h + # nolint
                time_since_last_failure + time_since_last_error + model + age, # nolint
                data = merged_data) # nolint
summary(linear_model)
# Model fit is very low with a multiple R-squared value of 0.009621, meaning that only about 0.96% of the variability in time_until_next_failure is explained by the model. # nolint
# Several predictors including volt, rotate, pressure_rolling_avg and vibration are statistically significant, but the model explains very little variance in the dependent variable (time_until_next_failure). # nolint


# We need better interactive features to improve the model so we will carry out subset selection to evaluate various subsets of our predictors to identify those that yield the best model. # nolint
install.packages("leaps")
library(leaps)
subset_model <- regsubsets(time_until_next_failure ~ volt + rotate + pressure + vibration, data = merged_data, nvmax = 4) # nolint
summary(subset_model)
plot(subset_model, scale = "adjr2")
# Best model: The subset model suggests that a combination of volt and rotate are the best fit. # nolint

# Given the poor fit of the linear regression model, we would normally look for better predictors and more complex feature interactions, but instead I used a subset # nolint
# selection model which suggests that a combination of voltage and rotate are the best feature fits to predict failure. I think this gives us justification to explore  # nolint
# Recurrent Neural Networks (RNNs), or more specifically LSTMs, as they are well-suited for capturing temporal dependencies and sequential patterns in time series  # nolint
# data. Unlike linear regression, which relies on a static relationship between variables, RNNs can leverage the sequential nature of sensor data, such as  # nolint
# fluctuations in voltage and rotation over time, to better predict time_until_next_failure. # nolint

# This connects well with the goal of predicting anomalies in sensor readings (e.g., pressure, voltage) that show potential machine failure.  # nolint
# By training an LSTM model on telemetry data to forecast sensor readings, we can capture more complex time-dependent relationships.  # nolint
# Then, by comparing the forecasted readings with historical failures, we can start to identify patterns and deviations that precede breakdowns.  # nolint
# This approach helps us to not only predict when a failure might occur but also detect anomalies in the sensor data that signal potential issues,  # nolint
# giving us a better solution than linear regression alone.