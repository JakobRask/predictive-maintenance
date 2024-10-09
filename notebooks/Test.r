install.packages("languageserver", lib = "C:/Users/rshaw/Desktop/EC Utbildning - Data Science/Kurs 9 - Project/Project/ds23_projektkurs/predictive-maintenance/notebooks") # nolint

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

# Check for missing values in the datasets
sum(is.na(telemetry))
sum(is.na(errors))
sum(is.na(maintenance))
sum(is.na(failures))
sum(is.na(machines))
colnames(telemetry)

# Check the datetime column in each dataset
class(errors$datetime)
class(maintenance$datetime)
class(failures$datetime)

library(lubridate)
# Convert datetime columns to POSIXct format
errors$datetime <- ymd_hms(errors$datetime)
maintenance$datetime <- ymd_hms(maintenance$datetime)
failures$datetime <- ymd_hms(failures$datetime)

class(errors$datetime)
class(maintenance$datetime)
class(failures$datetime)

install.packages("fuzzyjoin")
library(fuzzyjoin)

# Fuzzy join telemetry and errors with a 1-hour tolerance
merged_data <- difference_left_join(telemetry, errors, 
                                    by = c("machineID", "datetime"), 
                                    max_dist = dminutes(60))

# Fuzzy join the merged data with failures (1-hour tolerance)
merged_data <- difference_left_join(merged_data, failures, 
                                    by = c("machineID", "datetime"), 
                                    max_dist = dminutes(60))

# Fuzzy join the result with maintenance data (1-hour tolerance)
merged_data <- difference_left_join(merged_data, maintenance, 
                                    by = c("machineID", "datetime"), 
                                    max_dist = dminutes(60))

# Finally, merge with machine metadata (by machineID, no need for datetime)
merged_data <- left_join(merged_data, machines, by = "machineID")

head(merged_data)

# Fit a simple linear regression model
model <- lm(~ volt + rotate + pressure + vibration, data = telemetry) # nolint

# View the summary to check the statistical significance of features
summary(model)

install.packages("leaps")
library(leaps)

# Perform subset selection using regsubsets
subset_model <- regsubsets(voltage_pressure_interaction ~ volt + rotate + pressure + vibration, data = telemetry, nvmax = 4)

# View summary of the subset selection
summary(subset_model)

# Plot the results of the subset selection
plot(subset_model, scale = "adjr2")
