install.packages("languageserver", lib = "C:/Users/rshaw/Desktop/EC Utbildning - Data Science/Kurs 9 - Project/Project/ds23_projektkurs/predictive-maintenance/notebooks")

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

telemetry$datetime <- ymd_hms(telemetry$datetime)

# Some of the datasets have datetime columns in character format (chr). So before merging, we convert them to the same format as our telemetry data (POSIXct).
errors$datetime <- ymd_hms(errors$datetime)
failures$datetime <- ymd_hms(failures$datetime)
maintenance$datetime <- ymd_hms(maintenance$datetime)

invalid_telemetry_dates <- telemetry[is.na(ymd_hms(telemetry$datetime)), ]
head(invalid_telemetry_dates)

head(telemetry$datetime)
head(telemetry[is.na(telemetry$datetime), "datetime"])
raw_invalid_telemetry <- telemetry[is.na(ymd_hms(telemetry$datetime)), "datetime"]
head(raw_invalid_telemetry)
library(tidyr)
telemetry <- fill(telemetry, datetime, .direction = "downup")


# We use the left_join() function to combine the datasets on machineID and datetime. Since the telemetry data is the largest, we  merge the others onto it. Could also do this in SQL?
merged_data <- left_join(telemetry, errors, by = c("machineID", "datetime"))
merged_data <- left_join(telemetry, errors, by = c("machineID", "datetime"), relationship = "many-to-many")

merged_data <- left_join(merged_data, failures, by = c("machineID", "datetime"))
merged_data <- left_join(merged_data, maintenance, by = c("machineID", "datetime"), relationship = "many-to-many")
merged_data <- left_join(merged_data, machines, by = "machineID")

head(merged_data)

# Replace NAs in errorID and failure with "none" and maintenance component with "none"
merged_data$errorID[is.na(merged_data$errorID)] <- "none"
merged_data$failure[is.na(merged_data$failure)] <- "none"
merged_data$comp[is.na(merged_data$comp)] <- "none"

colnames(merged_data)

# Check the first few rows of comp.x and comp.y
head(merged_data[c("comp.x", "comp.y")])
# Check for matching timestamps in telemetry and maintenance
common_times <- intersect(telemetry$datetime, maintenance$datetime)
length(common_times)  # How many matching timestamps are there?

# Round datetime to the nearest hour
telemetry$datetime <- floor_date(telemetry$datetime, unit = "hour")
maintenance$datetime <- floor_date(maintenance$datetime, unit = "hour")

# Merge telemetry with maintenance again
merged_data <- left_join(telemetry, maintenance, by = c("machineID", "datetime"))
common_times_rounded <- intersect(telemetry$datetime, maintenance$datetime)
length(common_times_rounded)  # Check for improved matches


#Here we interaction term for voltage and pressure in telemetry data
telemetry <- telemetry %>%
  mutate(voltage_pressure_interaction = volt * pressure)

library(ggplot2)
library(lubridate)
#options(device = "windows")
ggplot(telemetry, aes(x = datetime, y = voltage_pressure_interaction)) +
  geom_line() +
  labs(title = "Voltage-Pressure Interaction Over Time", 
       x = "Time", 
       y = "Voltage * Pressure")


¨

# library(lubridate)
# str(telemetry)
# dev.list()

