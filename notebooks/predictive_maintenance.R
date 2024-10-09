library(dplyr)
library(lubridate)
library(tidyverse)

telemetry <- read.csv("data/raw/PdM_telemetry.csv")
errors <- read.csv("data/raw/PdM_errors.csv")
maintenance <- read.csv("data/raw/PdM_maint.csv")
failures <- read.csv("data/raw/PdM_failures.csv")
machines <- read.csv("data/raw/PdM_Machines.csv")

# Create interaction term for voltage and pressure in telemetry data
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

#library(lubridate)
#telemetry$datetime <- ymd_hms(telemetry$datetime)
#str(telemetry)
#dev.list()

