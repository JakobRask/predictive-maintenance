#Here we interaction term for voltage and pressure in telemetry data
telemetry <- telemetry %>%
  mutate(voltage_pressure_interaction = volt * pressure)

library(ggplot2)
library(lubridate)
#options(device = "windows") # nolint
ggplot(telemetry, aes(x = datetime, y = voltage_pressure_interaction)) +
  geom_line() +
  labs(title = "Voltage-Pressure Interaction Over Time",
       x = "Time",
       y = "Voltage * Pressure")