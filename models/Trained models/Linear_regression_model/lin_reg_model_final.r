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
write.csv(merged_data %>% slice(1:50), "C:/Users/rshaw/Desktop/EC Utbildning - Data Science/Kurs 9 - Project/Project/ds23_projektkurs/predictive-maintenance/models/Trained models/merged_data_first_50.csv", row.names = FALSE) #nolint

merged_data <- merged_data %>%
  mutate(datetime = as.POSIXct(datetime))
merged_data <- merged_data %>%
  arrange(machineID, datetime)

# Feature Engineering
# 1. Error count over last 24 hours
# 2. Anomaly Detection in Sensor Readings: Detect anomalies in pressure, voltage, vibration, and rotation #nolint
# 3. Time since last failure: How long ago was the last failure event?
# 4. Time since last error: How long ago was the last error event?

#1. Error count last 24hr
merged_data <- merged_data %>%
  group_by(machineID) %>%
  mutate(error_count_24h = rollapplyr(!is.na(errorID), width = 24, FUN = sum, fill = 0, align = "right")) %>% # nolint
  ungroup()

#2. Anomaly Detection in Sensor Readings
merged_data <- merged_data %>%
  group_by(machineID) %>%
  arrange(datetime) %>%
  mutate(pressure_anomaly = ifelse(abs(pressure - rollmean(pressure, k = 24, fill = NA, align = "right")) > 2*rollapply(pressure, width = 24, FUN = sd, fill = NA, align = "right"), 1, 0), #nolint
         volt_anomaly = ifelse(abs(volt - rollmean(volt, k = 24, fill = NA, align = "right")) > 2*rollapply(volt, width = 24, FUN = sd, fill = NA, align = "right"), 1, 0), #nolint
         vibration_anomaly = ifelse(abs(vibration - rollmean(vibration, k = 24, fill = NA, align = "right")) > 2*rollapply(vibration, width = 24, FUN = sd, fill = NA, align = "right"), 1, 0), #nolint
         rotate_anomaly = ifelse(abs(rotate - rollmean(rotate, k = 24, fill = NA, align = "right")) > 2*rollapply(rotate, width = 24, FUN = sd, fill = NA, align = "right"), 1, 0)) %>% # nolint
  ungroup()

#3. Time since last failure
merged_data <- merged_data %>%
  group_by(machineID) %>%
  mutate(time_since_last_failure = as.numeric(difftime(datetime, lag(datetime[failure != "No Failure"], default = first(datetime)), units = "hours"))) %>% # nolint
  ungroup()

#4. Time since last error in hours
merged_data <- merged_data %>%
  group_by(machineID) %>%
  mutate(time_since_last_error = as.numeric(difftime(datetime, lag(datetime[!is.na(errorID)], default = first(datetime)), units = "hours"))) %>% # nolint
  ungroup()

str(merged_data)
View(merged_data %>% slice(1:50))
write.csv(merged_data %>% slice(1:50), "C:/Users/rshaw/Desktop/EC Utbildning - Data Science/Kurs 9 - Project/Project/ds23_projektkurs/predictive-maintenance/models/Trained models/merged_data_w_eng_feat.csv", row.names = FALSE) #nolint

#EDA
# Step 1: Ensure datetime is in POSIXct format in both datasets
merged_data$datetime <- ymd_hms(merged_data$datetime)
maintenance$datetime <- ymd_hms(maintenance$datetime)

# Step 2: Summarize anomalies by machineID (already working code)
anomaly_summary <- merged_data %>%
  group_by(machineID) %>%
  summarize(
    pressure_anomalies = sum(pressure_anomaly == 1, na.rm = TRUE),
    volt_anomalies = sum(volt_anomaly == 1, na.rm = TRUE),
    vibration_anomalies = sum(vibration_anomaly == 1, na.rm = TRUE),
    rotate_anomalies = sum(rotate_anomaly == 1, na.rm = TRUE)
  )

View(anomaly_summary)

# Step 3: Join anomaly summary with machine metadata 
merged_anomalies <- anomaly_summary %>%
  left_join(machines, by = "machineID")

View(merged_anomalies)
write.csv(merged_anomalies, "C:/Users/rshaw/Desktop/EC Utbildning - Data Science/Kurs 9 - Project/Project/ds23_projektkurs/predictive-maintenance/models/Trained models/merged_anomalies.csv", row.names = FALSE) #nolint

# A. Summary statistics for anomalies and machine age
install.packages("summarytools")
library(summarytools)
dfSummary(merged_anomalies)

# B. Group the data by machine model and calculate mean anomalies
anomalies_by_model <- merged_anomalies %>%
  group_by(model) %>%
  summarise(
    avg_pressure_anomalies = mean(pressure_anomalies, na.rm = TRUE),
    avg_volt_anomalies = mean(volt_anomalies, na.rm = TRUE),
    avg_vibration_anomalies = mean(vibration_anomalies, na.rm = TRUE),
    avg_rotate_anomalies = mean(rotate_anomalies, na.rm = TRUE),
    avg_age = mean(age, na.rm = TRUE)
  )
print(anomalies_by_model)
View(anomalies_by_model)
write.csv(anomalies_by_model, "C:/Users/rshaw/Desktop/EC Utbildning - Data Science/Kurs 9 - Project/Project/ds23_projektkurs/predictive-maintenance/models/Trained models/anomalies_by_model.csv", row.names = FALSE) #nolint

# C. Compute correlation between anomalies and machine age
correlation_matrix <- merged_anomalies %>%
  select(pressure_anomalies, volt_anomalies, vibration_anomalies, rotate_anomalies, age) %>% #nolint
  cor(use = "complete.obs")
print(correlation_matrix)
View(correlation_matrix)
write.csv(correlation_matrix, "C:/Users/rshaw/Desktop/EC Utbildning - Data Science/Kurs 9 - Project/Project/ds23_projektkurs/predictive-maintenance/models/Trained models/anomalies_and_machine_age.csv", row.names = FALSE) #nolint

# 1. Anomaly Distribution Across Machines:
# Machines with high voltage anomalies are likely to have corresponding pressure issues. # nolint
# The range (difference between min and max) shows that some machines have significantly more anomalies (up to 384) than others (as few as 308), indicating variability in the anomaly count across machines.

# 2. Anomalies by Machine Model:
# Model 1 has the highest average pressure anomalies (358.8), which stands out compared to other models, 
# e.g., Model 4, which has 342.4 on average). This indicates that Model 1 might be more susceptible to pressure-related issues 
# and will likely require more frequent maintenace.

# 3. Anomalies and Machine Age: 
# Older machines are slightly more prone to pressure anomalies, but age does not seem to have a strong relationship with # nolint
# other types of anomalies (voltage, vibration and rotation). The correlation between machine age and pressure anomalies is +0.22, 
# which is a weak but positive correlation. This means that, generally, as machine age increases, the likelihood of pressure anomalies also increases.

# Plot machine age vs pressure anomalies
ggplot(merged_anomalies, aes(x = age, y = pressure_anomalies)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Relationship Between Machine Age and Pressure Anomalies",
       x = "Machine Age (years)", y = "Pressure Anomalies")

# Train/validation/test splits
# First we set the cutoff dates for the splits
train_cutoff <- as.Date("2015-09-30")       # Training data ends on 30 September 2015 #nolint
validation_cutoff <- as.Date("2015-11-30")  # Validation data ends on 30 November 2015 #nolint

train_data <- merged_data[merged_data$datetime <= train_cutoff, ]
validation_data <- merged_data[merged_data$datetime > train_cutoff & merged_data$datetime <= validation_cutoff, ] #nolint
test_data <- merged_data[merged_data$datetime > validation_cutoff, ]

#create comp 3 pressure variable as target variable instead of pressure

linear_model <- lm(pressure ~ volt + rotate + pressure_rolling_avg + vibration + error_count_24h + #nolint
                   time_since_last_failure + time_since_last_error + model + age + #nolint
                   pressure_anomaly + volt_anomaly + vibration_anomaly + rotate_anomaly, #nolint
                   data = train_data)
summary(linear_model)

validation_predictions <- predict(linear_model, newdata = validation_data)
test_predictions <- predict(linear_model, newdata = test_data)

validation_residuals <- validation_data$pressure - validation_predictions
test_residuals <- test_data$pressure - test_predictions

validation_mse <- mean(validation_residuals^2)
test_mse <- mean(test_residuals^2)

print(paste("Validation MSE:", validation_mse))
print(paste("Test MSE:", test_mse))

# This linear regression model is predicting pressure based on our set of features and sensor anomalies. # nolint
# It models normal pressure behavior while also checking if any anomalies have been detected in the sensor readings. #nolint
# By including the anomaly flags (pressure_anomaly, volt_anomaly, etc.) as features, the model explicitly learns how these abnormal #nolint
# sensor readings affect the pressure. This allows the model to better handle situations where anomalies influence pressure predictions. #nolint
# The model also detects anomalies indirectly by using high residuals (differences between predicted and actual pressure). #nolint
# However for comprehensive anomaly detection across all sensors, a dedicated RNN (or classification) model would be better. #nolint

# The model's residuals range from -55.066 to 56.611, indicating substantial variation in its ability to predict pressure accurately. #nolint
# This suggests that in many cases, the model struggles to correctly predict pressure, possibly due to unaccounted-for factors or anomalies. #nolint

# The pressure_anomaly feature has a strong positive effect (estimate: 0.221, p-value: 0.000578), highlighting that deviations in pressure are significant indicators of potential failure. #nolint
# However, the model is heavily focused on pressure and fails to account for anomalies in other #nolint
# sensors (voltage, vibration, rotation), limiting its scope. Also, the low adjusted R² value of 15.76% indicates that the model explains only a small #nolint
# portion of the variance in pressure, suggesting that it is not very robust in its pressure predictions and may require improvements to better capture the underlying patterns in the data. #nolint

# With time-series data from machine sensors, RNNs capture temporal dependencies, learning patterns over time #nolint
# from previous sensor readings like voltage, rotation, and pressure, making them well-suited for predictive maintenance. #nolint
# Advanced RNN variants, like LSTM (Long Short-Term Memory), handle nonlinear relationships #nolint
# better than linear models. This is useful when different sensor readings influence each other or when anomalies #nolint
# emerge gradually over time. By using an RNN, you can model not just individual data points but the sequences leading up #nolint
# to those points, improving the ability to predict anomalous behavior and/or failures. #nolint



