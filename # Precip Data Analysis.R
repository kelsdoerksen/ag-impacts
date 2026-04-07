# Precip Data Analysis

# Loading libraries
library(terra)
library(sf)
library(haven)      # for reading .dta files
library(dplyr)
library(tidyr)
# Load precip sample data
precip_data <- readRDS('/Volumes/PRO-G40/UCT-data/AgImpacts/DCPP_tas_CanESM5_obs-ERA5_fyear1.rds')

# Basic investigation of the data
str(precip_data, max.level = 2)

# Unwrap the 1965 data from ensemble_40
precip_1965 <- terra::unwrap(precip_data$ensemble_40$`1965`)

# Now look at it
plot(precip_1965)


# ===== BASIC EXPLORATION =====

# View monthly layer names (probably months 1-12)
names(precip_1965)

# Rename layers to be clearer
names(precip_1965) <- month.abb  # Jan, Feb, Mar, etc.
precip_1965

# Summary statistics for each month
summary(precip_1965)

# Global mean precipitation by month
global(precip_1965, "mean", na.rm = TRUE)


# ===== VISUALIZATION =====

# Plot all 12 months
plot(precip_1965, 
     main = paste("Monthly Precipitation -", month.abb),
     col = hcl.colors(100, "Blues", rev = TRUE))

# Plot just January
plot(precip_1965[[1]], 
     main = "January 1965 Precipitation",
     col = hcl.colors(100, "Blues", rev = TRUE))

# Interactive plot (if you want to zoom/pan)
library(leaflet)
library(terra)
plot(precip_1965[[1]])


# ===== TIME SERIES ANALYSIS =====

# Extract mean precipitation for entire region by month
monthly_mean <- global(precip_1965, "mean", na.rm = TRUE)
monthly_mean$month <- 1:12

# Plot seasonal pattern
plot(monthly_mean$month, monthly_mean$mean,
     type = "b",
     xlab = "Month",
     ylab = "Mean Precipitation",
     main = "Seasonal Pattern 1965",
     col = "steelblue",
     pch = 19)

# With ggplot2 (prettier)
library(ggplot2)
ggplot(monthly_mean, aes(x = month, y = mean)) +
  geom_line(color = "steelblue", size = 1) +
  geom_point(color = "steelblue", size = 3) +
  scale_x_continuous(breaks = 1:12, labels = month.abb) +
  theme_minimal() +
  labs(title = "Mean Precipitation by Month (1965)",
       x = "Month",
       y = "Precipitation (mm)")


# ===== EXTRACT VALUES AT SPECIFIC LOCATIONS =====

# Example: extract at a point (lon, lat)
point <- vect(cbind(-10, 20), crs = "EPSG:4326")
extract(precip_1965, point)