# ALY 6110: Data Management and Big Data
# Group 4: Khushi Doshi, Yash Tailor, Krutika Patel
# Method: Classification

cat("\014") # clears console
rm(list = ls()) # clears global environment
try(dev.off(dev.list()["RStudioGD"]), silent = TRUE) # clears plots
try(p_unload(p_loaded(), character.only = TRUE), silent = TRUE) # clears packages
options(scipen = 100) # disables scientific notation for entire R session

# Install necessary libraries if not already installed
#install.packages("janitor")
#install.packages("dplyr")
#install.packages("ggplot2")
#install.packages("scales")
#install.packages("corrplot")
#install.packages("naniar")
#install.packages("kernlab")
#install.packages("PRROC")

# Load libraries
library(janitor)
library(dplyr)
library(ggplot2)
library(scales)
library(corrplot)
library(naniar)
library(glmnet)
library(caret)
library(e1071)
library(randomForest)
library(kernlab)
library(PRROC)

# Load the dataset
spotify_youtube_dataset <- read.csv("Spotify_Youtube_Dataset.csv")

# Clean column names
spotify_youtube_dataset <- clean_names(spotify_youtube_dataset)

# Names of the columns
names(spotify_youtube_dataset)

# ------------------------ Missing values -------------------------------------------

# Check for missing values
sapply(spotify_youtube_dataset, function(x) sum(is.na(x)))

# Visualize missing data
gg_miss_var(spotify_youtube_dataset) +
  labs(title = "Missing Values by Variable", x = "Variables", y = "Number of Missing Values") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5))

# Handling Missing Data with imputation

# 1. Track-Specific Features: Impute with median
spotify_youtube_dataset <- spotify_youtube_dataset %>%
  mutate(across(c(danceability, energy, key, loudness, speechiness, acousticness, instrumentalness,
                  liveness, valence, tempo, duration_ms),
                ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# 2. Engagement Metrics: Add missing indicator columns and then impute
spotify_youtube_dataset <- spotify_youtube_dataset %>%
  mutate(views_missing = ifelse(is.na(views), 1, 0),
         likes_missing = ifelse(is.na(likes), 1, 0),
         comments_missing = ifelse(is.na(comments), 1, 0),
         stream_missing = ifelse(is.na(stream), 1, 0)) %>%
  mutate(views = ifelse(is.na(views), 0, views),    # Impute missing views with 0
         likes = ifelse(is.na(likes), 0, likes),    # Impute missing likes with 0
         comments = ifelse(is.na(comments), 0, comments),  # Impute missing comments with 0
         stream = ifelse(is.na(stream), 0, stream))  # Impute missing stream with 0

# Outlier detection and handling using 1st and 99th percentiles
cap_outliers <- function(x) {
  q1 <- quantile(x, 0.01)
  q99 <- quantile(x, 0.99)
  x <- ifelse(x < q1, q1, x)
  x <- ifelse(x > q99, q99, x)
  return(x)
}

# Apply the function to numerical columns
spotify_youtube_dataset <- spotify_youtube_dataset %>%
  mutate(across(where(is.numeric), cap_outliers))

# Min-Max Scaling for numerical columns
spotify_youtube_dataset <- spotify_youtube_dataset %>%
  mutate(across(where(is.numeric), ~ rescale(., to = c(0, 1))))

# Check for missing values
sapply(spotify_youtube_dataset, function(x) sum(is.na(x)))

# Overwrite the original dataset variable with the cleaned dataset only
spotify_youtube_dataset <- spotify_youtube_dataset

# Save the cleaned dataset to a CSV file
write.csv(spotify_youtube_dataset, "Spotify_Youtube_Dataset_Cleaned.csv", row.names = FALSE)

# ------------------------ Correlation and Visualizations -------------------------------------------

# Compute correlation matrix for numerical columns
numeric_columns <- select(spotify_youtube_dataset, where(is.numeric))
corr_matrix <- cor(numeric_columns, use = "complete.obs")

# Enhanced Correlation Matrix
corrplot(corr_matrix, method = "color", type = "upper", 
         col = colorRampPalette(c("#FFEDA0", "#FEB24C", "#F03B20"))(200),
         title = "Correlation Matrix", addCoef.col = "black",
         mar = c(0,0,1,0), tl.col = "black", tl.srt = 45, number.cex = 0.6)

# Scatter plot for Views vs Likes if they are highly correlated
ggplot(spotify_youtube_dataset, aes(x = views, y = likes)) +
  geom_point(alpha = 0.5, color = "dodgerblue") +
  labs(title = "Scatter plot of Views vs Likes", x = "Views", y = "Likes") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5)) +
  scale_x_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale()))


# ------------------------ Enhanced EDA Visualizations -------------------------------------------

# Example: Views vs Likes (Color by Stream for better effect)
ggplot(spotify_youtube_dataset, aes(x = views, y = likes, color = stream)) +
  geom_point(alpha = 0.6) +
  scale_color_gradient(low = "yellow", high = "red") +
  labs(title = "Views vs Likes (Colored by Stream)", x = "Views", y = "Likes") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5)) +
  scale_x_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale()))

# Example: Energy vs Danceability (Color by Views)
ggplot(spotify_youtube_dataset, aes(x = danceability, y = energy, color = views)) +
  geom_point(alpha = 0.6) +
  scale_color_gradient(low = "yellow", high = "blue") +
  labs(title = "Energy vs Danceability (Colored by Views)", x = "Danceability", y = "Energy") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5)) +
  scale_x_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale()))

# ------------------------ Density Plots -------------------------------------------

# Density plot of Danceability
ggplot(spotify_youtube_dataset, aes(x = danceability)) +
  geom_density(fill = "skyblue", alpha = 0.7) +
  labs(title = "Density Plot of Danceability", x = "Danceability", y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5)) +
  scale_x_continuous(labels = label_number(scale_cut = cut_short_scale()))

# Density plot of Energy
ggplot(spotify_youtube_dataset, aes(x = energy)) +
  geom_density(fill = "lightgreen", alpha = 0.7) +
  labs(title = "Density Plot of Energy", x = "Energy", y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5)) +
  scale_x_continuous(labels = label_number(scale_cut = cut_short_scale()))

# ------------------------ Bar Plots for Categorical Variables -------------------------------------------

# Frequency of Album Types
ggplot(spotify_youtube_dataset, aes(x = album_type)) +
  geom_bar(fill = "steelblue", color = "black") +
  labs(title = "Frequency of Album Types", x = "Album Type", y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5)) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale()))

# Calculate the top 10 artists based on total views
top_artists_views <- spotify_youtube_dataset %>%
  group_by(artist) %>%
  summarize(total_views = sum(views, na.rm = TRUE)) %>%
  arrange(desc(total_views)) %>%
  slice(1:10)  # Select the top 10 artists with the highest views

top_artists_views

# Plot the top 10 artists based on total views
ggplot(top_artists_views, aes(x = reorder(artist, total_views), y = total_views)) +
  geom_bar(stat = "identity", fill = "#FFA500", color = "black", width = 0.7) +
  coord_flip() +
  labs(title = "Top 10 Artists by Total Views", x = "Artist", y = "Total Views") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.title.x = element_text(size = 10),
    axis.title.y = element_text(size = 10),
    axis.text.x = element_text(size = 8),
    axis.text.y = element_text(size = 9),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_blank()
  ) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale()))


# -------------------------------------------------------------------------------

# Load necessary libraries
library(dplyr)
library(caret)
library(randomForest)
library(e1071)

# ------------------------ Step 1: Data Preparation -------------------------------------------

# Create engagement score and target variable
spotify_youtube_dataset <- spotify_youtube_dataset %>%
  mutate(engagement_score = views + likes + comments + stream,
         is_popular = ifelse(engagement_score > median(engagement_score, na.rm = TRUE), 1, 0))  # 1 = popular, 0 = not popular

# Convert 'is_popular' to factor for classification
spotify_youtube_dataset$is_popular <- as.factor(spotify_youtube_dataset$is_popular)

# Standardize numerical features for Logistic Regression and SVM
scaled_data <- spotify_youtube_dataset %>%
  mutate(across(where(is.numeric), ~ scale(.) %>% as.numeric()))

# ------------------------ Feature Selection -------------------------------------------
# Create engagement score and target variable
spotify_youtube_dataset <- spotify_youtube_dataset %>%
  mutate(engagement_score = views + likes + comments + stream,
         is_popular = ifelse(engagement_score > median(engagement_score, na.rm = TRUE), 1, 0))

spotify_youtube_dataset$is_popular <- as.factor(spotify_youtube_dataset$is_popular)

scaled_data <- spotify_youtube_dataset %>%
  mutate(across(where(is.numeric), ~ scale(.) %>% as.numeric()))

# Calculate correlation matrix and remove highly correlated features (>0.8)
cor_matrix <- cor(select(scaled_data, where(is.numeric)))
highly_correlated <- findCorrelation(cor_matrix, cutoff = 0.8)
scaled_data <- scaled_data %>% select(-all_of(highly_correlated))

# ------------------------ PCA with Explained Variance Threshold ------------------------
# Perform PCA, retaining components up to 95% variance
numeric_data <- select(scaled_data, where(is.numeric))
pca_result <- prcomp(numeric_data, center = TRUE, scale. = TRUE)
explained_variance <- summary(pca_result)$importance[2, ]
components_to_retain <- which(cumsum(explained_variance) >= 0.95)[1]

# Retain necessary PCA components based on explained variance
reduced_data <- data.frame(pca_result$x[, 1:components_to_retain], is_popular = scaled_data$is_popular)

# ------------------------ Data Splitting -------------------------------------------
set.seed(123)
trainIndex <- createDataPartition(reduced_data$is_popular, p = 0.8, list = FALSE)
train_data <- reduced_data[trainIndex, ]
test_data <- reduced_data[-trainIndex, ]

# ------------------------ Fit and Tune SVM Models (Linear, Radial, Polynomial) ------------------------
# Use a smaller subset for initial tuning to save memory
set.seed(123)
sampled_train_data <- train_data[sample(1:nrow(train_data), 5000), ]  # Reduce tuning set size

# Linear Kernel
tune_linear <- tune.svm(is_popular ~ ., data = sampled_train_data, type = 'C-classification',
                        kernel = 'linear', cost = c(.01, .1, .5, 1, 2.5, 5), scale = FALSE)

best_cost_linear <- tune_linear$best.parameters$cost

classifier_linear <- svm(is_popular ~ ., data = train_data, type = 'C-classification',
                         kernel = 'linear', cost = best_cost_linear, probability = TRUE)

y_pred_linear <- predict(classifier_linear, newdata = test_data[-ncol(test_data)])
linear_accuracy <- mean(y_pred_linear == test_data$is_popular)
cat("Tuned Linear Kernel SVM Accuracy:", linear_accuracy, "\n")

# Radial Kernel
tune_radial <- tune.svm(is_popular ~ ., data = sampled_train_data, type = 'C-classification',
                        kernel = 'radial', cost = c(.01, .1, .5, 1, 2.5, 5), gamma = c(0.1, .5, 1), scale = FALSE)
best_cost_radial <- tune_radial$best.parameters$cost
best_gamma_radial <- tune_radial$best.parameters$gamma

classifier_radial <- svm(is_popular ~ ., data = train_data, type = 'C-classification',
                         kernel = 'radial', cost = best_cost_radial, gamma = best_gamma_radial, probability = TRUE)

y_pred_radial <- predict(classifier_radial, newdata = test_data[-ncol(test_data)])
radial_accuracy <- mean(y_pred_radial == test_data$is_popular)
cat("Tuned Radial Kernel SVM Accuracy:", radial_accuracy, "\n")

# Polynomial Kernel
tune_polynomial <- tune.svm(is_popular ~ ., data = sampled_train_data, type = 'C-classification',
                            kernel = 'polynomial', cost = c(.01, .1, .5, 1, 2.5, 5), gamma = c(0.1, .5, 1), scale = FALSE)
best_cost_poly <- tune_polynomial$best.parameters$cost
best_gamma_poly <- tune_polynomial$best.parameters$gamma

classifier_poly <- svm(is_popular ~ ., data = train_data, type = 'C-classification',
                       kernel = 'polynomial', cost = best_cost_poly, gamma = best_gamma_poly, probability = TRUE)

y_pred_poly <- predict(classifier_poly, newdata = test_data[-ncol(test_data)])
poly_accuracy <- mean(y_pred_poly == test_data$is_popular)
cat("Tuned Polynomial Kernel SVM Accuracy:", poly_accuracy, "\n")

# ------------------------ Logistic Regression with Sparse Matrix ------------------------
x_train <- sparse.model.matrix(is_popular ~ ., data = train_data)[, -1]
y_train <- train_data$is_popular

x_test <- sparse.model.matrix(is_popular ~ ., data = test_data)[, -1]
y_test <- test_data$is_popular

cv_lasso <- cv.glmnet(x_train, as.numeric(y_train) - 1, family = "binomial", alpha = 1)
best_lambda <- cv_lasso$lambda.min
lasso_model <- glmnet(x_train, as.numeric(y_train) - 1, family = "binomial", alpha = 1, lambda = best_lambda)

test_pred_prob <- predict(lasso_model, s = best_lambda, newx = x_test, type = "response")
test_pred_class <- ifelse(test_pred_prob > 0.5, 1, 0)

# ------------------- Random Forest -----------------------------------
train_data$is_popular <- as.factor(train_data$is_popular)
test_data$is_popular <- as.factor(test_data$is_popular)

tune_grid <- expand.grid(mtry = c(2, 3))
train_control <- trainControl(method = "cv", number = 5)

rf_tuned <- train(is_popular ~ ., data = train_data, method = "rf", tuneGrid = tune_grid,
                  trControl = train_control, ntree = 50, importance = TRUE)

rf_predictions <- predict(rf_tuned, newdata = test_data)


#-------------------------Model Evaluation--------------------------------------


#Define a function to calculate performance metrics
calc_metrics <- function(true, pred) {
  cm <- confusionMatrix(pred, true)
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Pos Pred Value"] # Precision
  recall <- cm$byClass["Sensitivity"]       # Recall
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  return(c(accuracy, precision, recall, f1))
}

# ------------------------ SVM Model Performance Metrics ------------------------
# Linear SVM Metrics
y_pred_linear <- predict(classifier_linear, newdata = test_data[-ncol(test_data)])
linear_metrics <- calc_metrics(test_data$is_popular, y_pred_linear)
cat("Linear Kernel SVM Metrics:\n", linear_metrics, "\n")

# Radial SVM Metrics
y_pred_radial <- predict(classifier_radial, newdata = test_data[-ncol(test_data)])
radial_metrics <- calc_metrics(test_data$is_popular, y_pred_radial)
cat("Radial Kernel SVM Metrics:\n", radial_metrics, "\n")

# Polynomial SVM Metrics
y_pred_poly <- predict(classifier_poly, newdata = test_data[-ncol(test_data)])
poly_metrics <- calc_metrics(test_data$is_popular, y_pred_poly)
cat("Polynomial Kernel SVM Metrics:\n", poly_metrics, "\n")

# ------------------------ Logistic Regression Performance Metrics ------------------------
test_pred_class <- as.factor(ifelse(test_pred_prob > 0.5, 1, 0))
logistic_metrics <- calc_metrics(y_test, test_pred_class)
cat("Logistic Regression Metrics:\n", logistic_metrics, "\n")

# ------------------------ Random Forest Performance Metrics ------------------------
rf_metrics <- calc_metrics(test_data$is_popular, rf_predictions)
cat("Random Forest Metrics:\n", rf_metrics, "\n")

# ------------------------ Model Comparison Summary ------------------------
model_comparison <- data.frame(
  Model = c("SVM - Linear", "SVM - Radial", "SVM - Polynomial", "Logistic Regression", "Random Forest"),
  Accuracy = c(linear_metrics[1], radial_metrics[1], poly_metrics[1], logistic_metrics[1], rf_metrics[1]),
  Precision = c(linear_metrics[2], radial_metrics[2], poly_metrics[2], logistic_metrics[2], rf_metrics[2]),
  Recall = c(linear_metrics[3], radial_metrics[3], poly_metrics[3], logistic_metrics[3], rf_metrics[3]),
  F1_Score = c(linear_metrics[4], radial_metrics[4], poly_metrics[4], logistic_metrics[4], rf_metrics[4])
)

print("Model Comparison Summary:")
print(model_comparison)


 # Train the final model on the full training data

# Finalize and Save the Logistic Regression Model

# Train the final model on the full training data
final_x_train <- sparse.model.matrix(is_popular ~ ., data = train_data)[, -1]
final_y_train <- as.numeric(train_data$is_popular) - 1

# Cross-validation to get the best lambda again
set.seed(123)
final_cv_lasso <- cv.glmnet(final_x_train, final_y_train, family = "binomial", alpha = 1)
final_best_lambda <- final_cv_lasso$lambda.min

# Fit the final Logistic Regression model with Lasso Regularization using the best lambda
final_logistic_model <- glmnet(final_x_train, final_y_train, family = "binomial", alpha = 1, lambda = final_best_lambda)

# Save the model for future use
saveRDS(final_logistic_model, file = "final_logistic_model.rds")

# Predict on the test data for confirmation of final performance
final_x_test <- sparse.model.matrix(is_popular ~ ., data = test_data)[, -1]
final_test_pred_prob <- predict(final_logistic_model, s = final_best_lambda, newx = final_x_test, type = "response")
final_test_pred_class <- as.factor(ifelse(final_test_pred_prob > 0.5, 1, 0))

# Evaluate the final model performance on the test data
final_metrics <- calc_metrics(test_data$is_popular, final_test_pred_class)
cat("Final Logistic Regression Model Performance:\n")
cat("Accuracy:", final_metrics[1], "\n")
cat("Precision:", final_metrics[2], "\n")
cat("Recall:", final_metrics[3], "\n")
cat("F1 Score:", final_metrics[4], "\n")


# ------------- K Fold cross-validations --------------------


# Define the number of folds for cross-validation
set.seed(123)
k <- 5  # You can adjust k for different folds (e.g., 5 or 10)

# Prepare data for cross-validation
x_data <- sparse.model.matrix(is_popular ~ ., data = reduced_data)[, -1]
y_data <- as.numeric(reduced_data$is_popular) - 1

# Set up cross-validation control for k-folds
cv_control <- trainControl(method = "cv", number = k, verboseIter = TRUE)

# Perform k-fold cross-validation on Logistic Regression with Lasso
cv_model <- train(
  x = x_data, y = as.factor(y_data),
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, lambda = final_best_lambda),  # Lasso regularization
  trControl = cv_control,
  family = "binomial"
)

# Print cross-validation results
print("Cross-Validation Results:")
print(cv_model$results)

# Average cross-validation accuracy
cv_accuracy <- mean(cv_model$resample$Accuracy)
cat("Average Cross-Validation Accuracy:", cv_accuracy, "\n")

# Check consistency across folds
cat("Accuracy Standard Deviation Across Folds:", sd(cv_model$resample$Accuracy), "\n")

# k-fold cross-validation for SVM 


# Set up cross-validation control
set.seed(123)
cv_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)  # 5-fold cross-validation

# Perform cross-validation on SVM with Linear Kernel
svm_linear_cv <- train(
  is_popular ~ ., data = train_data,
  method = "svmLinear",
  trControl = cv_control,
  tuneGrid = expand.grid(C = c(.01, .1, .5, 1, 2.5, 5))
)

# Print SVM Linear CV Results
print("SVM Linear Kernel Cross-Validation Results:")
print(svm_linear_cv$results)

# Average cross-validation accuracy for SVM Linear
svm_linear_cv_accuracy <- mean(svm_linear_cv$resample$Accuracy)
cat("Average Cross-Validation Accuracy for SVM Linear:", svm_linear_cv_accuracy, "\n")

# Perform cross-validation on SVM with Radial Kernel
svm_radial_cv <- train(
  is_popular ~ ., data = train_data,
  method = "svmRadial",
  trControl = cv_control,
  tuneGrid = expand.grid(C = c(.01, .1, .5, 1, 2.5, 5), sigma = c(0.1, 0.5, 1))
)

# Print SVM Radial CV Results
print("SVM Radial Kernel Cross-Validation Results:")
print(svm_radial_cv$results)

# Average cross-validation accuracy for SVM Radial
svm_radial_cv_accuracy <- mean(svm_radial_cv$resample$Accuracy)
cat("Average Cross-Validation Accuracy for SVM Radial:", svm_radial_cv_accuracy, "\n")

# Perform cross-validation on SVM with Polynomial Kernel
svm_poly_cv <- train(
  is_popular ~ ., data = train_data,
  method = "svmPoly",
  trControl = cv_control,
  tuneGrid = expand.grid(C = c(.01, .1, .5, 1, 2.5, 5), 
                         degree = c(2, 3, 4),  # polynomial degrees to try
                         scale = c(0.01, 0.1, 1))  # scale parameter for polynomial kernel
)

# Print SVM Polynomial Kernel CV Results
print("SVM Polynomial Kernel Cross-Validation Results:")
print(svm_poly_cv$results)

# Average cross-validation accuracy for SVM Polynomial
svm_poly_cv_accuracy <- mean(svm_poly_cv$resample$Accuracy)
cat("Average Cross-Validation Accuracy for SVM Polynomial:", svm_poly_cv_accuracy, "\n")



# k-fold cross-validation for Random Forest

# Set up cross-validation control for Random Forest
cv_control_rf <- trainControl(method = "cv", number = 5, verboseIter = TRUE)  # 5-fold cross-validation

# Perform cross-validation on Random Forest
set.seed(123)
rf_cv <- train(
  is_popular ~ ., data = train_data,
  method = "rf",
  trControl = cv_control_rf,
  tuneGrid = expand.grid(mtry = c(2, 3, 4)),
  ntree = 50  # You can increase the number of trees if needed
)

# Print Random Forest CV Results
print("Random Forest Cross-Validation Results:")
print(rf_cv$results)

# Average cross-validation accuracy for Random Forest
rf_cv_accuracy <- mean(rf_cv$resample$Accuracy)
cat("Average Cross-Validation Accuracy for Random Forest:", rf_cv_accuracy, "\n")

